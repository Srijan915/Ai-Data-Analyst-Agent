"""PDF extraction and Streamlit registration with small reportlab fixtures."""
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import duckdb
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from app.file_handlers.detect_format import detect_format
from app.file_handlers.pdf_handler import _table_frame, load_pdf_bytes

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def fixture_bytes(name):
    return (FIXTURES / name).read_bytes()


def uploaded_pdf(name):
    item = BytesIO(fixture_bytes(name))
    item.name, item.size, item.file_id = name, len(item.getvalue()), name
    return item


def pdf_app(monkeypatch, name):
    monkeypatch.delenv("APP_ACCESS_PASSWORD", raising=False)
    upload = uploaded_pdf(name)
    monkeypatch.setattr(st, "file_uploader", lambda *a, **kw: upload if kw.get("accept_multiple_files") else None)
    app = AppTest.from_file(str(ROOT / "app/main.py")).run(timeout=25)
    assert not app.exception
    return app


def test_table_pdf_keeps_page_text_and_registerable_rows():
    parts = load_pdf_bytes(fixture_bytes("table_report.pdf"), "table_report.pdf")
    tables = [part for part in parts if "dataframe" in part]
    texts = [part for part in parts if "text" in part]
    assert [part["name"] for part in tables] == ["table_report_p2_t1"]
    assert tables[0]["dataframe"].to_dict("records") == [
        {"ID": "0012", "Amount": "10"}, {"ID": "0013", "Amount": "20"}]
    assert [part["name"] for part in texts] == ["table_report_p1_text", "table_report_p2_text"]


def test_text_only_and_no_text_pdf():
    text = load_pdf_bytes(fixture_bytes("text_report.pdf"), "text_report.pdf")
    assert len(text) == 1 and "Quarterly narrative" in text[0]["text"]
    assert "dataframe" not in text[0]
    blank = load_pdf_bytes(fixture_bytes("no_text_report.pdf"), "no_text_report.pdf")
    assert blank[0]["notice"] == "No extractable text found (OCR is not supported)"


def test_corrupt_pdf_has_clear_error():
    with pytest.raises(ValueError, match="Could not read PDF"):
        load_pdf_bytes(fixture_bytes("corrupt_report.pdf"), "corrupt_report.pdf")
    with pytest.raises(ValueError, match="TXT or PDF"):
        detect_format("unknown.png", b"image data")


def test_unclear_header_uses_generic_columns_without_dropping_first_row():
    frame = _table_frame([["ID", " id "], ["0012", "Ana"]])
    assert frame.columns.tolist() == ["column_1", "column_2"]
    assert frame.iloc[0].tolist() == ["ID", " id "]
    assert _table_frame([[None, "Amount"], ["0012", "10"]]).iloc[0, 1] == "Amount"


def test_page_failure_is_recorded_and_later_pages_continue():
    broken = Mock()
    broken.extract_tables.side_effect = ValueError("damaged page")
    good = Mock()
    good.extract_tables.return_value = [[["ID", "Amount"], ["0012", "10"]]]
    good.extract_text.return_value = "Later page text"
    document = MagicMock()
    document.pages = [broken, good]
    document.__enter__.return_value = document
    with patch("app.file_handlers.pdf_handler.pdfplumber.open", return_value=document):
        parts = load_pdf_bytes(b"%PDF-fake", "report.pdf")
    assert any(part.get("name") == "report_p2_t1" for part in parts)
    assert [part["page_errors"] for part in parts if "page_errors" in part] == [
        [{"name": "report.pdf", "page": 1, "message": "damaged page"}]]


def test_streamlit_pdf_table_and_text_use_separate_paths(monkeypatch):
    app = pdf_app(monkeypatch, "table_report.pdf")
    catalog = app.session_state.dataset_catalog
    assert [record.filename for record in catalog.profile.datasets] == ["table_report_p2_t1"]
    assert len(app.session_state.documents) == 2
    table = catalog.profile.datasets[0].table_name
    connection = duckdb.connect(str(catalog.path), read_only=True)
    try:
        assert connection.execute(f'SELECT "ID" FROM "{table}" ORDER BY "ID"').fetchall() == [
            ("0012",), ("0013",)]
    finally:
        connection.close()
        app.session_state.agent.release_dataset()


def test_streamlit_text_only_and_image_only_pdf(monkeypatch):
    app = pdf_app(monkeypatch, "text_report.pdf")
    assert app.session_state.dataset_catalog is None
    assert len(app.session_state.documents) == 1
    assert any("PDF text preview" in item.value for item in app.info)
    app = pdf_app(monkeypatch, "no_text_report.pdf")
    assert app.session_state.dataset_catalog is None
    assert any("OCR is not supported" in item.value for item in app.warning)


def test_streamlit_corrupt_pdf_and_failed_page_list(monkeypatch):
    app = pdf_app(monkeypatch, "corrupt_report.pdf")
    assert app.session_state.upload_error
    assert any("Could not read PDF" in item.value for item in app.error)
    app = pdf_app(monkeypatch, "text_report.pdf")
    app.session_state.pdf_page_errors = [{"name": "report.pdf", "page": 2, "message": "damaged page"}]
    app.run(timeout=15)
    assert any("Pages that failed" in item.value for item in app.warning)
    assert any("page 2: damaged page" in item.value for item in app.caption)
