"""Tiny format fixtures and Streamlit registration checks."""
from io import BytesIO
from pathlib import Path

import duckdb
import pytest
import streamlit as st
from openpyxl import Workbook
from pandas.testing import assert_frame_equal
import pandas as pd
from streamlit.testing.v1 import AppTest

from app.file_handlers.csv_handler import load_csv_bytes
from app.file_handlers.detect_format import detect_format, read_upload
from app.file_handlers.excel_handler import load_excel_bytes, sheet_names
from app.file_handlers.pdf_handler import load_pdf_bytes
from app.file_handlers.txt_handler import load_txt_bytes

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def upload(name):
    data = (FIXTURES / name).read_bytes()
    item = BytesIO(data)
    item.name, item.size, item.file_id = name, len(data), name
    return item


def test_csv_delimiter_and_latin1_keep_zero_padded_ids():
    frame = load_csv_bytes((FIXTURES / "semicolon_ids.csv").read_bytes(), "ids.csv")[0]["dataframe"]
    assert frame.customer_id.tolist() == ["0012", "0013"]
    assert frame.amount.tolist() == ["10", "20"]
    assert load_csv_bytes(b"name,city\nAna,caf\xe9\n", "latin.csv")[0]["dataframe"].city[0] == "caf\u00e9"


@pytest.mark.parametrize("delimiter,encoding", [(";", "utf-8"), ("\t", "utf-8-sig"),
                                                 ("|", "latin-1"), (",", "utf-8")])
def test_csv_delimiters_and_encodings_preserve_frames(delimiter, encoding):
    text = f"id{delimiter}city\n001{delimiter}café\n002{delimiter}Lima\n"
    frame = load_csv_bytes(text.encode(encoding), "places.csv")[0]["dataframe"]
    assert_frame_equal(frame, pd.DataFrame({"id": ["001", "002"],
                                             "city": ["café", "Lima"]}))


def test_unreadable_upload_and_unsupported_extension():
    class BrokenUpload:
        name = "broken.csv"
        size = 1

        def getvalue(self):
            raise OSError("unreadable")

    with pytest.raises(ValueError, match="Could not read"):
        read_upload(BrokenUpload())
    with pytest.raises(ValueError, match="empty"):
        read_upload(BytesIO())
    with pytest.raises(ValueError, match="Unsupported file type"):
        detect_format("image.png", b"plain text")


def test_pdf_table_text_and_corruption_fixtures():
    table = load_pdf_bytes((FIXTURES / "table_report.pdf").read_bytes(), "table_report.pdf")
    assert [part["name"] for part in table if "dataframe" in part] == ["table_report_p2_t1"]
    assert_frame_equal([part["dataframe"] for part in table if "dataframe" in part][0],
                       pd.DataFrame({"ID": ["0012", "0013"], "Amount": ["10", "20"]}))
    text = load_pdf_bytes((FIXTURES / "text_report.pdf").read_bytes(), "text_report.pdf")
    assert "Quarterly narrative" in text[0]["text"] and "dataframe" not in text[0]
    with pytest.raises(ValueError, match="Could not read PDF"):
        load_pdf_bytes((FIXTURES / "corrupt_report.pdf").read_bytes(), "corrupt_report.pdf")


def test_excel_signature_and_two_selected_sheets():
    source = upload("two_sheets.xlsx")
    before = source.getvalue()
    assert detect_format("wrong.csv", before)[0] == "xlsx"
    assert sheet_names(before) == ["Orders", "People"]
    parts = load_excel_bytes(before, source.name, ["Orders", "People"])
    assert [item["name"] for item in parts] == ["two_sheets.xlsx / Orders", "two_sheets.xlsx / People"]
    assert parts[0]["dataframe"].customer_id.tolist() == ["0012", "0013"]
    assert read_upload(source) == before and source.getvalue() == before


def test_excel_formula_text_is_not_silently_lost():
    book = Workbook()
    book.active.append(["id", "calculation"])
    book.active.append(["0012", "=1+2"])
    buffer = BytesIO()
    book.save(buffer)
    frame = load_excel_bytes(buffer.getvalue(), "formulas.xlsx", ["Sheet"])[0]["dataframe"]
    assert frame.iloc[0].tolist() == ["0012", "=1+2"]


def test_txt_table_or_complete_document():
    table = load_txt_bytes((FIXTURES / "structured_notes.txt").read_bytes(), "notes.txt")[0]
    assert table["dataframe"].item.tolist() == ["001", "", "002"]
    original = (FIXTURES / "document_notes.txt").read_bytes().decode("utf-8")
    assert load_txt_bytes(original.encode(), "document.txt") == [{"name": "document.txt", "text": original}]


def test_empty_binary_and_pdf_are_detected_clearly():
    with pytest.raises(ValueError, match="empty"):
        detect_format("empty.csv", b"")
    with pytest.raises(ValueError, match="not a supported"):
        detect_format("image.txt", b"\x00\x01binary")
    assert detect_format("disguised.csv", b"%PDF-1.7")[0] == "pdf"


@pytest.mark.parametrize("filename,expected_tables,expected_documents", [
    ("two_sheets.xlsx", 2, 0),
    ("structured_notes.txt", 1, 0),
    ("document_notes.txt", 0, 1),
    ("semicolon_ids.csv", 1, 0),
])
def test_streamlit_upload_registers_each_table(monkeypatch, filename, expected_tables, expected_documents):
    monkeypatch.delenv("APP_ACCESS_PASSWORD", raising=False)
    selected = upload(filename)
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: selected if kwargs.get("accept_multiple_files") else None)
    app = AppTest.from_file(str(ROOT / "app/main.py")).run(timeout=25)
    assert not app.exception
    assert not app.session_state.upload_error
    assert len(app.session_state.documents) == expected_documents
    catalog = app.session_state.dataset_catalog
    assert (len(catalog.profile.datasets) if catalog else 0) == expected_tables
    if catalog:
        connection = duckdb.connect(str(catalog.path), read_only=True)
        try:
            for record in catalog.profile.datasets:
                assert connection.execute(f'SELECT COUNT(*) FROM "{record.table_name}"').fetchone()[0] == record.row_count
        finally:
            connection.close()
            app.session_state.agent.release_dataset()
    else:
        assert app.session_state.agent is None
        assert any("TXT document preview" in item.value for item in app.info)


def test_sheet_selection_replaces_tables_and_keeps_text_ids(monkeypatch):
    monkeypatch.delenv("APP_ACCESS_PASSWORD", raising=False)
    selected = upload("two_sheets.xlsx")
    monkeypatch.setattr(st, "file_uploader", lambda *a, **kw: selected if kw.get("accept_multiple_files") else None)
    app = AppTest.from_file(str(ROOT / "app/main.py")).run(timeout=25)
    assert len(app.session_state.dataset_catalog.profile.datasets) == 2
    app.multiselect[0].set_value(["Orders"]).run(timeout=25)
    assert not app.exception
    catalog = app.session_state.dataset_catalog
    assert [item.filename for item in catalog.profile.datasets] == ["two_sheets.xlsx / Orders"]
    table = catalog.profile.datasets[0].table_name
    connection = duckdb.connect(str(catalog.path), read_only=True)
    try:
        assert connection.execute(f'SELECT customer_id FROM "{table}" ORDER BY customer_id').fetchall() == [("0012",), ("0013",)]
    finally:
        connection.close()
        app.session_state.agent.release_dataset()


def test_mixed_upload_keeps_document_beside_table(monkeypatch):
    monkeypatch.delenv("APP_ACCESS_PASSWORD", raising=False)
    selected = [upload("structured_notes.txt"), upload("document_notes.txt")]
    monkeypatch.setattr(st, "file_uploader", lambda *a, **kw: selected if kw.get("accept_multiple_files") else None)
    app = AppTest.from_file(str(ROOT / "app/main.py")).run(timeout=25)
    assert not app.exception
    assert len(app.session_state.dataset_catalog.profile.datasets) == 1
    assert app.session_state.documents[0]["text"].startswith("Meeting notes")
    app.session_state.agent.release_dataset()


def test_mismatched_extension_warns_and_uses_xlsx(monkeypatch):
    monkeypatch.delenv("APP_ACCESS_PASSWORD", raising=False)
    selected = upload("two_sheets.xlsx")
    selected.name = "renamed.csv"
    monkeypatch.setattr(st, "file_uploader", lambda *a, **kw: selected if kw.get("accept_multiple_files") else None)
    app = AppTest.from_file(str(ROOT / "app/main.py")).run(timeout=25)
    assert not app.exception
    assert len(app.session_state.dataset_catalog.profile.datasets) == 2
    assert any("using XLSX" in warning.value for warning in app.warning)
    app.session_state.agent.release_dataset()
