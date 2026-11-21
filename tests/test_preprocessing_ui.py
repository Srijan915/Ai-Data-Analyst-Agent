"""Real Streamlit upload and preprocessing flow, without provider calls."""
from io import BytesIO
from pathlib import Path

import streamlit as st
from streamlit.testing.v1 import AppTest

from app.core.dataset_catalog import CatalogSQLExecutor
from app.utils.security import validate_sql


ROOT = Path(__file__).resolve().parents[1]


def test_upload_preprocess_and_query_both_tables(monkeypatch):
    monkeypatch.delenv("APP_ACCESS_PASSWORD", raising=False)
    path = ROOT / "data/sample/customer_shopping_behavior.csv"
    upload = BytesIO(path.read_bytes())
    upload.name, upload.size, upload.file_id = path.name, len(upload.getvalue()), "preprocess-sample"
    monkeypatch.setattr(st, "file_uploader", lambda *a, **kw: [upload] if kw.get("accept_multiple_files") else None)
    app = AppTest.from_file(str(ROOT / "app/main.py")).run(timeout=30)
    assert not app.exception
    try:
        next(item for item in app.button if item.label == "Data Preprocess").click().run(timeout=30)
        assert not app.exception
        assert app.multiselect[0].options == ["Remove duplicates", "Handle missing values",
                                               "Correct data types", "Standardize text values",
                                               "Clean column names"]
        next(item for item in app.button if item.label == "Submit").click().run(timeout=30)
        assert app.session_state.dataset_catalog is None
        assert any("Select at least one" in item.value for item in app.info)
        app.multiselect[0].select("Remove duplicates").select("Clean column names").run(timeout=30)
        next(item for item in app.button if item.label == "Submit").click().run(timeout=30)
        assert not app.exception
        assert not app.error, [item.value for item in app.error]
        assert app.session_state.dataset_catalog is not None, ([item.value for item in app.info],
                                                               app.multiselect[0].value)
        catalog = app.session_state.dataset_catalog
        assert catalog.profile.table_names == ["dataset", "dataset_processed"]
        assert app.session_state.preprocess_report["original_rows"] == 3900
        assert app.session_state.preprocess_report["processed_rows"] == 3900
        executor = CatalogSQLExecutor(catalog)
        try:
            for table in ("dataset", "dataset_processed"):
                checked = validate_sql(f'SELECT COUNT(*) AS n FROM "{table}"', executor.table_names, 100)
                assert checked.is_valid
                assert executor.execute(checked).dataframe.iloc[0, 0] == 3900
            assert not validate_sql("SELECT * FROM preprocessing_runs", executor.table_names, 100).is_valid
        finally:
            executor.close()
    finally:
        if app.session_state.agent is not None:
            app.session_state.agent.release_dataset()
