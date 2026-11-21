"""Processed tables participate in the same catalog and SQL guard."""
from io import BytesIO

import duckdb

from app.core.dataset_catalog import DatasetCatalog
from app.preprocessing.change_report import build_change_report
from app.preprocessing.processor import process_dataframe
from app.utils.security import validate_sql


def test_processed_table_preserves_original_and_audits_each_run():
    catalog = DatasetCatalog()
    try:
        source = catalog.add(BytesIO(b"id,name\n001, Ada \n001, Ada \n002,Bob\n"), "people.csv")
        catalog.rename_single_table(source.dataset_id)
        original = catalog.read_table(source.dataset_id)
        frame, details = process_dataframe(original, ["Remove duplicates", "Standardize text values"])
        report = build_change_report(original, frame, details)
        first = catalog.register_processed(source.dataset_id, frame, report)
        assert catalog.profile.table_names == ["dataset", "dataset_processed"]
        assert len(catalog.read_table(source.dataset_id)) == 3
        assert len(catalog.read_table(first.dataset_id)) == 2
        assert catalog.read_table(source.dataset_id).id.tolist() == ["001", "001", "002"]
        second_frame, second_details = process_dataframe(original, ["Standardize text values"])
        second = catalog.register_processed(source.dataset_id, second_frame,
                                            build_change_report(original, second_frame, second_details))
        assert first.dataset_id != second.dataset_id
        assert len(catalog.read_table(second.dataset_id)) == 3
        assert catalog.profile.table_names == ["dataset", "dataset_processed"]
        conn = duckdb.connect(str(catalog.path), read_only=True)
        try:
            assert conn.execute("SELECT count(*) FROM preprocessing_runs").fetchone() == (2,)
        finally:
            conn.close()
    finally:
        catalog.close()


def test_processed_and_audit_table_visibility():
    catalog = DatasetCatalog()
    try:
        source = catalog.add(BytesIO(b"id,name\n001,Ada\n"), "people.csv")
        frame = catalog.read_table(source.dataset_id)
        processed, details = process_dataframe(frame, ["Clean column names"])
        catalog.register_processed(source.dataset_id, processed,
                                   build_change_report(frame, processed, details))
        assert "preprocessing_runs" not in catalog.profile.table_names
        assert any(name.endswith("_processed") for name in catalog.profile.table_names)
        assert not validate_sql("SELECT * FROM preprocessing_runs", catalog.profile.table_names, 100).is_valid
        assert validate_sql(f'SELECT * FROM "{catalog.profile.table_names[-1]}"',
                            catalog.profile.table_names, 100).is_valid
        catalog.remove(source.dataset_id)
        assert catalog.profile.table_names == []
    finally:
        catalog.close()


def test_processing_one_dataset_does_not_change_another():
    catalog = DatasetCatalog()
    try:
        first = catalog.add(BytesIO(b"id,value\n001, A \n"), "first.csv")
        second = catalog.add(BytesIO(b"id,value\n002, B \n"), "second.csv")
        other_before = catalog.read_table(second.dataset_id)
        original = catalog.read_table(first.dataset_id)
        processed, details = process_dataframe(original, ["Standardize text values"])
        catalog.register_processed(first.dataset_id, processed,
                                   build_change_report(original, processed, details))
        assert catalog.read_table(second.dataset_id).equals(other_before)
        assert len(catalog.profile.table_names) == 3
    finally:
        catalog.close()
