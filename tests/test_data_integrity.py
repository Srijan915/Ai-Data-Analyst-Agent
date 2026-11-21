"""Original storage, processed storage, joins and mocked SQL planning."""
from io import BytesIO
from unittest.mock import Mock

import duckdb
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from app.core.agent import AgentValidationError, DataAnalystAgent
from app.core.dataset_catalog import CatalogSQLExecutor, DatasetCatalog
from app.core.query_planner import QueryPlanner
from app.preprocessing.change_report import build_change_report
from app.preprocessing.processor import process_dataframe
from app.utils.security import validate_sql


def _process(catalog, record):
    original = catalog.read_table(record.dataset_id)
    cleaned, details = process_dataframe(original, ["Remove duplicates", "Standardize text values"])
    result = catalog.register_processed(record.dataset_id, cleaned,
                                        build_change_report(original, cleaned, details))
    return original, result


def test_original_audit_isolation_and_validated_query():
    catalog = DatasetCatalog()
    try:
        first = catalog.add(BytesIO(b"id,value\n001, A \n001, A \n"), "first.csv")
        second = catalog.add(BytesIO(b"id,value\n002, B \n"), "second.csv")
        saved_other = catalog.read_table(second.dataset_id)
        saved_original, processed = _process(catalog, first)
        assert_frame_equal(catalog.read_table(first.dataset_id), saved_original)
        assert_frame_equal(catalog.read_table(second.dataset_id), saved_other)
        with CatalogSQLExecutor(catalog) as executor:
            sql = f'SELECT id, value FROM "{processed.table_name}"'
            validated = validate_sql(sql, executor.table_names)
            assert validated.is_valid
            assert_frame_equal(executor.execute(validated).dataframe,
                               pd.DataFrame({"id": ["001"], "value": ["A"]}))
        with duckdb.connect(str(catalog.path), read_only=True) as connection:
            assert connection.execute("SELECT COUNT(*) FROM preprocessing_runs").fetchone() == (1,)
        assert not validate_sql("SELECT * FROM preprocessing_runs", catalog.profile.table_names).is_valid
    finally:
        catalog.close()


def test_confirmed_original_relationship_survives_preprocessing():
    catalog = DatasetCatalog()
    try:
        left = catalog.add(BytesIO(b"key,amount\n001,10\n002,20\n"), "sales.csv")
        right = catalog.add(BytesIO(b"key,category\n001,A\n002,B\n"), "items.csv")
        relation = catalog.profile.relationships[0]
        catalog.set_relationship_status(relation.left_table, relation.left_column,
                                        relation.right_table, relation.right_column, "confirmed")
        _process(catalog, left)
        assert any(item.status == "confirmed" for item in catalog.profile.relationships)
        sql = (f'SELECT i.category, SUM(CAST(s.amount AS INTEGER)) AS total '
               f'FROM "{left.table_name}" s JOIN "{right.table_name}" i '
               'ON s.key = i.key GROUP BY i.category ORDER BY i.category')
        with CatalogSQLExecutor(catalog) as executor:
            result = executor.execute(validate_sql(sql, executor.table_names)).dataframe
        assert result.set_index("category").total.to_dict() == {"A": 10, "B": 20}
    finally:
        catalog.close()


def test_mocked_gemini_sql_is_validated_before_duckdb_execution():
    catalog = DatasetCatalog()
    model = Mock()
    try:
        source = catalog.add(BytesIO(b"id,value\n001,10\n002,20\n"), "values.csv")
        _process(catalog, source)
        table = f"{source.table_name}_processed"
        model.generate_text.return_value = f'SELECT COUNT(*) AS n FROM "{table}"'
        agent = DataAnalystAgent(catalog.read_table(source.dataset_id), catalog.profile,
                                 dataset_catalog=catalog,
                                 query_planner=QueryPlanner(llm_client=model))
        try:
            result = agent.run("How many rows are in the processed dataset?",
                               generate_explanation=False, generate_chart_spec=False)
            assert result.validation.is_valid and result.query_result.dataframe.iloc[0, 0] == 2
            model.generate_text.return_value = 'SELECT COUNT(*) FROM preprocessing_runs'
            with pytest.raises(AgentValidationError):
                agent.run("How many rows are in the processed dataset?",
                          generate_explanation=False, generate_chart_spec=False)
            assert len(catalog.read_table(source.dataset_id)) == 2
        finally:
            agent.release_dataset()
    finally:
        catalog.close()
