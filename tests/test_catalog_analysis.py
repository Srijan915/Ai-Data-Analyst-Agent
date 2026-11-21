"""Multi-CSV planning, relationship authorization and actual DuckDB execution."""
import io
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from app.core.agent import AgentDataError, DataAnalystAgent
from app.core.dataset_catalog import CatalogSQLExecutor, DatasetCatalog
from app.core.chart_generator import generate_dataset_auto_visualizations
from app.core.join_validation import JoinValidationError, check_catalog_joins
from app.core.query_planner import QueryPlanner
from app.utils.security import validate_sql

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def catalog():
    catalog = DatasetCatalog()
    for name in ("customer_shopping_behavior.csv", "item_catalog.csv"):
        with (ROOT / "data/sample" / name).open("rb") as source:
            catalog.add(source, name)
    try:
        yield catalog
    finally:
        catalog.close()


def confirm(catalog):
    relation = catalog.profile.relationships[0]
    catalog.set_relationship_status(relation.left_table, relation.left_column,
                                    relation.right_table, relation.right_column, "confirmed")


def join_sql(catalog):
    sales, items = catalog.profile.datasets
    return (f'SELECT i."Item Category", SUM(s."Purchase Amount (USD)") AS total '
            f'FROM "{sales.table_name}" s JOIN "{items.table_name}" i '
            'ON s."Item Purchased" = i."Item Purchased" GROUP BY i."Item Category" ORDER BY i."Item Category"')


def test_catalog_agent_refreshes_confirmation_and_executes_real_join(catalog):
    model = Mock()
    model.generate_text.return_value = join_sql(catalog)
    first = catalog.profile.datasets[0].profile
    frame = pd.DataFrame(first.sample_rows)
    agent = DataAnalystAgent(frame, catalog.profile, dataset_catalog=catalog,
                             query_planner=QueryPlanner(llm_client=model))
    try:
        with pytest.raises(AgentDataError, match="confirmed"):
            agent.run("Join the datasets and show total purchase amount by item category")
        model.generate_text.assert_not_called()
        confirm(catalog)  # the agent already exists: it must not use a stale profile
        result = agent.run("Join the datasets and show total purchase amount by item category",
                           generate_explanation=False, generate_chart_spec=True)
        raw = pd.read_csv(ROOT / "data/sample/customer_shopping_behavior.csv")
        expected = raw.groupby("Category")["Purchase Amount (USD)"].sum().sort_index()
        actual = result.query_result.dataframe.set_index("Item Category").total.sort_index()
        assert actual.to_dict() == expected.to_dict()
        assert actual.sum() == raw["Purchase Amount (USD)"].sum()
        assert result.chart is not None
        prompt = model.generate_text.call_args.kwargs["prompt"]
        assert all(d.table_name in prompt for d in catalog.profile.datasets)
        assert '"status": "confirmed"' in prompt
        assert "Query ONLY the table `dataset`" not in model.generate_text.call_args.kwargs["system_prompt"]
        assert 'FROM "dataset"' not in prompt
    finally:
        agent.release_dataset()


@pytest.mark.parametrize("condition", [
    'ON s."Item Purchased" = i."Item Category"',
    'ON s."Item Purchased" = i."Item Purchased" OR 1=1',
    'ON LOWER(s."Item Purchased") = LOWER(i."Item Purchased")',
    'ON "Item Purchased" = "Item Purchased"',
    'ON i."Item Purchased" = i."Item Purchased"',
    'USING ("Item Purchased")',
    '',
])
def test_invalid_or_ambiguous_join_never_executes(catalog, condition):
    confirm(catalog)
    sales, items = catalog.profile.datasets
    sql = f'SELECT COUNT(*) FROM "{sales.table_name}" s JOIN "{items.table_name}" i {condition}'
    validation = validate_sql(sql, catalog.profile.table_names)
    assert validation.is_valid
    with CatalogSQLExecutor(catalog) as executor:
        with pytest.raises(JoinValidationError):
            executor.execute(validation)


def test_unconfirmed_and_revoked_relationship_rejected_at_executor(catalog):
    validation = validate_sql(join_sql(catalog), catalog.profile.table_names)
    with CatalogSQLExecutor(catalog) as executor:
        with pytest.raises(JoinValidationError, match="not confirmed"):
            executor.execute(validation)
        confirm(catalog)
        assert executor.execute(validation).row_count == 4
        r = catalog.profile.relationships[0]
        catalog.set_relationship_status(r.left_table, r.left_column, r.right_table, r.right_column, "rejected")
        with pytest.raises(JoinValidationError, match="not confirmed"):
            executor.execute(validation)


def test_cte_projection_preserves_keys_but_computed_key_does_not(catalog):
    confirm(catalog)
    sales, items = catalog.profile.datasets
    for expression, permitted in [('"Item Purchased"', True), ('LOWER("Item Purchased")', False)]:
        sql = (f'WITH purchases AS (SELECT {expression} AS item FROM "{sales.table_name}") '
               f'SELECT COUNT(*) FROM purchases p JOIN "{items.table_name}" i ON p.item = i."Item Purchased"')
        if permitted:
            check_catalog_joins(sql, catalog.profile)
        else:
            with pytest.raises(JoinValidationError):
                check_catalog_joins(sql, catalog.profile)


def test_unrelated_subquery_cannot_bypass_join_policy(catalog):
    confirm(catalog)
    sales, items = catalog.profile.datasets
    sql = f'SELECT (SELECT COUNT(*) FROM "{items.table_name}") FROM "{sales.table_name}"'
    with pytest.raises(JoinValidationError, match="connecting"):
        check_catalog_joins(sql, catalog.profile)


def test_many_to_many_relationship_cannot_be_confirmed():
    catalog = DatasetCatalog()
    try:
        catalog.add(io.BytesIO(b"key,a\n1,10\n1,20\n"), "a.csv")
        catalog.add(io.BytesIO(b"key,b\n1,30\n1,40\n"), "b.csv")
        with pytest.raises(ValueError, match="many-to-many"):
            confirm(catalog)
        assert catalog.profile.relationships[0].status == "suggested"
    finally:
        catalog.close()


def test_catalog_overview_queries_selected_table_and_labels_preview(catalog):
    sales, items = catalog.profile.datasets
    agent = DataAnalystAgent(pd.DataFrame(sales.profile.sample_rows), catalog.profile,
                             dataset_catalog=catalog)
    try:
        charts = generate_dataset_auto_visualizations(
            pd.DataFrame(items.profile.sample_rows), items.profile, agent=agent,
            table_name=items.table_name,
        )
        assert {"label": "Total Analyzed", "value": "25"} in charts[0]["metrics"]
        charts = generate_dataset_auto_visualizations(
            pd.DataFrame(sales.profile.sample_rows), sales.profile, agent=agent,
            table_name=sales.table_name,
        )
        histogram = next(chart for chart in charts if chart["chart_type"] == "histogram")
        assert histogram["sample_label"] == "Based on first 5 rows"
        assert {"label": "Total Analyzed", "value": "3,900"} in histogram["metrics"]
    finally:
        agent.release_dataset()
