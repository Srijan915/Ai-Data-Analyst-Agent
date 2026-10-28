"""Offline multi-dataset catalog and relationship safety tests."""
import io

import pytest

from app.core.dataset_catalog import CatalogSQLExecutor, DatasetCatalog
from app.utils.security import validate_sql


def test_catalog_keeps_duplicate_filenames_as_distinct_safe_tables():
    catalog = DatasetCatalog()
    try:
        first = catalog.add(io.BytesIO(b"id,value\n1,10\n"), "orders.csv")
        second = catalog.add(io.BytesIO(b"id,value\n2,20\n"), "orders.csv")
        assert first.dataset_id != second.dataset_id
        assert first.table_name != second.table_name
        assert all(name.startswith("ds_orders_") for name in catalog.profile.table_names)
        validation = validate_sql(
            f"SELECT SUM(value) AS total FROM {first.table_name} UNION SELECT SUM(value) FROM {second.table_name}",
            catalog.profile.table_names,
        )
        assert not validation.is_valid  # set operations remain intentionally unsupported
    finally:
        catalog.close()


def test_catalog_join_is_allowlisted_and_cleanup_is_session_private():
    catalog = DatasetCatalog()
    root = catalog.root
    try:
        orders = catalog.add(io.BytesIO(b"customer_id,total\n1,10\n2,20\n"), "orders.csv")
        customers = catalog.add(io.BytesIO(b"customer_id,name\n1,A\n2,B\n"), "customers.csv")
        suggestion = catalog.profile.relationships[0]
        assert suggestion.status == "suggested"
        assert suggestion.overlap_percentage == 100
        catalog.set_relationship_status(
            suggestion.left_table, suggestion.left_column,
            suggestion.right_table, suggestion.right_column, "confirmed",
        )
        assert catalog.profile.relationships[0].status == "confirmed"
        sql = f"SELECT SUM(o.total) AS total FROM {orders.table_name} o JOIN {customers.table_name} c ON o.customer_id = c.customer_id"
        validation = validate_sql(sql, catalog.profile.table_names)
        assert validation.is_valid
        with CatalogSQLExecutor(catalog) as executor:
            assert executor.execute(validation).dataframe.iloc[0, 0] == 30
        assert not validate_sql("SELECT * FROM unrelated", catalog.profile.table_names).is_valid
    finally:
        catalog.close()
    assert not root.exists()


def test_catalog_rejects_non_csv_name():
    catalog = DatasetCatalog()
    try:
        with pytest.raises(Exception):
            catalog.add(io.BytesIO(b"x\n1\n"), "not_csv.txt")
    finally:
        catalog.close()


def test_catalog_replacement_is_atomic_and_removal_invalidates_relationships():
    catalog = DatasetCatalog()
    try:
        orders = catalog.add(io.BytesIO(b"customer_id,total\n1,10\n"), "orders.csv")
        customers = catalog.add(io.BytesIO(b"customer_id,name\n1,A\n"), "customers.csv")
        assert catalog.profile.relationships
        with pytest.raises(Exception):
            catalog.replace(orders.dataset_id, io.BytesIO(b"x\n1\n"), "bad.txt")
        assert orders.dataset_id in {record.dataset_id for record in catalog.profile.datasets}
        replacement = catalog.replace(orders.dataset_id, io.BytesIO(b"order_id,total\n2,99\n"), "orders.csv")
        assert replacement.dataset_id != orders.dataset_id
        assert not catalog.profile.relationships
        catalog.remove(customers.dataset_id)
        assert [record.dataset_id for record in catalog.profile.datasets] == [replacement.dataset_id]
    finally:
        catalog.close()
