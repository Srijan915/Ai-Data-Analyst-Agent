"""Offline measurement of SQL validator behavior on fixed safety cases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.utils.security import validate_sql


ALLOWED_TABLES = ["orders", "customers"]
MAX_ROWS = 100
RESULTS_PATH = Path(__file__).with_name("security_results.json")


# Every statement in this list is expected to be rejected by the security
# policy.  The evaluator reports any statement that the real validator accepts.
UNSAFE_CASES = [
    {"label": "drop_table", "sql": "DROP TABLE orders"},
    {"label": "delete_rows", "sql": "DELETE FROM orders"},
    {"label": "update_rows", "sql": "UPDATE orders SET status = 'closed'"},
    {"label": "insert_rows", "sql": "INSERT INTO orders VALUES (1)"},
    {"label": "alter_table", "sql": "ALTER TABLE orders ADD COLUMN note TEXT"},
    {"label": "create_table", "sql": "CREATE TABLE copied AS SELECT * FROM orders"},
    {"label": "copy_to_file", "sql": "COPY orders TO 'orders.csv'"},
    {"label": "pragma", "sql": "PRAGMA table_info(orders)"},
    {"label": "attach_database", "sql": "ATTACH 'other.db' AS other"},
    {"label": "install_extension", "sql": "INSTALL httpfs"},
    {"label": "load_extension", "sql": "LOAD httpfs"},
    {"label": "read_csv", "sql": "SELECT * FROM read_csv('outside.csv')"},
    {"label": "read_csv_auto", "sql": "SELECT * FROM read_csv_auto('outside.csv')"},
    {"label": "read_parquet", "sql": "SELECT * FROM read_parquet('outside.parquet')"},
    {"label": "read_json_auto", "sql": "SELECT * FROM read_json_auto('outside.json')"},
    {"label": "sqlite_scan", "sql": "SELECT * FROM sqlite_scan('other.db', 'orders')"},
    {"label": "postgres_scan", "sql": "SELECT * FROM postgres_scan('conn', 'public', 'orders')"},
    {"label": "random_function", "sql": "SELECT RANDOM() FROM orders"},
    {"label": "version_function", "sql": "SELECT VERSION() FROM orders"},
    {"label": "current_database_function", "sql": "SELECT CURRENT_DATABASE() FROM orders"},
    {"label": "unallowed_table", "sql": "SELECT * FROM secrets"},
    {"label": "qualified_table", "sql": "SELECT * FROM main.orders"},
    {"label": "stacked_drop", "sql": "SELECT * FROM orders; DROP TABLE orders"},
    {"label": "comment_then_delete", "sql": "SELECT * FROM orders -- harmless\nDELETE FROM orders"},
    {"label": "block_comment_then_drop", "sql": "SELECT * FROM orders /* harmless */; DROP TABLE orders"},
    {"label": "union_policy_bypass", "sql": "SELECT * FROM orders UNION ALL SELECT * FROM customers"},
    {"label": "set_setting", "sql": "SET enable_external_access = true"},
    {"label": "vacuum", "sql": "VACUUM"},
]


# These are read-only queries over only the allowed tables.  A missing LIMIT
# is intentional: validation should inject MAX_ROWS rather than reject it.
SAFE_CASES = [
    {"label": "simple_select", "sql": "SELECT id, amount FROM orders"},
    {"label": "filter", "sql": "SELECT id FROM orders WHERE amount > 10"},
    {"label": "count", "sql": "SELECT COUNT(*) AS order_count FROM orders"},
    {"label": "aggregate", "sql": "SELECT SUM(amount) AS total_amount FROM orders"},
    {"label": "group_by", "sql": "SELECT status, COUNT(*) AS count FROM orders GROUP BY status"},
    {"label": "join", "sql": "SELECT o.id, c.name FROM orders AS o JOIN customers AS c ON o.customer_id = c.id"},
    {"label": "cte", "sql": "WITH recent AS (SELECT * FROM orders) SELECT * FROM recent"},
    {"label": "subquery", "sql": "SELECT * FROM (SELECT id, amount FROM orders) AS rows"},
    {"label": "case", "sql": "SELECT CASE WHEN amount > 10 THEN 'large' ELSE 'small' END AS size FROM orders"},
    {"label": "cast", "sql": "SELECT CAST(amount AS DOUBLE) AS numeric_amount FROM orders"},
    {"label": "coalesce", "sql": "SELECT COALESCE(status, 'unknown') AS status FROM orders"},
    {"label": "date_trunc", "sql": "SELECT DATE_TRUNC('month', order_date) AS month FROM orders"},
    {"label": "string_function", "sql": "SELECT UPPER(status) AS upper_status FROM orders"},
    {"label": "existing_limit", "sql": "SELECT id FROM orders LIMIT 10"},
    {"label": "aggregate_window", "sql": "SELECT id, SUM(amount) OVER (PARTITION BY customer_id) AS customer_total FROM orders"},
    {"label": "row_number_window", "sql": "SELECT id, ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY id) AS row_num FROM orders"},
]


def _validator_message(validation: Any) -> str:
    """Return the real validator outcome in a JSON-friendly form."""
    if validation.is_valid:
        return f"accepted: {validation.cleaned_sql}"
    return "; ".join(
        f"{error.code}: {error.message}"
        for error in validation.errors
    )


def run_security_eval() -> dict[str, Any]:
    """Run fixed unsafe and safe cases through the production validator."""
    unexpected_cases = []
    unsafe_blocked = 0
    safe_wrongly_blocked = 0

    for case in UNSAFE_CASES:
        validation = validate_sql(case["sql"], ALLOWED_TABLES, max_rows=MAX_ROWS)
        if validation.is_valid:
            unexpected_cases.append(
                {
                    "expected": "rejected",
                    **case,
                    "validator_message": _validator_message(validation),
                }
            )
        else:
            unsafe_blocked += 1

    for case in SAFE_CASES:
        validation = validate_sql(case["sql"], ALLOWED_TABLES, max_rows=MAX_ROWS)
        if not validation.is_valid:
            safe_wrongly_blocked += 1
            unexpected_cases.append(
                {
                    "expected": "accepted",
                    **case,
                    "validator_message": _validator_message(validation),
                }
            )

    return {
        "unsafe_total": len(UNSAFE_CASES),
        "unsafe_blocked": unsafe_blocked,
        "safe_total": len(SAFE_CASES),
        "safe_wrongly_blocked": safe_wrongly_blocked,
        "unexpected_cases": unexpected_cases,
    }


def main() -> None:
    """Run the offline evaluation and save the exact observed results."""
    results = run_security_eval()
    try:
        RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        print(f"warning: could not write {RESULTS_PATH}: {exc}")

    print(f"unsafe blocked: {results['unsafe_blocked']} of {results['unsafe_total']}")
    print(
        "safe wrongly blocked: "
        f"{results['safe_wrongly_blocked']} of {results['safe_total']}"
    )
    print("unexpected cases:")
    if not results["unexpected_cases"]:
        print("  none")
    for case in results["unexpected_cases"]:
        print(f"  {case['label']}: {case['validator_message']}")


if __name__ == "__main__":
    main()
