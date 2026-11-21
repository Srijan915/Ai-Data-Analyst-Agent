"""Real multi-upload UI workflow with only upload widgets and Gemini stubbed."""
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

from app.core import agent as agent_module
from app.core.query_planner import QueryPlanner

ROOT = Path(__file__).resolve().parents[1]


def button(app, label):
    return next(item for item in app.button if item.label == label)


def test_multiple_upload_confirm_analyze_revoke_and_reject(monkeypatch):
    monkeypatch.delenv("APP_ACCESS_PASSWORD", raising=False)
    uploads = []
    for name in ("customer_shopping_behavior.csv", "item_catalog.csv"):
        upload = BytesIO((ROOT / "data/sample" / name).read_bytes())
        upload.name = name
        upload.size = len(upload.getvalue())
        upload.file_id = name
        uploads.append(upload)
    monkeypatch.setattr(st, "file_uploader", lambda *a, **kw: uploads if kw.get("accept_multiple_files") else None)
    model = Mock()
    monkeypatch.setattr(agent_module, "QueryPlanner", lambda: QueryPlanner(llm_client=model))
    monkeypatch.setattr(agent_module, "explain_query_result", lambda **kw: f'The query returned {kw["query_result"].row_count} rows.')
    app = AppTest.from_file(str(ROOT / "app/main.py")).run(timeout=25)
    assert not app.exception
    catalog = app.session_state.dataset_catalog
    sales, items = catalog.profile.datasets
    assert sales.row_count == 3900 and items.row_count == 25
    assert any(sales.table_name in item.value for item in app.code)
    assert any(items.table_name in item.value for item in app.code)
    assert catalog.profile.relationships[0].status == "suggested"
    model.generate_text.return_value = (
        f'SELECT i."Item Category", SUM(s."Purchase Amount (USD)") AS total '
        f'FROM "{sales.table_name}" s JOIN "{items.table_name}" i '
        'ON s."Item Purchased" = i."Item Purchased" GROUP BY i."Item Category" ORDER BY i."Item Category"'
    )
    try:
        app.text_area[0].input("Join the datasets and show total purchase amount by item category")
        button(app, "Analyze question").click().run(timeout=20)
        assert not app.exception
        assert app.session_state.last_result is None
        model.generate_text.assert_not_called()
        assert any("confirmed" in message.value for message in app.warning)

        button(app, "Confirm join").click().run(timeout=20)
        assert not app.exception
        assert catalog.profile.relationships[0].status == "confirmed"
        button(app, "Analyze question").click().run(timeout=20)
        assert not app.exception
        result = app.session_state.last_result
        expected = pd.read_csv(ROOT / "data/sample/customer_shopping_behavior.csv").groupby("Category")["Purchase Amount (USD)"].sum().to_dict()
        assert result.query_result.dataframe.set_index("Item Category").total.to_dict() == expected
        assert result.explanation == "The query returned 4 rows."
        assert result.chart is not None
        assert any(item.value == result.validation.cleaned_sql for item in app.code)
        assert len(app.get("plotly_chart")) > 0

        button(app, "Revoke join").click().run(timeout=20)
        assert not app.exception
        assert app.session_state.last_result is None
        assert app.session_state.conversation_history == []
        assert catalog.profile.relationships[0].status == "rejected"
        # No explicit 'join' trigger: generated SQL must still be blocked.
        app.text_area[0].input("Show total purchase amount by item category")
        button(app, "Analyze question").click().run(timeout=20)
        assert not app.exception
        assert app.session_state.last_result is None
        assert any("not confirmed" in message.value for message in app.warning)
    finally:
        app.session_state.agent.release_dataset()
