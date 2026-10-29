"""Regression checks for production-safe Streamlit UI behavior."""

import ast
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from app.core.llm_client import (
    LLMConfigurationError,
    LLMCredentialsError,
    LLMNetworkError,
    LLMTimeoutError,
    LLMTLSVerificationError,
)
from app.main import _display_chart
from app.models.schemas import QueryResult
from app.ui.analysis import _planning_error_message, display_analysis_result

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAIN_MODULE = PROJECT_ROOT / "app" / "main.py"


def _main_source() -> str:
    return MAIN_MODULE.read_text(encoding="utf-8") + "\n" + (PROJECT_ROOT / "app/ui/analysis.py").read_text(encoding="utf-8")


def test_streamlit_ui_does_not_render_raw_exceptions():
    """Detailed exceptions must stay in server logs, not reach end users."""

    source = _main_source()
    tree = ast.parse(source)

    raw_exception_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "st"
        and node.func.attr == "exception"
    ]

    assert raw_exception_calls == []
    assert "could not be rendered: {exc}" not in source


def test_streamlit_ui_keeps_bounded_session_only_follow_up_context():
    """Follow-up context must be bounded, resettable, and planner-only."""

    source = _main_source()
    normalized_source = "".join(source.split())

    assert 'if "conversation_history" not in st.session_state:' in source
    assert "conversation_context=(" in source
    assert "history[-MAX_CONVERSATION_TURNS:]" in normalized_source
    assert "Clear follow-up context" in source

    # It is reset when either the active dataset or just the context is
    # cleared, and when a fresh upload successfully replaces the dataset.
    assert source.count(
        "st.session_state.conversation_history = []"
    ) >= 3


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (LLMCredentialsError("private-key"), "credentials are missing or invalid"),
        (LLMConfigurationError("private-model"), "model or request is unavailable"),
        (LLMNetworkError("private-network"), "could not be reached"),
        (LLMTimeoutError("private-timeout"), "request timed out"),
        (LLMTLSVerificationError("private-certificate"), "trusted TLS certificate"),
    ],
)
def test_planning_error_messages_are_specific_and_do_not_expose_details(
    error,
    expected,
):
    message = _planning_error_message(error)

    assert expected in message
    assert "private-" not in message


def test_display_chart_handles_numpy_integer_and_float_values():
    # Single-column scalar with numpy integer
    df_int = pd.DataFrame({"total_count": [np.int64(42)]})
    qr_int = QueryResult(dataframe=df_int, row_count=1, truncated=False)
    with patch("streamlit.html"), patch("streamlit.caption"), patch("streamlit.metric") as mock_metric:
        _display_chart(None, query_result=qr_int)
        mock_metric.assert_called_once_with(label="Total Count", value="42")

    # Single-column scalar with numpy float
    df_float = pd.DataFrame({"average_val": [np.float64(1234.56)]})
    qr_float = QueryResult(dataframe=df_float, row_count=1, truncated=False)
    with patch("streamlit.html"), patch("streamlit.caption"), patch("streamlit.metric") as mock_metric:
        _display_chart(None, query_result=qr_float)
        mock_metric.assert_called_once_with(label="Average Val", value="1,234.56")

    # Multi-column summary with both numpy integer and float
    df_multi = pd.DataFrame({
        "item_count": [np.int32(100)],
        "unit_rate": [np.float32(99.5)],
    })
    qr_multi = QueryResult(dataframe=df_multi, row_count=1, truncated=False)
    col1 = MagicMock()
    col2 = MagicMock()
    with patch("streamlit.html"), patch("streamlit.caption"), patch("streamlit.columns", return_value=[col1, col2]):
        _display_chart(None, query_result=qr_multi)
        col1.metric.assert_called_once_with(label="Item Count", value="100")
        col2.metric.assert_called_once_with(label="Unit Rate", value="99.50")


def test_display_analysis_result_chart_failure_shows_warning_and_renders_sql():
    mock_result = MagicMock()
    mock_result.validation = MagicMock(is_valid=True)
    mock_result.warnings = []
    mock_result.query_result = MagicMock(truncated=False, dataframe=None)
    mock_result.explanation = "Analysis successful"
    mock_result.chart = {"data": []}
    mock_result.sql = "SELECT 1"

    failing_renderer = MagicMock(side_effect=RuntimeError("Chart plotting crashed"))

    with patch("streamlit.html"), patch("streamlit.subheader"), patch("streamlit.container"), \
         patch("streamlit.caption"), patch("streamlit.expander"), patch("streamlit.code") as mock_code, \
         patch("streamlit.warning") as mock_warning:
        display_analysis_result(mock_result, chart_renderer=failing_renderer)

        failing_renderer.assert_called_once()
        mock_warning.assert_any_call("Chart could not be rendered")
        mock_code.assert_called_once_with("SELECT 1", language="sql")
