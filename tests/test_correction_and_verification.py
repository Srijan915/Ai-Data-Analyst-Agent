"""Bounded correction and deterministic result-grounding tests."""
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from app.core.agent import DataAnalystAgent
from app.core.data_profiler import profile_dataframe
from app.core.query_planner import QueryPlanner, QueryPlanningError
from app.core import query_planner as planner_module
from app.core.result_verifier import explanation_is_grounded, verify_query_result
from app.models.schemas import QueryResult


def test_malformed_model_sql_is_corrected_once_then_revalidated(monkeypatch):
    monkeypatch.setattr(planner_module, "config", SimpleNamespace(
        max_question_chars=2000, max_sql_correction_attempts=1,
    ))
    llm = Mock()
    llm.generate_text.side_effect = ["not SQL", "SELECT COUNT(*) AS total FROM dataset"]
    frame = pd.DataFrame({"value": [1, 2, 3]})
    agent = DataAnalystAgent(frame, profile_dataframe(frame), query_planner=QueryPlanner(llm_client=llm))
    result = agent.run("count records", generate_explanation=False, generate_chart_spec=False)
    assert result.query_result.dataframe.iloc[0, 0] == 3
    assert llm.generate_text.call_count == 2
    assert "SELECT COUNT" in result.validation.cleaned_sql.upper()


def test_correction_stops_at_configured_limit(monkeypatch):
    monkeypatch.setattr(planner_module, "config", SimpleNamespace(
        max_question_chars=2000, max_sql_correction_attempts=1,
    ))
    llm = Mock()
    llm.generate_text.side_effect = ["bad", "still bad", "unexpected third request"]
    frame = pd.DataFrame({"value": [1]})
    with pytest.raises(QueryPlanningError):
        QueryPlanner(llm_client=llm).plan("count values", profile_dataframe(frame))
    assert llm.generate_text.call_count == 2


def test_result_verifier_flags_empty_and_metadata_errors():
    empty = QueryResult(dataframe=pd.DataFrame({"total": []}), row_count=0, truncated=False)
    assert verify_query_result(empty).warnings
    inconsistent = QueryResult(dataframe=pd.DataFrame({"total": [1, 2]}), row_count=1, truncated=False)
    with pytest.raises(ValueError, match="metadata"):
        verify_query_result(inconsistent)


def test_unsupported_explanation_figure_is_not_grounded():
    result = QueryResult(dataframe=pd.DataFrame({"total": [10]}), row_count=1, truncated=False)
    assert explanation_is_grounded("The total is 10.", result)
    assert not explanation_is_grounded("The total is 999.", result)
