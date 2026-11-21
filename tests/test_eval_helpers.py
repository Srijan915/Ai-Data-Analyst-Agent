"""Offline tests for live-evaluation dataframe comparison helpers."""

import pandas as pd
import pytest

from evals.run_evals import normalize, results_match


def test_normalize_ignores_row_order():
    actual = pd.DataFrame({"product": ["B", "A"], "sales": [2, 1]})
    expected = pd.DataFrame({"product": ["A", "B"], "sales": [1, 2]})

    assert normalize(actual).equals(normalize(expected))
    assert results_match(actual, expected, allow_extra_columns=True) is True


def test_results_match_rounds_floats_to_six_decimal_places():
    actual = pd.DataFrame({"average": [1.23456789]})
    expected = pd.DataFrame({"average": [1.23456781]})

    assert results_match(actual, expected) is True


def test_results_match_can_use_published_two_decimal_precision():
    actual = pd.DataFrame({"average": [59.7643589744]})
    expected = pd.DataFrame({"average": [59.76]})

    assert results_match(actual, expected, decimal_places=2) is True


def test_results_match_treats_common_null_representations_as_equal():
    actual = pd.DataFrame({"value": [None, "present"]})
    expected = pd.DataFrame({"value": [float("nan"), "present"]})

    assert results_match(actual, expected) is True


def test_results_match_rejects_different_column_counts():
    actual = pd.DataFrame({"product": ["A"], "sales": [1]})
    expected = pd.DataFrame({"product": ["A"]})

    assert results_match(actual, expected) is False


def test_results_match_preserves_large_integer_precision():
    actual = pd.DataFrame({"id": [9007199254740992]})
    expected = pd.DataFrame({"id": [9007199254740993]})

    assert not results_match(actual, expected)


def test_results_match_accepts_equivalent_values_with_different_aliases():
    """Output aliases are explanatory metadata, not answer correctness."""
    actual = pd.DataFrame({"total": [10], "average": [5.0]})
    expected = pd.DataFrame({"total_purchase_amount": [10], "mean_purchase_amount": [5.0]})

    assert results_match(actual, expected) is True


def test_results_match_accepts_requested_named_columns_with_extra_fields():
    actual = pd.DataFrame({"category": ["A"], "age": [30], "total": [10]})
    expected = pd.DataFrame({"category": ["A"], "total": [10]})

    assert results_match(actual, expected, allow_extra_columns=True) is True


# ===========================================================================
# Evaluation runner error resilience and metric tests (mocked offline)
# ===========================================================================

def _make_provider_error():
    from app.core.agent import AgentPlanningError
    from app.core.query_planner import QueryPlanningError
    from app.core.llm_client import LLMAPIError

    err = AgentPlanningError("Could not generate SQL for the question: LLM failed")
    err.__cause__ = QueryPlanningError("The language model could not generate an analysis plan.")
    err.__cause__.__cause__ = LLMAPIError("503 Service Unavailable")
    return err


def _make_network_error():
    from app.core.agent import AgentPlanningError
    from app.core.query_planner import QueryPlanningError
    from app.core.llm_client import LLMNetworkError

    err = AgentPlanningError("Could not generate SQL for the question: LLM failed")
    err.__cause__ = QueryPlanningError("The language model could not generate an analysis plan.")
    err.__cause__.__cause__ = LLMNetworkError("Connection refused")
    return err


def _make_timeout_error():
    from app.core.agent import AgentPlanningError
    from app.core.query_planner import QueryPlanningError
    from app.core.llm_client import LLMTimeoutError

    err = AgentPlanningError("Could not generate SQL for the question: LLM failed")
    err.__cause__ = QueryPlanningError("The language model could not generate an analysis plan.")
    err.__cause__.__cause__ = LLMTimeoutError("The provider request timed out.")
    return err


def _make_planning_error():
    from app.core.agent import AgentPlanningError
    from app.core.query_planner import QueryPlanningError

    err = AgentPlanningError("Could not generate SQL for the question: Syntax error in model SQL")
    err.__cause__ = QueryPlanningError("The language model produced unparseable SQL.")
    return err


def test_run_evaluation_correct_answer(monkeypatch, tmp_path):
    """
    An answerable question whose agent result matches the gold standard
    must be classified as matched, marked correct, and produce 100% accuracy.
    """
    from evals.run_evals import run_evaluation

    csv_file = tmp_path / "sample.csv"
    csv_file.write_text("id,val\n1,10\n", encoding="utf-8")

    monkeypatch.setattr("evals.run_evals._require_gemini_api_key", lambda: None)
    monkeypatch.setattr("evals.run_evals.run_gold_sql", lambda sql, p: pd.DataFrame({"val": [10]}))
    monkeypatch.setattr("evals.run_evals.ask_agent", lambda q, p: pd.DataFrame({"val": [10]}))

    questions = [
        {"id": "q1", "split": "dev", "answerable": True, "question": "Total?", "gold_sql": "SELECT 10"}
    ]

    results = run_evaluation(questions, csv_file, runs=1)

    assert results["total_cases"] == 1
    assert results["correct_cases"] == 1
    assert results["incorrect_cases"] == 0
    assert results["provider_errors"] == 0
    assert results["planning_errors"] == 0
    assert results["evaluable_cases"] == 1
    assert results["accuracy_percent"] == 100.0
    assert results["case_results"][0]["outcome"] == "matched"
    assert results["case_results"][0]["correct"] is True


def test_run_evaluation_records_sql_and_full_mismatch_diagnostics(monkeypatch, tmp_path):
    """A failed case must be debuggable from the JSON result alone."""
    from evals.run_evals import run_evaluation

    csv_file = tmp_path / "sample.csv"
    csv_file.write_text("id,val\n1,10\n", encoding="utf-8")
    actual = pd.DataFrame({"model_total": [9]})
    actual.attrs["generated_sql"] = "SELECT SUM(val) AS model_total FROM dataset"

    monkeypatch.setattr("evals.run_evals._require_gemini_api_key", lambda: None)
    monkeypatch.setattr(
        "evals.run_evals.run_gold_sql",
        lambda sql, path: pd.DataFrame({"expected_total": [10]}),
    )
    monkeypatch.setattr("evals.run_evals.ask_agent", lambda q, p: actual)

    results = run_evaluation(
        [{"id": "q1", "split": "dev", "answerable": True, "question": "Total?", "gold_sql": "SELECT 10"}],
        csv_file,
        runs=1,
    )

    case = results["case_results"][0]
    assert case["generated_sql"] == "SELECT SUM(val) AS model_total FROM dataset"
    assert case["expected_result"]["rows"] == [{"expected_total": 10}]
    assert case["actual_result"]["rows"] == [{"model_total": 9}]
    assert case["failure_reason"] == "column_alias_mismatch_and_values_mismatch"


def test_run_evaluation_records_sql_when_validation_refuses(monkeypatch, tmp_path):
    from evals.run_evals import AgentRefused, run_evaluation

    csv_file = tmp_path / "sample.csv"
    csv_file.write_text("id,val\n1,10\n", encoding="utf-8")
    monkeypatch.setattr("evals.run_evals._require_gemini_api_key", lambda: None)
    monkeypatch.setattr("evals.run_evals.run_gold_sql", lambda sql, path: pd.DataFrame({"val": [10]}))
    monkeypatch.setattr(
        "evals.run_evals.ask_agent",
        lambda q, p: (_ for _ in ()).throw(AgentRefused("blocked", "SELECT bad FROM dataset")),
    )

    result = run_evaluation(
        [{"id": "q1", "split": "dev", "answerable": True, "question": "Total?", "gold_sql": "SELECT 10"}],
        csv_file,
        runs=1,
    )

    assert result["case_results"][0]["generated_sql"] == "SELECT bad FROM dataset"


def test_run_evaluation_incorrect_answer(monkeypatch, tmp_path):
    """
    Answerable questions with mismatched results or unjustified refusals, and
    unanswerable questions that the agent mistakenly answers, must all be counted
    as incorrect cases, included in evaluable cases, and produce 0% accuracy.
    """
    from evals.run_evals import run_evaluation, AgentRefused

    csv_file = tmp_path / "sample.csv"
    csv_file.write_text("id,val\n1,10\n", encoding="utf-8")

    monkeypatch.setattr("evals.run_evals._require_gemini_api_key", lambda: None)
    monkeypatch.setattr("evals.run_evals.run_gold_sql", lambda sql, p: pd.DataFrame({"val": [10]}))

    call_index = [0]
    responses = [
        pd.DataFrame({"val": [999]}),               # q1: mismatch
        AgentRefused("Unjustified refusal"),        # q2: wrongly refused answerable
        pd.DataFrame({"val": [10]}),                # u1: answered unanswerable
    ]

    def mock_ask(q, p):
        idx = call_index[0]
        call_index[0] += 1
        resp = responses[idx]
        if isinstance(resp, BaseException):
            raise resp
        return resp

    monkeypatch.setattr("evals.run_evals.ask_agent", mock_ask)

    questions = [
        {"id": "q1", "split": "dev", "answerable": True, "question": "Mismatch?", "gold_sql": "SELECT 10"},
        {"id": "q2", "split": "dev", "answerable": True, "question": "Wrong refuse?", "gold_sql": "SELECT 10"},
        {"id": "u1", "split": "dev", "answerable": False, "question": "Answered unans?", "gold_sql": None},
    ]

    results = run_evaluation(questions, csv_file, runs=1)

    assert results["total_cases"] == 3
    assert results["correct_cases"] == 0
    assert results["incorrect_cases"] == 3
    assert results["provider_errors"] == 0
    assert results["planning_errors"] == 0
    assert results["evaluable_cases"] == 3
    assert results["accuracy_percent"] == 0.0

    assert results["case_results"][0]["outcome"] == "result_mismatch"
    assert results["case_results"][0]["correct"] is False
    assert results["case_results"][1]["outcome"].startswith("agent_refused:")
    assert results["case_results"][1]["correct"] is False
    assert results["case_results"][2]["outcome"] == "agent_answered_unanswerable_question"
    assert results["case_results"][2]["correct"] is False


def test_run_evaluation_mixed_run(monkeypatch, tmp_path):
    """
    A mixed run containing correct answers, incorrect answers, planning errors,
    and provider failures must categorize each outcome accurately, exclude provider
    failures from evaluable cases, and compute accuracy only across evaluable cases.
    """
    from evals.run_evals import run_evaluation, AgentRefused

    csv_file = tmp_path / "sample.csv"
    csv_file.write_text("id,val\n1,10\n", encoding="utf-8")

    monkeypatch.setattr("evals.run_evals._require_gemini_api_key", lambda: None)
    monkeypatch.setattr(
        "evals.run_evals.run_gold_sql",
        lambda sql, path: pd.DataFrame({"val": [10]}),
    )

    provider_err = _make_provider_error()
    planning_err = _make_planning_error()
    refusal = AgentRefused("Unconfirmed column")

    call_index = [0]
    responses = [
        pd.DataFrame({"val": [10]}),       # Q1: answerable, correct match
        pd.DataFrame({"val": [999]}),      # Q2: answerable, result mismatch
        provider_err,                      # Q3: answerable, provider error (503)
        refusal,                           # Q4: unanswerable, safely refused
        pd.DataFrame({"val": [10]}),       # Q5: unanswerable, agent answered
        planning_err,                      # Q6: answerable, genuine planning error
    ]

    def mock_ask_agent(question, path):
        idx = call_index[0]
        call_index[0] += 1
        resp = responses[idx]
        if isinstance(resp, BaseException):
            raise resp
        return resp

    monkeypatch.setattr("evals.run_evals.ask_agent", mock_ask_agent)

    questions = [
        {"id": "q1", "split": "dev", "answerable": True, "question": "Total?", "gold_sql": "SELECT 10"},
        {"id": "q2", "split": "dev", "answerable": True, "question": "Average?", "gold_sql": "SELECT 10"},
        {"id": "q3", "split": "dev", "answerable": True, "question": "Count?", "gold_sql": "SELECT 10"},
        {"id": "q4", "split": "dev", "answerable": False, "question": "Unanswerable 1?", "gold_sql": None},
        {"id": "q5", "split": "dev", "answerable": False, "question": "Unanswerable 2?", "gold_sql": None},
        {"id": "q6", "split": "dev", "answerable": True, "question": "Max?", "gold_sql": "SELECT 10"},
    ]

    results = run_evaluation(questions, csv_file, runs=1)

    assert results["total_cases"] == 6
    assert results["correct_cases"] == 2      # q1 (matched) and q4 (refused)
    assert results["incorrect_cases"] == 2    # q2 (mismatch) and q5 (answered unanswerable)
    assert results["provider_errors"] == 1    # q3
    assert results["planning_errors"] == 1    # q6
    assert results["evaluable_cases"] == 5    # 6 total - 1 provider error
    assert results["accuracy_percent"] == 40.0  # 2 correct / 5 evaluable * 100

    # Verify individual case results
    outcomes = [r["outcome"] for r in results["case_results"]]
    assert outcomes[0] == "matched"
    assert outcomes[1] == "result_mismatch"
    assert outcomes[2] == "provider_error"
    assert outcomes[3].startswith("agent_refused:")
    assert outcomes[4] == "agent_answered_unanswerable_question"
    assert outcomes[5] == "planning_error"

    # Provider error does not count as incorrect analytical answer (correct is None)
    assert results["case_results"][2]["correct"] is None


def test_run_evaluation_provider_error_on_answerable_question(monkeypatch, tmp_path):
    """
    Provider errors on answerable questions must be recorded as provider_error,
    must not count as incorrect analytical answers, and must leave accuracy_percent
    as None when no evaluable cases remain.
    """
    from evals.run_evals import run_evaluation

    csv_file = tmp_path / "sample.csv"
    csv_file.write_text("id,val\n1,10\n", encoding="utf-8")

    err = _make_provider_error()
    monkeypatch.setattr("evals.run_evals._require_gemini_api_key", lambda: None)
    monkeypatch.setattr("evals.run_evals.run_gold_sql", lambda sql, p: pd.DataFrame({"val": [10]}))
    monkeypatch.setattr("evals.run_evals.ask_agent", lambda q, p: (_ for _ in ()).throw(err))

    questions = [
        {"id": "q1", "split": "dev", "answerable": True, "question": "Total?", "gold_sql": "SELECT 10"}
    ]

    results = run_evaluation(questions, csv_file, runs=1)

    assert results["total_cases"] == 1
    assert results["correct_cases"] == 0
    assert results["incorrect_cases"] == 0
    assert results["provider_errors"] == 1
    assert results["planning_errors"] == 0
    assert results["evaluable_cases"] == 0
    assert results["accuracy_percent"] is None
    assert results["case_results"][0]["outcome"] == "provider_error"
    assert results["case_results"][0]["correct"] is None


def test_run_evaluation_provider_error_on_unanswerable_question(monkeypatch, tmp_path):
    """
    Provider errors on unanswerable questions must be recorded as provider_error
    and must NEVER be falsely credited as a correct refusal.
    """
    from evals.run_evals import run_evaluation

    csv_file = tmp_path / "sample.csv"
    csv_file.write_text("id,val\n1,10\n", encoding="utf-8")

    err = _make_provider_error()
    monkeypatch.setattr("evals.run_evals._require_gemini_api_key", lambda: None)
    monkeypatch.setattr("evals.run_evals.ask_agent", lambda q, p: (_ for _ in ()).throw(err))

    questions = [
        {"id": "u1", "split": "dev", "answerable": False, "question": "Predict?", "gold_sql": None}
    ]

    results = run_evaluation(questions, csv_file, runs=1)

    assert results["total_cases"] == 1
    assert results["correct_cases"] == 0
    assert results["incorrect_cases"] == 0
    assert results["provider_errors"] == 1
    assert results["planning_errors"] == 0
    assert results["evaluable_cases"] == 0
    assert results["accuracy_percent"] is None
    assert results["case_results"][0]["outcome"] == "provider_error"
    assert results["case_results"][0]["correct"] is None


def test_run_evaluation_genuine_planning_error(monkeypatch, tmp_path):
    """
    Genuine planning failures (where the model returns invalid SQL) must be recorded
    as planning_error, counted in evaluable cases, and marked incorrect.
    """
    from evals.run_evals import run_evaluation

    csv_file = tmp_path / "sample.csv"
    csv_file.write_text("id,val\n1,10\n", encoding="utf-8")

    err = _make_planning_error()
    monkeypatch.setattr("evals.run_evals._require_gemini_api_key", lambda: None)
    monkeypatch.setattr("evals.run_evals.run_gold_sql", lambda sql, p: pd.DataFrame({"val": [10]}))
    monkeypatch.setattr("evals.run_evals.ask_agent", lambda q, p: (_ for _ in ()).throw(err))

    questions = [
        {"id": "q1", "split": "dev", "answerable": True, "question": "Total?", "gold_sql": "SELECT 10"}
    ]

    results = run_evaluation(questions, csv_file, runs=1)

    assert results["total_cases"] == 1
    assert results["correct_cases"] == 0
    assert results["incorrect_cases"] == 0
    assert results["planning_errors"] == 1
    assert results["provider_errors"] == 0
    assert results["evaluable_cases"] == 1
    assert results["accuracy_percent"] == 0.0
    assert results["case_results"][0]["outcome"] == "planning_error"
    assert results["case_results"][0]["correct"] is False


def test_run_evaluation_zero_evaluable_cases(monkeypatch, tmp_path):
    """
    When all runs encounter provider errors (or when no questions are run),
    evaluable_cases must be 0 and accuracy_percent must be None without division by zero.
    """
    from evals.run_evals import run_evaluation

    csv_file = tmp_path / "sample.csv"
    csv_file.write_text("id,val\n1,10\n", encoding="utf-8")

    err = _make_provider_error()
    monkeypatch.setattr("evals.run_evals._require_gemini_api_key", lambda: None)
    monkeypatch.setattr("evals.run_evals.run_gold_sql", lambda sql, p: pd.DataFrame({"val": [10]}))
    monkeypatch.setattr("evals.run_evals.ask_agent", lambda q, p: (_ for _ in ()).throw(err))

    questions = [
        {"id": "q1", "split": "dev", "answerable": True, "question": "Total?", "gold_sql": "SELECT 10"},
        {"id": "q2", "split": "dev", "answerable": True, "question": "Count?", "gold_sql": "SELECT 10"},
    ]

    results = run_evaluation(questions, csv_file, runs=1)

    assert results["total_cases"] == 2
    assert results["evaluable_cases"] == 0
    assert results["provider_errors"] == 2
    assert results["correct_cases"] == 0
    assert results["accuracy_percent"] is None

    # Also test empty question list
    results_empty = run_evaluation([], csv_file, runs=1)
    assert results_empty["total_cases"] == 0
    assert results_empty["evaluable_cases"] == 0
    assert results_empty["accuracy_percent"] is None


def test_run_evaluation_genuine_safe_refusal_on_unanswerable(monkeypatch, tmp_path):
    """
    Genuine safe refusals by the agent on unanswerable questions must be
    credited as correct_cases and achieve 100% accuracy.
    """
    from evals.run_evals import run_evaluation, AgentRefused

    csv_file = tmp_path / "sample.csv"
    csv_file.write_text("id,val\n1,10\n", encoding="utf-8")

    refused = AgentRefused("This question references a column not present in the dataset.")
    monkeypatch.setattr("evals.run_evals._require_gemini_api_key", lambda: None)
    monkeypatch.setattr("evals.run_evals.ask_agent", lambda q, p: (_ for _ in ()).throw(refused))

    questions = [
        {"id": "u1", "split": "dev", "answerable": False, "question": "Join non-existent?", "gold_sql": None}
    ]

    results = run_evaluation(questions, csv_file, runs=1)

    assert results["total_cases"] == 1
    assert results["correct_cases"] == 1
    assert results["incorrect_cases"] == 0
    assert results["provider_errors"] == 0
    assert results["planning_errors"] == 0
    assert results["evaluable_cases"] == 1
    assert results["accuracy_percent"] == 100.0
    assert results["case_results"][0]["correct"] is True
    assert "agent_refused:" in results["case_results"][0]["outcome"]


def test_run_evaluation_correct_accuracy_denominator(monkeypatch, tmp_path):
    """
    Accuracy denominator must strictly be evaluable_cases (total_cases - provider_errors).
    Provider errors must never reduce the calculated accuracy percentage by inflating
    the denominator.
    """
    from evals.run_evals import run_evaluation

    csv_file = tmp_path / "sample.csv"
    csv_file.write_text("id,val\n1,10\n", encoding="utf-8")

    monkeypatch.setattr("evals.run_evals._require_gemini_api_key", lambda: None)
    monkeypatch.setattr("evals.run_evals.run_gold_sql", lambda sql, p: pd.DataFrame({"val": [10]}))

    provider_503 = _make_provider_error()
    planning_err = _make_planning_error()

    # 10 questions: 4 correct, 2 incorrect (mismatch), 1 planning error, 3 provider errors
    # Total = 10, Provider = 3 -> Evaluable = 7.
    # Accuracy = 4 / 7 * 100 = 57.142857...
    # If denominator were wrongly total_cases (10), accuracy would be 40.0%.
    responses = [
        pd.DataFrame({"val": [10]}),   # 1: correct
        pd.DataFrame({"val": [10]}),   # 2: correct
        pd.DataFrame({"val": [10]}),   # 3: correct
        pd.DataFrame({"val": [10]}),   # 4: correct
        pd.DataFrame({"val": [999]}),  # 5: incorrect mismatch
        pd.DataFrame({"val": [999]}),  # 6: incorrect mismatch
        planning_err,                  # 7: planning error
        provider_503,                  # 8: provider error
        provider_503,                  # 9: provider error
        provider_503,                  # 10: provider error
    ]

    call_index = [0]
    def mock_ask(q, p):
        idx = call_index[0]
        call_index[0] += 1
        resp = responses[idx]
        if isinstance(resp, BaseException):
            raise resp
        return resp

    monkeypatch.setattr("evals.run_evals.ask_agent", mock_ask)

    questions = [
        {"id": f"q{i}", "split": "dev", "answerable": True, "question": f"Q{i}?", "gold_sql": "SELECT 10"}
        for i in range(1, 11)
    ]

    results = run_evaluation(questions, csv_file, runs=1)

    assert results["total_cases"] == 10
    assert results["correct_cases"] == 4
    assert results["incorrect_cases"] == 2
    assert results["planning_errors"] == 1
    assert results["provider_errors"] == 3
    assert results["evaluable_cases"] == 7
    assert results["accuracy_percent"] == pytest.approx((4 / 7) * 100.0)
    assert results["accuracy_percent"] != pytest.approx(40.0)  # Not 4 / 10


def test_run_evaluation_provider_failures_not_counted_as_incorrect(monkeypatch, tmp_path):
    """
    Provider outages (503), timeouts, and network failures must be recorded as
    provider_error, excluded from incorrect_cases, assigned correct=None in case results,
    and must not lower the accuracy percentage of other valid runs.
    """
    from evals.run_evals import run_evaluation

    csv_file = tmp_path / "sample.csv"
    csv_file.write_text("id,val\n1,10\n", encoding="utf-8")

    monkeypatch.setattr("evals.run_evals._require_gemini_api_key", lambda: None)
    monkeypatch.setattr("evals.run_evals.run_gold_sql", lambda sql, p: pd.DataFrame({"val": [10]}))

    provider_503 = _make_provider_error()
    provider_timeout = _make_timeout_error()
    provider_network = _make_network_error()

    # 2 correct answers, 3 provider failures (503, timeout, network)
    responses = [
        pd.DataFrame({"val": [10]}),  # correct
        provider_503,                 # 503 outage
        pd.DataFrame({"val": [10]}),  # correct
        provider_timeout,             # timeout
        provider_network,             # network error
    ]

    call_index = [0]
    def mock_ask(q, p):
        idx = call_index[0]
        call_index[0] += 1
        resp = responses[idx]
        if isinstance(resp, BaseException):
            raise resp
        return resp

    monkeypatch.setattr("evals.run_evals.ask_agent", mock_ask)

    questions = [
        {"id": "q1", "split": "dev", "answerable": True, "question": "Total?", "gold_sql": "SELECT 10"},
        {"id": "q2", "split": "dev", "answerable": True, "question": "503?", "gold_sql": "SELECT 10"},
        {"id": "q3", "split": "dev", "answerable": True, "question": "Count?", "gold_sql": "SELECT 10"},
        {"id": "q4", "split": "dev", "answerable": True, "question": "Timeout?", "gold_sql": "SELECT 10"},
        {"id": "q5", "split": "dev", "answerable": True, "question": "Network?", "gold_sql": "SELECT 10"},
    ]

    results = run_evaluation(questions, csv_file, runs=1)

    assert results["total_cases"] == 5
    assert results["correct_cases"] == 2
    assert results["incorrect_cases"] == 0      # Provider errors must NOT be counted as incorrect
    assert results["provider_errors"] == 3
    assert results["planning_errors"] == 0
    assert results["evaluable_cases"] == 2
    assert results["accuracy_percent"] == 100.0  # 2 correct / 2 evaluable = 100%

    # Verify that every provider failure has correct=None (not False)
    provider_cases = [r for r in results["case_results"] if r["outcome"] == "provider_error"]
    assert len(provider_cases) == 3
    for case in provider_cases:
        assert case["correct"] is None
