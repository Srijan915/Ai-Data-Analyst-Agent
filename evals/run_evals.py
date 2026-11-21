"""Small, reproducible helpers for live agent evaluations.

This module deliberately does not store credentials or call the agent at
import time.  Run it only after setting GEMINI_API_KEY in the environment.
"""

from __future__ import annotations

import argparse
import json
import os
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

if __name__ == "__main__":
    # Live evaluation accepts an explicitly supplied environment key only.
    # data_prep imports application configuration, which otherwise loads .env.
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"

from evals.data_prep import load_for_gold


class AgentRefused(RuntimeError):
    """Raised when the agent safely declines, clarifies, or blocks a question."""

    def __init__(self, message: str, generated_sql: str | None = None):
        super().__init__(message)
        self.generated_sql = generated_sql


# Reuse the prepared agent for questions against the same CSV.  Creating an
# agent profiles and prepares the dataset, so doing that once avoids work and
# keeps every question in an evaluation run on the same prepared data.
_AGENT_CACHE: dict[Path, Any] = {}
_GOLD_DATAFRAME_CACHE: dict[Path, pd.DataFrame] = {}


def _require_gemini_api_key() -> None:
    """Stop before a live evaluation when no key was supplied by the user."""
    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Set it in your environment before "
            "running a live evaluation."
        )


def _get_agent(csv_path: Path) -> Any:
    """Load one canonical evaluation CSV for the app agent."""
    resolved_path = csv_path.resolve()
    cached_agent = _AGENT_CACHE.get(resolved_path)
    if cached_agent is not None:
        return cached_agent

    # Imports stay here so comparison helpers remain fully offline and can be
    # tested without loading application configuration or an API client.
    from app.core.agent import DataAnalystAgent
    from app.core.data_profiler import profile_dataframe

    # This must use the same prepared dataframe as ``run_gold_sql``.  Gold
    # SQL has reviewed snake-case identifiers, so evaluating the model against
    # the upload's display headers would be a schema mismatch, not an answer.
    dataframe = _get_gold_dataframe(resolved_path).copy()

    agent = DataAnalystAgent(
        dataframe=dataframe,
        dataset_profile=profile_dataframe(dataframe),
    )
    _AGENT_CACHE[resolved_path] = agent
    return agent


def ask_agent(question: str, csv_path: str | Path) -> pd.DataFrame:
    """Run one question and return only its result dataframe.

    AgentRefused separates safe refusals, clarification requests, and SQL
    security blocks from normal answer mismatches in a live evaluation.
    """
    _require_gemini_api_key()

    # Import the exception types only for a real agent request.  This keeps
    # normalize() and results_match() offline-only functions.
    from app.core.agent import AgentDataError, AgentValidationError, AgentError

    agent = _get_agent(Path(csv_path))
    try:
        result = agent.run(
            question=question,
            generate_explanation=False,
            generate_chart_spec=False,
        )
    except (AgentDataError, AgentValidationError) as exc:
        raise AgentRefused(
            str(exc),
            generated_sql=agent.last_generated_sql,
        ) from None
    except AgentError as exc:
        exc.generated_sql = agent.last_generated_sql
        raise

    dataframe = result.query_result.dataframe.copy()
    # DataFrame.attrs survives the compatibility-preserving DataFrame return
    # and lets the harness retain the generated SQL without changing its
    # public helper contract or exposing it to ordinary application users.
    dataframe.attrs["generated_sql"] = result.sql
    return dataframe


def release_cached_agents() -> None:
    """Release resources held by CSV agents after a command-line run."""
    for agent in _AGENT_CACHE.values():
        agent.release_dataset()
    _AGENT_CACHE.clear()


def _get_gold_dataframe(csv_path: Path) -> pd.DataFrame:
    """Load each evaluation CSV once for DuckDB gold-query execution."""
    resolved_path = csv_path.resolve()
    cached_dataframe = _GOLD_DATAFRAME_CACHE.get(resolved_path)
    if cached_dataframe is None:
        cached_dataframe = load_for_gold(resolved_path)
        _GOLD_DATAFRAME_CACHE[resolved_path] = cached_dataframe
    return cached_dataframe


def run_gold_sql(sql: str, csv_path: str | Path) -> pd.DataFrame:
    """Run one reviewed gold query against the snake-case ``data`` table."""
    connection = duckdb.connect(":memory:")
    try:
        connection.register("data", _get_gold_dataframe(Path(csv_path)))
        return connection.execute(sql).fetchdf()
    finally:
        connection.close()


def _normalize_cell(value: Any, decimal_places: int) -> Any:
    """Make one cell comparable across ordering, float, and NULL differences."""
    if pd.isna(value):
        return None
    if isinstance(value, Integral) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, Real) and not isinstance(value, bool):
        return round(float(value), decimal_places)
    return value


def _row_sort_key(row: pd.Series) -> tuple[tuple[str, str], ...]:
    """Return a type-aware sortable key without changing result values."""
    return tuple(
        (type(value).__name__, repr(value))
        for value in row
    )


def normalize(dataframe: pd.DataFrame, decimal_places: int = 6) -> pd.DataFrame:
    """Return a copy with rounded floats, common NULLs, and sorted rows."""
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame")
    if decimal_places < 0:
        raise ValueError("decimal_places must be greater than or equal to zero")

    normalized = dataframe.copy().astype(object)
    for column in normalized.columns:
        normalized[column] = normalized[column].map(
            lambda value: _normalize_cell(value, decimal_places)
        )

    row_order = sorted(
        range(len(normalized)),
        key=lambda index: _row_sort_key(normalized.iloc[index]),
    )
    return normalized.iloc[row_order].reset_index(drop=True).astype(object)


def results_match(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    decimal_places: int = 6,
    allow_extra_columns: bool = False,
) -> bool:
    """Compare result values while ignoring output aliases and row order.

    SQL aliases are presentation metadata: a logically correct query can use
    ``total`` where gold SQL calls the same expression ``total_sales``.  The
    harness preserves alias differences in its diagnostics, but does not turn
    them into false model failures.  Column position and count remain part of
    the contract so differently shaped answers do not match accidentally.
    """
    if not isinstance(actual, pd.DataFrame) or not isinstance(expected, pd.DataFrame):
        return False
    if actual.shape[1] != expected.shape[1]:
        # A result may include harmless additional fields while still
        # containing every specifically requested, identically named output
        # column. Compare that projection rather than treating presentation
        # verbosity as a factual error. Ambiguous/renamed schemas still fail.
        expected_names = list(expected.columns)
        if not allow_extra_columns or (
            len(set(expected_names)) != len(expected_names)
            or any(name not in actual.columns for name in expected_names)
        ):
            return False
        actual = actual.loc[:, expected_names]
    comparable_actual = actual.copy()
    comparable_expected = expected.copy()
    comparable_actual.columns = range(comparable_actual.shape[1])
    comparable_expected.columns = range(comparable_expected.shape[1])
    return normalize(comparable_actual, decimal_places).equals(
        normalize(comparable_expected, decimal_places)
    )


def _dataframe_diagnostic(dataframe: pd.DataFrame | None) -> dict[str, Any] | None:
    """Return a JSON-safe full result representation for an evaluation case."""
    if dataframe is None:
        return None
    return {
        "columns": [str(column) for column in dataframe.columns],
        "dtypes": {str(column): str(dtype) for column, dtype in dataframe.dtypes.items()},
        "rows": json.loads(dataframe.to_json(orient="records", date_format="iso")),
    }


def _mismatch_reason(actual: pd.DataFrame, expected: pd.DataFrame) -> str:
    """Explain the first comparison boundary that prevented a match."""
    if actual.shape[1] != expected.shape[1]:
        return (
            "column_count_mismatch: "
            f"actual={actual.shape[1]}, expected={expected.shape[1]}"
        )
    if len(actual) != len(expected):
        return f"row_count_mismatch: actual={len(actual)}, expected={len(expected)}"
    if list(actual.columns) != list(expected.columns):
        return "column_alias_mismatch_and_values_mismatch"
    return "normalized_values_mismatch"


def _load_questions(questions_path: Path, split: str) -> list[dict[str, Any]]:
    """Load only the requested split from the reviewed question file."""
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    selected_questions = [
        question for question in questions if question.get("split") == split
    ]
    if not selected_questions:
        raise ValueError(f"No questions found for split: {split}")
    return selected_questions


def _is_provider_failure(exc: BaseException) -> bool:
    """Return True if exc was caused by an LLM provider or infrastructure failure."""
    from app.core.llm_client import (
        LLMAccessDeniedError,
        LLMAPIError,
        LLMClientError,
        LLMConfigurationError,
        LLMCredentialsError,
        LLMNetworkError,
        LLMThrottlingError,
    )

    provider_error_types = (
        LLMAPIError,
        LLMNetworkError,
        LLMThrottlingError,
        LLMCredentialsError,
        LLMAccessDeniedError,
        LLMConfigurationError,
        LLMClientError,
    )

    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, provider_error_types):
            return True
        current = current.__cause__ or current.__context__
    return False


def run_evaluation(
    questions: list[dict[str, Any]],
    csv_path: Path,
    runs: int,
) -> dict[str, Any]:
    """Compare live agent results with gold SQL across a fixed question list."""
    _require_gemini_api_key()
    from app.core.agent import AgentError, AgentPlanningError
    case_results = []

    for question_case in questions:
        for run_number in range(1, runs + 1):
            question = question_case["question"]
            expected: pd.DataFrame | None = None
            actual: pd.DataFrame | None = None
            generated_sql: str | None = None
            failure_reason: str | None = None
            if question_case["answerable"]:
                expected = run_gold_sql(question_case["gold_sql"], csv_path)
                try:
                    actual = ask_agent(question, csv_path)
                    generated_sql = actual.attrs.get("generated_sql")
                    # Reviewed gold SQL presents aggregate values to two
                    # decimal places.  Compare at that published precision so
                    # equivalent full-precision DuckDB aggregates are not
                    # scored as wrong answers solely for formatting.
                    correct = results_match(
                        actual,
                        expected,
                        decimal_places=2,
                        allow_extra_columns=False,
                    )
                    outcome = "matched" if correct else "result_mismatch"
                    if not correct:
                        failure_reason = _mismatch_reason(actual, expected)
                except AgentRefused as exc:
                    correct = False
                    outcome = f"agent_refused: {exc}"
                    generated_sql = exc.generated_sql
                    failure_reason = str(exc)
                except AgentPlanningError as exc:
                    generated_sql = getattr(exc, "generated_sql", None)
                    if _is_provider_failure(exc):
                        correct = None
                        outcome = "provider_error"
                    else:
                        correct = False
                        outcome = "planning_error"
                    failure_reason = str(exc)
                except AgentError as exc:
                    generated_sql = getattr(exc, "generated_sql", None)
                    correct = False
                    outcome = "agent_error"
                    failure_reason = str(exc)
            else:
                try:
                    actual = ask_agent(question, csv_path)
                    generated_sql = actual.attrs.get("generated_sql")
                    correct = False
                    outcome = "agent_answered_unanswerable_question"
                    failure_reason = "agent returned a result for an unanswerable question"
                except AgentRefused as exc:
                    correct = True
                    outcome = f"agent_refused: {exc}"
                    generated_sql = exc.generated_sql
                    failure_reason = str(exc)
                except AgentPlanningError as exc:
                    generated_sql = getattr(exc, "generated_sql", None)
                    if _is_provider_failure(exc):
                        correct = None
                        outcome = "provider_error"
                    else:
                        correct = False
                        outcome = "planning_error"
                    failure_reason = str(exc)
                except AgentError as exc:
                    generated_sql = getattr(exc, "generated_sql", None)
                    correct = False
                    outcome = "agent_error"
                    failure_reason = str(exc)

            case_results.append(
                {
                    "id": question_case["id"],
                    "run": run_number,
                    "answerable": question_case["answerable"],
                    "correct": correct,
                    "outcome": outcome,
                    "generated_sql": generated_sql,
                    "expected_result": _dataframe_diagnostic(expected),
                    "actual_result": _dataframe_diagnostic(actual),
                    "failure_reason": failure_reason,
                }
            )

    total_cases = len(case_results)
    correct_cases = sum(1 for result in case_results if result["correct"] is True)
    incorrect_cases = sum(
        1 for result in case_results
        if result["correct"] is False and result["outcome"] not in {"planning_error", "agent_error"}
    )
    provider_errors = sum(1 for result in case_results if result["outcome"] == "provider_error")
    planning_errors = sum(1 for result in case_results if result["outcome"] == "planning_error")
    evaluable_cases = total_cases - provider_errors
    accuracy_percent = (
        (100.0 * correct_cases / evaluable_cases)
        if evaluable_cases > 0
        else None
    )

    return {
        "total_cases": total_cases,
        "correct_cases": correct_cases,
        "incorrect_cases": incorrect_cases,
        "provider_errors": provider_errors,
        "planning_errors": planning_errors,
        "evaluable_cases": evaluable_cases,
        "accuracy_percent": accuracy_percent,
        "total_runs": total_cases,
        "correct_runs": correct_cases,
        "case_results": case_results,
    }


def main() -> None:
    """Run either one smoke question or a reviewed question split."""
    parser = argparse.ArgumentParser(description="Run live agent evaluations.")
    parser.add_argument("--csv", type=Path, required=True, help="CSV to evaluate.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run exactly one question once.",
    )
    parser.add_argument(
        "--question",
        default="How many rows are in this dataset?",
        help="Question used by --smoke.",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("evals/questions.json"),
        help="Reviewed question JSON used for split evaluations.",
    )
    parser.add_argument(
        "--split",
        choices=["dev", "test"],
        help="Question split to run when --smoke is not used.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Live agent runs per question for a split evaluation.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        help="Optional JSON path for the measured split results.",
    )
    args = parser.parse_args()

    if not args.csv.is_file():
        parser.error(f"CSV file does not exist: {args.csv}")
    if args.runs <= 0:
        parser.error("--runs must be greater than zero.")

    if args.smoke:
        from app.core.agent import AgentPlanningError
        try:
            dataframe = ask_agent(args.question, args.csv)
        except AgentRefused as exc:
            print(f"Agent refused the smoke question: {exc}")
            raise SystemExit(2) from None
        except AgentPlanningError as exc:
            if _is_provider_failure(exc):
                print(f"Provider error during smoke test: {exc}")
            else:
                print(f"Agent planning failed: {exc}")
            raise SystemExit(1) from None
        except RuntimeError as exc:
            parser.error(str(exc))
        finally:
            release_cached_agents()
        print(dataframe.to_string(index=False))
        return

    if args.split is None:
        parser.error("Use --smoke or provide --split dev/test.")
    if not args.questions.is_file():
        parser.error(f"Question file does not exist: {args.questions}")

    try:
        questions = _load_questions(args.questions, args.split)
        results = run_evaluation(questions, args.csv, args.runs)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    finally:
        release_cached_agents()

    results_path = args.results or Path(f"evals/accuracy_results_{args.split}.json")
    results_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"correct runs: {results['correct_cases']} of {results['total_cases']}")
    print(f"evaluable cases: {results['evaluable_cases']}")
    if results["provider_errors"]:
        print(f"provider errors: {results['provider_errors']}")
    if results["planning_errors"]:
        print(f"planning errors: {results['planning_errors']}")
    print(f"accuracy percent: {results['accuracy_percent']}")
    print(f"saved results: {results_path}")


if __name__ == "__main__":
    main()
