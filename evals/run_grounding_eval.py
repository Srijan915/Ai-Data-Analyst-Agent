"""Offline measurement of numeric grounding on fixed result/explanation pairs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from app.core.result_verifier import explanation_is_grounded
from app.models.schemas import QueryResult


RESULTS_PATH = Path(__file__).with_name("grounding_results.json")


@dataclass(frozen=True)
class GroundingCase:
    """One result frame with a supported and one-number-corrupted sentence."""

    label: str
    dataframe: pd.DataFrame
    correct_explanation: str
    corrupted_explanation: str


def _case(
    label: str,
    value: Any,
    correct_explanation: str,
    corrupted_explanation: str,
) -> GroundingCase:
    """Build a one-value result frame used by most deterministic cases."""
    return GroundingCase(
        label=label,
        dataframe=pd.DataFrame({"value": [value]}),
        correct_explanation=correct_explanation,
        corrupted_explanation=corrupted_explanation,
    )


def build_grounding_cases() -> list[GroundingCase]:
    """Return thirty fixed cases spanning common numeric presentation formats."""
    cases = [
        _case("count", 7, "There are 7 records.", "There are 8 records."),
        _case("zero_count", 0, "The count is 0.", "The count is 2."),
        _case("integer", 42, "The total is 42.", "The total is 43."),
        _case("negative_integer", -5, "The change is -5.", "The change is -6."),
        _case("average", 12.5, "The average is 12.5.", "The average is 13.5."),
        _case("decimal", 3.14159, "The value is 3.14159.", "The value is 3.14158."),
        _case("quarter_percent", 0.25, "The share is 25%.", "The share is 26%."),
        _case("decimal_percent", 0.075, "The rate is 7.5%.", "The rate is 7.6%."),
        _case("currency", 1250, "Revenue is $1,250.", "Revenue is $1,251."),
        _case("currency_decimal", 99.95, "The cost is $99.95.", "The cost is $99.96."),
        _case("thousands", 1500, "The total is 1,500.", "The total is 1,501."),
        _case("large_thousands", 12500, "The total is 12,500.", "The total is 12,501."),
        _case("thousands_suffix", 12000, "The total is 12K.", "The total is 13K."),
        _case("decimal_thousands_suffix", 2500, "The total is 2.5K.", "The total is 2.6K."),
        _case("millions_suffix", 3000000, "The total is 3M.", "The total is 4M."),
        _case("decimal_millions_suffix", 1250000, "The total is 1.25M.", "The total is 1.26M."),
        _case("billions_suffix", 2000000000, "The total is 2B.", "The total is 3B."),
        _case("fraction", 0.125, "The ratio is 0.125.", "The ratio is 0.126."),
        _case("large_integer", 999999, "The total is 999999.", "The total is 999998."),
        _case("embedded_number_string", "Region 1234", "The code is 1,234.", "The code is 1,235."),
        _case("year", 2024, "The year is 2024.", "The year is 2025."),
        _case("whole_float", 1.0, "The result is 1.", "The result is 2."),
        _case("whole_percent", 100, "The completion is 100%.", "The completion is 101%."),
        _case("negative_percent", -0.1, "The change is -10%.", "The change is -11%."),
        _case("negative_thousands_suffix", -3000, "The change is -3K.", "The change is -4K."),
        _case("currency_millions", 2500000, "Revenue is $2.5M.", "Revenue is $2.6M."),
        _case("rounded_decimal", 10.25, "The average is 10.25.", "The average is 10.26."),
        _case("string_number", "batch-12", "The batch number is 12.", "The batch number is 13."),
        _case("plus_sign", 15, "The adjustment is +15.", "The adjustment is +16."),
    ]
    row_count_case = GroundingCase(
        label="row_count",
        dataframe=pd.DataFrame({"name": ["A", "B", "C"]}),
        correct_explanation="The result has 3 rows.",
        corrupted_explanation="The result has 4 rows.",
    )
    cases.append(row_count_case)
    assert len(cases) == 30
    return cases


def _query_result(dataframe: pd.DataFrame) -> QueryResult:
    """Construct the same result object used by the application pipeline."""
    return QueryResult(
        dataframe=dataframe,
        row_count=len(dataframe),
        truncated=False,
    )


def run_grounding_eval() -> dict[str, Any]:
    """Measure catches on corrupt sentences and false alarms on correct ones."""
    cases = build_grounding_cases()
    misses = []
    false_alarms = []

    for case in cases:
        result = _query_result(case.dataframe)
        correct_is_grounded = explanation_is_grounded(
            case.correct_explanation,
            result,
        )
        corrupted_is_grounded = explanation_is_grounded(
            case.corrupted_explanation,
            result,
        )

        if not correct_is_grounded:
            false_alarms.append(
                {
                    "label": case.label,
                    "sentence": case.correct_explanation,
                    "reason": "A correct numeric claim was rejected.",
                }
            )
        if corrupted_is_grounded:
            misses.append(
                {
                    "label": case.label,
                    "sentence": case.corrupted_explanation,
                    "reason": "The changed number was accepted as grounded.",
                }
            )

    return {
        "corrupted_total": len(cases),
        "corrupted_caught": len(cases) - len(misses),
        "correct_total": len(cases),
        "correct_false_alarms": len(false_alarms),
        "misses": misses,
        "false_alarms": false_alarms,
    }


def main() -> None:
    """Run the offline evaluation and save the exact observed results."""
    results = run_grounding_eval()
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print(
        "corrupted explanations caught: "
        f"{results['corrupted_caught']} of {results['corrupted_total']}"
    )
    print(
        "correct explanation false alarms: "
        f"{results['correct_false_alarms']} of {results['correct_total']}"
    )
    print("misses:")
    if not results["misses"]:
        print("  none")
    for case in results["misses"]:
        print(f"  {case['label']}: {case['sentence']} -- {case['reason']}")
    print("false alarms:")
    if not results["false_alarms"]:
        print("  none")
    for case in results["false_alarms"]:
        print(f"  {case['label']}: {case['sentence']} -- {case['reason']}")


if __name__ == "__main__":
    main()
