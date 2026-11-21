"""Execute every answerable gold query and show a compact result preview."""

from __future__ import annotations

import json
from pathlib import Path

from evals.run_evals import run_gold_sql


QUESTIONS_PATH = Path(__file__).with_name("questions.json")
SAMPLE_CSV = Path("data/sample/customer_shopping_behavior.csv")


def main() -> None:
    """Fail if a reviewed gold query errors or returns no rows."""
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    executed_count = 0

    for question_case in questions:
        if not question_case["answerable"]:
            continue

        result = run_gold_sql(question_case["gold_sql"], SAMPLE_CSV)
        if result.empty:
            raise ValueError(f"Gold query returned no rows: {question_case['id']}")

        executed_count += 1
        print(f"{question_case['id']}: {question_case['question']}")
        print(result.head().to_string(index=False))

    print(f"gold queries executed successfully: {executed_count}")


if __name__ == "__main__":
    main()
