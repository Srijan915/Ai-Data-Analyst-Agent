"""Offline safety and answerability benchmark CLI."""

from __future__ import annotations

import sys
from app.core.offline_benchmark import benchmark_cases, run_offline_benchmark


def main() -> None:
    results = run_offline_benchmark()
    total = results["total"]
    passed = results["passed"]
    failed = results["failed"]
    print(f"offline benchmark: {passed} of {total} passed (failed: {failed})")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
