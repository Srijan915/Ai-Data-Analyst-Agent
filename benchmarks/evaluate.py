"""Measure preprocessing correctness and median runtime on seeded copies."""
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import json
import platform
from pathlib import Path
from statistics import median
import sys
from time import perf_counter

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from app.preprocessing.processor import process_dataframe

DATASETS = Path(__file__).resolve().parent / "datasets"
EXPECTED = Path(__file__).resolve().parent / "expected_outputs"
REPORTS = ROOT / "reports"
REPEATS = 5


def _duplicate_metrics(expected, processed):
    injected = set(expected["duplicate_indices"])
    if not injected:
        return None, None, None
    removed = set(range(expected["dataset_rows"])) - set(processed.index)
    correct = len(injected & removed)
    precision = correct / len(removed) if removed else 0.0
    recall = correct / len(injected)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _type_accuracy(frame, details, expected):
    if "Correct data types" not in expected["selected_operations"]:
        return None
    wanted = set(expected["convertible_columns"])
    actual = {item["column"] for item in details["converted_columns"]}
    return sum((column in wanted) == (column in actual) for column in frame.columns) / len(frame.columns)


def _text_success(processed, details, expected):
    cells = expected["whitespace_cells"]
    if not cells:
        return None
    renamed = {item["from"]: item["to"] for item in details["renamed_columns"]}
    correct = sum(cell["row"] in processed.index and
                  processed.at[cell["row"], renamed.get(cell["column"], cell["column"])] == cell["expected"]
                  for cell in cells)
    return correct / len(cells)


def _same_value(before, after, operation, converted):
    if str(before) == str(after):
        return True
    if isinstance(before, str) and not before.strip() and pd.isna(after):
        return True
    if "Standardize text values" in operation and isinstance(before, str) and before.strip() == str(after):
        return True
    if converted:
        try:
            return Decimal(str(before)) == Decimal(str(after))
        except (InvalidOperation, ValueError):
            return str(before) == str(after)[:10]
    return False


def _integrity_violations(frame, processed, details, expected):
    renamed = {item["from"]: item["to"] for item in details["renamed_columns"]}
    converted = {item["column"] for item in details["converted_columns"]}
    operations = expected["selected_operations"]
    violations = 0
    for index in processed.index:
        for column in frame.columns:
            before, after = frame.at[index, column], processed.at[index, renamed.get(column, column)]
            if not _same_value(before, after, operations, column in converted and column != "Customer ID"):
                violations += 1
    for index in expected["leading_zero_indices"]:
        if index in processed.index and str(processed.at[index, "customer_id" if "Clean column names" in operations
                                          else "Customer ID"]) != frame.at[index, "Customer ID"]:
            violations += 1
    return violations


def evaluate_one(name):
    expected = json.loads((EXPECTED / f"{name}.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(DATASETS / f"{name}.csv", dtype=str, keep_default_na=False)
    durations = []
    for _ in range(REPEATS):
        start = perf_counter()
        processed, details = process_dataframe(frame, expected["selected_operations"])
        durations.append(perf_counter() - start)
    seconds = median(durations)
    precision, recall, f1 = _duplicate_metrics(expected, processed)
    return {"dataset": name, "dataset_rows": len(frame), "seed": expected["seed"],
            "run_date_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": platform.python_version(), "pandas_version": pd.__version__,
            "machine": platform.platform(), "repeats": REPEATS,
            "duplicate_precision": precision, "duplicate_recall": recall,
            "duplicate_f1": f1, "type_conversion_accuracy": _type_accuracy(frame, details, expected),
            "text_success_rate": _text_success(processed, details, expected),
            "data_integrity_violations": _integrity_violations(frame, processed, details, expected),
            "median_seconds": seconds, "records_per_second": len(frame) / seconds}


def main():
    names = ("duplicates", "whitespace", "convertible", "messy_names", "combined")
    missing = [name for name in names if not (DATASETS / f"{name}.csv").exists()
               or not (EXPECTED / f"{name}.json").exists()]
    if missing:
        raise FileNotFoundError("Generate benchmark copies first: python benchmarks/run_benchmark.py")
    results = [evaluate_one(name) for name in names]
    REPORTS.mkdir(exist_ok=True)
    pd.DataFrame(results).to_csv(REPORTS / "benchmark_results.csv", index=False)
    (REPORTS / "benchmark_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    for row in results:
        print(f"{row['dataset']}: {row['median_seconds']:.4f}s median, "
              f"{row['records_per_second']:.0f} rows/s, integrity violations={row['data_integrity_violations']}")
    print("Exact-duplicate detection is expected to score about 1.0 by design: this is a correctness check, not a competitive result.")


if __name__ == "__main__":
    main()
