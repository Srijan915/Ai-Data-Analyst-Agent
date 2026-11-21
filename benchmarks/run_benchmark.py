"""Create reproducible benchmark copies; never write to the source dataset."""
import json
import random
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/sample/customer_shopping_behavior.csv"
DATASETS = Path(__file__).resolve().parent / "datasets"
EXPECTED = Path(__file__).resolve().parent / "expected_outputs"
SEED = 42
NUMERIC_COLUMNS = ["Age", "Purchase Amount (USD)", "Review Rating", "Previous Purchases"]
OPERATIONS = ["Remove duplicates", "Handle missing values", "Correct data types",
              "Standardize text values", "Clean column names"]


def _add_whitespace(frame, count, rng):
    positions = rng.sample(range(len(frame)), count)
    for index in positions:
        frame.at[index, "Category"] = f"  {frame.at[index, 'Category']}  "
    return [{"row": index, "column": "Category", "expected": frame.at[index, "Category"].strip()}
            for index in positions]


def _add_duplicates(frame, count, rng):
    selected = rng.sample(range(len(frame)), count)
    start = len(frame)
    return pd.concat([frame, frame.iloc[selected]], ignore_index=True), list(range(start, start + count))


def _variant(base, name, rng):
    frame = base.copy(deep=True)
    expected = {"seed": SEED, "source_rows": len(base), "duplicate_indices": [],
                "whitespace_cells": [], "messy_columns": {}, "leading_zero_indices": [],
                "convertible_columns": NUMERIC_COLUMNS}
    if name in {"whitespace", "combined"}:
        expected["whitespace_cells"] = _add_whitespace(frame, 50, rng)
    if name in {"convertible", "combined"}:
        indices = rng.sample(range(len(frame)), 20)
        for index in indices:
            frame.at[index, "Customer ID"] = "00" + frame.at[index, "Customer ID"]
        expected["leading_zero_indices"] = indices
    if name in {"duplicates", "combined"}:
        frame, expected["duplicate_indices"] = _add_duplicates(frame, 30, rng)
    if name in {"messy_names", "combined"}:
        expected["messy_columns"] = {"Category": " Category $ ", "Gender": "GENDER"}
        frame.rename(columns=expected["messy_columns"], inplace=True)
        expected["whitespace_cells"] = [{**cell, "column": expected["messy_columns"][cell["column"]]}
                                        for cell in expected["whitespace_cells"]]
    expected["selected_operations"] = {
        "duplicates": [OPERATIONS[0]], "whitespace": [OPERATIONS[3]],
        "convertible": [OPERATIONS[2]], "messy_names": [OPERATIONS[4]],
        "combined": OPERATIONS,
    }[name]
    expected["dataset_rows"] = len(frame)
    return frame, expected


def generate():
    random.seed(SEED)
    rng = random.Random(SEED)
    base = pd.read_csv(SOURCE, dtype=str, keep_default_na=False)
    if base.duplicated().any():
        raise ValueError("The benchmark requires a source without exact duplicate rows.")
    DATASETS.mkdir(parents=True, exist_ok=True)
    EXPECTED.mkdir(parents=True, exist_ok=True)
    for name in ("duplicates", "whitespace", "convertible", "messy_names", "combined"):
        frame, expected = _variant(base, name, rng)
        frame.to_csv(DATASETS / f"{name}.csv", index=False)
        (EXPECTED / f"{name}.json").write_text(json.dumps(expected, indent=2), encoding="utf-8")
        print(f"{name}: {len(frame)} rows; expected output recorded")


if __name__ == "__main__":
    generate()
