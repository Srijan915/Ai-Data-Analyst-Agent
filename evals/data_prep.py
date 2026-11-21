"""Canonical dataset preparation shared by both sides of live evaluation."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from app.core.data_loader import load_csv
from app.core.execution_preparation import prepare_execution_dataframe


def _snake_case(column_name: str) -> str:
    """Convert one raw CSV column name to a DuckDB-friendly identifier."""
    snake_case_name = re.sub(r"[^a-z0-9]+", "_", column_name.lower())
    return re.sub(r"_+", "_", snake_case_name).strip("_")


def load_for_gold(csv_path: str | Path) -> pd.DataFrame:
    """Return the exact prepared schema used by the live-evaluation agent.

    Gold queries deliberately use stable snake-case identifiers.  The agent
    must therefore receive the *same* canonical names and typed values; using
    ``pandas.read_csv`` for gold SQL while using the application loader for
    the agent made comparisons depend on two different datasets.
    """
    path = Path(csv_path)
    with path.open("rb") as source:
        dataframe = load_csv(source)

    dataframe = prepare_execution_dataframe(dataframe)
    dataframe = dataframe.copy()
    dataframe.columns = [_snake_case(str(column)) for column in dataframe.columns]
    if dataframe.columns.duplicated().any():
        duplicates = dataframe.columns[dataframe.columns.duplicated()].unique()
        raise ValueError(
            "Evaluation column normalization produced duplicate names: "
            + ", ".join(map(str, duplicates))
        )
    return dataframe
