"""Deterministic checks around executed results and generated narratives."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from app.models.schemas import QueryResult


@dataclass(frozen=True)
class ResultVerification:
    warnings: list[str] = field(default_factory=list)


def verify_query_result(result: QueryResult) -> ResultVerification:
    """Reject internally inconsistent result metadata before presentation."""
    if not isinstance(result, QueryResult):
        raise TypeError("result must be a QueryResult")
    frame = result.dataframe
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("result dataframe is invalid")
    if frame.columns.duplicated().any():
        raise ValueError("result has duplicate column names")
    if result.row_count < len(frame):
        raise ValueError("result row metadata is inconsistent")
    warnings = []
    if frame.empty:
        warnings.append("The validated query returned no rows; check filters or data coverage.")
    if result.truncated:
        warnings.append("The displayed result reached the configured row limit.")
    return ResultVerification(warnings=warnings)


def explanation_is_grounded(explanation: str, result: QueryResult) -> bool:
    """Allow only numeric claims visible in the bounded result values.

    This is intentionally conservative: a generated explanation with a new
    numeric claim is suppressed instead of being presented as a fact.
    """
    if not isinstance(explanation, str) or not explanation.strip():
        return False
    values = {str(result.row_count)}
    for value in result.dataframe.to_numpy().ravel():
        if pd.isna(value):
            continue
        values.add(str(value))
        if isinstance(value, (int, float)):
            values.add(f"{value:g}")
    claims = re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:%|[kKmMbB])?", explanation)
    for claim in claims:
        normalized = claim.rstrip("%kKmMbB")
        if claim.endswith("%") or normalized not in values:
            return False
    return True
