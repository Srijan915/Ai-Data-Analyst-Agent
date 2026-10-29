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

    values: set[str] = {str(result.row_count)}
    numeric_values: set[float] = {float(result.row_count)}

    def _register_number(num_str: str) -> None:
        values.add(num_str)
        try:
            num = float(num_str)
            numeric_values.add(num)
            if num.is_integer():
                values.add(str(int(num)))
            values.add(f"{num:g}")
        except (ValueError, OverflowError):
            pass

    for value in result.dataframe.to_numpy().ravel():
        if pd.isna(value):
            continue
        values.add(str(value))
        if isinstance(value, (int, float)):
            _register_number(str(value))
        elif isinstance(value, str):
            cleaned_str = re.sub(r"(?<=\d),(?=\d)", "", value)
            for embedded in re.findall(r"[-+]?\d+(?:\.\d+)?", cleaned_str):
                _register_number(embedded)

    # Strip structural list numbering / markdown headers (e.g. "1. ", "2) ", "### 1. ")
    text = re.sub(r"(?m)^\s*(?:#+\s*)?\d+[\.\)]\s+", "", explanation)
    # Strip common ordinals (e.g. "1st", "2nd", "3rd", "4th") so ranks are not treated as data claims
    text = re.sub(r"\b\d+(?:st|nd|rd|th)\b", "", text, flags=re.IGNORECASE)
    # Remove digit grouping commas (e.g. 1,000 -> 1000)
    text = re.sub(r"(?<=\d),(?=\d)", "", text)

    claims = re.findall(r"(?<![A-Za-z0-9_])[-+]?\d+(?:\.\d+)?(?:%|[kKmMbB])?", text)
    for claim in claims:
        if claim in values:
            continue

        raw = claim.rstrip("%kKmMbB")
        if raw in values:
            continue

        try:
            val = float(raw)
        except (ValueError, OverflowError):
            return False

        suffix = claim[len(raw):].lower()
        if suffix == "k":
            val *= 1_000
        elif suffix == "m":
            val *= 1_000_000
        elif suffix == "b":
            val *= 1_000_000_000

        candidates = [val]
        if suffix == "%":
            candidates.append(val / 100.0)

        matched = False
        for candidate in candidates:
            if candidate.is_integer() and str(int(candidate)) in values:
                matched = True
                break
            if f"{candidate:g}" in values:
                matched = True
                break
            if any(abs(candidate - nv) < 1e-5 for nv in numeric_values):
                matched = True
                break

        if not matched:
            return False

    return True
