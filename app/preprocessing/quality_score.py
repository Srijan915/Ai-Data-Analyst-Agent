"""Five transparent structural-quality dimensions, each scored 0–100."""
import re

import pandas as pd

from app.preprocessing.processor import missing_mask, safe_conversion


def missing_count(frame: pd.DataFrame) -> int:
    return sum(int(missing_mask(frame.iloc[:, index]).sum()) for index in range(len(frame.columns)))


def _type_score(frame: pd.DataFrame) -> float:
    if not len(frame.columns):
        return 0.0
    flagged = sum(safe_conversion(frame.iloc[:, index], str(name))[0] is not None
                  for index, name in enumerate(frame.columns))
    return 100 * (len(frame.columns) - flagged) / len(frame.columns)


def _text_score(frame: pd.DataFrame) -> float:
    clean = total = 0
    for index in range(len(frame.columns)):
        values = frame.iloc[:, index]
        if not (pd.api.types.is_object_dtype(values) or pd.api.types.is_string_dtype(values)):
            continue
        for value in values[~missing_mask(values)]:
            if isinstance(value, str):
                total += 1
                clean += value == value.strip()
    return 100 * clean / total if total else 100.0


def _schema_score(frame: pd.DataFrame) -> float:
    if not len(frame.columns):
        return 0.0
    seen, clean = set(), 0
    for name in frame.columns:
        label = str(name)
        clean += bool(re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", label)) and label not in seen
        seen.add(label)
    return 100 * clean / len(frame.columns)


def quality_score(frame: pd.DataFrame) -> dict[str, float]:
    rows, columns = frame.shape
    cells = rows * columns
    dimensions = {
        "Completeness": 100 * (cells - missing_count(frame)) / cells if cells else 0.0,
        "Uniqueness": 100 * len(frame.drop_duplicates()) / rows if rows else 0.0,
        "Type consistency": _type_score(frame),
        "Text consistency": _text_score(frame),
        "Schema cleanliness": _schema_score(frame),
    }
    return {**dimensions, "Overall": sum(dimensions.values()) / 5}
