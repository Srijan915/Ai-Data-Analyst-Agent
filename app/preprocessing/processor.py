"""Five opt-in operations; every run starts from an untouched copy."""
from decimal import Decimal, InvalidOperation
import re

import pandas as pd

OPERATIONS = (
    "Remove duplicates", "Handle missing values", "Correct data types",
    "Standardize text values", "Clean column names",
)


def missing_mask(series: pd.Series) -> pd.Series:
    return series.isna() | series.map(lambda value: isinstance(value, str) and not value.strip())


def _identifier_name(name: str) -> bool:
    return bool(re.search(r"(?:^|[^a-z])(id|code|sku|zip|postal|account|phone)(?:$|[^a-z])",
                          name.lower()) or re.search(r"(?:id|code|sku|zip|postal)$", name.lower()))


def safe_conversion(series: pd.Series, name: str):
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return None, "already typed"
    if _identifier_name(name):
        return None, "identifier or code column"
    mask = missing_mask(series)
    values = series[~mask]
    if values.empty:
        return None, "no non-missing values"
    if values.map(lambda value: bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(value)))).all():
        try:
            return pd.to_datetime(series.where(~mask), format="%Y-%m-%d", errors="raise"), "ISO date"
        except (ValueError, TypeError):
            return None, "invalid ISO date"
    if values.map(lambda value: bool(re.match(r"[+-]?0\d", str(value).strip()))).any():
        return None, "leading-zero identifier"
    try:
        converted = pd.to_numeric(series.where(~mask), errors="raise")
        for original, number in zip(values, converted[~mask]):
            if pd.isna(number) or Decimal(str(original).strip()) != Decimal(str(number)):
                return None, "numeric conversion would lose a value or precision"
    except (ValueError, TypeError, InvalidOperation, OverflowError):
        return None, "not consistently numeric or strict ISO dates"
    return converted, "numeric"


def _remove_duplicates(frame: pd.DataFrame, details: dict) -> None:
    before = len(frame)
    frame.drop_duplicates(inplace=True)
    details["duplicates_removed"] = before - len(frame)


def _handle_missing(frame: pd.DataFrame) -> None:
    for index in range(len(frame.columns)):
        values = frame.iloc[:, index]
        frame.isetitem(index, values.mask(missing_mask(values), pd.NA))


def _correct_types(frame: pd.DataFrame, details: dict) -> None:
    for index, name in enumerate(frame.columns):
        converted, reason = safe_conversion(frame.iloc[:, index], str(name))
        if converted is None:
            details["skipped"].append({"operation": "Correct data types", "column": str(name), "reason": reason})
            continue
        frame.isetitem(index, converted)
        details["converted_columns"].append({"column": str(name), "type": reason})


def _standardize_text(frame: pd.DataFrame, details: dict) -> None:
    for index in range(len(frame.columns)):
        values = frame.iloc[:, index]
        if not (pd.api.types.is_object_dtype(values) or pd.api.types.is_string_dtype(values)):
            continue
        cleaned = values.map(lambda value: value.strip() if isinstance(value, str) else value)
        details["text_values_changed"] += sum(
            isinstance(value, str) and value != cleaned.iloc[position]
            for position, value in enumerate(values)
        )
        frame.isetitem(index, cleaned)


def _clean_names(frame: pd.DataFrame, details: dict) -> None:
    used, names = set(), []
    for old in frame.columns:
        base = re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", str(old).strip().lower())).strip("_") or "column"
        new, suffix = base, 2
        while new in used:
            new, suffix = f"{base}_{suffix}", suffix + 1
        used.add(new)
        names.append(new)
        if str(old) != new:
            details["renamed_columns"].append({"from": str(old), "to": new})
    frame.columns = names


def process_dataframe(original: pd.DataFrame, selected: list[str]):
    if any(operation not in OPERATIONS for operation in selected):
        raise ValueError("Unknown preprocessing operation.")
    frame = original.copy(deep=True)
    details = {"selected_operations": list(selected), "completed": [], "skipped": [],
               "duplicates_removed": 0, "converted_columns": [],
               "text_values_changed": 0, "renamed_columns": []}
    actions = {OPERATIONS[0]: lambda: _remove_duplicates(frame, details),
               OPERATIONS[1]: lambda: _handle_missing(frame),
               OPERATIONS[2]: lambda: _correct_types(frame, details),
               OPERATIONS[3]: lambda: _standardize_text(frame, details),
               OPERATIONS[4]: lambda: _clean_names(frame, details)}
    for operation in OPERATIONS:
        if operation in selected:
            actions[operation]()
            details["completed"].append(operation)
    return frame, details
