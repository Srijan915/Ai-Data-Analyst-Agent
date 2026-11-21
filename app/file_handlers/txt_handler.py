"""Delimited TXT tables or intact document text."""
import csv
from collections import Counter

import pandas as pd

from app.file_handlers.csv_handler import decode_text
from app.utils.dataset_resources import check_dataframe_limits


def _table_delimiter(lines: list[str]) -> str | None:
    nonempty = [line for line in lines if line.strip()]
    if len(nonempty) < 2:
        return None
    for delimiter in ("\t", ",", ";", "|"):
        counts = Counter(line.count(delimiter) for line in nonempty)
        count, frequency = counts.most_common(1)[0]
        if count >= 1 and frequency > len(nonempty) / 2:
            return delimiter
    return None


def load_txt_bytes(data: bytes, name: str) -> list[dict]:
    text = decode_text(data)
    lines = text.splitlines()
    delimiter = _table_delimiter(lines)
    if delimiter is None:
        return [{"name": name, "text": text}]
    try:
        rows = [next(csv.reader([line], delimiter=delimiter, strict=True), [])
                for line in lines]
    except csv.Error:
        return [{"name": name, "text": text}]
    header_index = next((index for index, row in enumerate(rows) if row), None)
    if header_index is None:
        return [{"name": name, "text": text}]
    width = max(len(row) for row in rows)
    header = rows[header_index]
    headers = header + [f"extra_{index}" for index in range(1, width - len(header) + 1)]
    values = [row + [""] * (width - len(row))
              for index, row in enumerate(rows) if index != header_index]
    frame = pd.DataFrame(values, columns=headers, dtype=str)
    if frame.empty:
        return [{"name": name, "text": text}]
    check_dataframe_limits(frame)
    return [{"name": name, "dataframe": frame}]
