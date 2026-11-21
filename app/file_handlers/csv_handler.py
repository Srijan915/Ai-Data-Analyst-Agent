"""Faithful small CSV parsing before the existing registration stage."""
import csv
from io import StringIO

import pandas as pd
from pandas.errors import EmptyDataError, ParserError

from app.utils.dataset_resources import check_dataframe_limits


def decode_text(data: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return data.decode(encoding).lstrip("\ufeff")
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not decode this text file.")


def delimiter_for(text: str) -> str:
    try:
        return csv.Sniffer().sniff(text[:4096], delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def load_csv_bytes(data: bytes, name: str) -> list[dict]:
    text = decode_text(data)
    delimiter = delimiter_for(text)
    try:
        rows = csv.reader(StringIO(text), delimiter=delimiter, strict=True)
        header = next(rows, None)
        if not header or not any(cell.strip() for cell in header):
            raise ValueError(f"{name} has no CSV header.")
        if any(row and len(row) != len(header) for row in rows):
            raise ValueError(f"{name} has rows with different column counts.")
        frame = pd.read_csv(StringIO(text), sep=delimiter, dtype=str,
                            keep_default_na=False, skip_blank_lines=False)
        if frame.empty:
            raise ValueError(f"{name} has no data rows.")
        check_dataframe_limits(frame)
    except (csv.Error, EmptyDataError, ParserError, UnicodeError) as exc:
        raise ValueError(f"Could not parse {name} as CSV: {exc}") from exc
    return [{"name": name, "dataframe": frame}]
