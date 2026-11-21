"""Extract PDF tables and page text without OCR or SQL access to documents."""
from io import BytesIO
from pathlib import Path
import re

import pandas as pd
import pdfplumber

from app.utils.dataset_resources import check_dataframe_limits


def _table_frame(table) -> pd.DataFrame | None:
    rows = [list(row or []) for row in table]
    width = max((len(row) for row in rows), default=0)
    if not width or not any(cell is not None and str(cell).strip() for row in rows for cell in row):
        return None
    normalized = [row + [None] * (width - len(row)) for row in rows]
    first = normalized[0]
    keys = [re.sub(r"\s+", " ", str(cell or "")).strip().casefold() for cell in first]
    use_header = len(rows) > 1 and all(keys) and len(set(keys)) == width
    columns = [str(cell) for cell in first] if use_header else [f"column_{n}" for n in range(1, width + 1)]
    frame = pd.DataFrame(normalized[1:] if use_header else normalized, columns=columns, dtype=str)
    check_dataframe_limits(frame)
    return frame


def _page_parts(page, stem: str, page_number: int) -> list[dict]:
    parts = []
    for table_number, table in enumerate(page.extract_tables() or [], start=1):
        frame = _table_frame(table)
        if frame is not None:
            parts.append({"name": f"{stem}_p{page_number}_t{table_number}", "dataframe": frame})
    text = page.extract_text() or ""
    if text.strip():
        parts.append({"name": f"{stem}_p{page_number}_text", "text": text,
                      "format": "pdf"})
    return parts


def load_pdf_bytes(data: bytes, name: str) -> list[dict]:
    stem = re.sub(r"[^A-Za-z0-9_]+", "_", Path(name).stem).strip("_") or "report"
    try:
        document = pdfplumber.open(BytesIO(data))
    except Exception as exc:
        raise ValueError(f"Could not read PDF {name}: {exc}") from exc
    parts, failed_pages = [], []
    with document:
        for index in range(len(document.pages)):
            try:
                parts.extend(_page_parts(document.pages[index], stem, index + 1))
            except Exception as exc:
                failed_pages.append({"name": name, "page": index + 1,
                                     "message": str(exc) or type(exc).__name__})
    if failed_pages:
        parts.append({"name": name, "page_errors": failed_pages})
    if not parts:
        parts.append({"name": name, "notice": "No extractable text found (OCR is not supported)"})
    return parts
