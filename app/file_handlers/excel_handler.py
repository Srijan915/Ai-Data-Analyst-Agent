"""Selected XLSX worksheets, each kept as a separate text-valued table."""
from io import BytesIO

import pandas as pd

from app.utils.dataset_resources import check_dataframe_limits


def sheet_names(data: bytes) -> list[str]:
    try:
        with pd.ExcelFile(BytesIO(data), engine="openpyxl") as workbook:
            return workbook.sheet_names
    except (OSError, ValueError, ImportError) as exc:
        raise ValueError(f"Could not read XLSX workbook: {exc}") from exc


def load_excel_bytes(data: bytes, name: str, selected_sheets: list[str]) -> list[dict]:
    available = sheet_names(data)
    if not selected_sheets:
        raise ValueError(f"Select at least one sheet from {name}.")
    if any(sheet not in available for sheet in selected_sheets):
        raise ValueError(f"{name} contains an unknown sheet selection.")
    results = []
    for sheet in selected_sheets:
        try:
            frame = pd.read_excel(BytesIO(data), sheet_name=sheet,
                                  dtype=str, keep_default_na=False, engine="openpyxl",
                                  engine_kwargs={"data_only": False})
            if frame.empty:
                raise ValueError(f"{name} / {sheet} has no data rows.")
            check_dataframe_limits(frame)
        except (OSError, ImportError, KeyError) as exc:
            raise ValueError(f"Could not read {name} / {sheet}: {exc}") from exc
        results.append({"name": f"{name} / {sheet}", "dataframe": frame})
    return results
