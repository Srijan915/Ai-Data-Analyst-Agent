"""A plain report suitable for the UI and the session audit row."""
from datetime import datetime, timezone
import json
import uuid

import pandas as pd

from app.preprocessing.quality_score import missing_count, quality_score


def build_change_report(original: pd.DataFrame, processed: pd.DataFrame, details: dict) -> dict:
    before, after = quality_score(original), quality_score(processed)
    return {
        **details,
        "missing_before": missing_count(original),
        "missing_after": missing_count(processed),
        "original_rows": len(original), "processed_rows": len(processed),
        "original_columns": len(original.columns), "processed_columns": len(processed.columns),
        "score_before": before, "score_after": after,
        "score_difference": after["Overall"] - before["Overall"],
        "before_preview": original.head(5).copy(),
        "after_preview": processed.head(5).copy(),
    }


def audit_values(dataset_id: str, original_table: str, processed_table: str, report: dict) -> tuple:
    summary_keys = ("completed", "skipped", "duplicates_removed", "missing_before",
                    "missing_after", "converted_columns", "text_values_changed",
                    "renamed_columns", "original_columns", "processed_columns")
    summary = {key: report[key] for key in summary_keys}
    return (
        str(uuid.uuid4()), dataset_id, original_table, processed_table,
        json.dumps(report["selected_operations"]),
        report["score_before"]["Overall"], report["score_after"]["Overall"],
        report["original_rows"], report["processed_rows"], json.dumps(summary),
        datetime.now(timezone.utc).isoformat(),
    )
