"""Optional preprocessing keeps original data and uses conservative conversions."""
import pandas as pd
from pandas.testing import assert_frame_equal
from io import StringIO

from app.preprocessing.change_report import build_change_report
from app.preprocessing.processor import OPERATIONS, process_dataframe
from app.preprocessing.quality_score import quality_score


def test_operations_are_exactly_the_five_choices():
    assert OPERATIONS == ("Remove duplicates", "Handle missing values", "Correct data types",
                          "Standardize text values", "Clean column names")


def test_duplicate_removal_only_exact_rows():
    original = pd.DataFrame({"name": ["Ada", "Ada", "Ada "]})
    cleaned, details = process_dataframe(original, [OPERATIONS[0]])
    assert len(cleaned) == 2 and details["duplicates_removed"] == 1
    assert len(original) == 3


def test_missing_values_are_marked_without_filling_or_dropping():
    original = pd.DataFrame({"x": ["", "   ", "ok", None]})
    cleaned, _ = process_dataframe(original, [OPERATIONS[1]])
    assert cleaned.x.isna().sum() == 3 and len(cleaned) == 4
    assert quality_score(original)["Completeness"] == quality_score(cleaned)["Completeness"]


def test_safe_type_conversion_preserves_ids_and_ambiguous_dates():
    original = pd.DataFrame({"customer_id": ["001", "002"], "Code": ["03", "04"],
                             "amount": ["1.25", "2.50"], "day": ["2024-03-04", "2024-03-05"],
                             "ambiguous": ["03/04/2024", "04/05/2024"]})
    cleaned, details = process_dataframe(original, [OPERATIONS[2]])
    assert cleaned.customer_id.tolist() == ["001", "002"]
    assert cleaned.Code.tolist() == ["03", "04"]
    assert cleaned.amount.tolist() == [1.25, 2.5]
    assert pd.api.types.is_datetime64_any_dtype(cleaned.day)
    assert cleaned.ambiguous.tolist() == original.ambiguous.tolist()
    assert {entry["column"] for entry in details["converted_columns"]} == {"amount", "day"}


def test_text_trim_and_column_names_do_not_change_other_cells():
    original = pd.DataFrame([[" Ada ", "X", 4]], columns=[" Full Name ", "Full-Name", "Total $"])
    cleaned, details = process_dataframe(original, [OPERATIONS[3], OPERATIONS[4]])
    assert cleaned.columns.tolist() == ["full_name", "full_name_2", "total"]
    assert cleaned.iloc[0].tolist() == ["Ada", "X", 4]
    assert details["text_values_changed"] == 1 and len(details["renamed_columns"]) == 3
    assert original.columns[0] == " Full Name " and original.iloc[0, 0] == " Ada "


def test_quality_dimensions_and_report():
    original = pd.DataFrame({" Bad Name ": [" 1 ", " 1 "]})
    before = quality_score(original)
    assert before == {"Completeness": 100, "Uniqueness": 50, "Type consistency": 0,
                      "Text consistency": 0, "Schema cleanliness": 0, "Overall": 30}
    cleaned, details = process_dataframe(original, [OPERATIONS[0], OPERATIONS[3], OPERATIONS[4]])
    report = build_change_report(original, cleaned, details)
    assert report["score_before"] == before and report["score_after"]["Overall"] > 30
    assert report["original_rows"] == 2 and report["processed_rows"] == 1
    assert quality_score(pd.DataFrame())["Overall"] >= 0


def test_each_run_starts_from_original():
    original = pd.DataFrame({"x": [" A ", " A "]})
    trimmed, _ = process_dataframe(original, [OPERATIONS[3]])
    duplicate_only, _ = process_dataframe(original, [OPERATIONS[0]])
    assert trimmed.x.tolist() == ["A", "A"]
    assert duplicate_only.x.tolist() == [" A "]


def test_empty_selection_and_repeated_run_leave_original_unchanged():
    original = pd.DataFrame({"ID": ["001", "001"], " Name ": [" Ada ", " Ada "]})
    saved = original.copy(deep=True)
    unchanged, details = process_dataframe(original, [])
    assert_frame_equal(unchanged, saved)
    assert details["completed"] == []
    first, _ = process_dataframe(original, list(OPERATIONS))
    second, _ = process_dataframe(original, list(OPERATIONS))
    assert_frame_equal(first, second)
    assert_frame_equal(original, saved)
    assert first.id.tolist() == ["001"]


def test_invalid_numeric_and_ambiguous_date_are_skipped():
    original = pd.DataFrame({"amount": ["10", "not a number"],
                             "date": ["03/04/2024", "04/05/2024"],
                             "zip_code": ["00123", "00456"]})
    cleaned, details = process_dataframe(original, ["Correct data types"])
    assert_frame_equal(cleaned, original)
    assert {item["column"] for item in details["skipped"]} == set(original.columns)


def test_download_csv_round_trip_matches_processed_frame():
    original = pd.DataFrame({" ID ": ["001", "002"], " Full Name ": [" Ada ", " Bob "]})
    processed, _ = process_dataframe(original, ["Standardize text values", "Clean column names"])
    downloaded = pd.read_csv(StringIO(processed.to_csv(index=False)), dtype=str,
                             keep_default_na=False)
    assert_frame_equal(downloaded, processed.reset_index(drop=True))
