"""Hand-calculated structural quality checks."""
import pandas as pd
from pandas.testing import assert_frame_equal

from app.preprocessing.processor import process_dataframe
from app.preprocessing.quality_score import quality_score


def test_known_scores_and_repeatability():
    frame = pd.DataFrame({" Bad Name ": [" 10 ", " 10 "], "id": ["001", "001"]})
    expected = {"Completeness": 100.0, "Uniqueness": 50.0,
                "Type consistency": 50.0, "Text consistency": 50.0,
                "Schema cleanliness": 50.0, "Overall": 60.0}
    assert quality_score(frame) == expected
    assert quality_score(frame) == quality_score(frame.copy(deep=True))


def test_blank_normalization_does_not_inflate_completeness():
    frame = pd.DataFrame({"name": ["", "  ", "Ada", None]})
    saved = frame.copy(deep=True)
    cleaned, _ = process_dataframe(frame, ["Handle missing values"])
    assert quality_score(frame)["Completeness"] == 25.0
    assert quality_score(cleaned)["Completeness"] == 25.0
    assert_frame_equal(frame, saved)


def test_empty_dataframe_scores_are_bounded_and_stable():
    for frame in (pd.DataFrame(), pd.DataFrame(columns=["name"])):
        score = quality_score(frame)
        assert list(score) == ["Completeness", "Uniqueness", "Type consistency",
                               "Text consistency", "Schema cleanliness", "Overall"]
        assert all(0 <= value <= 100 for value in score.values())
        assert score == quality_score(frame)
