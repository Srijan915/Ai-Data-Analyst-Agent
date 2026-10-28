"""Deterministic answerability gate tests."""
import pandas as pd

from app.core.answerability import Answerability, assess_answerability
from app.core.data_profiler import profile_dataframe
from app.models.schemas import CatalogProfile, DatasetMetadata


def test_answerability_rejects_explicit_missing_column():
    profile = profile_dataframe(pd.DataFrame({"sales": [1, 2]}))
    result = assess_answerability('sum "profit"', profile)
    assert result.outcome == Answerability.UNANSWERABLE


def test_answerability_requests_schema_for_vague_question():
    profile = profile_dataframe(pd.DataFrame({"sales": [1, 2]}))
    assert assess_answerability("please help", profile).outcome == Answerability.NEEDS_CLARIFICATION


def test_answerability_requires_confirmed_relationship_for_join():
    profile = profile_dataframe(pd.DataFrame({"customer_id": [1]}))
    catalog = CatalogProfile(datasets=[
        DatasetMetadata(dataset_id="1", filename="orders.csv", table_name="orders", uploaded_at="now", row_count=1, profile=profile),
        DatasetMetadata(dataset_id="2", filename="customers.csv", table_name="customers", uploaded_at="now", row_count=1, profile=profile),
    ])
    assert assess_answerability("join orders and customers by customer_id", catalog).outcome == Answerability.NEEDS_CLARIFICATION
