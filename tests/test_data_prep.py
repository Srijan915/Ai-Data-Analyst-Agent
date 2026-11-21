"""Offline tests for preparing the sample dataset for gold SQL."""

from pathlib import Path

from evals.data_prep import load_for_gold


SAMPLE_CSV = Path("data/sample/customer_shopping_behavior.csv")


def test_load_for_gold_strips_bom_and_normalizes_column_names():
    dataframe = load_for_gold(SAMPLE_CSV)

    assert len(dataframe) == 3900
    assert dataframe.columns.tolist() == [
        "customer_id",
        "age",
        "gender",
        "item_purchased",
        "category",
        "purchase_amount_usd",
        "location",
        "size",
        "color",
        "season",
        "review_rating",
        "subscription_status",
        "shipping_type",
        "discount_applied",
        "promo_code_used",
        "previous_purchases",
        "payment_method",
        "frequency_of_purchases",
    ]
