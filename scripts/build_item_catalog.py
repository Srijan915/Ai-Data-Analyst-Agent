"""Reproduce the MIT-licensed sample's item dimension without inventing data."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build_item_catalog() -> Path:
    source = ROOT / "data/sample/customer_shopping_behavior.csv"
    destination = ROOT / "data/sample/item_catalog.csv"
    with source.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    items = {}
    for row in rows:
        item, category = row["Item Purchased"], row["Category"]
        if not item or not category or (item in items and items[item] != category):
            raise ValueError("The source does not define one category per nonempty item.")
        items[item] = category
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Item Purchased", "Item Category"])
        writer.writerows(sorted(items.items()))
    return destination


if __name__ == "__main__":
    print(build_item_catalog())
