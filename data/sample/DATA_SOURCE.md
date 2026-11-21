# Customer Shopping Behavior dataset source

This repository includes `customer_shopping_behavior.csv` from Amlan Mohanty's
[customer-trends-data-analysis-SQL-Python-PowerBI repository](https://github.com/amlanmohanty1/customer-trends-data-analysis-SQL-Python-PowerBI).

Copyright © 2025 Amlan Mohanty. The source repository and this copied dataset
are provided under the MIT License. See the adjacent [LICENSE](LICENSE) file.

`item_catalog.csv` is a derived two-column dimension from that same file, under
the same license: each distinct `Item Purchased` and its existing `Category`,
renamed to `Item Category`. It has 25 rows, one per item, and contains no invented
keys, categories, or new observations. `scripts/build_item_catalog.py` validates
that each item has one nonempty category and regenerates the sorted output.
The source has 3,900 rows and 18 columns. The shared item key provides a genuine
many-to-one relationship; see [the join contract](../../docs/DATASETS.md).
