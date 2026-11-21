# Sample datasets and relationship contract

Both files are covered by [the adjacent MIT license](../data/sample/LICENSE),
with original attribution in [DATA_SOURCE.md](../data/sample/DATA_SOURCE.md).

| CSV | Rows | Columns | Role |
| --- | --- | --- | --- |
| `customer_shopping_behavior.csv` | 3,900 | 18 | Original customer/purchase records |
| `item_catalog.csv` | 25 | 2 | Derived item-to-category lookup |

The lookup takes distinct existing `Item Purchased` / `Category` pairs and
renames `Category` to `Item Category`. No keys or descriptive data are invented.
This is a normalization demonstration, not new enrichment: the source already
has category information. The duplicate category name is avoided so relationship
discovery proposes the intended item key rather than a many-to-many category join.

Reproduce from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts/build_item_catalog.py
```

The generator fails if an item/category is empty or one item maps to more than
one category. It writes deterministic sorted rows. The relationship is:

```text
customer_shopping_behavior.Item Purchased (many purchase rows)
    -> item_catalog.Item Purchased (one non-null row per item)
```

All source items are covered by this derived lookup. Inner joining on this key
preserves purchase row count and total purchase amount for this sample. A join on
category would repeat rows and inflate totals; that is not the authorized key.
Other user uploads may have partial key coverage, so inspect unmatched rows and
the chosen join type before interpreting an aggregate.

The UI displays actual physical names, which include a content hash; never
hardcode those names. The model receives both schemas and confirmation status.
Confirm **Item Purchased**, then ask for total purchase amount by **Item Category
from item_catalog** to exercise a real join. Revoking confirmation must refuse
further joined queries and clear the previous result.

`tests/test_catalog_analysis.py` and `tests/test_catalog_ui.py` use these actual
files. Expected totals are independently calculated from the original source
with Pandas, not hardcoded into application answers. They cover profile refresh,
exact-key enforcement, revoked/unconfirmed relationships, incorrect predicates,
CTE lineage, many-to-many rejection, selected-table charts, preview labels, and
the upload-to-render workflow. The provider is stubbed in those tests; they do
not measure live Gemini accuracy.
