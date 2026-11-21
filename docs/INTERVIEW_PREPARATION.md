# Interview preparation

## One-minute walkthrough

“This project lets someone upload CSV data and ask analytical questions in
plain English. A Gemini planner proposes DuckDB SQL from a bounded schema
profile. SQLGlot checks the proposed query against read-only, table, function,
and row-limit rules before a resource-bounded DuckDB executor runs it. The UI
shows the resulting table, visualization, explanation when available, and the
validated SQL actually executed. A numeric grounding check can withhold an
explanation when it contains unsupported numbers.”

## Demonstration

For a format demonstration, use `tests/fixtures/two_sheets.xlsx` to show sheet
selection and two catalog tables. `structured_notes.txt` becomes a table; the
free-form `document_notes.txt` remains complete in the session with a preview.
Both handlers preserve text values before the existing SQL preparation stage.
For PDFs, upload `tests/fixtures/table_report.pdf`: the page-two table is
registered separately from the page text. Use `text_report.pdf` to show a
document preview and `no_text_report.pdf` to explain the no-OCR boundary.

1. Start the app with `python -m streamlit run app/main.py` and upload
   `data/sample/customer_shopping_behavior.csv`.
2. Ask “What is the total purchase amount by category?” Inspect the result,
   the chart, and **Generated SQL**. Explain that the SQL is validated and may
   include an enforced row limit.
3. Ask “What is the monthly purchase trend over order dates?” The sample has
   no date column, so the deterministic gate should explain that the data is
   missing. Clear follow-up context in the sidebar or clear the dataset in the workspace.
4. Discuss the tradeoffs: Gemini can produce wrong but safe SQL; a numeric
   grounding check cannot verify all semantic claims; profiles and bounded
   result rows are sent to the provider.

The first question requires a working Gemini key and provider access. The
missing-date example is checked before planning, but loading a dataset is still
required. If access is unavailable, use the automated Streamlit workflow tests
with their stubbed model and the saved evaluation reports; describe them as
offline verification rather than a live demonstration.

## Questions worth preparing for

**Why SQLGlot instead of prompt instructions alone?** Model output is untrusted.
Parsing and allowlisting statements, functions and table references give the
executor a deterministic boundary independent of the prompt. DuckDB resource
and external-access controls add another layer.

**How are joins controlled?** The session catalog proposes relationships from
schema and value overlap. A person confirms a relationship to guide the planner.
The confirmation step checks full-table key uniqueness and nulls. A separate
SQLGlot scope-based guard resolves join columns to physical keys and enforces the
confirmed pairs before execution, including direct executor calls. Computed or
ambiguous keys and unsupported joins fail closed. Revocation invalidates cached
results and is rechecked at execution. Suggestions and approvals still do not
prove that a join is meaningful for a business question.

**How do you handle incorrect or unavailable model output?** Invalid plans may
receive bounded correction. Provider failures have typed error messages and
are reported separately from answer accuracy. Security and conversion checks
still apply after correction. Reviewers should inspect the generated SQL even
when execution succeeds.

**What does grounding mean here?** Recognized numeric claims are compared with
result values and row count, and unsupported explanations are withheld. It
does not verify every nonnumeric statement, causality, or whether the SQL answers
the intended question.

**What did you measure?** A historical one-run dev evaluation scored 18/20
(90%) with no provider errors, using the earlier question wording and comparison
policy. The small fixed security set blocked 28/28 unsafe queries, with 0/16
safe queries falsely blocked; the grounding set caught 30/30 corruptions with
0/30 false alarms. All 45 gold SQL queries executed. Held-out live accuracy
and run-to-run consistency were not measured. Full context is in the
[validation report](VALIDATION_REPORT.md).

**What would you improve next?** Evaluate a broader range of datasets, joins,
date columns, and real user questions; measure consistency and held-out
accuracy; and assess the privacy and operational requirements before a larger
deployment. The answerability and grounding rules are intentionally limited.

## Multi-file demonstration and implementation story

Upload `customer_shopping_behavior.csv` and `item_catalog.csv` together. Show both
physical schemas and the shared **Item Purchased** key. Confirm it and ask:
“Join the two datasets on Item Purchased and show total Purchase Amount (USD) by
Item Category from item_catalog.” Inspect the four category totals, chart, and
executed SQL. Revoke the relationship and repeat to show enforcement. The item
dimension is a documented projection of the original licensed sample, not an
independently collected dataset. See [DATASETS.md](DATASETS.md).

**What broke before?** The prompt told Gemini to query only `dataset` while the
catalog exposed hashed physical names. The agent retained an old relationship
snapshot after confirmation. Overview charts also looked up a nonexistent catalog
attribute and fell back to `dataset`. The fixes aligned the names, refreshed the
profile, and added execution-time authorization rather than widening the allowlist.
UI reruns now preserve a draft question during relationship confirmation.

**How do you substantiate the result?** The multi-file test compares actual DuckDB
join outputs against independently grouped source rows, checks that totals are
not duplicated, and tests revocation, wrong keys, OR predicates, computed keys,
unqualified keys, unsupported joins, and many-to-many confirmation. A Streamlit
test exercises the same flow with only upload widgets and the provider stubbed.
The real browser smoke uploads files and clicks confirmation without stubs.
New live Gemini join accuracy is not measured without process credentials.

**Why not support every SQL join shape?** Direct key lineage is auditable; computed
or ambiguous lineage needs a stronger semantic model. Refusing unsupported shapes
is safer than guessing authorization. Full-table unique-key checking is more
reliable than using sampled profile counts, but a valid key still cannot prove
that a metric is additive or that a multi-join aggregate is correct.

Use [README.md](../README.md) for current counts. The 18/20 live
score belongs to the earlier single-table dev run and is retained without chasing
the two accepted output-column mismatches.

## Preprocessing interview notes

**Why Pandas?** Each cleanup operation is a small, readable DataFrame operation.
The uploaded table is copied first, so an optional run cannot edit the original.

**Why DuckDB?** It queries local tabular data with SQL, works with Pandas, and
needs no server. Single small CSV queries use an in-memory connection. Catalogs
and processed tables use a temporary, session-owned DuckDB file.

**How does Gemini fit?** It proposes SQL using a bounded schema description and
the user's question. Its output is treated as untrusted text. SQLGlot parses it,
allows read-only SQL over registered tables, and checks joins against confirmed
relationships before DuckDB executes it.

**How does preprocessing work?** The user selects from five operations and
submits. Each run starts from a deep copy of the original table. Exact duplicate
rows can be removed; blanks can become missing; safe numeric/ISO-date columns
can convert; text can be trimmed; column names can become unique snake_case.
The new table is `<original_table>_processed`; the source stays unchanged.

**How is quality scored?** Completeness is non-missing cells divided by cells;
uniqueness is distinct rows divided by rows; type consistency is columns
not flagged as safely convertible text divided by columns; text consistency is
trimmed non-missing text cells divided by non-missing text cells; schema
cleanliness is unique snake_case names divided by columns. Each becomes 0–100;
overall is their mean. This cannot measure factual accuracy.

**How is integrity protected?** Leading-zero IDs, code columns, ambiguous
dates and values that would lose numeric precision are not converted. The
original table and uploaded bytes remain untouched. The processed table goes
through the normal SQL allowlist; `preprocessing_runs` stays outside it.

**How is benchmarking done?** A seed-42 script writes copies of the sample CSV
with known injected errors. Five runs per case are timed with `perf_counter()`;
the median, rows per second, correctness rates and integrity violations are
saved as CSV and JSON. The source CSV is never rewritten.

### Ten short questions

1. **Why is the model not allowed to execute SQL directly?** Its SQL can be unsafe or wrong; parsing and allowlisting happen before execution.
2. **What happens if Gemini returns `DROP TABLE`?** The planner rejects it; only supported read-only statements reach execution.
3. **Can a generated query read `preprocessing_runs`?** No. It is not in the registered-table allowlist.
4. **What if a user preprocesses twice?** Both runs start from the original; the processed table is replaced and each run gets an audit row.
5. **Why keep IDs as text?** Numeric conversion can remove leading zeros and change identifier meaning.
6. **Why skip `03/04/2024`?** Its day/month order is ambiguous; only strict ISO dates convert.
7. **Does a higher quality score mean correct business data?** No. It measures structural cleanliness only.
8. **Can the app read scanned PDFs?** No. PDF extraction requires text or tables; OCR is not included.
9. **How are joins authorized?** A user confirms a suggested key, and the SQL guard checks that approval again at execution.
10. **Why is duplicate F1 near 1.0?** Exact injected copies are deterministic; the metric verifies correctness, not model skill.
