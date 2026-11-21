# QueryPilotAI

## Overview

A Streamlit dashboard for asking questions about uploaded tables. Gemini proposes SQL;
SQLGlot validates it; DuckDB executes it. Review the result table, explanation,
Plotly chart and executed SQL, then export the returned rows as CSV.

## Tech stack

Python, Streamlit, Pandas, DuckDB, Gemini, SQLGlot and Plotly. `openpyxl` reads
XLSX sheets; `pdfplumber` extracts tables and text from text-based PDFs. The
application has no separate frontend or permanent database service.

## Run the app (PowerShell)

Run these commands from the repository root. Use Python 3.11+ or reuse the
existing virtual environment. No Node.js or frontend build is needed.

```powershell
# Only for a new environment:
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Launch the dashboard:
$env:PYTHON_DOTENV_DISABLED = '1'
.\.venv\Scripts\python.exe -m streamlit run app/main.py
```

Open **http://localhost:8501**. Without a Gemini key, uploads, profiling,
relationship controls and dataset charts work; natural-language SQL generation
needs a key supplied privately in the process environment before launch.

| Environment variable | Purpose |
| --- | --- |
| `GEMINI_API_KEY` | Required for live Gemini requests. Never commit the value. |
| `LLM_PROVIDER` | `gemini` (default). |
| `LLM_MODEL_ID` | Defaults to `gemini-2.5-flash`; use a model available to your account. |
| `PYTHON_DOTENV_DISABLED` | `1` prevents automatic reading of local `.env` files. |
| `APP_ACCESS_PASSWORD` | Optional shared access gate for a private demo. |
| `MAX_UPLOAD_SIZE_MB`, `MAX_QUERY_RESULT_ROWS` | Upload and returned-result limits. |

See [app/config.py](app/config.py) for defaults and [OPERATIONS.md](OPERATIONS.md)
for resource limits and deployment settings. `.env` and Streamlit secrets are
ignored by Git. These commands disable automatic dotenv loading.

## Demonstrate the workflow

1. Upload `data/sample/customer_shopping_behavior.csv` using **Upload CSV, XLSX,
   TXT or PDF files**. Inspect the row counts, schema, preview and dataset charts.
2. Ask **What is the total purchase amount by category?** and click **Analyze
   question**. Live generation requires Gemini access.
3. Inspect the explanation, result table, visualization and expandable SQL.
   **Export results as CSV** downloads the returned table, including any row limit.
4. For joins, select both that CSV and `data/sample/item_catalog.csv`. Inspect the
   suggested `Item Purchased` relationship and click **Confirm join**.
5. Ask **Join the datasets and show total purchase amount by item category**.
   Unconfirmed joins are refused. Revoking a relationship clears stale results
   and follow-up context.
6. Use **Explore Dataset** for automatic charts and the custom visualizer.
   Preview-based charts are labelled; ask a question for complete-table analysis.
7. **Clear active dataset** releases the active data. **Clear follow-up context**
   keeps the data while starting a fresh conversation.

Optional cleanup: after uploading a table, click **Data Preprocess**, select the
source dataset if there are several, choose any of the five operations, and click
**Submit**. An empty selection changes nothing. Every run starts with the
original table. The result appears as `<original_table>_processed` alongside it,
with before/after previews, a change report, five quality dimensions, and a
**Download Cleaned Dataset** CSV button. Resubmitting replaces that processed
table and adds an audit row. The original stays queryable. For the sample CSV,
upload `customer_shopping_behavior.csv`, preprocess it, then ask questions about
`dataset` or `dataset_processed` (Gemini access required for natural-language SQL).

The operations are exact-row duplicate removal, marking blank cells missing,
conservative numeric/strict ISO-date conversion, trimming text values, and
lowercase snake_case column names. IDs, leading-zero values and ambiguous dates
stay as text. The quality score averages completeness, uniqueness, type
consistency, text consistency and schema cleanliness; it measures structural
cleanliness, not real-world accuracy. Unstructured documents cannot be preprocessed.

## Five preprocessing operations and quality score

| Operation | Effect |
| --- | --- |
| Remove duplicates | Remove later copies of exact rows; report rows removed. |
| Handle missing values | Mark empty and whitespace-only cells missing; do not fill or drop rows. |
| Correct data types | Convert only whole columns that safely parse as numbers or strict `YYYY-MM-DD` dates; preserve IDs and leading zeros. |
| Standardize text values | Trim text values without changing case. |
| Clean column names | Make unique lowercase snake_case names without changing cells. |

The same five formulas score the original and processed DataFrames, each from
0 to 100. Empty denominators are handled explicitly.

| Dimension | Formula |
| --- | --- |
| Completeness | `100 × non-missing cells / all cells`; blank and whitespace-only strings count as missing. |
| Uniqueness | `100 × (rows − exact duplicate rows) / rows`. |
| Type consistency | `100 × columns not flagged as safely convertible text / all columns`; a text column is flagged when all non-missing values safely parse as numeric or strict ISO dates. |
| Text consistency | `100 × non-missing text cells without leading/trailing whitespace / non-missing text cells`. |
| Schema cleanliness | `100 × unique lowercase snake_case column names / all columns`. |

**Overall = the arithmetic mean of the five dimensions.** This describes
structural cleanliness, not whether the source values are factually correct.

For format checks, upload `tests/fixtures/two_sheets.xlsx` and select its sheets;
each selected sheet becomes a table. `structured_notes.txt` becomes a table,
while `document_notes.txt` stays a document with a text preview. The document
is not queryable with SQL. Delimited CSV/TXT values and Excel cells are read as
text first, preserving IDs such as `0012`. If an extension disagrees with the
detected bytes, the dashboard warns and uses the detected format. Excel formulas
are retained as formula text; they are not evaluated during upload.

For PDFs, upload `tests/fixtures/table_report.pdf`. Its page-two table becomes
`table_report_p2_t1`; page text appears separately as a document preview.
`text_report.pdf` stays document text, while `no_text_report.pdf` reports that
OCR is unsupported. A damaged page appears under **Pages that failed**, and
other pages continue. Only extracted tables are queryable with SQL.

## Supported formats and limits

| Format | Behavior |
| --- | --- |
| CSV | Detects common delimiters and decodes UTF-8, UTF-8 with BOM or Latin-1; preserves text IDs. |
| XLSX | Lets the user select sheets; each selected sheet becomes a table. Formulas remain text. |
| TXT | Delimited lines become a table; free-form text remains a document preview. |
| PDF | Extracts tables and text from text-based pages; only extracted tables become queryable. |

Scanned or image-only PDFs need OCR, which is not supported. Free-form text is
not forced into a table. Upload, memory, row, query-time and result-size limits
are enforced; see [OPERATIONS.md](OPERATIONS.md).

The dashboard uses a dark green and warm cream theme, standard Streamlit controls and
Plotly charts. There is no separate React interface or 3D canvas.

## Architecture and workflow

```text
Streamlit → format detection/loading → table profile → answerability → Gemini SQL
          → SQLGlot + confirmed-join checks → DuckDB
          → result verification → table, explanation, chart, CSV export
```

| Location | Responsibility |
| --- | --- |
| `app/main.py`, `app/ui/` | Streamlit entry point, session controls and presentation |
| `app/file_handlers/` | CSV, XLSX, TXT and PDF parsing; text previews |
| `app/preprocessing/` | Optional Pandas cleanup, quality scoring and change report |
| `app/core/` | Loading, catalog, planning, execution, verification and charts |
| `app/utils/security.py` | SQLGlot read-only SQL validation |
| `app/config.py`, `.streamlit/config.toml` | Runtime limits and dashboard theme |
| `tests/`, `evals/` | Tests, security/grounding checks and SQL evaluation |
| `data/sample/` | Demonstration CSVs and their licenses |

## DuckDB storage

Streamlit session state holds the agent, profiles, catalog and recent results.
Small standard CSVs are registered as `dataset` in a new in-memory DuckDB connection
for each execution. Large CSVs, Excel sheets, structured TXT, PDF tables and multi-file catalogs use existing temporary
DuckDB files with bounded resources, not permanent application storage. Catalog
names use `ds_<sanitized filename stem>_<content hash>` with collision suffixes.
The first preprocessing run on a single CSV moves its full original table into
the session catalog under the existing `dataset` SQL name. Processed tables use
the same SQL validator and confirmed-join rules. The catalog also keeps
`preprocessing_runs` metadata, which is excluded from Gemini's table allowlist.
Each run writes one row with the source and processed table names, selected
operations, before/after overall scores, row counts, summary and timestamp.
The original table is unchanged; another submission replaces only its
`<original_table>_processed` table. The catalog is a temporary DuckDB file owned
by the Streamlit session and released on clear/expiry. There is no durable
cross-session history or recovery after session loss. Before preprocessing, a
small single CSV uses the existing in-memory per-query DuckDB path.

The existing `app/api/` compatibility adapter and its tests remain available;
the Streamlit application does not need an API server. See
[architecture and workflow](docs/ARCHITECTURE.md) for module boundaries,
relationship rules and result verification.

## Security and limitations

- SQLGlot rejects writes, multiple statements, unauthorized tables/functions and
  external file access. DuckDB execution has resource limits and external access disabled.
- Joins require explicitly confirmed keys. Confirmation checks full-table key
  uniqueness and nulls; execution rechecks current approvals.
- Answerability checks reject common requests for missing data. Numeric grounding
  can withhold unsupported explanations; neither check proves business correctness.
- Schema samples, questions and bounded results may be sent to Gemini. Provider
  quota, access, timeout and network failures are reported in the UI.
- CSV, XLSX, TXT and PDF uploads are supported. Unstructured TXT and PDF page
  text are preview-only. Scanned PDFs need OCR, which is not included. Relationship discovery is heuristic;
  composite and many-to-many relationships are not automatically approved.
- Custom catalog charts use preview rows. Exports contain returned rows,
  not necessarily every source row. Session data is temporary.

## Run tests and evaluations

```powershell
$env:PYTHON_DOTENV_DISABLED = '1'
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m evals.run_security_eval
.\.venv\Scripts\python.exe -m evals.run_grounding_eval
.\.venv\Scripts\python.exe -m evals.validate_questions
```

Pytest includes Streamlit runtime tests, real DuckDB queries, provider-error
handling and API tests. UI integration tests substitute the model response and
upload widget; they are not measurements of live Gemini accuracy.

Current verification (2026-11-19): **599 passed, 1 skipped, 82.58% coverage**
with one existing Starlette/httpx deprecation warning. The skipped test is the
opt-in live Gemini check; no Gemini key was available in the process environment.
The focused preprocessing, format, integrity, UI and confirmed-join run passed
**39 tests**. A real-browser Streamlit smoke check passed for upload, join
confirmation, desktop/mobile layout and missing-date refusal.

Streamlit restoration verification (2026-11-19): **553 tests passed**, **81.60%
coverage**, with one existing Starlette/httpx deprecation warning. The real-browser
upload, relationship confirmation, responsive layout and missing-date refusal
checks also passed without provider calls.

Multi-format upload verification (2026-11-19): **565 tests passed**, **81.51%
coverage**, with the same deprecation warning. The new format tests cover sheet
selection, distinct catalog tables, TXT documents, leading-zero IDs, formula
text, mismatched extensions and unchanged upload bytes. The browser CSV workflow
was also rerun successfully.

PDF extraction verification (2026-11-19): **573 tests passed**, **81.89%
coverage**, with the same deprecation warning. The PDF fixtures exercise table
and text pages, a corrupt file, a page without extractable text, and recovery
after a page extraction error. A real-browser check also passed for table, text,
no-OCR and corrupt-file behavior. No OCR or live Gemini requests were used.
The tiny PDFs are checked in; `scripts/build_pdf_fixtures.py` can regenerate
them with reportlab, which is needed only for fixture generation.

## Reproduce preprocessing benchmarks

```powershell
$env:PYTHON_DOTENV_DISABLED = '1'
.\.venv\Scripts\python.exe benchmarks/run_benchmark.py
.\.venv\Scripts\python.exe benchmarks/evaluate.py
```

The generator copies `customer_shopping_behavior.csv` with seed 42 and records
injected errors in `benchmarks/expected_outputs/`; it does not modify the source.
The evaluator runs each case five times and writes median timing, records per
second, correctness rates, and integrity violations to
`reports/benchmark_results.csv` and `.json`. Blank cells preserved as numeric
nulls are treated as equivalent. Exact-duplicate F1 near 1.0 is a correctness
check by design, not a competitive benchmark. Live Gemini is never used.

Recorded run on Windows 11, Python 3.13.0 and pandas 2.3.2, seed 42:

| Case | Rows | Median seconds | Rows/second | Correctness | Integrity violations |
| --- | ---: | ---: | ---: | --- | ---: |
| Duplicates | 3,930 | 0.0039 | 1,013,409 | Precision/recall/F1 1.0 | 0 |
| Whitespace | 3,900 | 0.1693 | 23,037 | Text success 1.0 | 0 |
| Convertible types | 3,900 | 0.0910 | 42,848 | Type accuracy 1.0 | 0 |
| Messy names | 3,900 | 0.0003 | 15,390,687 | Not separately scored | 0 |
| Combined | 3,930 | 0.7253 | 5,419 | All measured rates 1.0 | 0 |

These are preprocessing benchmark measurements, separate from pytest results.
See the exact values and run metadata in `reports/benchmark_results.csv`.

Historical evaluation: **28/28** unsafe SQL blocked, **30/30** numeric corruptions
detected, **45/45** gold queries validated, and **18/20** live dev answers. These
fixed project-authored cases are not a general accuracy or security guarantee.
See [validation details](docs/VALIDATION_REPORT.md) and
[evaluation commands and comparison rules](evals/README.md).

## Browser smoke check and screenshots

```powershell
$env:PYTHON_DOTENV_DISABLED = '1'
.\.venv\Scripts\python.exe scripts/capture_ui.py
```

The optional script uses local Chrome/Edge and the installed `websockets` module.
It starts Streamlit, uploads both sample files, confirms a join, checks desktop
and mobile layout, exercises a missing-date refusal and writes
`screenshots/streamlit-*.png`. No provider requests are made. Alternatively,
follow the demonstration steps and capture the workspace, result and SQL panels
with your browser. Retained older screenshots document earlier UI versions.
