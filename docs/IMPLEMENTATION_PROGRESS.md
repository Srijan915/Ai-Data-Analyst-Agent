# Implementation progress

## Baseline (2026-10-28)

- Branch: `feature/gemini-multi-dataset`; pre-existing uncommitted Gemini migration edits were preserved.
- Focused Gemini/operations run: 24 passed, 3 failed. Two failures were finish-reason normalization; one was an obsolete legacy-provider test. The focused subset alone is not expected to meet the repository-wide coverage threshold.
- Current architecture provides a secure single CSV `DiskDataset`, profiling, SQLGlot validation, DuckDB execution, and Streamlit UI.

## Active: Phase A — Gemini migration

- Normalized Gemini finish reasons using enum/object `name`, rather than enum string rendering.
- Updated production operation test and fresh-install dependency to `google-genai==2.33.0`.
- SDK inspection confirms `HttpRetryOptions(attempts=1)` means one total attempt and therefore no retries. The SDK accepts a single request timeout; the configured connection and read timeout values are combined until the SDK exposes separate controls.
- Focused migration verification: `27 passed` (`test_llm_client.py` and `test_production_operations.py`, offline, `--no-cov`).

## Phase B–D (implemented core)

- Added session-private `DatasetCatalog`, separate collision-resistant table names, multi-file Streamlit upload, catalog-aware SQL execution, cleanup, and relationship suggestions that require explicit confirmation before a multi-table question can be planned.
- Added structured deterministic answerability classification (`ANSWERABLE`, `NEEDS_CLARIFICATION`, `UNANSWERABLE`) for catalog questions.
- Added offline catalog and answerability tests.
- Full suite after the initial gate integration: `459 passed, 15 failed`, all failures were legacy single-table tests preempted by the new gate. The gate was narrowed to multi-table catalogs to preserve the established single-table validation behavior. Next: rerun the full suite, then document remaining roadmap limitations honestly.

## Verification (2026-10-28)

- Full suite: `474 passed`, coverage `86.79%` (threshold: 70%).
- Dependency check: `pip check` passed.
- No live Gemini request was made.
- Remaining roadmap work not represented as complete: bounded SQL correction/independent result verification, the requested 50–100 case benchmark and its metrics, individual dataset replacement/removal controls in the UI, and an end-to-end clean-install/deployment review. Existing historical deployment claims and URLs in legacy documentation need a separate verified deployment refresh.

## Remaining roadmap completion (2026-10-28)

- Added `MAX_SQL_CORRECTION_ATTEMPTS` (default 1). Only malformed planner output is repaired; every repair returns through the existing SQLGlot validation and executor limits.
- Added deterministic result metadata checks and numeric-claim grounding before displaying an explanation.
- Added transactional catalog replacement and individual removal controls. Failed replacements preserve the existing table; catalog relationship suggestions are recomputed after changes.
- Added a reproducible 50-case offline benchmark (`app.core.offline_benchmark`): 50/50 passed for deterministic answerability and SQL validation/rejection checks. It makes no Gemini request and does not measure model accuracy.
- Next/final verification: full suite, dependency/diff checks, deployment-path review, and clean-install feasibility.
- Clean-install check initially found and corrected a resolver conflict: `google-genai==2.33.0` requires Pydantic `>=2.12.5`; requirements now pins `pydantic==2.12.5`.

## Final local verification (2026-10-28)

- Recreated `.clean-install-check` from scratch with Python 3.13. Python 3.14
  initially selected a pandas source build, so it was not used as the clean
  verification interpreter; Python 3.13 resolved binary wheels for the pinned
  requirements.
- Fresh `pip install -r requirements.txt` completed with
  `google-genai==2.33.0` and `pydantic==2.12.5`; `pip check` reported no broken
  requirements.
- Full fresh-environment suite: `480 passed` in 22.54 seconds; coverage:
  `85.77%` (required minimum: `70%`).
- Offline benchmark: `50/50` passed. It is deterministic and makes no Gemini
  request, so it is not a live-model accuracy result.
- `git diff --check` passed with no whitespace errors. The final working tree
  retains the intended Gemini, multi-dataset, correction/verification,
  benchmark, documentation, and test changes; no commit was made.
- Removed active references to the historical legacy-provider deployment. The optional
  health probe now requires `DEPLOYMENT_HEALTH_URL` and skips clearly when it
  is not configured. No literal credentials were found in tracked project
  files; scan matches were the documented placeholder and the runtime
  environment-variable reference. The MIT license and copyright attribution
  remain intact.
- Docker verification: **NOT APPLICABLE — Docker deployment intentionally not
  used.** No live Gemini request was made; live-provider behavior needs an
  authorized Gemini credential and explicit execution before it can be claimed.

## Expanded release-readiness verification (2026-10-28)

- Re-read this progress record and reviewed branch `feature/gemini-multi-dataset`,
  the latest commit, Git status, the complete working diff, ignored artifacts,
  and the temporary `.clean-install-check` environment. Existing uncommitted
  work and generated verification artifacts were preserved.
- In the fresh environment, `python --version` reported `Python 3.13.0` and
  pip reported `26.2.2`; imports of the application modules, including
  `app.main`, succeeded. `pip check` again reported no broken requirements.
- Focused Gemini missing-credential and production-operation checks passed:
  `9 passed`. This confirms a missing key is handled lazily and locally, with
  no provider request during startup/upload paths.
- The documented local Streamlit command was started without a provider call.
  Its `/_stcore/health` endpoint returned `200 OK`, startup logs had no errors,
  and the temporary process was stopped cleanly.
- A second full fresh-environment run passed: `480 passed` in 18.96 seconds,
  with `85.77%` coverage. Focused offline UI/session/catalog/correction tests
  passed `31/31`; the deterministic benchmark again passed `50/50`.
- Static security review confirmed SQL is parsed in DuckDB dialect before
  execution, allows only read-only SELECT/WITH statements and allowed tables/
  functions, rejects multiple/destructive/external-access constructs, clamps
  result limits, and is backed by read-only DuckDB connections with external
  access disabled. Correction returns through that same validator.
- Privacy/data-flow review confirmed that Gemini receives schema/profile
  metadata and up to five bounded sample rows for planning, and validated SQL
  plus up to 20 bounded result rows for explanation; it does not receive the
  CSV file itself. The README already documents this limitation, lack of
  automatic redaction, prompt-injection residual risk, and log-access needs.
- Docker deployment: **NOT APPLICABLE — Docker deployment intentionally not
  used.** **BLOCKED:** live Gemini verification awaits explicit authorization
  and securely configured credentials; no live request was made.

## Non-Docker deployment decision (2026-10-28)

- The optional Docker deployment path was intentionally removed: `Dockerfile`,
  `.dockerignore`, the Docker-only CI job, and Docker Dependabot updates were
  deleted. No application source, data workflow, tests, health endpoint, or
  Gemini integration was removed.
- The selected deployment path is direct Python/Streamlit. The repository
  already has the required `requirements.txt`, root `.streamlit/config.toml`,
  and `app/main.py` entry point for a Streamlit Community Cloud deployment.
  Documentation now describes deploying with Python 3.13, private platform
  secrets, and the existing optional `DEPLOYMENT_HEALTH_URL` probe.
- Post-removal verification: the full Python 3.13 suite passed `480` tests in
  19.93 seconds at `85.77%` coverage (minimum: `70%`), and the deterministic
  offline benchmark passed `50/50` cases.
- No hosted deployment, Docker build, or live Gemini request was performed as
  part of this documentation/configuration change.
