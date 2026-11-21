# Evaluation scripts

All commands below run from the project root with `.venv` activated.

## Offline checks

- `python -m evals.run_security_eval` runs fixed safe and unsafe SQL through
  the production SQL validator, prints the observed outcomes, and writes
  `evals/security_results.json`.
- `python -m evals.run_grounding_eval` runs fixed result/explanation pairs
  through the production numeric-grounding check and writes
  `evals/grounding_results.json`.
- `python -m evals.validate_questions` executes every answerable gold SQL
  query in `evals/questions.json` against the local sample CSV and prints a
  result head for each one.

## Live accuracy harness

`python -m evals.run_evals --csv data/sample/customer_shopping_behavior.csv --smoke`
runs one live question once. It exits before loading the agent if
`GEMINI_API_KEY` is absent from the process environment. The CLI disables
automatic `.env` loading before importing application configuration.

After reviewing every question and gold SQL query, run a split with:

```powershell
python -m evals.run_evals --csv data/sample/customer_shopping_behavior.csv --split dev --runs 1
python -m evals.run_evals --csv data/sample/customer_shopping_behavior.csv --split test --runs 3
```

The harness uses `evals.data_prep.load_for_gold` for both the agent and gold
SQL: production CSV loading and execution preparation followed by evaluation
column names normalized to snake case. Comparison ignores aliases and row
order, rounds aggregate values to two decimals, and requires equal column
counts. It saves the measured result to
`evals/accuracy_results_dev.json` or `evals/accuracy_results_test.json`.

Never place real credentials in this directory or commit generated live result
files until you have reviewed them.
