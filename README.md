# Foresight AutoML

A dataset-agnostic system for explainable predictive modeling and decision support.

Upload any tabular CSV, pick a target column, and Foresight AutoML will:

1. Validate and profile the data (quality issues, likely target leakage)
2. Detect the problem type (binary, multiclass, regression)
3. Train baselines and candidate models with 5-fold cross-validation
4. Evaluate the best model once on a held-out test set
5. Explain what the model relies on (SHAP + permutation importance)
6. Produce a plain-English HTML report

> Status: Phase 1 in progress.

## Setup (Windows PowerShell)

```powershell
python -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run the app

```powershell
venv\Scripts\python.exe -m streamlit run app.py
```
Local development: The app runs locally by default, and uploaded files are processed in memory rather than persisted by the application.

Then open http://localhost:8501. The app is only reachable from your own computer,
and uploaded files stay in memory (they are never written to disk).

## Benchmarks

Download the datasets listed in `benchmarks/datasets.json` into `benchmarks/data/`
(not committed), then run:

```powershell
venv\Scripts\python.exe -m benchmarks.run_benchmarks
```

This writes `benchmarks/results.csv` and one HTML report per dataset to
`outputs/benchmark_reports/`. Results from the latest run (seed 42, 5-fold CV,
20% held-out test set):

| Dataset | Rows | Task | Metric | Baseline | Simple model | Best model (CV mean ± std) | Test |
|---|---|---|---|---|---|---|---|
| Telco churn (IBM) | 7,043 | binary | ROC-AUC | 0.500 | 0.858 | Logistic regression 0.858 ± 0.013 | 0.847 |
| German credit (UCI) | 1,000 | binary | ROC-AUC | 0.500 | 0.771 | Random forest 0.791 ± 0.039 | 0.782 |
| Bank marketing (UCI) | 45,211 | binary, imbalanced (11.7%) | PR-AUC | 0.117 | 0.398 | LightGBM 0.443 ± 0.011 | 0.464 |
| Bike sharing, hourly (UCI) | 17,379 | regression | RMSE (lower is better) | 182.2 | 142.6 | LightGBM 41.2 ± 1.6 | 39.0 |

Each dataset runs end to end in 7-30 seconds on a laptop (including the decision analysis below).

**From scores to decisions (binary datasets).** Calibration and the cut-off are chosen on
out-of-fold predictions for the training data, then checked once on the test set:

| Dataset | Costs (false alarm : miss) | Calibrated? (test Brier) | Cut-off | Test cost per 1,000 rows: cut-off 0.50 → chosen | Positives caught | Lift in top 10% |
|---|---|---|---|---|---|---|
| Telco churn | 1 : 1 | No (0.137) | 0.53 | 198.7 → 199.4 | 54% | 2.8× |
| German credit | 1 : 5 (documented) | Yes, 0.165 → 0.158 | 0.16 | 830 → 530 | 92% | 2.5× |
| Bank marketing | 1 : 1 | Yes, 0.147 → 0.080 | 0.53 | 104.3 → 103.8 | 19% | 4.5× |

With the German credit cost matrix, the chosen cut-off (0.16) matches the theoretical
optimum for calibrated probabilities, 1 / (1 + 5) ≈ 0.17, and cuts the test-set cost by 36%.
With equal costs the chosen cut-off stays near 0.50, as expected; on Telco it was marginally
worse than 0.50 on the test set, which the report states rather than hides.

**Optional tuning.** `--tune` (or the "Also tune" checkbox in the app) adds Optuna-tuned
random forest and LightGBM. Tuning runs inside **nested cross-validation**: each outer fold
gets its own search on its training part only, so tuned scores are directly comparable with
the untuned ones and not optimistic. Tuned results are written to `benchmarks/results_tuned.csv`.

```powershell
venv\Scripts\python.exe -m benchmarks.run_benchmarks --tune
```

Latest tuned run (up to 20 trials and 30 seconds per search):

| Dataset | Metric | Untuned best (CV) | Tuned best, nested CV | Test: untuned → tuned | Run time |
|---|---|---|---|---|---|
| Telco churn | ROC-AUC | Logistic regression 0.858 ± 0.013 | LightGBM 0.867 ± 0.011 | 0.847 → 0.858 | 4.5 min |
| German credit | ROC-AUC | Random forest 0.791 ± 0.039 | LightGBM 0.797 ± 0.044 | 0.782 → 0.788 | 2.5 min |
| Bank marketing | PR-AUC | LightGBM 0.443 ± 0.011 | LightGBM 0.453 ± 0.014 | 0.464 → 0.468 | 6.5 min |
| Bike sharing | RMSE (lower is better) | LightGBM 41.2 ± 1.6 | LightGBM 38.9 ± 1.0 | 39.0 → 36.5 | 6.3 min |

Tuning helped on every dataset, but on the three classification datasets the gain is within
one standard deviation; the clearest improvement is bike demand (RMSE −5% in CV, −6% on test).
The 30-second limit was reached on Telco, bank and bike, so a rerun can give slightly
different tuned numbers (it ran fewer than 20 trials there).

**Optional run tracking (MLflow, local only).** Tick "Save this run to the local MLflow log"
in the app, or add `--track` to the benchmark command. Each run's settings, CV and test
metrics, data profile, importance table, report and model are stored in `mlflow.db` and
`mlruns/` (both git-ignored). List recent runs with:

```powershell
venv\Scripts\python.exe -m foresight.tracking
```

MLflow sends usage telemetry over the internet by default; `foresight/tracking.py` switches
it off before MLflow is loaded, and is the only module allowed to import MLflow. A test
checks this in a real (non-test) environment with all network connections blocked.

**Leakage handling.** The automatic check excluded `Churn Value` and `Churn Reason`
(Telco). Three known leaks are dropped in `datasets.json`, each with a documented reason:
`Churn Score` (another model's prediction; flagged by the warning tier at single-column
ROC-AUC 0.942), `duration` (bank; only known after the call), and `casual` + `registered`
(bike; they sum to the target). The single-column check cannot catch leaks that are
spread across columns or depend on timing, so a human review step is still needed.

## Run tests

```powershell
venv\Scripts\python.exe -m pytest
```
