"""Optional, local-only experiment tracking with MLflow.

Each run records its settings, every CV and test metric, the data profile
summary, the importance table, the HTML report and the fitted model, so runs
can be compared later. Nothing leaves this computer:

- MLflow sends usage telemetry over the internet BY DEFAULT. The environment
  variables below switch it off, and they must be set BEFORE MLflow is
  imported. That is why this is the only module allowed to import MLflow
  (a test enforces it).
- Runs are stored in a local SQLite file (config.MLFLOW_DB).
- No data rows are logged: only aggregated results and column names.
"""

import os

# --- Must come before "import mlflow" ----------------------------------------
os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"   # no usage data sent to MLflow
os.environ["DO_NOT_TRACK"] = "true"               # the general opt-out, as a backup
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "true"   # silence an import-time message

import json  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import joblib  # noqa: E402
import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

from foresight import config  # noqa: E402
from foresight.ingest import neutralize_formula_injection  # noqa: E402

MAX_PARAM_LENGTH = 500   # MLflow rejects very long parameter values


def _tracking_uri() -> str:
    return f"sqlite:///{config.MLFLOW_DB.as_posix()}"


def _experiment_id(client: MlflowClient) -> str:
    """Create the experiment on first use, with artifacts inside the project."""
    experiment = client.get_experiment_by_name(config.MLFLOW_EXPERIMENT)
    if experiment is not None:
        return experiment.experiment_id
    config.MLFLOW_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    return client.create_experiment(config.MLFLOW_EXPERIMENT,
                                    artifact_location=config.MLFLOW_ARTIFACT_DIR.as_uri())


def _param(value) -> str:
    text = str(value)
    return text if len(text) <= MAX_PARAM_LENGTH else text[: MAX_PARAM_LENGTH - 1] + "…"


def log_run(result, explanation=None, decision=None, report_html: str = "",
            dataset_name: str = "") -> str:
    """Log one finished run. Returns the MLflow run id."""
    client = MlflowClient(tracking_uri=_tracking_uri())
    experiment_id = _experiment_id(client)
    run = client.create_run(experiment_id,
                            run_name=f"{dataset_name or 'dataset'} - {result.best.name}")
    run_id = run.info.run_id
    try:
        # --- Settings -------------------------------------------------------
        params = {
            "dataset": dataset_name,
            "target": result.target,
            "problem_type": result.problem_type,
            "positive_class": result.positive_class,
            "rows_train": result.n_train,
            "rows_test": result.n_test,
            "columns_used": len(result.profile.numeric_features)
                            + len(result.profile.categorical_features),
            "columns_excluded": len(result.profile.excluded),
            "selection_metric": result.selection_metric,
            "best_model": result.best.name,
            "tuned": result.tuned,
            "seed": config.RANDOM_SEED,
            "cv_folds": config.CV_FOLDS,
            "test_size": config.TEST_SIZE,
        }
        for key, value in result.best.tuned_params.items():
            params[f"tuned.{key}"] = value
        if decision is not None and decision.applicable:
            params.update({"cost_false_alarm": decision.cost_fp, "cost_missed": decision.cost_fn,
                           "calibrated": decision.calibrated, "threshold": decision.threshold})
        for key, value in params.items():
            client.log_param(run_id, key, _param(value))

        # --- Metrics --------------------------------------------------------
        for r in result.model_results:
            if r.error:
                continue
            for metric, value in r.cv_mean.items():
                client.log_metric(run_id, f"cv.{r.key}.{metric}.mean", value)
                client.log_metric(run_id, f"cv.{r.key}.{metric}.std", r.cv_std[metric])
        for metric, value in result.test_scores.items():
            client.log_metric(run_id, f"test.{metric}", value)
        client.log_metric(run_id, "runtime_seconds", result.runtime_seconds)
        if decision is not None and decision.applicable:
            chosen = decision.test_chosen
            client.log_metric(run_id, "test.brier", decision.test_brier_calibrated
                              if decision.calibrated else decision.test_brier_raw)
            client.log_metric(run_id, "test.cost_per_1000",
                              chosen.cost_per_1000(decision.cost_fp, decision.cost_fn))
            client.log_metric(run_id, "test.recall_at_threshold", chosen.recall)

        # --- Files ----------------------------------------------------------
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            profile_rows = [{"column": c.name, "kind": c.kind, "used": c.use,
                             "missing_share": round(c.missing_share, 4),
                             "flags": c.flags, "reason": c.reason}
                            for c in result.profile.columns]
            (tmp / "profile.json").write_text(json.dumps(profile_rows, indent=2, default=str),
                                              encoding="utf-8")
            if explanation is not None:
                # Column names come from the upload: guard against formula injection.
                neutralize_formula_injection(explanation.importance).to_csv(
                    tmp / "importance.csv", index=False)
            if report_html:
                (tmp / "report.html").write_text(report_html, encoding="utf-8")
            joblib.dump(result.pipeline, tmp / "model.joblib")
            for path in tmp.iterdir():
                client.log_artifact(run_id, str(path))

        client.set_terminated(run_id, "FINISHED")
    except Exception:
        client.set_terminated(run_id, "FAILED")
        raise
    return run_id


def recent_runs(limit: int = 20) -> pd.DataFrame:
    """A table of recent runs (newest first), for comparing results."""
    if not config.MLFLOW_DB.exists():
        return pd.DataFrame()
    mlflow.set_tracking_uri(_tracking_uri())
    runs = mlflow.search_runs(experiment_names=[config.MLFLOW_EXPERIMENT],
                              max_results=limit, order_by=["start_time DESC"])
    if runs.empty:
        return runs
    keep = ["start_time", "tags.mlflow.runName", "params.problem_type",
            "params.best_model", "params.tuned", "params.selection_metric"]
    keep += sorted(c for c in runs.columns if c.startswith("metrics.test."))
    return runs[[c for c in keep if c in runs.columns]]


if __name__ == "__main__":
    # venv\Scripts\python.exe -m foresight.tracking  -> show recent runs
    table = recent_runs()
    print("No runs logged yet." if table.empty else table.to_string(index=False))
