"""Tests for foresight/tracking.py (local MLflow tracking, telemetry off)."""

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from foresight import config
from foresight.decision import analyze_decision
from foresight.explain import explain_model
from foresight.report import build_report
from foresight.train import run_training

N = 300
ROW_ID_PREFIX = "CUSTOMER-ROW-"   # unique per row: row contents, must never be logged

# Variables that make MLflow think it runs inside tests/CI (it then skips
# telemetry on its own). The subprocess test removes them, so it checks the
# real-world situation, where only OUR settings keep telemetry off.
MLFLOW_TEST_MARKERS = [
    "PYTEST_CURRENT_TEST", "GITHUB_ACTIONS", "CI", "CIRCLECI", "GITLAB_CI", "JENKINS_URL",
    "TRAVIS", "TF_BUILD", "BITBUCKET_BUILD_NUMBER", "CODEBUILD_BUILD_ARN", "BUILDKITE",
    "TEAMCITY_VERSION", "CLOUD_RUN_EXECUTION", "RUNBOT_HOST_URL", "RUNBOT_BUILD_NAME",
    "RUNBOT_WORKER_ID", "MLFLOW_DISABLE_TELEMETRY", "DO_NOT_TRACK",
]


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MLFLOW_DB", tmp_path / "mlflow.db")
    monkeypatch.setattr(config, "MLFLOW_ARTIFACT_DIR", tmp_path / "mlruns")
    return tmp_path


def churn_df(n=N):
    rng = np.random.default_rng(config.RANDOM_SEED)
    df = pd.DataFrame({
        "customer_ref": [f"{ROW_ID_PREFIX}{i:05d}" for i in range(n)],
        "tenure": rng.normal(24, 10, n),
        "plan": rng.choice(["basic", "pro"], n),
    })
    logit = -0.12 * (df["tenure"] - 24) + (df["plan"] == "basic")
    df["churn"] = np.where(logit + rng.logistic(size=n) > 0.3, "yes", "no")
    return df


@pytest.fixture(scope="module")
def finished():
    result = run_training(churn_df(), "churn", "binary", positive_class="yes")
    explanation = explain_model(result)
    decision = analyze_decision(result, cost_fp=1, cost_fn=5)
    report = build_report(result, explanation, "demo.csv", decision=decision)
    return result, explanation, decision, report


def run_data(store, run_id):
    from mlflow.tracking import MlflowClient

    from foresight import tracking
    client = MlflowClient(tracking_uri=tracking._tracking_uri())
    return client, client.get_run(run_id)


# ---------------------------------------------------------------------------
# What gets logged
# ---------------------------------------------------------------------------
def test_logs_real_numbers(store, finished):
    from foresight import tracking

    result, explanation, decision, report = finished
    run_id = tracking.log_run(result, explanation, decision, report, "demo.csv")
    _, run = run_data(store, run_id)

    assert run.info.status == "FINISHED"
    assert run.data.params["best_model"] == result.best.name
    assert run.data.params["threshold"] == str(decision.threshold)
    assert run.data.params["cost_missed"] == "5.0"
    m = run.data.metrics
    assert m["test.roc_auc"] == pytest.approx(result.test_scores["roc_auc"])
    best = result.best
    assert m[f"cv.{best.key}.roc_auc.mean"] == pytest.approx(best.cv_mean["roc_auc"])
    assert m[f"cv.{best.key}.roc_auc.std"] == pytest.approx(best.cv_std["roc_auc"])
    assert m["test.cost_per_1000"] == pytest.approx(
        decision.test_chosen.cost_per_1000(1, 5))


def test_logs_files_inside_project_store(store, finished):
    from foresight import tracking

    run_id = tracking.log_run(*finished, dataset_name="demo.csv")
    client, _ = run_data(store, run_id)
    names = {f.path for f in client.list_artifacts(run_id)}
    assert names == {"profile.json", "importance.csv", "report.html", "model.joblib"}
    # Files are stored under the configured folder, nowhere else.
    assert any((store / "mlruns").rglob("report.html"))


def test_no_data_rows_logged(store, finished):
    # Aggregated results (column names, category labels in explanations) are
    # fine, as in the report. Row contents (IDs, exact values) are not.
    from foresight import tracking

    result = finished[0]
    tracking.log_run(*finished, dataset_name="demo.csv")
    exact_values = [repr(v) for v in result.X_test["tenure"].head(20)]
    text_files = [p for p in (store / "mlruns").rglob("*")
                  if p.suffix in (".json", ".csv", ".html")]
    assert len(text_files) == 3
    for path in text_files:
        text = path.read_text(encoding="utf-8")
        assert ROW_ID_PREFIX not in text, path.name
        assert not any(v in text for v in exact_values), path.name
    profile = next((store / "mlruns").rglob("profile.json"))
    columns = {row["column"] for row in json.loads(profile.read_text(encoding="utf-8"))}
    assert columns == {"customer_ref", "tenure", "plan"}


def test_recent_runs(store, finished):
    from foresight import tracking

    assert tracking.recent_runs().empty            # nothing logged yet
    tracking.log_run(*finished, dataset_name="a.csv")
    tracking.log_run(*finished, dataset_name="b.csv")
    table = tracking.recent_runs()
    assert len(table) == 2
    assert "metrics.test.roc_auc" in table.columns


def test_regression_run_without_decision(store):
    from foresight import tracking

    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=N)})
    df["y"] = 3 * df["x"] + rng.normal(size=N)
    result = run_training(df, "y", "regression")
    run_id = tracking.log_run(result)
    _, run = run_data(store, run_id)
    assert "test.rmse" in run.data.metrics
    assert "threshold" not in run.data.params


# ---------------------------------------------------------------------------
# No internet: telemetry off, and only tracking.py may import MLflow
# ---------------------------------------------------------------------------
SUBPROCESS_SCRIPT = r"""
import socket, sys
def blocked(*args, **kwargs):
    raise RuntimeError("network access attempted")
socket.socket.connect = blocked
socket.socket.connect_ex = blocked
socket.create_connection = blocked

mode = sys.argv[1]
if mode == "control":
    # Plain MLflow import in this environment: telemetry would be ON.
    import mlflow
    from mlflow.telemetry.utils import is_telemetry_disabled
    print("disabled" if is_telemetry_disabled() else "enabled")
else:
    from pathlib import Path
    import numpy as np, pandas as pd
    from foresight import config
    from foresight import tracking
    from mlflow.telemetry import get_telemetry_client
    from mlflow.telemetry.utils import is_telemetry_disabled
    assert is_telemetry_disabled(), "telemetry is on"
    assert get_telemetry_client() is None, "telemetry client exists"
    config.MLFLOW_DB = Path(sys.argv[2]) / "mlflow.db"
    config.MLFLOW_ARTIFACT_DIR = Path(sys.argv[2]) / "mlruns"
    from foresight.train import run_training
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=200)})
    df["y"] = np.where(df["x"] + rng.normal(size=200) > 0, "a", "b")
    tracking.log_run(run_training(df, "y", "binary"))
    print("disabled")
"""


def run_outside_test_env(*args):
    env = {k: v for k, v in os.environ.items() if k not in MLFLOW_TEST_MARKERS}
    return subprocess.run([sys.executable, "-c", SUBPROCESS_SCRIPT, *args],
                          cwd=config.PROJECT_ROOT, env=env, capture_output=True,
                          text=True, timeout=300)


def test_control_plain_mlflow_would_send_telemetry():
    # Proves the subprocess environment is "real world": without our
    # settings, MLflow would have telemetry switched on.
    done = run_outside_test_env("control")
    assert done.stdout.strip() == "enabled", done.stderr[-2000:]


def test_tracking_module_disables_telemetry_and_uses_no_network(tmp_path):
    done = run_outside_test_env("tracking", str(tmp_path))
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip().endswith("disabled")
    assert (tmp_path / "mlflow.db").exists()


def test_only_tracking_module_imports_mlflow():
    sources = [config.PROJECT_ROOT / "app.py",
               *(config.PROJECT_ROOT / "foresight").glob("*.py"),
               *(config.PROJECT_ROOT / "benchmarks").glob("*.py")]
    for path in sources:
        if path.name == "tracking.py":
            continue
        text = path.read_text(encoding="utf-8")
        assert "import mlflow" not in text and "from mlflow" not in text, path.name
