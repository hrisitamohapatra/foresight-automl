"""Tests for benchmarks/run_benchmarks.py, using small synthetic files."""

import json

import numpy as np
import pandas as pd
import pytest

from benchmarks import run_benchmarks as rb
from foresight import config
from foresight.train import run_training

N = 300


@pytest.fixture
def data_dir(tmp_path):
    rng = np.random.default_rng(config.RANDOM_SEED)
    d = tmp_path / "data"
    (d / "sub").mkdir(parents=True)

    # Standard CSV, binary.
    churn = pd.DataFrame({"tenure": rng.normal(24, 10, N), "plan": rng.choice(["a", "b"], N)})
    churn["churn"] = np.where(-0.15 * (churn["tenure"] - 24) + rng.logistic(size=N) > 0.5,
                              "Yes", "No")
    churn.to_csv(d / "churn.csv", index=False)

    # Space-separated, no header, numeric 1/2 target (like german.data).
    credit = pd.DataFrame({0: rng.choice(["A11", "A12"], N), 1: rng.integers(6, 48, N)})
    credit[2] = np.where(credit[1] + rng.normal(0, 10, N) > 30, 2, 1)
    credit.to_csv(d / "sub" / "credit.data", sep=" ", header=False, index=False)

    # Semicolon-separated with a column to drop (like bank-full.csv).
    bank = pd.DataFrame({"age": rng.integers(20, 70, N), "duration": rng.integers(0, 900, N)})
    bank["y"] = np.where(bank["age"] + rng.normal(0, 15, N) > 50, "yes", "no")
    bank.to_csv(d / "bank.csv", sep=";", index=False)
    return d


SPECS = [
    {"name": "churn", "file": "churn.csv", "target": "churn", "positive_class": "Yes"},
    {"name": "credit", "file": "sub/credit.data", "sep": "whitespace", "header": False,
     "columns": ["account", "months", "risk"], "target": "risk",
     "target_labels": {"1": "good", "2": "bad"}, "positive_class": "bad",
     "costs": {"false_alarm": 1, "missed": 5}},
    {"name": "bank", "file": "bank.csv", "sep": ";", "target": "y",
     "positive_class": "yes", "drop_columns": ["duration"]},
]


# ---------------------------------------------------------------------------
# Reading datasets
# ---------------------------------------------------------------------------
def test_whitespace_no_header_file(data_dir):
    df = rb.read_dataset(SPECS[1], data_dir)
    assert list(df.columns) == ["account", "months", "risk"]
    assert set(df["risk"]) == {"good", "bad"}


def test_semicolon_file_and_drop_columns(data_dir):
    df = rb.read_dataset(SPECS[2], data_dir)
    assert list(df.columns) == ["age", "y"]


def test_missing_drop_column_rejected(data_dir):
    spec = dict(SPECS[2], drop_columns=["nope"])
    with pytest.raises(rb.BenchmarkError, match="not found"):
        rb.read_dataset(spec, data_dir)


@pytest.mark.parametrize("bad_file", ["../secret.csv", "../../etc/passwd", "missing.csv"])
def test_paths_must_stay_in_data_folder(data_dir, bad_file):
    (data_dir.parent / "secret.csv").write_text("a,b\n1,2\n")
    with pytest.raises(rb.BenchmarkError):
        rb.read_dataset(dict(SPECS[0], file=bad_file), data_dir)


def test_files_go_through_app_limits(data_dir, monkeypatch):
    monkeypatch.setattr(config, "MAX_ROWS", 100)
    with pytest.raises(Exception, match="rows"):
        rb.read_dataset(SPECS[0], data_dir)


# ---------------------------------------------------------------------------
# Results come from real runs
# ---------------------------------------------------------------------------
def test_results_match_a_direct_run(data_dir, tmp_path):
    results_file = tmp_path / "results.csv"
    results = rb.run_all(SPECS, data_dir, results_file, save_report=False)
    assert list(results["dataset"]) == ["churn", "credit", "bank"]
    assert (results["error"] == "").all()

    # Re-run churn directly: the benchmark numbers must be exactly these.
    direct = run_training(pd.read_csv(data_dir / "churn.csv"), "churn", "binary",
                          positive_class="Yes")
    row = results.iloc[0]
    assert row["rows"] == N
    assert row["metric"] == direct.selection_metric
    assert row["best_score"] == pytest.approx(direct.best.cv_mean[direct.selection_metric])
    assert row["test_score"] == pytest.approx(direct.test_scores[direct.selection_metric])
    assert row["baseline_score"] == pytest.approx(0.5)


def test_costs_used_for_threshold(data_dir, tmp_path):
    results = rb.run_all(SPECS[:2], data_dir, tmp_path / "results.csv", save_report=False)
    churn, credit = results.iloc[0], results.iloc[1]
    assert churn["costs_fp_fn"] == "1.0:1.0"
    assert credit["costs_fp_fn"] == "1:5"
    # Expensive misses -> a lower cut-off than the 1:1 default would suggest.
    assert credit["threshold"] < 0.5
    assert credit["test_cost_per_1000_chosen"] > 0


def test_results_csv_columns(data_dir, tmp_path):
    results_file = tmp_path / "results.csv"
    rb.run_all(SPECS[:1], data_dir, results_file, save_report=False)
    written = pd.read_csv(results_file)
    for col in ("dataset", "rows", "problem_type", "baseline_score", "simple_model_score",
                "best_score", "best_std", "test_score", "total_seconds"):
        assert col in written.columns
    assert written["total_seconds"].iloc[0] > 0


def test_failure_recorded_not_hidden(data_dir, tmp_path):
    specs = [dict(SPECS[0], name="broken", file="does_not_exist.csv"), SPECS[0]]
    results = rb.run_all(specs, data_dir, tmp_path / "results.csv", save_report=False)
    assert "not found" in results.iloc[0]["error"]
    assert results.iloc[1]["error"] == ""


def test_results_csv_formula_injection_neutralized(data_dir, tmp_path):
    df = pd.read_csv(data_dir / "churn.csv")
    df["=leak()"] = (df["churn"] == "Yes").astype(int)
    df.to_csv(data_dir / "churn.csv", index=False)

    results_file = tmp_path / "results.csv"
    results = rb.run_all(SPECS[:1], data_dir, results_file, save_report=False)
    assert results.iloc[0]["leaky_columns_flagged"] == "=leak()"
    text = results_file.read_text(encoding="utf-8")
    assert ",'=leak()," in text


def test_track_logs_each_run(data_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MLFLOW_DB", tmp_path / "mlflow.db")
    monkeypatch.setattr(config, "MLFLOW_ARTIFACT_DIR", tmp_path / "mlruns")
    rb.run_all(SPECS[:2], data_dir, tmp_path / "results.csv", save_report=False, track=True)
    from foresight import tracking
    runs = tracking.recent_runs()
    assert len(runs) == 2
    assert set(runs["tags.mlflow.runName"].str.split(" - ").str[0]) == {"churn.csv", "credit.csv"}


def test_datasets_file_is_valid():
    specs = rb.load_specs()
    names = [s["name"] for s in specs]
    assert len(names) == len(set(names))
    for spec in specs:
        assert {"name", "file", "target"} <= set(spec)
