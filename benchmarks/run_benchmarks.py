"""Run Foresight AutoML on several public datasets and record the results.

Usage (from the project folder):
    venv\\Scripts\\python.exe -m benchmarks.run_benchmarks
    venv\\Scripts\\python.exe -m benchmarks.run_benchmarks --only bank_marketing

Datasets are described in benchmarks/datasets.json and read from
benchmarks/data/ (download them yourself; that folder is not committed).

Every dataset goes through the same path as the app: converted to CSV bytes,
then load_csv -> prepare_target -> run_training -> explain_model. Every
number in results.csv comes from these real runs; nothing is hardcoded.
"""

import argparse
import io
import json
import time
from pathlib import Path

import pandas as pd

from foresight import config
from foresight.explain import explain_model
from foresight.ingest import (
    IngestError,
    detect_problem_type,
    load_csv,
    neutralize_formula_injection,
    prepare_target,
)
from foresight.report import build_report
from foresight.train import TrainError, run_training

BENCH_DIR = Path(__file__).resolve().parent
DATA_DIR = BENCH_DIR / "data"
DATASETS_FILE = BENCH_DIR / "datasets.json"
RESULTS_FILE = BENCH_DIR / "results.csv"
REPORTS_DIR = config.OUTPUT_DIR / "benchmark_reports"


class BenchmarkError(Exception):
    """A problem with a dataset entry. The message is safe to print."""


# ---------------------------------------------------------------------------
# Reading a dataset
# ---------------------------------------------------------------------------
def _resolve_inside(data_dir: Path, relative: str) -> Path:
    """The dataset path, which must stay inside the data folder."""
    data_dir = data_dir.resolve()
    path = (data_dir / relative).resolve()
    if not path.is_relative_to(data_dir):
        raise BenchmarkError(f"File path leaves the data folder: {relative}")
    if not path.is_file():
        raise BenchmarkError(f"File not found in the data folder: {relative}")
    return path


def read_dataset(spec: dict, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Read one dataset through the app's own CSV ingestion."""
    path = _resolve_inside(data_dir, spec["file"])
    raw = path.read_bytes()

    sep = spec.get("sep", ",")
    has_header = spec.get("header", True)
    if sep != "," or not has_header:
        # Not a standard CSV: parse with the given options, then convert to a
        # standard CSV so it still goes through load_csv's checks.
        frame = pd.read_csv(
            io.BytesIO(raw),
            sep=r"\s+" if sep == "whitespace" else sep,
            header=0 if has_header else None,
            names=spec.get("columns"),
        )
        raw = frame.to_csv(index=False).encode("utf-8")

    df = load_csv(raw, "dataset.csv")

    missing = [c for c in spec.get("drop_columns", []) if c not in df.columns]
    if missing:
        raise BenchmarkError(f"Columns to drop not found: {missing}")
    df = df.drop(columns=spec.get("drop_columns", []))

    if "target_labels" in spec:
        mapping = spec["target_labels"]
        df[spec["target"]] = df[spec["target"]].map(lambda v: mapping.get(str(v), v))
    return df


# ---------------------------------------------------------------------------
# Running one dataset
# ---------------------------------------------------------------------------
def run_one(spec: dict, data_dir: Path = DATA_DIR, save_report: bool = True) -> dict:
    """Train and evaluate one dataset. Returns one results row."""
    started = time.perf_counter()
    df = read_dataset(spec, data_dir)
    clean, n_dropped = prepare_target(df, spec["target"])

    problem_type = spec.get("problem_type")
    if problem_type is None:
        detected = detect_problem_type(clean[spec["target"]])
        if detected.ambiguous:
            raise BenchmarkError("Problem type is ambiguous; set 'problem_type' in datasets.json.")
        problem_type = detected.problem_type

    result = run_training(clean, spec["target"], problem_type,
                          positive_class=spec.get("positive_class"),
                          n_dropped_target=n_dropped)
    train_seconds = time.perf_counter() - started
    explanation = explain_model(result)

    if save_report:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        html = build_report(result, explanation, dataset_name=f"{spec['name']}.csv")
        (REPORTS_DIR / f"{spec['name']}.html").write_text(html, encoding="utf-8")

    key = result.selection_metric
    baseline = result.result_for("dummy")
    simple = result.result_for("linear")
    best = result.best
    return {
        "dataset": spec["name"],
        "rows": len(clean),
        "columns": df.shape[1] - 1,
        "problem_type": problem_type,
        "imbalanced": result.profile.is_imbalanced,
        "metric": key,
        "baseline_score": baseline.cv_mean.get(key),
        "simple_model_score": simple.cv_mean.get(key),
        "simple_model_std": simple.cv_std.get(key),
        "best_model": best.name,
        "best_score": best.cv_mean[key],
        "best_std": best.cv_std[key],
        "test_score": result.test_scores[key],
        "columns_used": len(result.profile.numeric_features) + len(result.profile.categorical_features),
        "leaky_columns_flagged": "; ".join(result.profile.leaky_columns),
        "columns_warned": "; ".join(c.name for c in result.profile.columns
                                    if "strong_single_predictor" in c.flags),
        "top_drivers": "; ".join(explanation.top_drivers[:3]),
        "train_seconds": round(train_seconds, 1),
        "total_seconds": round(time.perf_counter() - started, 1),
        "error": "",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def load_specs(path: Path = DATASETS_FILE) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_all(specs: list[dict], data_dir: Path = DATA_DIR, results_file: Path = RESULTS_FILE,
            save_report: bool = True) -> pd.DataFrame:
    rows = []
    for spec in specs:
        print(f"Running {spec['name']} ...", flush=True)
        try:
            row = run_one(spec, data_dir, save_report)
            print(f"  done in {row['total_seconds']}s: best {row['best_model']}, "
                  f"{row['metric']} CV {row['best_score']:.3f} ± {row['best_std']:.3f}, "
                  f"test {row['test_score']:.3f}", flush=True)
        except (BenchmarkError, IngestError, TrainError) as err:
            # Record the failure honestly instead of skipping it silently.
            print(f"  FAILED: {err}", flush=True)
            row = {"dataset": spec["name"], "error": str(err)}
        rows.append(row)

    results = pd.DataFrame(rows)
    # Protect against spreadsheet formula injection: column names from the
    # data (e.g. in leaky_columns_flagged) end up in this CSV.
    neutralize_formula_injection(results).to_csv(results_file, index=False)
    return results


def main():
    parser = argparse.ArgumentParser(description="Run Foresight AutoML benchmarks.")
    parser.add_argument("--only", nargs="+", help="names of datasets to run")
    args = parser.parse_args()

    specs = load_specs()
    if args.only:
        specs = [s for s in specs if s["name"] in args.only]
        if not specs:
            parser.error("none of the given dataset names are in datasets.json")

    results = run_all(specs)
    print(f"\nWrote {len(results)} rows to {RESULTS_FILE.relative_to(config.PROJECT_ROOT)}")
    print(f"Reports saved in {REPORTS_DIR.relative_to(config.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
