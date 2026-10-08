"""Run the full modeling workflow on one dataset.

  1. Encode the target and split off a held-out TEST set (20%, stratified).
  2. Profile the TRAINING set only (column choices, leakage check).
  3. 5-fold cross-validation for every model on the training set.
  4. Pick the best model by its mean CV score (the test set plays no part).
  5. Refit the best model on the whole training set.
  6. Score it ONCE on the test set.
  7. Optionally save it to the outputs folder.
"""

import re
import time
import uuid
import warnings
from dataclasses import dataclass, field
from typing import Callable

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split

from foresight import config
from foresight.ingest import IngestError, check_target_for_problem_type
from foresight.models import get_metrics, get_models, score_model, selection_metric
from foresight.preprocess import build_pipeline
from foresight.profile import DataProfile, profile_data
from foresight.tune import TUNABLE, nested_cv, search, tuned_spec


class TrainError(Exception):
    """A problem that stops training. The message is safe to show users."""


@dataclass
class ModelResult:
    key: str
    name: str
    role: str                         # "baseline", "simple", "candidate"
    cv_mean: dict[str, float] = field(default_factory=dict)
    cv_std: dict[str, float] = field(default_factory=dict)
    fit_seconds: float = 0.0          # average training time per fold
    error: str = ""                   # set if this model failed
    tuned_params: dict = field(default_factory=dict)   # final tuned settings, if any


@dataclass
class TrainResult:
    run_id: str
    target: str
    problem_type: str
    class_labels: list | None         # original label for each code 0, 1, ...
    positive_class: object            # binary only: the label encoded as 1
    profile: DataProfile
    n_train: int
    n_test: int
    n_dropped_target: int             # rows dropped for a missing target
    selection_metric: str
    model_results: list[ModelResult]
    best_key: str
    test_scores: dict[str, float]
    pipeline: object                  # the fitted best pipeline
    X_train: pd.DataFrame
    y_train: np.ndarray
    X_test: pd.DataFrame
    y_test: np.ndarray
    runtime_seconds: float
    included_overrides: list[str] = field(default_factory=list)
    best_spec: object = None          # ModelSpec that builds the best model (unfitted)
    tuned: bool = False               # was Optuna tuning switched on?
    tuning_hit_time_limit: bool = False

    @property
    def best(self) -> ModelResult:
        return self.result_for(self.best_key)

    def result_for(self, key: str) -> ModelResult:
        return next(r for r in self.model_results if r.key == key)


# ---------------------------------------------------------------------------
# Target encoding
# ---------------------------------------------------------------------------
def default_positive_class(y: pd.Series):
    """The rarer class: usually the event of interest (churned, defaulted)."""
    counts = y.value_counts()
    rarest = counts[counts == counts.min()].index
    return sorted(rarest, key=str)[-1]   # deterministic if there is a tie


def encode_target(y: pd.Series, problem_type: str, positive_class=None):
    """Turn class labels into codes 0..k-1 (binary: positive class = 1).

    Returns (encoded y, class_labels, positive_class). class_labels[i] is the
    original label for code i. Regression targets are returned as floats.
    """
    if problem_type == "regression":
        return y.astype(float).to_numpy(), None, None

    labels = sorted(y.unique(), key=str)
    if problem_type == "binary":
        if positive_class is None:
            positive_class = default_positive_class(y)
        if positive_class not in labels:
            raise TrainError("The chosen positive class is not a value of the target.")
        negative = next(label for label in labels if label != positive_class)
        labels = [negative, positive_class]

    codes = {label: i for i, label in enumerate(labels)}
    return y.map(codes).to_numpy(dtype=int), labels, positive_class


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _cv_splitter(problem_type: str):
    if problem_type == "regression":
        return KFold(config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)
    return StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)


def _apply_overrides(profile: DataProfile, include: list[str]) -> list[str]:
    """Let the user keep columns that were excluded ONLY for likely leakage."""
    applied = []
    for col in profile.columns:
        if col.name in include and not col.use and "likely_leakage" in col.flags:
            col.use = True
            col.reason = ""
            col.flags.append("user_included")
            applied.append(col.name)
    return applied


def _is_better(score: float, best: float | None, higher_is_better: bool) -> bool:
    if best is None:
        return True
    return score > best if higher_is_better else score < best


# ---------------------------------------------------------------------------
# Split and profile (shared by the UI preview and the real run)
# ---------------------------------------------------------------------------
def split_data(df: pd.DataFrame, target: str, problem_type: str, positive_class=None):
    """Check the target, encode it, and split off the held-out test set.

    Deterministic (fixed seed), so calling it twice gives the same split.
    Returns X_train, X_test, y_train, y_test, class_labels, positive_class.
    """
    try:
        check_target_for_problem_type(df[target], problem_type)
    except IngestError as err:
        raise TrainError(str(err)) from None

    X = df.drop(columns=[target])
    y, class_labels, positive_class = encode_target(df[target], problem_type, positive_class)
    stratify = None if problem_type == "regression" else y
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=stratify,
        random_state=config.RANDOM_SEED,
    )
    return X_train, X_test, y_train, y_test, class_labels, positive_class


def _profile_training(X_train, y_train, class_labels, problem_type) -> DataProfile:
    # Profile with the original labels so the report shows real class names.
    labels = pd.Series(y_train) if class_labels is None else \
        pd.Series([class_labels[c] for c in y_train])
    return profile_data(X_train, labels, problem_type)


def preview_profile(df: pd.DataFrame, target: str, problem_type: str,
                    positive_class=None) -> DataProfile:
    """The exact profile run_training will compute, without training.

    Lets the UI show data checks and leakage flags before training starts.
    """
    X_train, _, y_train, _, class_labels, _ = split_data(df, target, problem_type,
                                                         positive_class)
    return _profile_training(X_train, y_train, class_labels, problem_type)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_training(
    df: pd.DataFrame,
    target: str,
    problem_type: str,
    positive_class=None,
    include_columns: list[str] = (),
    n_dropped_target: int = 0,
    save: bool = False,
    on_progress: Callable[[str, float], None] | None = None,
    tune: bool = False,
) -> TrainResult:
    """Train, compare, select, and test. `df` should come from ingest.prepare_target.

    tune=True also adds Optuna-tuned versions of random forest and LightGBM,
    scored with nested cross-validation (see foresight/tune.py).
    """
    started = time.perf_counter()

    def progress(message: str, fraction: float):
        if on_progress:
            on_progress(message, fraction)

    # 1. Encode the target and split off the held-out test set.
    X_train, X_test, y_train, y_test, class_labels, positive_class = split_data(
        df, target, problem_type, positive_class)

    # 2. Profile the training data only.
    progress("Profiling the data and checking for leakage", 0.05)
    profile = _profile_training(X_train, y_train, class_labels, problem_type)
    included = _apply_overrides(profile, list(include_columns))
    if not profile.numeric_features and not profile.categorical_features:
        raise TrainError("No usable input columns remain after the data checks, "
                         "so no model can be trained.")

    # 3. Cross-validate every model on the training set.
    metrics = get_metrics(problem_type)
    scoring = {m.key: m.scorer for m in metrics}
    specs = get_models(problem_type, balanced=profile.is_imbalanced)
    cv = _cv_splitter(problem_type)
    results: list[ModelResult] = []
    metric_key = selection_metric(problem_type, profile.is_imbalanced)
    selection = next(m for m in metrics if m.key == metric_key)
    cv_share = 0.45 if tune else 0.70   # share of the progress bar for plain CV

    for i, spec in enumerate(specs):
        progress(f"Cross-validating: {spec.name}", 0.10 + cv_share * i / len(specs))
        result = ModelResult(spec.key, spec.name, spec.role)
        pipeline = build_pipeline(profile, spec.make(), scale=spec.scale)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                cv_out = cross_validate(pipeline, X_train, y_train, cv=cv,
                                        scoring=scoring, error_score="raise")
        except Exception:
            # One model failing should not stop the others.
            result.error = "This model could not be trained on this data."
            results.append(result)
            continue
        for m in metrics:
            fold_scores = m.sign * cv_out[f"test_{m.key}"]
            result.cv_mean[m.key] = float(np.mean(fold_scores))
            result.cv_std[m.key] = float(np.std(fold_scores))
        result.fit_seconds = float(np.mean(cv_out["fit_time"]))
        results.append(result)

    # 3b. Optional: tuned versions, scored by nested CV on the SAME outer folds,
    #     so tuned and untuned scores are directly comparable.
    hit_time_limit = False
    if tune:
        tunable = [s for s in specs if s.family in TUNABLE]
        for i, spec in enumerate(tunable):
            progress(f"Tuning (nested cross-validation): {spec.name}",
                     0.55 + 0.25 * i / len(tunable))
            result = ModelResult(f"{spec.key}_tuned", f"{spec.name} (tuned)", "candidate")
            try:
                nested = nested_cv(spec, profile, X_train, y_train, problem_type,
                                   metrics, selection.scorer, cv)
            except Exception:
                result.error = "This model could not be tuned on this data."
                results.append(result)
                continue
            for m in metrics:
                result.cv_mean[m.key] = float(np.mean(nested.fold_scores[m.key]))
                result.cv_std[m.key] = float(np.std(nested.fold_scores[m.key]))
            result.fit_seconds = nested.fit_seconds
            hit_time_limit |= nested.hit_time_limit
            results.append(result)

    # 4. Pick the best model by mean CV score. Models are listed simplest
    #    first (tuned versions last) and a model must be strictly better to
    #    replace the leader, so ties go to the simpler model.
    higher = selection.higher_is_better
    best_key, best_score = None, None
    for r in results:
        if not r.error and _is_better(r.cv_mean[metric_key], best_score, higher):
            best_key, best_score = r.key, r.cv_mean[metric_key]
    if best_key is None:
        raise TrainError("None of the models could be trained on this data.")

    # 5. Refit the best model on the full training set.
    progress("Training the best model on all training data", 0.85)
    if best_key.endswith("_tuned"):
        # One more search, now on the whole training set, for the final settings.
        base_spec = next(s for s in specs if f"{s.key}_tuned" == best_key)
        found = search(base_spec, profile, X_train, y_train, problem_type, selection.scorer)
        hit_time_limit |= found.hit_time_limit
        best_spec = tuned_spec(base_spec, found.params)
        next(r for r in results if r.key == best_key).tuned_params = found.params
    else:
        best_spec = next(s for s in specs if s.key == best_key)
    best_pipeline = build_pipeline(profile, best_spec.make(), scale=best_spec.scale)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        best_pipeline.fit(X_train, y_train)

    # 6. The one and only evaluation on the held-out test set.
    progress("Evaluating on the held-out test set", 0.95)
    test_scores = score_model(best_pipeline, X_test, y_test, problem_type)

    result = TrainResult(
        run_id=uuid.uuid4().hex,
        target=target,
        problem_type=problem_type,
        class_labels=class_labels,
        positive_class=positive_class,
        profile=profile,
        n_train=len(X_train),
        n_test=len(X_test),
        n_dropped_target=n_dropped_target,
        selection_metric=metric_key,
        model_results=results,
        best_key=best_key,
        test_scores=test_scores,
        pipeline=best_pipeline,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        runtime_seconds=time.perf_counter() - started,
        included_overrides=included,
        best_spec=best_spec,
        tuned=tune,
        tuning_hit_time_limit=hit_time_limit,
    )

    # 7. Save the fitted model (optional).
    if save:
        save_model(result)
    progress("Done", 1.0)
    return result


# ---------------------------------------------------------------------------
# Saving and loading models
# ---------------------------------------------------------------------------
# joblib files can run code when loaded, so we only ever load files this app
# saved itself: the name is built from a run id we generated, and the path
# must be inside OUTPUT_DIR.
_RUN_ID = re.compile(r"^[0-9a-f]{32}$")


def _model_path(run_id: str):
    if not _RUN_ID.match(str(run_id)):
        raise TrainError("Invalid model id.")
    output_dir = config.OUTPUT_DIR.resolve()
    path = (output_dir / f"model_{run_id}.joblib").resolve()
    if path.parent != output_dir:   # defense in depth
        raise TrainError("Invalid model id.")
    return path


def save_model(result: TrainResult):
    """Save the fitted pipeline plus what is needed to use it."""
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = _model_path(result.run_id)
    joblib.dump({
        "pipeline": result.pipeline,
        "target": result.target,
        "problem_type": result.problem_type,
        "class_labels": result.class_labels,
        "positive_class": result.positive_class,
        "input_columns": list(result.X_train.columns),
        "sklearn_version": sklearn.__version__,
    }, path)
    return path


def load_model(run_id: str) -> dict:
    """Load a model this app saved earlier, by its run id."""
    path = _model_path(run_id)
    if not path.is_file():
        raise TrainError("No saved model was found with that id.")
    return joblib.load(path)
