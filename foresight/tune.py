"""Optional hyperparameter tuning with Optuna, evaluated by nested CV.

Why nested? If we tuned on the same folds we then report, the reported score
would be optimistic: the settings were picked because they did well on those
very folds. Instead, for each OUTER fold we run a fresh Optuna search using
only that fold's training part (split again into inner folds), then score
the result on the outer fold's held-out part. The average over outer folds
is an honest estimate of "tune, then train".

The final tuned model is found by one more search on the whole training set.
The test set is never used here.
"""

import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import optuna
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

from foresight import config
from foresight.models import ModelSpec
from foresight.preprocess import build_pipeline

TUNABLE = ("random_forest", "lightgbm")

optuna.logging.set_verbosity(optuna.logging.WARNING)   # keep the console quiet


# ---------------------------------------------------------------------------
# Search spaces (ranges chosen to be safe defaults for tabular data)
# ---------------------------------------------------------------------------
def suggest_params(family: str, trial) -> dict:
    if family == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 128, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "subsample_freq": 1,   # needed for subsample to take effect
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
    if family == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 4, 32, log=True),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20, log=True),
            "max_features": trial.suggest_float("max_features", 0.2, 1.0),
        }
    raise ValueError(f"No search space for model type: {family}")


def tuned_spec(spec: ModelSpec, params: dict) -> ModelSpec:
    """A ModelSpec that builds the base model with the tuned settings."""
    return ModelSpec(
        key=f"{spec.key}_tuned",
        name=f"{spec.name} (tuned)",
        role="candidate",
        scale=spec.scale,
        make=lambda: spec.make().set_params(**params),
        family=spec.family,
    )


def _inner_cv(problem_type: str, folds: int):
    cls = KFold if problem_type == "regression" else StratifiedKFold
    return cls(folds, shuffle=True, random_state=config.RANDOM_SEED)


# ---------------------------------------------------------------------------
# One search
# ---------------------------------------------------------------------------
@dataclass
class SearchResult:
    params: dict
    n_trials: int
    hit_time_limit: bool


def search(spec: ModelSpec, profile, X, y, problem_type: str, scorer) -> SearchResult:
    """Optuna search on (X, y) only, scored by inner cross-validation.

    `scorer` is a scikit-learn scorer where higher is always better (error
    metrics are negated), so the study always maximizes.
    """
    inner = _inner_cv(problem_type, config.TUNE_INNER_FOLDS)

    def objective(trial):
        params = suggest_params(spec.family, trial)
        pipeline = build_pipeline(profile, spec.make().set_params(**params), scale=spec.scale)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            return float(np.mean(cross_val_score(pipeline, X, y, cv=inner, scoring=scorer)))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.RANDOM_SEED),
    )
    started = time.perf_counter()
    study.optimize(objective, n_trials=config.TUNE_TRIALS, timeout=config.TUNE_TIMEOUT_SECONDS)
    elapsed = time.perf_counter() - started

    # Rebuild the full parameter set (including fixed ones like subsample_freq).
    params = suggest_params(spec.family, optuna.trial.FixedTrial(study.best_params))
    n_done = len(study.trials)
    return SearchResult(params, n_done,
                        hit_time_limit=n_done < config.TUNE_TRIALS
                        and elapsed >= config.TUNE_TIMEOUT_SECONDS)


# ---------------------------------------------------------------------------
# Nested cross-validation
# ---------------------------------------------------------------------------
@dataclass
class NestedResult:
    fold_scores: dict[str, list[float]] = field(default_factory=dict)  # metric -> per fold
    fit_seconds: float = 0.0          # mean time per outer fold (search + fit)
    n_trials: list[int] = field(default_factory=list)
    hit_time_limit: bool = False


def nested_cv(spec: ModelSpec, profile, X, y, problem_type: str, metrics,
              selection_scorer, outer_cv) -> NestedResult:
    """Honest CV scores for "tune on the training part, then fit".

    `metrics` are models.Metric objects; scores use the same sign convention
    as the untuned CV (errors reported as positive numbers).
    """
    out = NestedResult(fold_scores={m.key: [] for m in metrics})
    times = []
    for train_idx, val_idx in outer_cv.split(X, y):
        started = time.perf_counter()
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]

        found = search(spec, profile, X_tr, y_tr, problem_type, selection_scorer)
        out.n_trials.append(found.n_trials)
        out.hit_time_limit |= found.hit_time_limit

        pipeline = build_pipeline(profile, spec.make().set_params(**found.params),
                                  scale=spec.scale)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            pipeline.fit(X_tr, y_tr)
        for m in metrics:
            out.fold_scores[m.key].append(float(m.sign * m.scorer(pipeline, X_val, y_val)))
        times.append(time.perf_counter() - started)
    out.fit_seconds = float(np.mean(times))
    return out
