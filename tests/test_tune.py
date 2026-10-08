"""Tests for foresight/tune.py and tuning inside run_training."""

import numpy as np
import optuna
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from foresight import config, train, tune
from foresight.decision import analyze_decision
from foresight.explain import explain_model
from foresight.models import get_metrics, get_models
from foresight.train import run_training

N = 300


@pytest.fixture(autouse=True)
def small_budget(monkeypatch):
    # Few trials keep the tests fast; the logic is the same as with 20.
    monkeypatch.setattr(config, "TUNE_TRIALS", 3)
    monkeypatch.setattr(config, "TUNE_TIMEOUT_SECONDS", 60)


def churn_df(n=N):
    rng = np.random.default_rng(config.RANDOM_SEED)
    df = pd.DataFrame({
        "tenure": rng.normal(24, 10, n),
        "calls": rng.poisson(1.5, n).astype(float),
        "plan": rng.choice(["basic", "pro"], n),
    })
    logit = -0.12 * (df["tenure"] - 24) + 0.8 * (df["calls"] - 1.5) + (df["plan"] == "basic")
    df["churn"] = np.where(logit + rng.logistic(size=n) > 0.3, "yes", "no")
    return df


# ---------------------------------------------------------------------------
# Search spaces and single searches
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("problem_type", ["binary", "multiclass", "regression"])
@pytest.mark.parametrize("family", tune.TUNABLE)
def test_search_space_is_valid(family, problem_type):
    spec = next(s for s in get_models(problem_type) if s.key == family)
    trial = optuna.trial.FixedTrial({
        "n_estimators": 150, "learning_rate": 0.05, "num_leaves": 31,
        "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8,
        "reg_lambda": 1.0, "max_depth": 8, "min_samples_leaf": 2, "max_features": 0.5,
    })
    params = tune.suggest_params(family, trial)
    model = spec.make().set_params(**params)   # rejects unknown parameter names
    for key, value in params.items():
        assert model.get_params()[key] == value


def test_tuned_spec_keeps_family_and_settings():
    base = next(s for s in get_models("binary") if s.key == "lightgbm")
    spec = tune.tuned_spec(base, {"n_estimators": 123, "learning_rate": 0.1})
    assert spec.key == "lightgbm_tuned" and spec.family == "lightgbm"
    assert spec.name == "LightGBM (tuned)"
    assert spec.make().get_params()["n_estimators"] == 123


def test_search_is_reproducible():
    df = churn_df()
    X, y = df.drop(columns="churn"), (df["churn"] == "yes").to_numpy(dtype=int)
    profile = train.preview_profile(df, "churn", "binary", "yes")
    spec = next(s for s in get_models("binary") if s.key == "lightgbm")
    scorer = get_metrics("binary")[0].scorer
    a = tune.search(spec, profile, X, y, "binary", scorer)
    b = tune.search(spec, profile, X, y, "binary", scorer)
    assert a.params == b.params
    assert a.n_trials == 3 and not a.hit_time_limit


def test_time_limit_stops_search(monkeypatch):
    monkeypatch.setattr(config, "TUNE_TRIALS", 1000)
    monkeypatch.setattr(config, "TUNE_TIMEOUT_SECONDS", 2)
    df = churn_df()
    X, y = df.drop(columns="churn"), (df["churn"] == "yes").to_numpy(dtype=int)
    profile = train.preview_profile(df, "churn", "binary", "yes")
    spec = next(s for s in get_models("binary") if s.key == "random_forest")
    found = tune.search(spec, profile, X, y, "binary", get_metrics("binary")[0].scorer)
    assert found.n_trials < 1000
    assert found.hit_time_limit


# ---------------------------------------------------------------------------
# Nested CV: tuning never sees the fold it is scored on
# ---------------------------------------------------------------------------
def test_nested_search_never_sees_validation_fold(monkeypatch):
    df = churn_df()
    X, y = df.drop(columns="churn"), (df["churn"] == "yes").to_numpy(dtype=int)
    profile = train.preview_profile(df, "churn", "binary", "yes")
    spec = next(s for s in get_models("binary") if s.key == "lightgbm")
    outer = StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)

    seen = []
    real_search = tune.search

    def spy(spec, profile, X_part, y_part, problem_type, scorer):
        seen.append(set(X_part.index))
        return real_search(spec, profile, X_part, y_part, problem_type, scorer)

    monkeypatch.setattr(tune, "search", spy)
    metrics = get_metrics("binary")
    result = tune.nested_cv(spec, profile, X, y, "binary", metrics, metrics[0].scorer, outer)

    assert len(seen) == config.CV_FOLDS                 # one search per outer fold
    for searched, (_, val_idx) in zip(seen, outer.split(X, y)):
        assert searched.isdisjoint(X.index[val_idx])    # validation rows never seen
    assert len(result.fold_scores["roc_auc"]) == config.CV_FOLDS


# ---------------------------------------------------------------------------
# Tuning inside run_training
# ---------------------------------------------------------------------------
def test_tuning_off_by_default():
    result = run_training(churn_df(), "churn", "binary", positive_class="yes")
    assert not result.tuned
    assert not any(r.key.endswith("_tuned") for r in result.model_results)


def test_tuning_adds_tuned_candidates():
    result = run_training(churn_df(), "churn", "binary", positive_class="yes", tune=True)
    keys = [r.key for r in result.model_results]
    assert keys == ["dummy", "linear", "random_forest", "lightgbm",
                    "random_forest_tuned", "lightgbm_tuned"]
    for key in ("random_forest_tuned", "lightgbm_tuned"):
        r = result.result_for(key)
        assert not r.error
        assert set(r.cv_mean) == {"roc_auc", "pr_auc", "f1", "accuracy"}


def weak_lightgbm_only(monkeypatch):
    """Only LightGBM, with deliberately poor defaults so tuning clearly wins."""
    from foresight.models import ModelSpec

    def lineup(problem_type, balanced=False):
        base = next(s for s in get_models(problem_type, balanced) if s.key == "lightgbm")
        return [ModelSpec("lightgbm", "LightGBM", "candidate", False,
                          lambda: base.make().set_params(n_estimators=5, num_leaves=2,
                                                         learning_rate=0.01))]
    monkeypatch.setattr(train, "get_models", lineup)


def test_tuned_winner_end_to_end(monkeypatch):
    weak_lightgbm_only(monkeypatch)
    calls = []
    real_score = train.score_model
    monkeypatch.setattr(train, "score_model",
                        lambda m, X, y, pt: calls.append(1) or real_score(m, X, y, pt))

    result = run_training(churn_df(), "churn", "binary", positive_class="yes", tune=True)
    assert result.best_key == "lightgbm_tuned"
    assert len(calls) == 1                      # test set still used exactly once
    assert result.best.tuned_params             # final settings recorded
    assert result.best_spec.family == "lightgbm"
    model = result.pipeline.named_steps["model"]
    for k, v in result.best.tuned_params.items():
        assert model.get_params()[k] == v       # final model really uses them
    exp = explain_model(result)
    assert exp.shap_available
    assert analyze_decision(result, 1, 5).applicable


def test_tuning_ignores_the_test_set(monkeypatch):
    weak_lightgbm_only(monkeypatch)   # so the tuned path (final search) really runs
    clean = run_training(churn_df(), "churn", "binary", positive_class="yes", tune=True)
    real_split = train.train_test_split

    def corrupted(*args, **kwargs):
        X_tr, X_te, y_tr, y_te = real_split(*args, **kwargs)
        return X_tr, X_te, y_tr, 1 - y_te

    monkeypatch.setattr(train, "train_test_split", corrupted)
    flipped = run_training(churn_df(), "churn", "binary", positive_class="yes", tune=True)
    assert clean.best_key == "lightgbm_tuned"
    assert flipped.best_key == clean.best_key
    assert flipped.best.cv_mean == pytest.approx(clean.best.cv_mean)
    assert flipped.best.tuned_params == clean.best.tuned_params


def test_regression_tuning():
    rng = np.random.default_rng(config.RANDOM_SEED)
    df = pd.DataFrame({"size": rng.normal(100, 20, N), "rooms": rng.integers(1, 6, N)})
    df["price"] = 3 * df["size"] + 10 * df["rooms"] + rng.normal(scale=20, size=N)
    result = run_training(df, "price", "regression", tune=True)
    tuned = result.result_for("lightgbm_tuned")
    assert tuned.cv_mean["rmse"] > 0


def test_report_mentions_tuning():
    from foresight.report import build_report

    # Normal line-up (the report always expects the baseline and simple model).
    result = run_training(churn_df(), "churn", "binary", positive_class="yes", tune=True)
    page = build_report(result, explain_model(result))
    assert "nested cross-validation" in page
    assert "LightGBM (tuned)" in page   # shown in the model comparison table


def test_report_lists_final_tuned_settings(monkeypatch):
    from foresight.report import tuning_context

    weak_lightgbm_only(monkeypatch)
    result = run_training(churn_df(), "churn", "binary", positive_class="yes", tune=True)
    text = tuning_context(result)["final_params"]
    assert "n_estimators = " in text and "learning_rate = " in text
