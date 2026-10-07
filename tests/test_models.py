"""Tests for foresight/models.py."""

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from foresight import config
from foresight.models import (
    METRICS,
    get_metrics,
    get_models,
    score_model,
    selection_metric,
)

N = 200
PROBLEM_TYPES = ["binary", "multiclass", "regression"]


def make_data(problem_type):
    rng = np.random.default_rng(config.RANDOM_SEED)
    X = rng.normal(size=(N, 4))
    signal = X[:, 0] + 0.5 * X[:, 1]
    if problem_type == "binary":
        y = (signal > 0).astype(int)
    elif problem_type == "multiclass":
        y = np.digitize(signal, [-0.5, 0.5])   # classes 0, 1, 2
    else:
        y = 3 * signal + rng.normal(scale=0.1, size=N)
    return X, y


# ---------------------------------------------------------------------------
# Model line-up
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("problem_type", PROBLEM_TYPES)
def test_lineup(problem_type):
    specs = get_models(problem_type)
    assert [s.key for s in specs] == ["dummy", "linear", "random_forest", "lightgbm"]
    assert [s.role for s in specs] == ["baseline", "simple", "candidate", "candidate"]
    # Only the linear model needs scaled inputs.
    assert [s.scale for s in specs] == [False, True, False, False]


@pytest.mark.parametrize("problem_type", PROBLEM_TYPES)
def test_every_model_fits_and_predicts(problem_type):
    X, y = make_data(problem_type)
    for spec in get_models(problem_type):
        model = spec.make().fit(X, y)
        assert len(model.predict(X)) == N
        if problem_type != "regression":
            proba = model.predict_proba(X)
            assert proba.shape == (N, len(np.unique(y)))


def test_make_returns_fresh_estimators():
    spec = get_models("binary")[2]
    assert spec.make() is not spec.make()


def test_balanced_sets_class_weight():
    for spec in get_models("binary", balanced=True)[1:]:
        assert spec.make().get_params()["class_weight"] == "balanced"
    for spec in get_models("binary", balanced=False)[1:]:
        assert spec.make().get_params()["class_weight"] is None


@pytest.mark.parametrize("key", ["random_forest", "lightgbm"])
def test_models_are_reproducible(key):
    X, y = make_data("regression")
    spec = next(s for s in get_models("regression") if s.key == key)
    a = spec.make().fit(X, y).predict(X)
    b = spec.make().fit(X, y).predict(X)
    # Parallel tree averaging can differ in the last floating-point digit
    # (~1e-15), so compare to 12 significant digits rather than bit-for-bit.
    np.testing.assert_allclose(a, b, rtol=1e-12)


def test_unknown_problem_type():
    with pytest.raises(ValueError):
        get_models("clustering")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def test_metric_keys():
    assert [m.key for m in get_metrics("binary")] == ["roc_auc", "pr_auc", "f1", "accuracy"]
    assert [m.key for m in get_metrics("multiclass")] == ["f1_macro", "accuracy"]
    assert [m.key for m in get_metrics("regression")] == ["mae", "rmse", "r2"]


def test_every_metric_has_a_description():
    for metrics in METRICS.values():
        for m in metrics:
            assert len(m.description) > 20


def test_selection_metrics_exist():
    for problem_type in PROBLEM_TYPES:
        keys = {m.key for m in get_metrics(problem_type)}
        assert selection_metric(problem_type, False) in keys
        assert selection_metric(problem_type, True) in keys


def test_selection_metric_choice():
    assert selection_metric("binary", False) == "roc_auc"
    assert selection_metric("binary", True) == "pr_auc"
    assert selection_metric("multiclass", True) == "f1_macro"
    assert selection_metric("regression", False) == "rmse"


class Perfect(BaseEstimator):
    """Returns the true answers, to check metric directions."""

    def fit(self, X, y):
        self.y_ = np.asarray(y)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return self.y_

    def predict_proba(self, X):
        return np.eye(len(self.classes_))[self.y_]


class PerfectClassifier(ClassifierMixin, Perfect):
    pass


class PerfectRegressor(RegressorMixin, Perfect):
    pass


def test_perfect_classifier_scores():
    X, y = make_data("binary")
    scores = score_model(PerfectClassifier().fit(X, y), X, y, "binary")
    assert scores == {"roc_auc": 1.0, "pr_auc": 1.0, "f1": 1.0, "accuracy": 1.0}


def test_perfect_regressor_errors_are_zero():
    X, y = make_data("regression")
    scores = score_model(PerfectRegressor().fit(X, y), X, y, "regression")
    assert scores["mae"] == pytest.approx(0) and scores["rmse"] == pytest.approx(0)
    assert scores["r2"] == pytest.approx(1)


def test_dummy_baseline_scores():
    X, y = make_data("binary")
    dummy = get_models("binary")[0].make().fit(X, y)
    scores = score_model(dummy, X, y, "binary")
    assert scores["roc_auc"] == pytest.approx(0.5)


def test_errors_reported_as_positive_numbers():
    X, y = make_data("regression")
    dummy = get_models("regression")[0].make().fit(X, y)
    scores = score_model(dummy, X, y, "regression")
    assert scores["mae"] > 0 and scores["rmse"] >= scores["mae"]
    assert scores["r2"] == pytest.approx(0, abs=1e-9)


def test_real_model_beats_baseline():
    X, y = make_data("binary")
    specs = {s.key: s for s in get_models("binary")}
    lgbm = score_model(specs["lightgbm"].make().fit(X, y), X, y, "binary")
    dummy = score_model(specs["dummy"].make().fit(X, y), X, y, "binary")
    assert lgbm["roc_auc"] > dummy["roc_auc"]
