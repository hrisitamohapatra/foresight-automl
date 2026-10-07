"""Tests for foresight/explain.py."""

import numpy as np
import pandas as pd
import pytest

from foresight import config, train
from foresight.explain import explain_model
from foresight.models import get_models
from foresight.train import run_training

N = 400


def churn_df(n=N):
    """tenure lowers churn, complaints raise it, plan 'basic' raises it, noise does nothing."""
    rng = np.random.default_rng(config.RANDOM_SEED)
    df = pd.DataFrame({
        "tenure": rng.normal(24, 10, size=n),
        "complaints": rng.poisson(1.5, size=n).astype(float),
        "noise": rng.normal(size=n),
        "plan": rng.choice(["basic", "pro", "family"], size=n),
    })
    logit = (-0.15 * (df["tenure"] - 24) + 1.0 * (df["complaints"] - 1.5)
             + 1.5 * (df["plan"] == "basic") - 0.5)
    df["churn"] = np.where(logit + rng.logistic(size=n) > 0, "yes", "no")
    return df


def only_model(monkeypatch, key):
    """Restrict the line-up to one model so it is guaranteed to be the best."""
    def lineup(problem_type, balanced=False):
        return [s for s in get_models(problem_type, balanced) if s.key == key]
    monkeypatch.setattr(train, "get_models", lineup)


@pytest.fixture(scope="module")
def churn_explanation():
    result = run_training(churn_df(), "churn", "binary", positive_class="yes")
    return result, explain_model(result)


# ---------------------------------------------------------------------------
# Rankings
# ---------------------------------------------------------------------------
def test_one_row_per_used_column(churn_explanation):
    _, exp = churn_explanation
    assert sorted(exp.importance["column"]) == ["complaints", "noise", "plan", "tenure"]
    # One-hot pieces are summed back: no "plan = basic" rows.
    assert not exp.importance["column"].str.contains(" = ").any()


def test_shares_sum_to_one(churn_explanation):
    _, exp = churn_explanation
    assert exp.shap_available
    assert exp.importance["shap_share"].sum() == pytest.approx(1.0)


def test_noise_ranks_last_in_both_methods(churn_explanation):
    _, exp = churn_explanation
    table = exp.importance
    assert table["column"].iloc[-1] == "noise"
    assert table.sort_values("perm_mean")["column"].iloc[0] == "noise"
    assert "noise" not in exp.top_drivers[:3]


def test_methods_agree_on_planted_signal(churn_explanation):
    _, exp = churn_explanation
    assert exp.top_overlap >= 3
    assert "fairly reliable" in exp.agreement_text


def test_agreement_never_compares_all_columns(churn_explanation):
    # 4 columns used -> compare the top 3, so disagreement is possible.
    _, exp = churn_explanation
    assert exp.top_k == 3


def test_small_influence_gets_no_direction(monkeypatch):
    # With a 50% threshold, only a column holding over half the influence keeps
    # its direction; every other column is reported as "Very little influence."
    monkeypatch.setattr(config, "DIRECTION_MIN_SHARE", 0.5)
    exp = explain_model(run_training(churn_df(), "churn", "binary", positive_class="yes"))
    small = exp.importance[exp.importance["shap_share"] < 0.5]
    assert len(small) >= 3
    assert (small["direction"] == "Very little influence.").all()


def test_permutation_uses_selection_metric(churn_explanation):
    result, exp = churn_explanation
    assert exp.perm_metric == "ROC-AUC"
    assert exp.perm_rows == result.n_test


# ---------------------------------------------------------------------------
# Direction hints
# ---------------------------------------------------------------------------
def directions(exp):
    return dict(zip(exp.importance["column"], exp.importance["direction"]))


def test_numeric_directions(churn_explanation):
    d = directions(churn_explanation[1])
    assert d["complaints"] == "Higher values go with more likely 'yes'."
    assert d["tenure"] == "Higher values go with less likely 'yes'."


def test_categorical_direction(churn_explanation):
    d = directions(churn_explanation[1])
    assert d["plan"].startswith("'basic' goes with more likely 'yes'")


def test_language_is_not_causal(churn_explanation):
    _, exp = churn_explanation
    text = " ".join(exp.importance["direction"]) + exp.agreement_text + " ".join(exp.notes)
    for word in ("cause", "causes", "because of", "leads to", "drives"):
        assert word not in text.lower()


def test_regression_directions():
    rng = np.random.default_rng(config.RANDOM_SEED)
    df = pd.DataFrame({"size": rng.normal(100, 20, size=N), "noise": rng.normal(size=N)})
    df["price"] = 3 * df["size"] + rng.normal(scale=10, size=N)
    exp = explain_model(run_training(df, "price", "regression"))
    assert exp.top_drivers[0] == "size"
    assert directions(exp)["size"] == "Higher values go with higher predicted price."


def test_multiclass_has_importance_but_no_directions():
    rng = np.random.default_rng(config.RANDOM_SEED)
    df = pd.DataFrame({"x1": rng.normal(size=N), "x2": rng.normal(size=N)})
    # Noise added so x1 is a strong but not perfect (leaky-looking) predictor.
    df["tier"] = pd.cut(df["x1"] + rng.normal(scale=0.3, size=N), [-np.inf, -0.5, 0.5, np.inf],
                        labels=["bronze", "silver", "gold"]).astype(str)
    exp = explain_model(run_training(df, "tier", "multiclass"))
    assert exp.shap_available
    assert exp.top_drivers[0] == "x1"
    assert (exp.importance["direction"] == "").all()


# ---------------------------------------------------------------------------
# Every model type can be explained
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key", ["linear", "random_forest", "lightgbm"])
def test_each_model_type(key, monkeypatch):
    only_model(monkeypatch, key)
    result = run_training(churn_df(), "churn", "binary")
    assert result.best_key == key
    exp = explain_model(result)
    assert exp.shap_available
    assert exp.top_drivers[-1] != "complaints"   # a real driver is not last


def test_forest_uses_smaller_shap_sample(monkeypatch):
    only_model(monkeypatch, "random_forest")
    monkeypatch.setattr(config, "SHAP_SAMPLE_ROWS_FOREST", 30)
    exp = explain_model(run_training(churn_df(), "churn", "binary"))
    assert exp.shap_rows == 30


def test_dummy_best_model(monkeypatch):
    only_model(monkeypatch, "dummy")
    exp = explain_model(run_training(churn_df(), "churn", "binary"))
    assert not exp.shap_available
    assert any("baseline" in n for n in exp.notes)
    assert (exp.importance["perm_mean"] == 0).all()


def test_shap_failure_falls_back(monkeypatch):
    from foresight import explain

    def broken(*args, **kwargs):
        raise RuntimeError("shap failed")

    monkeypatch.setattr(explain, "_raw_shap_values", broken)
    exp = explain_model(run_training(churn_df(), "churn", "binary"))
    assert not exp.shap_available
    assert any("could not be computed" in n for n in exp.notes)
    assert "no cross-check" in exp.agreement_text
    assert len(exp.importance) == 4
