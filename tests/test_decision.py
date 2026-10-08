"""Tests for foresight/decision.py."""

import numpy as np
import pandas as pd
import pytest

from foresight import config, decision, train
from foresight.decision import (
    DecisionError,
    analyze_decision,
    choose_threshold,
    gains_table,
    outcome_at,
)
from foresight.models import get_models
from foresight.train import run_training

N = 800


def churn_df(n=N, positive_share=0.15, seed=config.RANDOM_SEED):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "tenure": rng.normal(24, 10, n),
        "calls": rng.poisson(1.5, n).astype(float),
        "plan": rng.choice(["basic", "pro"], n),
    })
    logit = -0.12 * (df["tenure"] - 24) + 0.8 * (df["calls"] - 1.5) + 1.0 * (df["plan"] == "basic")
    score = logit + rng.logistic(size=n)
    df["churn"] = np.where(score > np.quantile(score, 1 - positive_share), "yes", "no")
    return df


@pytest.fixture(scope="module")
def imbalanced_run():
    # 15% positives -> "imbalanced" -> class_weight="balanced", which pushes
    # probabilities upward: a case where calibration should help.
    return run_training(churn_df(), "churn", "binary", positive_class="yes")


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
def test_outcome_counts():
    y = np.array([1, 1, 0, 0, 1])
    p = np.array([0.9, 0.4, 0.6, 0.1, 0.7])
    o = outcome_at(y, p, 0.5)
    assert (o.tp, o.fp, o.tn, o.fn) == (2, 1, 1, 1)
    assert o.recall == pytest.approx(2 / 3)
    assert o.precision == pytest.approx(2 / 3)
    assert o.flagged_share == pytest.approx(3 / 5)
    assert o.cost(cost_fp=1, cost_fn=5) == 1 * 1 + 1 * 5
    assert o.cost_per_1000(1, 5) == pytest.approx(1200)


def test_threshold_follows_costs():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=5000)
    y = (rng.uniform(size=5000) < p).astype(int)   # perfectly calibrated scores
    # Theory for calibrated scores: best cut-off = cost_fp / (cost_fp + cost_fn).
    assert choose_threshold(y, p, 1, 1) == pytest.approx(0.5, abs=0.05)
    assert choose_threshold(y, p, 1, 5) == pytest.approx(1 / 6, abs=0.05)
    assert choose_threshold(y, p, 5, 1) == pytest.approx(5 / 6, abs=0.05)


def test_threshold_ties_prefer_default():
    y = np.array([0, 1])
    p = np.array([0.0, 1.0])   # every cut-off is perfect
    assert choose_threshold(y, p, 1, 1) == 0.5


def test_gains_perfect_and_random():
    y = np.array([1] * 20 + [0] * 80)
    perfect = gains_table(y, y.astype(float))
    assert perfect.loc[perfect.pct_acted_on == 20, "pct_positives_reached"].item() == 100
    assert perfect.loc[perfect.pct_acted_on == 20, "lift"].item() == pytest.approx(5)

    rng = np.random.default_rng(0)
    y_big = rng.integers(0, 2, 20_000)
    random_scores = gains_table(y_big, rng.uniform(size=20_000))
    at_50 = random_scores.loc[random_scores.pct_acted_on == 50, "pct_positives_reached"].item()
    assert at_50 == pytest.approx(50, abs=2)
    assert random_scores["pct_positives_reached"].is_monotonic_increasing
    assert random_scores["lift"].iloc[-1] == pytest.approx(1)


@pytest.mark.parametrize("costs", [(0, 1), (1, -2), (float("nan"), 1),
                                   (float("inf"), 1), (1, 10**12), ("5", 1)])
def test_invalid_costs_rejected(costs, imbalanced_run):
    with pytest.raises(DecisionError):
        analyze_decision(imbalanced_run, *costs)


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------
def test_analysis_basics(imbalanced_run):
    a = analyze_decision(imbalanced_run, cost_fp=1, cost_fn=5)
    assert a.applicable
    assert a.calibration_method == "sigmoid"   # fewer than 5,000 training rows
    assert a.test_chosen.n == imbalanced_run.n_test
    assert a.test_flag_none.flagged_share == 0
    assert a.test_flag_all.flagged_share == 1
    # Chosen on training data, so it can only be <= the default there.
    assert a.oof_cost_chosen <= a.oof_cost_default
    assert len(a.gains) == 100


def test_calibration_helps_balanced_weights(imbalanced_run):
    a = analyze_decision(imbalanced_run)
    assert a.calibrated
    assert a.oof_brier_calibrated < a.oof_brier_raw
    assert a.test_brier_calibrated < a.test_brier_raw   # confirmed on unseen data


def test_expensive_misses_lower_the_threshold(imbalanced_run):
    cautious = analyze_decision(imbalanced_run, cost_fp=1, cost_fn=10)
    strict = analyze_decision(imbalanced_run, cost_fp=10, cost_fn=1)
    assert cautious.threshold < strict.threshold
    assert cautious.test_chosen.recall > strict.test_chosen.recall


def test_choices_ignore_the_test_set(monkeypatch):
    clean = analyze_decision(run_training(churn_df(), "churn", "binary",
                                          positive_class="yes"), 1, 5)
    real_split = train.train_test_split

    def corrupted_split(*args, **kwargs):
        X_tr, X_te, y_tr, y_te = real_split(*args, **kwargs)
        return X_tr, X_te, y_tr, 1 - y_te   # flip every test label

    monkeypatch.setattr(train, "train_test_split", corrupted_split)
    corrupted = analyze_decision(run_training(churn_df(), "churn", "binary",
                                              positive_class="yes"), 1, 5)
    assert corrupted.threshold == clean.threshold
    assert corrupted.calibrated == clean.calibrated
    assert corrupted.oof_brier_raw == pytest.approx(clean.oof_brier_raw)


def test_isotonic_for_large_training_sets(monkeypatch, imbalanced_run):
    monkeypatch.setattr(config, "CALIBRATION_ISOTONIC_MIN_ROWS", 100)
    assert analyze_decision(imbalanced_run).calibration_method == "isotonic"


def test_reproducible(imbalanced_run):
    a = analyze_decision(imbalanced_run, 1, 5)
    b = analyze_decision(imbalanced_run, 1, 5)
    assert a.threshold == b.threshold
    assert a.test_brier_calibrated == pytest.approx(b.test_brier_calibrated, rel=1e-9)


# ---------------------------------------------------------------------------
# When it does not apply
# ---------------------------------------------------------------------------
def test_regression_not_applicable():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=300)})
    df["y"] = df["x"] * 3 + rng.normal(size=300)
    a = analyze_decision(run_training(df, "y", "regression"))
    assert not a.applicable and "yes/no" in a.reason


def test_dummy_best_not_applicable(monkeypatch):
    monkeypatch.setattr(train, "get_models",
                        lambda pt, balanced=False: get_models(pt, balanced)[:1])
    a = analyze_decision(run_training(churn_df(), "churn", "binary", positive_class="yes"))
    assert not a.applicable and "baseline" in a.reason
