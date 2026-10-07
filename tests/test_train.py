"""Tests for foresight/train.py."""

import numpy as np
import pandas as pd
import pytest

from foresight import config, train
from foresight.models import ModelSpec, get_models
from foresight.train import (
    TrainError,
    default_positive_class,
    encode_target,
    load_model,
    run_training,
    save_model,
)

N = 300


def binary_df(seed=config.RANDOM_SEED, n=N, minority_share=0.3):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "tenure": rng.normal(24, 10, size=n),
        "monthly_fee": rng.normal(60, 15, size=n),
        "plan": rng.choice(["basic", "pro", "family"], size=n),
    })
    score = -0.1 * X["tenure"] + 0.05 * X["monthly_fee"] + rng.normal(size=n)
    churned = score > np.quantile(score, 1 - minority_share)
    X["churn"] = np.where(churned, "yes", "no")
    return X


def regression_df(n=N):
    rng = np.random.default_rng(config.RANDOM_SEED)
    df = pd.DataFrame({
        "size": rng.normal(100, 20, size=n),
        "rooms": rng.integers(1, 6, size=n),
        "area": rng.choice(["a", "b", "c"], size=n),
    })
    df["price"] = 3 * df["size"] + 10 * df["rooms"] + rng.normal(scale=20, size=n)
    return df


def multiclass_df(n=N):
    rng = np.random.default_rng(config.RANDOM_SEED)
    df = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    df["tier"] = pd.cut(df["x1"] + rng.normal(scale=0.3, size=n), [-np.inf, -0.5, 0.5, np.inf],
                        labels=["bronze", "silver", "gold"]).astype(str)
    return df


# ---------------------------------------------------------------------------
# Target encoding
# ---------------------------------------------------------------------------
def test_default_positive_is_rarer_class():
    assert default_positive_class(pd.Series(["no"] * 70 + ["yes"] * 30)) == "yes"


def test_binary_encoding_positive_is_one():
    y = pd.Series(["no", "yes", "no"])
    codes, labels, positive = encode_target(y, "binary")
    assert positive == "yes" and labels == ["no", "yes"]
    assert codes.tolist() == [0, 1, 0]


def test_positive_class_override():
    y = pd.Series(["no"] * 7 + ["yes"] * 3)
    codes, labels, positive = encode_target(y, "binary", positive_class="no")
    assert labels == ["yes", "no"] and codes[0] == 1


def test_invalid_positive_class_rejected():
    with pytest.raises(TrainError):
        encode_target(pd.Series(["a", "b"]), "binary", positive_class="c")


# ---------------------------------------------------------------------------
# End-to-end runs
# ---------------------------------------------------------------------------
def test_binary_run():
    result = run_training(binary_df(), "churn", "binary")
    assert [r.key for r in result.model_results] == ["dummy", "linear", "random_forest", "lightgbm"]
    for r in result.model_results:
        assert not r.error
        assert set(r.cv_mean) == {"roc_auc", "pr_auc", "f1", "accuracy"}
        assert set(r.cv_std) == set(r.cv_mean)
    assert result.selection_metric == "roc_auc"
    assert result.result_for("dummy").cv_mean["roc_auc"] == pytest.approx(0.5)
    assert result.best_key != "dummy"
    assert result.best.cv_mean["roc_auc"] > 0.7
    assert set(result.test_scores) == {"roc_auc", "pr_auc", "f1", "accuracy"}
    assert result.positive_class == "yes"


def test_split_sizes_and_stratification():
    result = run_training(binary_df(), "churn", "binary")
    assert result.n_train == 240 and result.n_test == 60
    assert result.y_train.mean() == pytest.approx(result.y_test.mean(), abs=0.02)


def test_imbalanced_binary_uses_pr_auc():
    result = run_training(binary_df(minority_share=0.1), "churn", "binary")
    assert result.profile.is_imbalanced
    assert result.selection_metric == "pr_auc"


def test_regression_run():
    result = run_training(regression_df(), "price", "regression")
    assert result.selection_metric == "rmse"
    assert result.best_key != "dummy"
    assert result.best.cv_mean["rmse"] < result.result_for("dummy").cv_mean["rmse"]
    assert result.test_scores["r2"] > 0.5


def test_multiclass_run_with_text_labels():
    result = run_training(multiclass_df(), "tier", "multiclass")
    assert sorted(result.class_labels) == ["bronze", "gold", "silver"]
    assert result.selection_metric == "f1_macro"
    assert result.best.cv_mean["f1_macro"] > result.result_for("dummy").cv_mean["f1_macro"]


def test_runs_are_reproducible():
    a = run_training(binary_df(), "churn", "binary")
    b = run_training(binary_df(), "churn", "binary")
    for ra, rb in zip(a.model_results, b.model_results):
        for key in ra.cv_mean:
            assert ra.cv_mean[key] == pytest.approx(rb.cv_mean[key], rel=1e-9)
    assert a.best_key == b.best_key
    assert a.test_scores == pytest.approx(b.test_scores, rel=1e-9)


# ---------------------------------------------------------------------------
# The held-out test set is never used for model selection
# ---------------------------------------------------------------------------
def test_test_set_used_exactly_once(monkeypatch):
    calls = []
    real_score_model = train.score_model

    def spy(model, X, y, problem_type):
        calls.append(X)
        return real_score_model(model, X, y, problem_type)

    monkeypatch.setattr(train, "score_model", spy)
    result = run_training(binary_df(), "churn", "binary")

    assert len(calls) == 1                            # one test evaluation
    assert calls[0].index.equals(result.X_test.index)  # ... on the test rows
    assert set(result.X_train.index).isdisjoint(result.X_test.index)


def test_selection_ignores_test_set(monkeypatch):
    # Corrupt the test labels: if they influenced selection, the chosen model
    # would change. It must not.
    clean = run_training(binary_df(), "churn", "binary")
    real_split = train.train_test_split

    def corrupted_split(*args, **kwargs):
        X_tr, X_te, y_tr, y_te = real_split(*args, **kwargs)
        return X_tr, X_te, y_tr, 1 - y_te

    monkeypatch.setattr(train, "train_test_split", corrupted_split)
    corrupted = run_training(binary_df(), "churn", "binary")
    assert corrupted.best_key == clean.best_key
    for a, b in zip(clean.model_results, corrupted.model_results):
        assert a.cv_mean == pytest.approx(b.cv_mean)


# ---------------------------------------------------------------------------
# Columns, leakage overrides, failures
# ---------------------------------------------------------------------------
def test_leaky_column_excluded_then_overridden():
    df = binary_df()
    df["refund_issued"] = (df["churn"] == "yes").astype(int)
    default = run_training(df, "churn", "binary")
    assert "refund_issued" in default.profile.excluded
    assert default.included_overrides == []

    forced = run_training(df, "churn", "binary", include_columns=["refund_issued"])
    assert forced.included_overrides == ["refund_issued"]
    assert "refund_issued" in forced.profile.numeric_features


def test_override_cannot_include_id_columns():
    df = binary_df()
    df["customer_id"] = [f"C{i}" for i in range(N)]
    result = run_training(df, "churn", "binary", include_columns=["customer_id"])
    assert "customer_id" in result.profile.excluded
    assert result.included_overrides == []


def test_no_usable_columns():
    df = pd.DataFrame({"id": [f"r{i}" for i in range(N)], "y": [0, 1] * (N // 2)})
    with pytest.raises(TrainError, match="No usable input columns"):
        run_training(df, "y", "binary")


def test_target_checked_before_training():
    df = binary_df()
    with pytest.raises(TrainError, match="numeric"):
        run_training(df, "churn", "regression")


def test_failing_model_does_not_stop_others(monkeypatch):
    class Broken:
        def get_params(self, deep=True):
            return {}

        def fit(self, X, y):
            raise RuntimeError("boom")

    def models_with_broken(problem_type, balanced=False):
        return get_models(problem_type, balanced)[:2] + [
            ModelSpec("broken", "Broken", "candidate", False, Broken)]

    monkeypatch.setattr(train, "get_models", models_with_broken)
    result = run_training(binary_df(), "churn", "binary")
    assert result.result_for("broken").error
    assert "boom" not in result.result_for("broken").error   # no internals shown
    assert result.best_key == "linear"


def test_preview_profile_matches_training_profile():
    df = binary_df()
    df["refund_issued"] = (df["churn"] == "yes").astype(int)
    preview = train.preview_profile(df, "churn", "binary")
    trained = run_training(df, "churn", "binary").profile
    assert preview.excluded == trained.excluded
    assert preview.leaky_columns == trained.leaky_columns == ["refund_issued"]
    assert [c.leakage_score for c in preview.columns] == \
        [c.leakage_score for c in trained.columns]


def test_progress_reported():
    updates = []
    run_training(binary_df(), "churn", "binary",
                 on_progress=lambda msg, frac: updates.append(frac))
    assert updates[-1] == 1.0
    assert updates == sorted(updates)


# ---------------------------------------------------------------------------
# Saving and loading
# ---------------------------------------------------------------------------
def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    result = run_training(binary_df(), "churn", "binary", save=True)
    files = list(tmp_path.iterdir())
    assert [f.name for f in files] == [f"model_{result.run_id}.joblib"]

    bundle = load_model(result.run_id)
    assert bundle["target"] == "churn"
    preds = bundle["pipeline"].predict(result.X_test)
    np.testing.assert_array_equal(preds, result.pipeline.predict(result.X_test))


@pytest.mark.parametrize("bad_id", ["../../secret", "..\\model", "abc", "",
                                    "0" * 31 + "g", "/etc/passwd"])
def test_load_rejects_bad_ids(bad_id, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    with pytest.raises(TrainError, match="Invalid model id"):
        load_model(bad_id)


def test_load_missing_model(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    with pytest.raises(TrainError, match="No saved model"):
        load_model("0" * 32)
