"""Tests for foresight/profile.py."""

import numpy as np
import pandas as pd
import pytest

from foresight import config
from foresight.profile import profile_data, to_numeric_safe

N = 400


@pytest.fixture
def rng():
    return np.random.default_rng(config.RANDOM_SEED)


def binary_data(rng, n=N):
    """Honest features that relate to the target only loosely."""
    signal = rng.normal(size=n)
    y = pd.Series((signal + rng.normal(scale=1.5, size=n) > 0).astype(int), name="target")
    X = pd.DataFrame({
        "signal": signal,
        "noise": rng.normal(size=n),
        "region": rng.choice(["north", "south", "east", "west"], size=n),
    })
    return X, y


def column(profile, name):
    return next(c for c in profile.columns if c.name == name)


# ---------------------------------------------------------------------------
# Column types
# ---------------------------------------------------------------------------
def test_basic_types(rng):
    X, y = binary_data(rng)
    p = profile_data(X, y, "binary")
    assert column(p, "signal").kind == "numeric"
    assert column(p, "region").kind == "categorical"
    assert p.numeric_features == ["signal", "noise"]
    assert p.categorical_features == ["region"]
    assert p.excluded == {}


def test_numbers_stored_as_text(rng):
    X, y = binary_data(rng)
    values = rng.normal(100, 20, size=N).round(2).astype(str).astype(object)
    values[:5] = " "   # blanks, like Telco's TotalCharges
    X["charges"] = values
    p = profile_data(X, y, "binary")
    assert column(p, "charges").kind == "numeric_text"
    assert "charges" in p.numeric_features
    assert p.numeric_text_features == ["charges"]
    assert any("stores numbers as text" in w for w in p.warnings)


def test_to_numeric_safe():
    result = to_numeric_safe(pd.Series([" 1.5", "", "abc", None, "2"]))
    assert result.iloc[0] == 1.5 and result.iloc[4] == 2.0
    assert result.iloc[1:4].isna().all()


def test_date_column_excluded(rng):
    X, y = binary_data(rng)
    X["signup"] = pd.date_range("2020-01-01", periods=N).strftime("%Y-%m-%d")
    p = profile_data(X, y, "binary")
    assert column(p, "signup").kind == "datetime"
    assert "signup" in p.excluded


def test_month_names_are_categories_not_dates(rng):
    X, y = binary_data(rng)
    X["month"] = rng.choice(["January", "March", "June"], size=N)
    p = profile_data(X, y, "binary")
    assert column(p, "month").kind == "categorical"


# ---------------------------------------------------------------------------
# Quality flags
# ---------------------------------------------------------------------------
def test_constant_and_empty_columns_excluded(rng):
    X, y = binary_data(rng)
    X["always_one"] = 1
    X["empty"] = np.nan
    p = profile_data(X, y, "binary")
    assert "constant" in column(p, "always_one").flags
    assert "all_missing" in column(p, "empty").flags
    assert {"always_one", "empty"} <= set(p.excluded)


def test_text_id_excluded(rng):
    X, y = binary_data(rng)
    X["customer_code"] = [f"C{i:05d}" for i in rng.permutation(N)]
    p = profile_data(X, y, "binary")
    assert "id_like" in column(p, "customer_code").flags


def test_integer_id_by_name_excluded(rng):
    X, y = binary_data(rng)
    X["customerID"] = rng.choice(10**6, size=N, replace=False)
    p = profile_data(X, y, "binary")
    assert "id_like" in column(p, "customerID").flags


def test_row_counter_excluded(rng):
    X, y = binary_data(rng)
    # A shuffled 1..N counter with some rows missing, like a training split.
    X["seq"] = rng.permutation(np.arange(1, int(N * 1.25) + 1))[:N]
    p = profile_data(X, y, "binary")
    assert "id_like" in column(p, "seq").flags


def test_continuous_and_large_integers_not_ids(rng):
    X, y = binary_data(rng)
    X["price"] = rng.normal(50, 10, size=N)                         # unique decimals
    X["income"] = rng.choice(np.arange(10_000, 500_000), size=N, replace=False)
    p = profile_data(X, y, "binary")
    assert "id_like" not in column(p, "price").flags
    assert "id_like" not in column(p, "income").flags
    assert "price" in p.numeric_features and "income" in p.numeric_features


def test_high_cardinality_flagged_but_kept(rng):
    X, y = binary_data(rng)
    X["city"] = rng.choice([f"city_{i}" for i in range(80)], size=N)
    p = profile_data(X, y, "binary")
    assert "high_cardinality" in column(p, "city").flags
    assert "city" in p.categorical_features


def test_high_missing_flagged(rng):
    X, y = binary_data(rng)
    X.loc[X.index[: int(N * 0.6)], "noise"] = np.nan
    p = profile_data(X, y, "binary")
    assert "high_missing" in column(p, "noise").flags
    assert "noise" in p.numeric_features


def test_duplicate_rows_counted(rng):
    X, y = binary_data(rng)
    X2 = pd.concat([X, X.iloc[:10]], ignore_index=True)
    y2 = pd.concat([y, y.iloc[:10]], ignore_index=True)
    assert profile_data(X2, y2, "binary").n_duplicate_rows == 10


# ---------------------------------------------------------------------------
# Class imbalance
# ---------------------------------------------------------------------------
def test_imbalance_detected(rng):
    X, _ = binary_data(rng)
    y = pd.Series([1] * 40 + [0] * (N - 40))
    p = profile_data(X, y, "binary", check_leakage=False)
    assert p.is_imbalanced
    assert p.minority_share == pytest.approx(0.10)
    assert any("imbalanced" in w for w in p.warnings)


def test_balanced_not_flagged(rng):
    X, y = binary_data(rng)
    p = profile_data(X, y, "binary")
    assert not p.is_imbalanced
    assert set(p.class_balance) == {"0", "1"}


def test_regression_has_no_class_balance(rng):
    X, _ = binary_data(rng)
    y = pd.Series(rng.normal(size=N))
    p = profile_data(X, y, "regression")
    assert p.class_balance is None and not p.is_imbalanced


# ---------------------------------------------------------------------------
# A planted leaky column is flagged
# ---------------------------------------------------------------------------
def test_honest_features_not_flagged(rng):
    X, y = binary_data(rng)
    p = profile_data(X, y, "binary")
    assert p.leaky_columns == []
    assert column(p, "signal").leakage_score < config.LEAKAGE_SCORE_THRESHOLD


def test_leaky_numeric_column_flagged_binary(rng):
    X, y = binary_data(rng)
    X["refund_amount"] = y * 50 + rng.normal(scale=1, size=N)   # known only after churn
    p = profile_data(X, y, "binary")
    assert p.leaky_columns == ["refund_amount"]
    assert "refund_amount" in p.excluded
    assert "refund_amount" not in p.numeric_features
    assert any("leakage" in w for w in p.warnings)


def test_strong_column_warned_but_kept(rng):
    # Separated by ~2.2 standard deviations: single-column ROC-AUC ~0.94,
    # between the warning (0.90) and exclusion (0.97) thresholds.
    X, y = binary_data(rng)
    X["vendor_score"] = y * 2 + rng.normal(scale=0.9, size=N)
    p = profile_data(X, y, "binary")
    col = column(p, "vendor_score")
    assert config.LEAKAGE_WARNING_THRESHOLD <= col.leakage_score < config.LEAKAGE_SCORE_THRESHOLD
    assert "strong_single_predictor" in col.flags
    assert col.use and "vendor_score" in p.numeric_features
    assert p.leaky_columns == []
    assert any("Check column 'vendor_score'" in w for w in p.warnings)


def test_leaky_text_column_flagged_binary(rng):
    X, y = binary_data(rng)
    X["account_status"] = np.where(y == 1, "closed", "active")
    p = profile_data(X, y, "binary")
    assert "account_status" in p.leaky_columns


def test_leaky_missingness_flagged(rng):
    # The value is only filled in after the outcome happens.
    X, y = binary_data(rng)
    X["cancel_reason_code"] = np.where(y == 1, rng.integers(1, 5, size=N), np.nan)
    p = profile_data(X, y, "binary")
    assert "cancel_reason_code" in p.leaky_columns


def test_leakage_check_with_text_labels(rng):
    X, y = binary_data(rng)
    labels = y.map({0: "stayed", 1: "churned"})
    X["refund_amount"] = y * 50 + rng.normal(scale=1, size=N)
    p = profile_data(X, labels, "binary")
    assert p.leaky_columns == ["refund_amount"]
    assert set(p.class_balance) == {"stayed", "churned"}


def test_leaky_column_flagged_regression(rng):
    X, _ = binary_data(rng)
    y = pd.Series(rng.normal(100, 15, size=N))
    X["final_invoice"] = y * 1.1 + rng.normal(scale=0.5, size=N)
    p = profile_data(X, y, "regression")
    assert p.leaky_columns == ["final_invoice"]


def test_leaky_column_flagged_multiclass(rng):
    X, _ = binary_data(rng)
    y = pd.Series(rng.choice(["bronze", "silver", "gold"], size=N))
    X["tier_code"] = y.map({"bronze": 1, "silver": 2, "gold": 3})
    p = profile_data(X, y, "multiclass")
    assert p.leaky_columns == ["tier_code"]


def test_leakage_check_samples_large_data(rng, monkeypatch):
    monkeypatch.setattr(config, "LEAKAGE_SAMPLE_ROWS", 100)
    X, y = binary_data(rng)
    X["refund_amount"] = y * 50 + rng.normal(scale=1, size=N)
    p = profile_data(X, y, "binary")
    assert p.leaky_columns == ["refund_amount"]
    assert p.n_rows == N   # the profile itself still covers all rows


def test_input_not_modified(rng):
    X, y = binary_data(rng)
    X["charges"] = rng.normal(size=N).astype(str).astype(object)
    X_before, y_before = X.copy(), y.copy()
    profile_data(X, y, "binary")
    pd.testing.assert_frame_equal(X, X_before)
    pd.testing.assert_series_equal(y, y_before)
