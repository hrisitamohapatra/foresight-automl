"""Tests for foresight/preprocess.py."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from foresight import config
from foresight.preprocess import (
    build_pipeline,
    build_preprocessor,
    feature_names_and_sources,
)
from foresight.profile import profile_data

N = 200


@pytest.fixture
def data():
    rng = np.random.default_rng(config.RANDOM_SEED)
    X = pd.DataFrame({
        "age": rng.normal(40, 10, size=N),
        "income": rng.normal(50_000, 8_000, size=N),
        "charges": rng.normal(70, 5, size=N).round(2).astype(str).astype(object),
        "region": rng.choice(["north", "south", "east"], size=N).astype(object),
        "plan": rng.choice(["basic", "pro"], size=N).astype(object),
    })
    X.loc[:9, "age"] = np.nan          # some missing numbers
    X.loc[10:14, "region"] = np.nan    # some missing categories
    X.loc[15:17, "charges"] = " "      # blank numbers stored as text
    y = pd.Series(rng.integers(0, 2, size=N))
    return X, y


def preprocessor_for(scale=False):
    return build_preprocessor(
        numeric=["age", "income", "charges"],
        categorical=["region", "plan"],
        numeric_text=["charges"],
        scale=scale,
    )


# ---------------------------------------------------------------------------
# Basic behaviour
# ---------------------------------------------------------------------------
def test_output_is_complete_numbers(data):
    X, _ = data
    out = preprocessor_for().fit_transform(X)
    assert out.dtype == float
    assert not np.isnan(out).any()


def test_feature_names_match_output(data):
    X, _ = data
    prep = preprocessor_for().fit(X)
    names, sources = feature_names_and_sources(prep)
    assert len(names) == len(sources) == prep.transform(X).shape[1]
    assert names == [
        "age", "income", "charges",
        "age (missing)", "charges (missing)",
        "region = (missing)", "region = east", "region = north", "region = south",
        "plan = basic", "plan = pro",
    ]
    assert set(sources) == {"age", "income", "charges", "region", "plan"}


def test_scaling_only_when_requested(data):
    X, _ = data
    unscaled = preprocessor_for(scale=False).fit_transform(X)
    scaled = preprocessor_for(scale=True).fit_transform(X)
    assert unscaled[:, 1].mean() == pytest.approx(X["income"].mean(), rel=1e-6)
    assert scaled[:, 1].mean() == pytest.approx(0, abs=1e-9)
    assert scaled[:, 1].std() == pytest.approx(1, rel=1e-6)


def test_unlisted_columns_dropped(data):
    X, _ = data
    X = X.assign(customer_id=range(N))
    prep = build_preprocessor(numeric=["age"], categorical=["plan"]).fit(X)
    _, sources = feature_names_and_sources(prep)
    assert "customer_id" not in sources


def test_no_features_rejected():
    with pytest.raises(ValueError):
        build_preprocessor(numeric=[], categorical=[])


def test_input_not_modified(data):
    X, _ = data
    before = X.copy()
    preprocessor_for(scale=True).fit_transform(X)
    pd.testing.assert_frame_equal(X, before)


# ---------------------------------------------------------------------------
# Messy and unseen data
# ---------------------------------------------------------------------------
def test_unseen_category_and_new_missing_values(data):
    X, _ = data
    prep = preprocessor_for().fit(X)
    new = X.head(3).copy()
    new["region"] = ["mars", np.nan, "north"]   # unseen + missing
    new["income"] = np.nan                       # never missing in training
    out = prep.transform(new)
    assert out.shape[1] == prep.transform(X).shape[1]
    assert not np.isnan(out).any()


def test_mixed_type_categories(data):
    X, _ = data
    X["flag"] = pd.Series([True, "x", 1.5, np.nan] * (N // 4), dtype=object)
    prep = build_preprocessor(numeric=["age"], categorical=["flag"]).fit(X)
    assert not np.isnan(prep.transform(X)).any()


def test_rare_categories_grouped():
    rng = np.random.default_rng(config.RANDOM_SEED)
    X = pd.DataFrame({"city": rng.choice([f"c{i}" for i in range(300)], size=2000)})
    prep = build_preprocessor(numeric=[], categorical=["city"]).fit(X)
    names, _ = feature_names_and_sources(prep)
    assert len(names) <= config.ONEHOT_MAX_CATEGORIES
    assert "city = (other)" in names


def test_all_missing_column_in_training_keeps_shape(data):
    X, _ = data
    train = X.copy()
    train["age"] = np.nan
    prep = preprocessor_for().fit(train)
    assert prep.transform(X).shape[1] == prep.transform(train).shape[1]


def test_build_pipeline_from_profile(data):
    X, y = data
    profile = profile_data(X, y, "binary")
    pipe = build_pipeline(profile, DummyClassifier(), scale=True).fit(X, y)
    assert len(pipe.predict(X)) == N


# ---------------------------------------------------------------------------
# Preprocessing is fit only on training folds
# ---------------------------------------------------------------------------
class RecordFitRows(BaseEstimator, TransformerMixin):
    """Spy step: remembers which rows (by index) it was fitted on."""

    def fit(self, X, y=None):
        self.fit_index_ = list(X.index)
        return self

    def transform(self, X):
        return X


class CountingModel(BaseEstimator, ClassifierMixin):
    """Spy model: remembers how many rows reached it during fit."""

    def fit(self, X, y):
        self.n_fit_rows_ = X.shape[0]
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.full(X.shape[0], self.classes_[0])


def test_preprocessing_fit_only_on_training_folds(data):
    X, y = data
    # Make age very different in some rows, so a median learned from all
    # rows would differ from one learned from the training folds only.
    X = X.copy()
    X.loc[150:, "age"] = 1_000.0

    pipe = Pipeline([
        ("spy", RecordFitRows()),
        ("pipeline", Pipeline([("preprocess", preprocessor_for(scale=True)),
                               ("model", CountingModel())])),
    ])
    cv = StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)
    result = cross_validate(pipe, X, y, cv=cv, return_estimator=True,
                            return_indices=True)

    full_median = X["age"].median()
    for fold, fitted in enumerate(result["estimator"]):
        train_idx = result["indices"]["train"][fold]
        test_idx = result["indices"]["test"][fold]
        train_rows = X.iloc[train_idx]

        # 1. The pipeline saw exactly the training rows, none of the held-out ones.
        seen = fitted.named_steps["spy"].fit_index_
        assert sorted(seen) == sorted(train_rows.index)
        assert set(seen).isdisjoint(X.index[test_idx])

        # 2. The learned median and scaling come from the training fold only.
        prep = fitted.named_steps["pipeline"].named_steps["preprocess"]
        num = prep.named_steps["columns"].named_transformers_["num"]
        learned_median = num.named_steps["impute"].statistics_[0]
        assert learned_median == pytest.approx(train_rows["age"].median())
        filled_age = train_rows["age"].fillna(train_rows["age"].median())
        assert num.named_steps["scale"].mean_[0] == pytest.approx(filled_age.mean())

        # 3. The model was trained on training-fold rows only.
        assert fitted.named_steps["pipeline"].named_steps["model"].n_fit_rows_ == len(train_idx)

    # Sanity check: at least one fold's median differs from the all-rows median,
    # so this test would catch preprocessing fit on all data.
    medians = [
        e.named_steps["pipeline"].named_steps["preprocess"].named_steps["columns"]
        .named_transformers_["num"].named_steps["impute"].statistics_[0]
        for e in result["estimator"]
    ]
    assert any(m != pytest.approx(full_median) for m in medians)
