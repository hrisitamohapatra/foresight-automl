"""Leakage-safe preprocessing built from scikit-learn Pipelines.

Everything that learns from data (medians, scaling, category lists) lives
inside a Pipeline. When cross-validation fits the Pipeline on training folds,
those values are learned from the training folds only; the validation fold
and the held-out test set are only ever *transformed*, never fitted on.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from foresight import config
from foresight.profile import DataProfile, to_numeric_safe

MISSING_LABEL = "(missing)"
OTHER_LABEL = "(other)"


# ---------------------------------------------------------------------------
# Stateless helpers. They learn nothing from the data, so they cannot leak.
# (Defined at module level so fitted pipelines can be saved with joblib.)
# ---------------------------------------------------------------------------
def _numeric_text_to_numbers(X: pd.DataFrame, columns=()) -> pd.DataFrame:
    """Convert numbers stored as text (e.g. "12.5") into real numbers."""
    X = X.copy()
    for col in columns:
        X[col] = to_numeric_safe(X[col])
    return X


def _as_category_text(X: pd.DataFrame) -> pd.DataFrame:
    """Turn every categorical value into text and mark missing values.

    Mixed values like True / "x" / 1.5 would otherwise confuse the encoder.
    """
    return X.astype("string").fillna(MISSING_LABEL).astype(object)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def build_preprocessor(numeric: list[str], categorical: list[str],
                       numeric_text: list[str] = (), scale: bool = False) -> Pipeline:
    """Build an (unfitted) preprocessing pipeline.

    numeric       columns used as numbers (may include numeric_text columns)
    categorical   columns used as categories
    numeric_text  subset of `numeric` stored as text in the raw data
    scale         standardize numbers (needed for linear models, not trees)

    Any column not listed is dropped.
    """
    if not numeric and not categorical:
        raise ValueError("No input columns to build a model from.")

    numeric_steps = [
        # Median is robust to outliers. add_indicator adds a 0/1 column per
        # feature that had missing values, because "missing" can be
        # informative. keep_empty_features keeps the output shape stable
        # even if a training fold happens to be all-missing for a column.
        ("impute", SimpleImputer(strategy="median", add_indicator=True,
                                 keep_empty_features=True)),
    ]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))

    categorical_steps = [
        ("as_text", FunctionTransformer(_as_category_text)),
        ("encode", OneHotEncoder(
            handle_unknown="infrequent_if_exist",   # unseen categories -> (other)
            min_frequency=config.ONEHOT_MIN_FREQUENCY,
            max_categories=config.ONEHOT_MAX_CATEGORIES,
            sparse_output=False,
            dtype=float,
        )),
    ]

    transformers = []
    if numeric:
        transformers.append(("num", Pipeline(numeric_steps), list(numeric)))
    if categorical:
        transformers.append(("cat", Pipeline(categorical_steps), list(categorical)))

    return Pipeline([
        ("to_numbers", FunctionTransformer(_numeric_text_to_numbers,
                                           kw_args={"columns": list(numeric_text)})),
        ("columns", ColumnTransformer(transformers, remainder="drop",
                                      sparse_threshold=0)),
    ])


def build_pipeline(profile: DataProfile, model, scale: bool = False) -> Pipeline:
    """Full pipeline: preprocessing (from the profile) followed by a model."""
    preprocessor = build_preprocessor(
        numeric=profile.numeric_features,
        categorical=profile.categorical_features,
        numeric_text=profile.numeric_text_features,
        scale=scale,
    )
    return Pipeline([("preprocess", preprocessor), ("model", model)])


# ---------------------------------------------------------------------------
# Readable feature names (used by the explanations)
# ---------------------------------------------------------------------------
def feature_names_and_sources(preprocessor: Pipeline) -> tuple[list[str], list[str]]:
    """For a FITTED preprocessor, describe every output column.

    Returns two lists of equal length:
      names    readable names, e.g. "age", "age (missing)", "region = north"
      sources  the original input column each output came from
    so one-hot columns can be added back up per original column.
    """
    column_transformer = preprocessor.named_steps["columns"]
    names, sources = [], []

    for name, transformer, cols in column_transformer.transformers_:
        if name == "num":
            for col in cols:
                names.append(col)
                sources.append(col)
            indicator = transformer.named_steps["impute"].indicator_
            if indicator is not None:
                for i in indicator.features_:
                    names.append(f"{cols[i]} (missing)")
                    sources.append(cols[i])

        elif name == "cat":
            encoder = transformer.named_steps["encode"]
            for i, col in enumerate(cols):
                rare = encoder.infrequent_categories_[i]
                rare = set(rare) if rare is not None else set()
                # Frequent categories come first, then one "(other)" column.
                for category in encoder.categories_[i]:
                    if category not in rare:
                        names.append(f"{col} = {category}")
                        sources.append(col)
                if rare:
                    names.append(f"{col} = {OTHER_LABEL}")
                    sources.append(col)

    return names, sources
