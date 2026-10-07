"""Profile the training data: column types, quality issues, and leakage.

IMPORTANT: call profile_data() on the TRAINING split only. The held-out test
set must never influence any decision, including which columns to use.
"""

import re
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import resample

from foresight import config

# Column kinds:
#   "numeric"       real numbers
#   "numeric_text"  numbers stored as text (converted during preprocessing)
#   "categorical"   text categories or True/False
#   "datetime"      dates (not used as inputs in Phase 1)


@dataclass
class ColumnProfile:
    name: str
    kind: str
    missing_share: float
    n_unique: int
    flags: list[str] = field(default_factory=list)
    use: bool = True                    # will the model use this column?
    reason: str = ""                    # plain-English reason if not used
    leakage_score: float | None = None  # single-column score, if checked


@dataclass
class DataProfile:
    n_rows: int
    n_columns: int
    problem_type: str
    columns: list[ColumnProfile]
    class_balance: dict | None     # {class label: share of rows}
    minority_share: float | None
    is_imbalanced: bool
    n_duplicate_rows: int
    warnings: list[str]

    def _names(self, kinds):
        return [c.name for c in self.columns if c.use and c.kind in kinds]

    @property
    def numeric_features(self) -> list[str]:
        """All columns used as numbers (including numbers stored as text)."""
        return self._names(("numeric", "numeric_text"))

    @property
    def numeric_text_features(self) -> list[str]:
        """The subset of numeric_features that must be converted from text."""
        return self._names(("numeric_text",))

    @property
    def categorical_features(self) -> list[str]:
        return self._names(("categorical",))

    @property
    def excluded(self) -> dict[str, str]:
        return {c.name: c.reason for c in self.columns if not c.use}

    @property
    def leaky_columns(self) -> list[str]:
        return [c.name for c in self.columns if "likely_leakage" in c.flags]


# ---------------------------------------------------------------------------
# Column type helpers
# ---------------------------------------------------------------------------
def to_numeric_safe(series: pd.Series) -> pd.Series:
    """Convert text such as " 12.5 " to numbers. Blanks and junk become NaN.

    This is stateless (it learns nothing from the data), so it is safe to
    apply to training and test data alike.
    """
    text = series.astype("string").str.strip()
    return pd.to_numeric(text, errors="coerce").astype(float)


def _non_blank(series: pd.Series) -> pd.Series:
    text = series.dropna().astype("string").str.strip()
    return text[text != ""]


def _is_numeric_text(series: pd.Series) -> bool:
    values = _non_blank(series)
    if values.empty:
        return False
    parsed = pd.to_numeric(values, errors="coerce")
    return parsed.notna().mean() >= config.NUMERIC_TEXT_RATIO


def _is_datetime_text(series: pd.Series) -> bool:
    values = _non_blank(series)
    if values.empty:
        return False
    sample = values.sample(min(200, len(values)), random_state=config.RANDOM_SEED)
    # Require a digit: words like "March" or "Monday" parse as dates but are
    # really categories.
    has_digit = sample.str.contains(r"\d", regex=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        except (ValueError, TypeError, OverflowError):
            return False
    return (parsed.notna() & has_digit).mean() >= config.DATETIME_PARSE_RATIO


def _column_kind(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "categorical"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if _is_numeric_text(series):
        return "numeric_text"
    if _is_datetime_text(series):
        return "datetime"
    return "categorical"


def _name_looks_like_id(name: str) -> bool:
    # Split "customerID", "Customer_Id", "ROW-ID" into lowercase words.
    words = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", name)
    return any(w.lower() in config.ID_NAME_HINTS for w in words)


def _looks_like_id(name: str, series: pd.Series, kind: str) -> bool:
    values = series.dropna()
    if len(values) == 0 or values.nunique() / len(values) < config.ID_LIKE_UNIQUE_RATIO:
        return False
    if kind == "categorical":
        # Unique text in (almost) every row: codes, names, emails...
        return True
    if kind in ("numeric", "numeric_text"):
        numbers = (to_numeric_safe(values) if kind == "numeric_text" else values).dropna()
        numbers = numbers.to_numpy(dtype=float)
        if len(numbers) == 0 or not np.all(np.mod(numbers, 1) == 0):
            return False   # decimals like prices are real measurements
        if _name_looks_like_id(name):
            return True
        # Whole numbers densely covering a range, like 1, 2, 3... (a row
        # counter). Training data is a random subset, so allow gaps.
        span = numbers.max() - numbers.min() + 1
        return len(np.unique(numbers)) / span >= 0.5
    return False


# ---------------------------------------------------------------------------
# Leakage check
# ---------------------------------------------------------------------------
def _single_column_score(x: pd.Series, y: np.ndarray, problem_type: str,
                         kind: str) -> float | None:
    """Cross-validated score of a small tree that sees ONLY this column."""
    if kind == "categorical":
        X = x.astype("string").fillna("__missing__").astype(object).to_frame()
        prep = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    else:
        X = x.astype(float).to_frame()
        # add_indicator lets the tree use "was this value missing?", which
        # can itself be a source of leakage.
        prep = SimpleImputer(strategy="median", add_indicator=True)

    seed = config.RANDOM_SEED
    folds = config.LEAKAGE_CV_FOLDS
    if problem_type == "regression":
        model = DecisionTreeRegressor(min_samples_leaf=5, random_state=seed)
        cv = KFold(folds, shuffle=True, random_state=seed)
        scoring = "r2"
    else:
        model = DecisionTreeClassifier(min_samples_leaf=5, random_state=seed)
        cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
        scoring = "roc_auc" if problem_type == "binary" else "balanced_accuracy"

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_val_score(make_pipeline(prep, model), X, y, cv=cv,
                                     scoring=scoring, error_score="raise")
    except ValueError:
        return None   # e.g. a class too rare to split; skip this column
    return float(np.mean(scores))


def _leakage_sample(X: pd.DataFrame, y: pd.Series, problem_type: str):
    """Use at most LEAKAGE_SAMPLE_ROWS rows (keeping class shares)."""
    if len(X) <= config.LEAKAGE_SAMPLE_ROWS:
        return X, y
    stratify = None if problem_type == "regression" else y
    return resample(X, y, n_samples=config.LEAKAGE_SAMPLE_ROWS, replace=False,
                    stratify=stratify, random_state=config.RANDOM_SEED)


_METRIC_NAMES = {"binary": "ROC-AUC", "multiclass": "balanced accuracy",
                 "regression": "R2"}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def profile_data(X: pd.DataFrame, y: pd.Series, problem_type: str,
                 check_leakage: bool = True) -> DataProfile:
    """Profile training features X and training target y.

    Decides which columns the model will use and collects plain-English
    warnings for the report. X and y are not modified.
    """
    warnings_list: list[str] = []
    columns: list[ColumnProfile] = []

    for name in X.columns:
        series = X[name]
        kind = _column_kind(series)
        col = ColumnProfile(
            name=name,
            kind=kind,
            missing_share=float(series.isna().mean()),
            n_unique=int(series.nunique(dropna=True)),
        )

        if col.missing_share == 1.0:
            col.flags.append("all_missing")
            col.use, col.reason = False, "Every value is missing."
        elif col.n_unique <= 1:
            col.flags.append("constant")
            col.use, col.reason = False, "It has the same value in every row."
        elif kind == "datetime":
            col.flags.append("date")
            col.use, col.reason = False, ("It looks like a date. Dates are not "
                                          "used as inputs in this version.")
        elif _looks_like_id(name, series, kind):
            col.flags.append("id_like")
            col.use, col.reason = False, ("It looks like an ID (almost every value "
                                          "is unique), so it has no general pattern "
                                          "to learn.")

        if col.use and col.missing_share > config.HIGH_MISSING_THRESHOLD:
            col.flags.append("high_missing")
            warnings_list.append(
                f"Column '{name}' is {col.missing_share:.0%} empty. Missing values "
                "are filled in automatically, but results for it are less reliable."
            )
        if (col.use and kind == "categorical"
                and col.n_unique > config.HIGH_CARDINALITY_THRESHOLD):
            col.flags.append("high_cardinality")
            warnings_list.append(
                f"Column '{name}' has {col.n_unique} different values. Rare values "
                "are grouped together as 'other'."
            )
        if kind == "numeric_text" and col.use:
            col.flags.append("numbers_as_text")
            warnings_list.append(
                f"Column '{name}' stores numbers as text. It is converted to numbers; "
                "blank or invalid entries are treated as missing."
            )
        columns.append(col)

    # --- Likely target leakage (only for columns still in use) -------------
    if check_leakage:
        X_s, y_s = _leakage_sample(X, y, problem_type)
        # Classification labels may be text ("yes"/"no"); turn them into codes
        # 0, 1, 2... so every scorer handles them the same way.
        y_arr = y_s.to_numpy() if problem_type == "regression" else pd.factorize(y_s)[0]
        for col in columns:
            if not col.use:
                continue
            x = X_s[col.name]
            if col.kind == "numeric_text":
                x = to_numeric_safe(x)
            score = _single_column_score(x, y_arr, problem_type, col.kind)
            col.leakage_score = score
            if score is not None and score >= config.LEAKAGE_SCORE_THRESHOLD:
                col.flags.append("likely_leakage")
                col.use = False
                col.reason = (
                    f"On its own it predicts the target almost perfectly "
                    f"({_METRIC_NAMES[problem_type]} {score:.3f}). This often means "
                    "it contains information only known after the outcome. "
                    "Check whether it would really be available at prediction time."
                )
                warnings_list.append(f"Possible target leakage: column '{col.name}' "
                                     "was excluded. " + col.reason)
            elif score is not None and score >= config.LEAKAGE_WARNING_THRESHOLD:
                col.flags.append("strong_single_predictor")
                warnings_list.append(
                    f"Check column '{col.name}': on its own it predicts the target very "
                    f"well ({_METRIC_NAMES[problem_type]} {score:.3f}). It is still used, "
                    "but make sure it is not derived from the outcome (for example "
                    "another model's score, or a value recorded after the event)."
                )

    # --- Class balance (classification only) --------------------------------
    class_balance, minority_share, is_imbalanced = None, None, False
    if problem_type in ("binary", "multiclass"):
        shares = y.value_counts(normalize=True)
        class_balance = {str(k): float(v) for k, v in shares.items()}
        minority_share = float(shares.min())
        is_imbalanced = minority_share < config.IMBALANCE_THRESHOLD
        if is_imbalanced:
            warnings_list.append(
                f"The classes are imbalanced: the rarest one is only "
                f"{minority_share:.1%} of rows. Accuracy alone would be misleading, "
                "so metrics that focus on the rare class are used."
            )

    # --- Duplicate rows -------------------------------------------------------
    n_dupes = int(X.assign(__target__=y.to_numpy()).duplicated().sum())
    if n_dupes:
        warnings_list.append(
            f"{n_dupes} rows are exact duplicates. This can make results look "
            "slightly better than they really are."
        )

    if not any(c.use for c in columns):
        warnings_list.append("No usable input columns remain after the checks.")

    return DataProfile(
        n_rows=len(X),
        n_columns=X.shape[1],
        problem_type=problem_type,
        columns=columns,
        class_balance=class_balance,
        minority_share=minority_share,
        is_imbalanced=is_imbalanced,
        n_duplicate_rows=n_dupes,
        warnings=warnings_list,
    )
