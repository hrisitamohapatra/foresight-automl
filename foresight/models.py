"""Model line-up and metric definitions.

Every run compares:
  - a dummy baseline       (the score you get without learning anything)
  - a simple linear model  (the "is a complex model even needed?" check)
  - random forest and LightGBM candidates
The linear model also competes as a candidate, so it can win.

Classification targets are expected to be encoded as integers 0..k-1 by
train.py; for binary problems 1 is the "positive" class (e.g. churned).
"""

from dataclasses import dataclass
from typing import Callable

from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score, get_scorer, make_scorer

from foresight import config


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
@dataclass
class ModelSpec:
    key: str                  # short id, e.g. "lightgbm" or "lightgbm_tuned"
    name: str                 # display name for the UI and report
    role: str                 # "baseline", "simple", or "candidate"
    scale: bool               # does it need standardized numbers?
    make: Callable[[], object]  # returns a fresh, unfitted estimator
    family: str = ""          # model type, shared by tuned variants (e.g. "lightgbm")

    def __post_init__(self):
        if not self.family:
            self.family = self.key


def get_models(problem_type: str, balanced: bool = False) -> list[ModelSpec]:
    """The models to compare for this problem type.

    balanced=True (for imbalanced classes) makes models weight the rare
    class more, so they do not just predict the common class.
    """
    seed = config.RANDOM_SEED
    weight = "balanced" if balanced else None

    if problem_type in ("binary", "multiclass"):
        return [
            ModelSpec("dummy", "Baseline (class shares)", "baseline", False,
                      lambda: DummyClassifier(strategy="prior")),
            ModelSpec("linear", "Logistic regression", "simple", True,
                      lambda: LogisticRegression(max_iter=2000, class_weight=weight,
                                                 random_state=seed)),
            ModelSpec("random_forest", "Random forest", "candidate", False,
                      lambda: RandomForestClassifier(n_estimators=200, class_weight=weight,
                                                     n_jobs=-1, random_state=seed)),
            ModelSpec("lightgbm", "LightGBM", "candidate", False,
                      lambda: LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                             class_weight=weight, random_state=seed,
                                             deterministic=True, force_row_wise=True,
                                             verbose=-1)),
        ]

    if problem_type == "regression":
        return [
            ModelSpec("dummy", "Baseline (average value)", "baseline", False,
                      lambda: DummyRegressor(strategy="mean")),
            ModelSpec("linear", "Linear regression (ridge)", "simple", True,
                      lambda: Ridge(alpha=1.0)),
            ModelSpec("random_forest", "Random forest", "candidate", False,
                      lambda: RandomForestRegressor(n_estimators=200, n_jobs=-1,
                                                    random_state=seed)),
            ModelSpec("lightgbm", "LightGBM", "candidate", False,
                      lambda: LGBMRegressor(n_estimators=300, learning_rate=0.05,
                                            random_state=seed, deterministic=True,
                                            force_row_wise=True, verbose=-1)),
        ]

    raise ValueError(f"Unknown problem type: {problem_type}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
@dataclass
class Metric:
    key: str
    name: str
    scorer: object            # a scikit-learn scorer: scorer(model, X, y)
    higher_is_better: bool
    sign: int                 # sklearn reports errors as negatives; flip back
    description: str          # plain English, for the report


def _scorer(name):
    return get_scorer(name)


# zero_division=0: a model that never predicts a class scores 0, silently.
_F1 = make_scorer(f1_score, zero_division=0)
_F1_MACRO = make_scorer(f1_score, average="macro", zero_division=0)

METRICS = {
    "binary": [
        Metric("roc_auc", "ROC-AUC", _scorer("roc_auc"), True, 1,
               "How well the model ranks positive cases above negative ones "
               "(0.5 = random guessing, 1.0 = perfect)."),
        Metric("pr_auc", "PR-AUC", _scorer("average_precision"), True, 1,
               "How well the model finds the positive cases without false alarms. "
               "Most informative when positives are rare."),
        Metric("f1", "F1", _F1, True, 1,
               "Balance between catching positive cases and avoiding false alarms, "
               "at the default 50% cut-off."),
        Metric("accuracy", "Accuracy", _scorer("accuracy"), True, 1,
               "Share of all predictions that are correct. Can be misleading "
               "when one class is rare."),
    ],
    "multiclass": [
        Metric("f1_macro", "Macro F1", _F1_MACRO, True, 1,
               "F1 averaged over all classes equally, so rare classes count as "
               "much as common ones."),
        Metric("accuracy", "Accuracy", _scorer("accuracy"), True, 1,
               "Share of all predictions that are correct."),
    ],
    "regression": [
        Metric("mae", "MAE", _scorer("neg_mean_absolute_error"), False, -1,
               "Average size of the prediction error, in the target's units."),
        Metric("rmse", "RMSE", _scorer("neg_root_mean_squared_error"), False, -1,
               "Like MAE but punishes large errors more, in the target's units."),
        Metric("r2", "R2", _scorer("r2"), True, 1,
               "Share of the target's variation the model explains "
               "(0 = no better than the average, 1 = perfect)."),
    ],
}


def get_metrics(problem_type: str) -> list[Metric]:
    return METRICS[problem_type]


def selection_metric(problem_type: str, is_imbalanced: bool) -> str:
    """Key of the metric used to pick the best model."""
    if problem_type == "binary" and is_imbalanced:
        return config.SELECTION_METRIC["binary_imbalanced"]
    return config.SELECTION_METRIC[problem_type]


def score_model(fitted_model, X, y, problem_type: str) -> dict[str, float]:
    """Score a fitted model on (X, y) with every metric for the problem type.

    Uses the same scorers as cross-validation, so CV and test numbers are
    directly comparable. Error metrics are returned as positive numbers.
    """
    return {m.key: float(m.sign * m.scorer(fitted_model, X, y))
            for m in get_metrics(problem_type)}
