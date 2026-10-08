"""Turn binary predictions into decisions: calibration, thresholds, lift.

Rigor rules followed here:
- Every CHOICE (use calibration or not, which cut-off) is made on
  out-of-fold predictions for the TRAINING data: each training row is scored
  by a model that never saw it. The test set plays no part in these choices.
- Calibration is learned inside each outer CV fold (nested), so its benefit
  is measured honestly.
- The test set is then used once, to report what the chosen setup achieves.

Binary problems only: thresholds and lift need a single "positive" class.
"""

import math
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from foresight import config
from foresight.preprocess import build_pipeline
from foresight.train import TrainResult


class DecisionError(Exception):
    """Invalid decision settings. The message is safe to show users."""


# ---------------------------------------------------------------------------
# Small building blocks
# ---------------------------------------------------------------------------
@dataclass
class Outcome:
    """What happens if we flag every row with probability >= threshold."""
    threshold: float
    tp: int   # flagged and really positive  (caught)
    fp: int   # flagged but not positive     (false alarm)
    tn: int   # not flagged, not positive
    fn: int   # not flagged but positive     (missed)

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def flagged_share(self) -> float:
        return (self.tp + self.fp) / self.n

    @property
    def recall(self) -> float:
        """Share of all positives that were caught."""
        positives = self.tp + self.fn
        return self.tp / positives if positives else 0.0

    @property
    def precision(self) -> float:
        """Share of flagged rows that really are positive."""
        flagged = self.tp + self.fp
        return self.tp / flagged if flagged else 0.0

    def cost(self, cost_fp: float, cost_fn: float) -> float:
        return self.fp * cost_fp + self.fn * cost_fn

    def cost_per_1000(self, cost_fp: float, cost_fn: float) -> float:
        return 1000 * self.cost(cost_fp, cost_fn) / self.n


def outcome_at(y: np.ndarray, proba: np.ndarray, threshold: float) -> Outcome:
    y = np.asarray(y).astype(bool)
    flagged = np.asarray(proba) >= threshold
    return Outcome(
        threshold=float(threshold),
        tp=int(np.sum(flagged & y)),
        fp=int(np.sum(flagged & ~y)),
        tn=int(np.sum(~flagged & ~y)),
        fn=int(np.sum(~flagged & y)),
    )


def choose_threshold(y, proba, cost_fp: float, cost_fn: float) -> float:
    """The cut-off from THRESHOLD_GRID with the lowest total cost.

    Ties go to the cut-off closest to 0.5 (the usual default).
    """
    best_t, best_cost = None, None
    for t in sorted(config.THRESHOLD_GRID, key=lambda t: (abs(t - 0.5), t)):
        cost = outcome_at(y, proba, t).cost(cost_fp, cost_fn)
        if best_cost is None or cost < best_cost:
            best_t, best_cost = t, cost
    return best_t


def gains_table(y, proba) -> pd.DataFrame:
    """Cumulative gains and lift, from highest to lowest predicted risk.

    Row k: if we act on the top k% of rows, what share of all positives do we
    reach, and how many times better is that than picking rows at random?
    """
    y = np.asarray(y).astype(int)
    order = np.argsort(-np.asarray(proba), kind="stable")
    captured = np.cumsum(y[order])
    total = captured[-1]
    rows = []
    for pct in range(1, 101):
        n_top = max(1, math.ceil(len(y) * pct / 100))
        share = captured[n_top - 1] / total if total else 0.0
        rows.append({"pct_acted_on": pct, "pct_positives_reached": 100 * share,
                     "lift": share / (pct / 100)})
    return pd.DataFrame(rows)


def check_costs(cost_fp: float, cost_fn: float) -> None:
    for value in (cost_fp, cost_fn):
        if not isinstance(value, (int, float)) or not math.isfinite(value) \
                or value <= 0 or value > config.MAX_COST:
            raise DecisionError(
                f"Costs must be positive numbers no larger than {config.MAX_COST:,}.")


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
@dataclass
class DecisionAnalysis:
    applicable: bool
    reason: str = ""                       # why not applicable, if so
    positive_class: object = None
    cost_fp: float = 1.0
    cost_fn: float = 1.0
    # Calibration
    calibration_method: str = ""           # "sigmoid" or "isotonic"
    calibrated: bool = False               # chosen on training data
    oof_brier_raw: float = float("nan")
    oof_brier_calibrated: float = float("nan")
    test_brier_raw: float = float("nan")
    test_brier_calibrated: float = float("nan")
    reliability: dict = field(default_factory=dict)   # name -> (predicted, actual)
    # Threshold
    threshold: float = 0.5                 # chosen on training data
    oof_cost_chosen: float = float("nan")
    oof_cost_default: float = float("nan")
    test_chosen: Outcome | None = None
    test_default: Outcome | None = None    # at 0.5
    test_flag_none: Outcome | None = None
    test_flag_all: Outcome | None = None
    # Lift
    gains: pd.DataFrame | None = None
    model: object = None                   # the model behind these decisions


def _not_applicable(reason: str) -> DecisionAnalysis:
    return DecisionAnalysis(applicable=False, reason=reason)


def analyze_decision(result: TrainResult, cost_fp: float = 1.0,
                     cost_fn: float = 1.0) -> DecisionAnalysis:
    """Calibrate, choose a cost-based threshold, and measure lift.

    cost_fp  cost of a false alarm (flagging a row that is not positive)
    cost_fn  cost of a miss (not flagging a row that is positive)
    """
    if result.problem_type != "binary":
        return _not_applicable("Thresholds and lift apply to yes/no predictions only.")
    if result.best_key == "dummy":
        return _not_applicable("No model beat the baseline, so there are no useful "
                               "scores to set a threshold on.")
    check_costs(cost_fp, cost_fn)

    X_train, y_train = result.X_train, result.y_train
    X_test, y_test = result.X_test, result.y_test
    seed = config.RANDOM_SEED
    outer_cv = StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=seed)

    # Rebuild the (unfitted) winning pipeline, including any tuned settings.
    spec = result.best_spec

    def fresh_pipeline():
        return build_pipeline(result.profile, spec.make(), scale=spec.scale)

    method = ("isotonic" if result.n_train >= config.CALIBRATION_ISOTONIC_MIN_ROWS
              else "sigmoid")

    def calibrated_model():
        return CalibratedClassifierCV(
            fresh_pipeline(), method=method,
            cv=StratifiedKFold(config.CALIBRATION_CV_FOLDS, shuffle=True, random_state=seed))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)

        # 1. Out-of-fold probabilities on the TRAINING data, raw and calibrated.
        oof_raw = cross_val_predict(fresh_pipeline(), X_train, y_train, cv=outer_cv,
                                    method="predict_proba")[:, 1]
        oof_cal = cross_val_predict(calibrated_model(), X_train, y_train, cv=outer_cv,
                                    method="predict_proba")[:, 1]

        # 2. Use calibration only if it helps on the training data.
        brier_raw = brier_score_loss(y_train, oof_raw)
        brier_cal = brier_score_loss(y_train, oof_cal)
        use_calibration = brier_cal < brier_raw
        oof = oof_cal if use_calibration else oof_raw

        # Final calibrated model, fitted on all training data.
        calibrated = calibrated_model().fit(X_train, y_train)

    # 3. Choose the cut-off on the training out-of-fold scores.
    threshold = choose_threshold(y_train, oof, cost_fp, cost_fn)

    # 4. The one evaluation on the test set.
    p_raw = result.pipeline.predict_proba(X_test)[:, 1]
    p_cal = calibrated.predict_proba(X_test)[:, 1]
    p = p_cal if use_calibration else p_raw

    def curve(proba):
        actual, predicted = calibration_curve(y_test, proba, n_bins=config.CALIBRATION_BINS,
                                              strategy="quantile")
        return predicted.tolist(), actual.tolist()

    return DecisionAnalysis(
        applicable=True,
        positive_class=result.positive_class,
        cost_fp=float(cost_fp),
        cost_fn=float(cost_fn),
        calibration_method=method,
        calibrated=bool(use_calibration),
        oof_brier_raw=float(brier_raw),
        oof_brier_calibrated=float(brier_cal),
        test_brier_raw=float(brier_score_loss(y_test, p_raw)),
        test_brier_calibrated=float(brier_score_loss(y_test, p_cal)),
        reliability={"Original": curve(p_raw), "Calibrated": curve(p_cal)},
        threshold=threshold,
        oof_cost_chosen=outcome_at(y_train, oof, threshold).cost(cost_fp, cost_fn),
        oof_cost_default=outcome_at(y_train, oof, 0.5).cost(cost_fp, cost_fn),
        test_chosen=outcome_at(y_test, p, threshold),
        test_default=outcome_at(y_test, p, 0.5),
        test_flag_none=outcome_at(y_test, p, np.inf),
        test_flag_all=outcome_at(y_test, p, -np.inf),
        gains=gains_table(y_test, p),
        model=calibrated if use_calibration else result.pipeline,
    )
