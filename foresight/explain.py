"""Explain what the final model relies on.

Two independent methods, both computed on HELD-OUT TEST rows:
  - SHAP: how much each column moves individual predictions, on average.
  - Permutation importance: how much the score drops when a column is
    shuffled (i.e. when the model can no longer use it).
If both methods rank the same columns highly, we can be more confident.

These describe what the MODEL relies on. They do not show that a column
causes the outcome.
"""

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance

from foresight import config
from foresight.models import get_metrics
from foresight.preprocess import feature_names_and_sources
from foresight.train import TrainResult

TREE_MODELS = ("random_forest", "lightgbm")


@dataclass
class Explanation:
    # One row per column the model uses, most important first. Columns:
    #   column, shap_share, shap_mean_abs, perm_mean, perm_std, direction
    importance: pd.DataFrame
    shap_available: bool
    shap_rows: int                 # test rows used for SHAP
    perm_rows: int                 # test rows used for permutation importance
    perm_metric: str               # metric whose drop is measured
    top_overlap: int               # shared columns in both top-k lists
    top_k: int
    agreement_text: str
    notes: list[str] = field(default_factory=list)

    @property
    def top_drivers(self) -> list[str]:
        return self.importance["column"].head(self.top_k).tolist()


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------
def _raw_shap_values(model_key, model, X_sample, X_background):
    """SHAP values for the transformed sample.

    Returns (n_rows, n_features) for binary/regression, or
    (n_rows, n_features, n_classes) for multiclass.
    """
    if model_key in TREE_MODELS:
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(X_sample, check_additivity=False)
    elif model_key == "linear":
        explainer = shap.LinearExplainer(model, X_background)
        values = explainer.shap_values(X_sample)
    else:
        return None   # e.g. the dummy baseline uses no inputs

    # Different SHAP versions/models return a list per class or one array.
    if isinstance(values, list):
        values = np.stack(values, axis=-1)
    values = np.asarray(values, dtype=float)
    # Binary models sometimes return both classes; keep the positive (1).
    if values.ndim == 3 and values.shape[2] == 2:
        values = values[:, :, 1]
    return values


def _to_columns(values, sources, columns):
    """Add up SHAP values of all outputs from the same original column.

    (SHAP values are additive, so e.g. "plan = basic" + "plan = pro" gives
    the total effect of "plan".)
    """
    mapping = np.zeros((len(sources), len(columns)))
    for i, src in enumerate(sources):
        mapping[i, columns.index(src)] = 1.0
    if values.ndim == 2:
        return values @ mapping                              # (n, columns)
    return np.einsum("nfk,fc->nck", values, mapping)         # (n, columns, k)


def _directions(result, col_shap, Xt, names, sources, columns):
    """Plain-English direction hints (binary and regression only)."""
    if result.problem_type == "binary":
        up = f"more likely '{result.positive_class}'"
        down = f"less likely '{result.positive_class}'"
    else:
        up = f"higher predicted {result.target}"
        down = f"lower predicted {result.target}"

    numeric = set(result.profile.numeric_features)
    out = {}
    for j, col in enumerate(columns):
        s = col_shap[:, j]
        if col in numeric:
            x = Xt[:, names.index(col)]
            if np.std(x) == 0 or np.std(s) == 0:
                continue
            rho = pd.Series(x).corr(pd.Series(s), method="spearman")
            if rho >= config.DIRECTION_MIN_CORRELATION:
                out[col] = f"Higher values go with {up}."
            elif rho <= -config.DIRECTION_MIN_CORRELATION:
                out[col] = f"Higher values go with {down}."
            else:
                out[col] = "No simple up/down pattern."
        else:
            # Average effect for rows in each category.
            effects = {}
            for i, (name, src) in enumerate(zip(names, sources)):
                rows = Xt[:, i] == 1
                if src == col and rows.sum() >= 5:
                    effects[name[len(col) + 3:]] = s[rows].mean()  # strip "col = "
            if len(effects) >= 2:
                high = max(effects, key=effects.get)
                low = min(effects, key=effects.get)
                if effects[high] > 0 > effects[low]:
                    out[col] = f"'{high}' goes with {up}; '{low}' goes with {down}."
    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def explain_model(result: TrainResult) -> Explanation:
    pipeline = result.pipeline
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    names, sources = feature_names_and_sources(preprocessor)
    columns = [c.name for c in result.profile.columns if c.use]
    rng = np.random.default_rng(config.RANDOM_SEED)
    notes: list[str] = []

    # --- SHAP on a sample of test rows --------------------------------------
    max_rows = (config.SHAP_SAMPLE_ROWS_FOREST if result.best_key == "random_forest"
                else config.SHAP_SAMPLE_ROWS)
    n_shap = min(max_rows, result.n_test)
    shap_idx = rng.choice(result.n_test, size=n_shap, replace=False)
    bg_idx = rng.choice(result.n_train, size=min(config.SHAP_BACKGROUND_ROWS,
                                                  result.n_train), replace=False)
    Xt = preprocessor.transform(result.X_test.iloc[shap_idx])
    Xt_background = preprocessor.transform(result.X_train.iloc[bg_idx])

    shap_mean_abs = np.zeros(len(columns))
    directions: dict[str, str] = {}
    shap_available = False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            values = _raw_shap_values(result.best_key, model, Xt, Xt_background)
        if values is not None:
            col_shap = _to_columns(values, sources, columns)
            axes = (0,) if col_shap.ndim == 2 else (0, 2)
            shap_mean_abs = np.abs(col_shap).mean(axis=axes)
            shap_available = True
            if col_shap.ndim == 2:
                directions = _directions(result, col_shap, Xt, names, sources, columns)
                total_shap = shap_mean_abs.sum()
                for j, col in enumerate(columns):
                    if total_shap > 0 and shap_mean_abs[j] / total_shap < config.DIRECTION_MIN_SHARE:
                        directions[col] = "Very little influence."
    except Exception:
        notes.append("SHAP values could not be computed for this model; "
                     "the ranking uses permutation importance only.")

    if result.best_key == "dummy":
        notes.append("The best model was the baseline, which ignores all inputs. "
                     "No column showed a pattern the models could learn reliably.")

    # --- Permutation importance on test rows --------------------------------
    n_perm = min(config.PERMUTATION_SAMPLE_ROWS, result.n_test)
    perm_idx = rng.choice(result.n_test, size=n_perm, replace=False)
    metric = next(m for m in get_metrics(result.problem_type)
                  if m.key == result.selection_metric)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        perm = permutation_importance(
            pipeline, result.X_test.iloc[perm_idx], result.y_test[perm_idx],
            scoring=metric.scorer, n_repeats=config.PERMUTATION_REPEATS,
            random_state=config.RANDOM_SEED,
        )
    # Scorers are "higher is better", so a positive value = score got worse.
    perm_by_col = dict(zip(result.X_test.columns, zip(perm.importances_mean,
                                                      perm.importances_std)))

    # --- Combine ---------------------------------------------------------------
    total = shap_mean_abs.sum()
    table = pd.DataFrame({
        "column": columns,
        "shap_share": shap_mean_abs / total if total > 0 else 0.0,
        "shap_mean_abs": shap_mean_abs,
        "perm_mean": [float(perm_by_col[c][0]) for c in columns],
        "perm_std": [float(perm_by_col[c][1]) for c in columns],
        "direction": [directions.get(c, "") for c in columns],
    })
    sort_by = "shap_mean_abs" if shap_available else "perm_mean"
    table = table.sort_values(sort_by, ascending=False, kind="stable").reset_index(drop=True)

    # --- Do the two methods agree on the top drivers? ------------------------
    # Compare at most (number of columns - 1); comparing ALL columns would
    # always "agree" and tell us nothing.
    k = min(config.TOP_K_DRIVERS, max(1, len(columns) - 1))
    top_perm = set(table.sort_values("perm_mean", ascending=False, kind="stable")
                   ["column"].head(k))
    top_shap = set(table["column"].head(k))
    overlap = len(top_perm & top_shap)
    if not shap_available:
        agreement = "Only permutation importance was available, so there is no cross-check."
    elif overlap >= max(1, (k + 1) // 2):
        agreement = (f"SHAP and permutation importance share {overlap} of their top {k} "
                     "columns, so this ranking is fairly reliable.")
    else:
        agreement = (f"SHAP and permutation importance share only {overlap} of their "
                     f"top {k} columns. Treat the exact ranking with caution.")

    return Explanation(
        importance=table,
        shap_available=shap_available,
        shap_rows=n_shap if shap_available else 0,
        perm_rows=n_perm,
        perm_metric=metric.name,
        top_overlap=overlap,
        top_k=k,
        agreement_text=agreement,
        notes=notes,
    )
