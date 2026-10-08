"""Build a plain-English, self-contained HTML report.

Security:
- Jinja2 autoescape is ON, so uploaded text (column names, category values,
  dataset names) is always shown as text, never interpreted as HTML.
- Text in report tables is also protected against spreadsheet formula
  injection, in case someone copies a table into Excel.
- Charts are embedded as images, so the report needs no external files.

Every number comes from the TrainResult / Explanation objects of a real run.
"""

import base64
import io
import re
from datetime import datetime
from pathlib import Path

import lightgbm
import matplotlib
import shap
import sklearn

matplotlib.use("Agg")   # draw charts to images, without opening windows
import matplotlib.pyplot as plt  # noqa: E402
from jinja2 import Environment, FileSystemLoader  # noqa: E402

from foresight import config  # noqa: E402
from foresight.decision import DecisionAnalysis  # noqa: E402
from foresight.explain import Explanation  # noqa: E402
from foresight.ingest import neutralize_cell, safe_display_name  # noqa: E402
from foresight.models import get_metrics  # noqa: E402
from foresight.train import TrainResult  # noqa: E402

TEMPLATE_DIR = Path(__file__).parent / "templates"

# autoescape=True for every template, regardless of file extension.
_env = Environment(
    loader=FileSystemLoader(TEMPLATE_DIR),
    autoescape=True,
    trim_blocks=True,
    lstrip_blocks=True,
)

ROLE_LABELS = {"baseline": "Baseline", "simple": "Simple model", "candidate": "Candidate"}
PROBLEM_LABELS = {"binary": "Yes/no prediction (binary classification)",
                  "multiclass": "Category prediction (multiclass classification)",
                  "regression": "Number prediction (regression)"}
MAX_DRIVER_ROWS = 10


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def fmt(value: float) -> str:
    """Format a metric value: 3 decimals, or thousands separators if large."""
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    text = f"{value:,.3f}"
    return "0.000" if text == "-0.000" else text   # no confusing "negative zero"


def cell(text) -> str:
    """Make uploaded text safe for a table cell (formula injection guard).

    HTML escaping is done separately by Jinja2's autoescape.
    """
    return neutralize_cell(str(text))


def _chart_label(text: str, max_length: int = 40) -> str:
    # "$" would switch matplotlib into math mode; show it literally.
    text = str(text).replace("$", r"\$")
    return text if len(text) <= max_length else text[: max_length - 1] + "…"


def _figure_to_base64(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=config.COLOR_TEXT)
    ax.xaxis.label.set_color(config.COLOR_TEXT)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def importance_chart(explanation: Explanation) -> str:
    table = explanation.importance.head(MAX_DRIVER_ROWS)
    if explanation.shap_available:
        values, xlabel = table["shap_share"] * 100, "Share of the model's reliance (%), SHAP"
    else:
        values, xlabel = table["perm_mean"], "Score drop when shuffled (permutation)"
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(table) + 1))
    ax.barh([_chart_label(c) for c in table["column"]][::-1], list(values)[::-1],
            color=config.CHART_BLUES[1])
    ax.set_xlabel(xlabel)
    _style_axes(ax)
    return _figure_to_base64(fig)


def model_chart(result: TrainResult) -> str:
    ok = [r for r in result.model_results if not r.error]
    metric = next(m for m in get_metrics(result.problem_type) if m.key == result.selection_metric)
    colors = [config.CHART_BLUES[0] if r.key == result.best_key else
              config.CHART_BLUES[4] if r.role == "baseline" else config.CHART_BLUES[2]
              for r in ok]
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(ok) + 1))
    ax.barh([_chart_label(r.name) for r in ok][::-1],
            [r.cv_mean[metric.key] for r in ok][::-1],
            xerr=[r.cv_std[metric.key] for r in ok][::-1],
            color=colors[::-1], ecolor=config.COLOR_TEXT, capsize=3)
    direction = "higher is better" if metric.higher_is_better else "lower is better"
    ax.set_xlabel(f"{metric.name}, cross-validation mean ± std ({direction})")
    _style_axes(ax)
    return _figure_to_base64(fig)


def reliability_chart(decision: DecisionAnalysis) -> str:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot([0, 1], [0, 1], linestyle="--", color=config.CHART_BLUES[4], label="Perfectly reliable")
    styles = {"Original": (config.CHART_BLUES[2], "o"), "Calibrated": (config.CHART_BLUES[0], "s")}
    for name, (predicted, actual) in decision.reliability.items():
        color, marker = styles[name]
        ax.plot(predicted, actual, marker=marker, color=color, label=name)
    ax.set_xlabel("Predicted probability (test rows, grouped)")
    ax.set_ylabel("Actual share that were positive")
    ax.legend(frameon=False)
    _style_axes(ax)
    return _figure_to_base64(fig)


def gains_chart(decision: DecisionAnalysis) -> str:
    gains = decision.gains
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot([0, 100], [0, 100], linestyle="--", color=config.CHART_BLUES[4], label="Random order")
    ax.plot(gains["pct_acted_on"], gains["pct_positives_reached"],
            color=config.CHART_BLUES[0], label="Model order (highest risk first)")
    ax.set_xlabel("% of rows acted on")
    ax.set_ylabel(f"% of all '{_chart_label(decision.positive_class)}' cases reached")
    ax.legend(frameon=False)
    _style_axes(ax)
    return _figure_to_base64(fig)


# ---------------------------------------------------------------------------
# Decision section text
# ---------------------------------------------------------------------------
def num(value: float) -> str:
    """Compact number for costs: 1, 5, 2.5, 1,000."""
    return f"{value:,.2f}".rstrip("0").rstrip(".")


def decision_context(decision: DecisionAnalysis | None, charts: bool = True) -> dict | None:
    """Plain-English text (and optionally chart images) for the decision section."""
    if decision is None:
        return None
    if not decision.applicable:
        return {"applicable": False, "reason": decision.reason}

    pos = decision.positive_class
    fp, fn = decision.cost_fp, decision.cost_fn
    method = "Platt/sigmoid" if decision.calibration_method == "sigmoid" else "isotonic"
    if decision.calibrated:
        calibration = (
            f"The probabilities were calibrated ({method} method) because, on the training "
            f"data, this made them more accurate: Brier score {fmt(decision.oof_brier_raw)} → "
            f"{fmt(decision.oof_brier_calibrated)} (lower is better). On the test set: "
            f"{fmt(decision.test_brier_raw)} → {fmt(decision.test_brier_calibrated)}."
        )
    else:
        calibration = (
            f"Calibration ({method} method) was tested but did not make the probabilities "
            f"more accurate on the training data (Brier score {fmt(decision.oof_brier_raw)} "
            f"vs {fmt(decision.oof_brier_calibrated)}), so the original probabilities are used."
        )

    def row(name, outcome):
        positives = outcome.tp + outcome.fn
        return {
            "name": name,
            "flagged": f"{outcome.flagged_share:.0%}",
            "caught": f"{outcome.tp} of {positives} ({outcome.recall:.0%})",
            "false_alarms": f"{outcome.fp}",
            "missed": f"{outcome.fn}",
            "cost": num(outcome.cost_per_1000(fp, fn)),
        }

    chosen, default = decision.test_chosen, decision.test_default
    chosen_cost, default_cost = chosen.cost_per_1000(fp, fn), default.cost_per_1000(fp, fn)
    if decision.threshold == 0.5:
        comparison = "The chosen cut-off is the same as the usual default of 0.50."
    elif chosen_cost < default_cost:
        comparison = (f"On the test set, this cut-off cost {num(chosen_cost)} per 1,000 rows, "
                      f"compared with {num(default_cost)} at the default cut-off of 0.50.")
    else:
        comparison = (f"On the test set, this cut-off cost {num(chosen_cost)} per 1,000 rows, "
                      f"compared with {num(default_cost)} at the default of 0.50, so the "
                      "improvement seen on the training data did not carry over here.")

    gains = decision.gains.set_index("pct_acted_on")
    gains_lines = [
        f"Acting on the top {pct}% highest-scoring rows reaches "
        f"{gains.loc[pct, 'pct_positives_reached']:.0f}% of all '{pos}' cases "
        f"({gains.loc[pct, 'lift']:.1f}× better than picking rows at random)."
        for pct in (10, 20, 30, 50)
    ]

    return {
        "applicable": True,
        "positive_class": pos,
        "calibration": calibration,
        "costs": (f"A missed '{pos}' case was counted as costing {num(fn)} and a false alarm "
                  f"(flagging a row that is not '{pos}') as costing {num(fp)}."),
        "threshold": (f"The cut-off with the lowest total cost on the training data is "
                      f"{decision.threshold:.2f}: rows scoring {decision.threshold:.2f} or higher "
                      f"are flagged as likely '{pos}'."),
        "comparison": comparison,
        "rows": [row(f"Chosen cut-off ({decision.threshold:.2f})", chosen),
                 row("Default cut-off (0.50)", default),
                 row("Flag nobody", decision.test_flag_none),
                 row("Flag everyone", decision.test_flag_all)],
        "gains_lines": gains_lines,
        "reliability_chart": reliability_chart(decision) if charts else "",
        "gains_chart": gains_chart(decision) if charts else "",
    }


# ---------------------------------------------------------------------------
# Plain-English text
# ---------------------------------------------------------------------------
def default_question(result: TrainResult) -> str:
    target = result.target
    if result.problem_type == "binary":
        return (f"Which records are most likely to be '{result.positive_class}' "
                f"for {target}, and what does the model rely on to tell?")
    if result.problem_type == "multiclass":
        return f"Which {target} category does each record most likely belong to, and why?"
    return f"What value of {target} should we expect for each record, and what is it based on?"


def headline(result: TrainResult) -> str:
    metric = next(m for m in get_metrics(result.problem_type) if m.key == result.selection_metric)
    test_value = result.test_scores[metric.key]
    text = (f"On {result.n_test:,} held-out rows that played no part in building it, "
            f"the {result.best.name} model scored {metric.name} {fmt(test_value)}")
    if not metric.higher_is_better:
        text += " (lower is better)"
    baseline = result.result_for("dummy")
    if result.best_key != "dummy" and not baseline.error:
        text += (f". For comparison, a baseline that ignores all inputs scores "
                 f"{fmt(baseline.cv_mean[metric.key])} in cross-validation")
    return text + "."


def caveats(result: TrainResult, explanation: Explanation) -> list[str]:
    metric = next(m for m in get_metrics(result.problem_type) if m.key == result.selection_metric)
    key = metric.key
    best = result.best
    items = [
        "The explanations show what the model relies on, not what causes the outcome. "
        "A column can matter to the model because it is linked to the real reason.",
        "Results assume future data looks like this data. Performance can drop if "
        "customers, products, or conditions change over time.",
    ]
    if result.best_key == "dummy":
        items.append("No model beat the baseline. The inputs may not contain a "
                     "usable signal for this target.")
    simple = result.result_for("linear")
    if result.best_key not in ("linear", "dummy") and not simple.error:
        gap = abs(best.cv_mean[key] - simple.cv_mean[key])
        if gap <= best.cv_std[key]:
            items.append(f"The simple model ({simple.name}) scores within the normal "
                         f"variation of the best model ({fmt(simple.cv_mean[key])} vs "
                         f"{fmt(best.cv_mean[key])}). It may be preferable because it "
                         "is easier to explain.")
    worse = (best.cv_mean[key] - result.test_scores[key]) * (1 if metric.higher_is_better else -1)
    if worse > 2 * max(best.cv_std[key], 1e-12):
        items.append("The test score is noticeably worse than cross-validation "
                     "suggested, so expect results on new data to vary.")
    if result.n_test < 200:
        items.append(f"The test set is small ({result.n_test} rows), so the test "
                     "score is uncertain.")
    if result.profile.is_imbalanced:
        items.append(f"The classes are imbalanced (rarest class: "
                     f"{result.profile.minority_share:.1%} of rows). Accuracy alone "
                     "would be misleading.")
    for name in result.included_overrides:
        items.append(f"Column '{name}' was flagged as possible target leakage but "
                     "was kept at the user's request. If it is not known at "
                     "prediction time, real-world results will be worse than reported.")
    if result.tuning_hit_time_limit:
        items.append("Tuning reached its time limit before finishing all trials, so "
                     "re-running may give slightly different tuned results.")
    if result.n_dropped_target:
        items.append(f"{result.n_dropped_target:,} rows were removed because the "
                     "target value was missing.")
    return items


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------
def build_context(result: TrainResult, explanation: Explanation,
                  dataset_name: str = "", decision_question: str = "",
                  decision: DecisionAnalysis | None = None) -> dict:
    """Everything the template needs, already formatted as text."""
    metrics = get_metrics(result.problem_type)

    model_rows = []
    for r in result.model_results:
        model_rows.append({
            "name": r.name,
            "role": ROLE_LABELS[r.role],
            "is_best": r.key == result.best_key,
            "error": r.error,
            "scores": [f"{fmt(r.cv_mean[m.key])} ± {fmt(r.cv_std[m.key])}"
                       if not r.error else "-" for m in metrics],
            "fit_seconds": f"{r.fit_seconds:.2f}",
        })

    test_rows = [{"name": m.name, "value": fmt(result.test_scores[m.key]),
                  "description": m.description,
                  "is_selection": m.key == result.selection_metric} for m in metrics]

    driver_rows = []
    for _, row in explanation.importance.head(MAX_DRIVER_ROWS).iterrows():
        driver_rows.append({
            "column": cell(row["column"]),
            "share": f"{row['shap_share']:.0%}" if explanation.shap_available else "-",
            "perm": f"{fmt(row['perm_mean'])} ± {fmt(row['perm_std'])}",
            "direction": row["direction"],
        })

    excluded_rows = [{"column": cell(name), "reason": reason}
                     for name, reason in result.profile.excluded.items()]

    profile = result.profile
    class_rows = []
    if profile.class_balance:
        class_rows = [{"label": cell(label), "share": f"{share:.1%}"}
                      for label, share in profile.class_balance.items()]

    return {
        "palette": {
            "primary": config.COLOR_PRIMARY,
            "background": config.COLOR_BACKGROUND,
            "secondary": config.COLOR_SECONDARY_BACKGROUND,
            "text": config.COLOR_TEXT,
            "warning": config.COLOR_WARNING,
            "dark_blue": config.CHART_BLUES[0],
        },
        "dataset_name": safe_display_name(dataset_name) if dataset_name else "Uploaded data",
        "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "question": decision_question.strip() or default_question(result),
        "target": result.target,
        "problem_label": PROBLEM_LABELS[result.problem_type],
        "positive_class": result.positive_class,
        "headline": headline(result),
        "best_name": result.best.name,
        "selection_metric": next(m.name for m in metrics if m.key == result.selection_metric),
        "metric_names": [m.name for m in metrics],
        "model_rows": model_rows,
        "test_rows": test_rows,
        "driver_rows": driver_rows,
        "n_used_columns": len(explanation.importance),
        "agreement": explanation.agreement_text,
        "explain_notes": explanation.notes,
        "excluded_rows": excluded_rows,
        "class_rows": class_rows,
        "data_warnings": profile.warnings,
        "caveats": caveats(result, explanation),
        "n_rows": result.n_train + result.n_test,
        "n_train": result.n_train,
        "n_test": result.n_test,
        "n_columns": profile.n_columns,
        "test_share": f"{config.TEST_SIZE:.0%}",
        "cv_folds": config.CV_FOLDS,
        "seed": config.RANDOM_SEED,
        "shap_rows": explanation.shap_rows,
        "perm_rows": explanation.perm_rows,
        "perm_repeats": config.PERMUTATION_REPEATS,
        "perm_metric": explanation.perm_metric,
        "runtime": f"{result.runtime_seconds:.1f}",
        "versions": f"scikit-learn {sklearn.__version__}, LightGBM {lightgbm.__version__}, "
                    f"SHAP {shap.__version__}",
        "importance_chart": importance_chart(explanation),
        "model_chart": model_chart(result),
        "decision": decision_context(decision),
        "tuning": tuning_context(result),
    }


def tuning_context(result: TrainResult) -> dict | None:
    """Method notes for Optuna tuning, if it was switched on."""
    if not result.tuned:
        return None
    params = result.best.tuned_params
    return {
        "trials": config.TUNE_TRIALS,
        "timeout": config.TUNE_TIMEOUT_SECONDS,
        "inner_folds": config.TUNE_INNER_FOLDS,
        "final_params": ", ".join(
            f"{k} = {v:.3g}" if isinstance(v, float) else f"{k} = {v}"
            for k, v in params.items()
        ),
    }


def build_report(result: TrainResult, explanation: Explanation,
                 dataset_name: str = "", decision_question: str = "",
                 decision: DecisionAnalysis | None = None) -> str:
    """Return the full report as an HTML string."""
    context = build_context(result, explanation, dataset_name, decision_question, decision)
    return _env.get_template("report.html.j2").render(**context)


_RUN_ID = re.compile(r"^[0-9a-f]{32}$")


def save_report(html: str, run_id: str) -> Path:
    """Write the report to the outputs folder, named by the run id."""
    if not _RUN_ID.match(str(run_id)):
        raise ValueError("Invalid run id.")
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = config.OUTPUT_DIR / f"report_{run_id}.html"
    path.write_text(html, encoding="utf-8")
    return path
