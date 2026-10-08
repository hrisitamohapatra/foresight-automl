"""Foresight AutoML: Streamlit app.

Start it with:
    venv\\Scripts\\python.exe -m streamlit run app.py

Security notes:
- The uploaded file stays in memory (Streamlit discards it when the session
  ends or the file is removed). It is never saved to disk.
- Uploaded text (column names, values) can contain Markdown or HTML. It is
  shown in tables (st.dataframe, which never interprets it), or escaped with
  md_escape() / chart_label() before being shown anywhere else.
- Raw HTML is used ONLY for the fixed page style and header below
  (PAGE_STYLE, HEADER_HTML), which are constants written in this file and
  never contain uploaded text. A test enforces this.
- Errors show friendly messages only. .streamlit/config.toml also hides
  error details in the browser as a backstop.
"""

import html
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from foresight import config
from foresight.decision import DecisionAnalysis, DecisionError, analyze_decision
from foresight.explain import Explanation, explain_model
from foresight.ingest import (
    IngestError,
    detect_problem_type,
    load_csv,
    prepare_target,
    safe_display_name,
)
from foresight.models import get_metrics
from foresight.narrate import Narrative, api_key, summarize
from foresight.report import build_report, caveats, decision_context, fmt, headline
from foresight.train import (
    TrainError,
    TrainResult,
    default_positive_class,
    preview_profile,
    run_training,
)

GENERIC_ERROR = ("Something went wrong while processing this data. "
                 "Please check the file and try again.")

PROBLEM_NAMES = {
    "binary": "yes/no prediction (binary classification)",
    "multiclass": "category prediction (multiclass classification)",
    "regression": "number prediction (regression)",
}
KIND_NAMES = {"numeric": "Number", "numeric_text": "Number (stored as text)",
              "categorical": "Category", "datetime": "Date"}
FLAG_NAMES = {"high_missing": "many missing values", "high_cardinality": "many categories",
              "numbers_as_text": "numbers stored as text", "user_included": "kept by you",
              "strong_single_predictor": "very strong on its own: check for leakage"}

STEPS = ["Upload data", "Choose target", "Check data", "Train models", "Results"]
DEMO_NAME = "demo_churn (synthetic).csv"

# ---------------------------------------------------------------------------
# Fixed page style and header. The ONLY raw HTML in the app: constants that
# never contain uploaded text (enforced by tests/test_app.py).
# ---------------------------------------------------------------------------
PAGE_STYLE = """
<style>
  .block-container { padding-top: 0; max-width: 1240px; }
  header[data-testid="stHeader"] { background: transparent; }
  /* Let the hero and the top menu span the whole main area while the content
     column keeps its width: measure the main area as a container, then pull
     these two elements out to its edges (cqw = % of that width). */
  [data-testid="stMain"] { container-type: inline-size; }

  /* Hero header: light-to-deep-blue gradient with a fine vertical line texture.
     Square corners on purpose. Fonts are system fonts (no web-font downloads). */
  .fs-hero {
    position: relative; border-radius: 0; overflow: hidden;
    margin-inline: calc(50% - 50cqw);            /* full width of the main area */
    min-height: 330px; padding: 34px 34px 38px;
    display: flex; flex-direction: column; justify-content: flex-end;
    background-image:
      repeating-linear-gradient(90deg, rgba(255,255,255,0.07) 0 1px, transparent 1px 7px),
      linear-gradient(180deg, #FFFFFF 0%, #DCE6F5 10%, #8DB3EE 30%, #3E7BDA 55%,
                      #1E64D4 76%, #0B3D91 100%);
    box-shadow: 0 18px 40px -24px rgba(11, 61, 145, 0.55);
  }
  .fs-body { text-align: center; color: #FFFFFF; }
  .fs-badge {
    display: inline-block; padding: 4px 12px; margin-top: 14px; font-size: 0.78rem;
    letter-spacing: 0.4px; color: #0B3D91; background: rgba(255,255,255,0.72);
    border: 1px solid rgba(30,100,212,0.35); border-radius: 0;
  }
  .fs-title { margin: 0; font-size: 3rem; font-weight: 600; line-height: 1.1;
              letter-spacing: -0.8px; color: #FFFFFF;
              text-shadow: 0 2px 18px rgba(11, 61, 145, 0.45); }
  .fs-title em { font-family: Georgia, "Times New Roman", serif; font-weight: 400;
                 font-style: italic; letter-spacing: -0.2px; }
  .fs-sub { margin: 12px auto 0; max-width: 640px; font-size: 0.98rem;
            line-height: 1.55; color: rgba(255,255,255,0.86); }
  @media (max-width: 640px) {
    .fs-hero { padding: 24px 20px 28px; min-height: 260px; }
    .fs-title { font-size: 2.1rem; }
  }

  /* Top menu (Analyze / About): centered, uppercase, square underline. */
  .stTabs [role="tablist"] { display: flex; justify-content: center; gap: 4px;
                             border-bottom: 1px solid #DCE6F5;
                             margin-inline: calc(50% - 50cqw);   /* full width */
                             padding-inline: calc(50cqw - 50%); }
  .stTabs [data-testid="stTab"] { padding: 10px 22px; border-radius: 0; }
  .stTabs [data-testid="stTab"] p { font-size: 0.86rem; font-weight: 600;
                                    letter-spacing: 1.4px; text-transform: uppercase; }
  /* Tabs inside the results keep normal, left-aligned labels. */
  .stTabs .stTabs [role="tablist"] { justify-content: flex-start;
                                     margin-inline: 0; padding-inline: 0; }
  .stTabs .stTabs [data-testid="stTab"] p { font-size: 0.95rem; letter-spacing: 0;
                                            text-transform: none; }

  [data-testid="stMetric"] {
    background: #F3F7FD; border: 1px solid #DCE6F5; border-left: 3px solid #1E64D4;
    border-radius: 0; padding: 12px 16px;
  }
  h2, h3, h4 { color: #0B3D91; }
</style>
"""

HEADER_HTML = """
<div class="fs-hero">
  <div class="fs-body">
    <h1 class="fs-title">Foresight <em>AutoML</em></h1>
    <span class="fs-badge">Dataset-agnostic &middot; Explainable &middot; Leakage-safe</span>
    <p class="fs-sub">Upload any CSV and get a validated, explained predictive model
      and a decision-ready report in minutes.</p>
  </div>
</div>
"""


# ---------------------------------------------------------------------------
# Safe display helpers
# ---------------------------------------------------------------------------
_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>~$=:&])")


def md_escape(text) -> str:
    """Show uploaded text literally inside Markdown.

    Backslash-escapes every Markdown/HTML special character, so text such as
    "![x](http://evil/img.png)" or "**bold**" cannot become an image, a link,
    or formatting. ":" and "." are escaped so URLs are not auto-linked.
    """
    text = str(text).replace("\r", " ").replace("\n", " ")
    return _MD_SPECIAL.sub(r"\\\1", text)


def chart_label(text, max_length: int = 60) -> str:
    """Plotly understands a few HTML tags in labels; show them as text."""
    text = str(text)
    if len(text) > max_length:
        text = text[: max_length - 1] + "…"
    return html.escape(text)


def centered():
    """A centered column (about 60% of the page) for the setup steps."""
    _, middle, _ = st.columns([1, 2.6, 1])
    return middle


# ---------------------------------------------------------------------------
# Demo data (synthetic, generated with a fixed seed: not real customers)
# ---------------------------------------------------------------------------
def make_demo_csv(n: int = 1500) -> bytes:
    """A synthetic churn dataset with a few realistic traps:
    an ID column, numbers stored as text with blanks, and a leaky column."""
    rng = np.random.default_rng(config.RANDOM_SEED)
    tenure = rng.integers(1, 72, n)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
    internet = rng.choice(["Fiber optic", "DSL", "None"], n, p=[0.45, 0.40, 0.15])
    calls = rng.poisson(1.6, n)
    monthly = np.round(np.clip(rng.normal(65, 20, n), 15, 130), 2)
    logit = (-2.0 + 1.6 * (contract == "Month-to-month") - 0.035 * tenure + 0.45 * calls
             + 0.012 * (monthly - 65) + 0.5 * (internet == "Fiber optic"))
    churned = rng.uniform(size=n) < 1 / (1 + np.exp(-logit))
    total = (monthly * tenure).round(2).astype(str).astype(object)
    total[rng.choice(n, 12, replace=False)] = " "          # blanks, as in real exports
    df = pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(n)],
        "tenure_months": tenure,
        "contract": contract,
        "internet_service": internet,
        "monthly_charges": monthly,
        "total_charges": total,
        "support_calls": calls,
        "refund_issued": churned.astype(int),               # only known after churn: a leak
        "churned": np.where(churned, "Yes", "No"),
    })
    return df.to_csv(index=False).encode("utf-8")


# ---------------------------------------------------------------------------
# Charts (shades of blue)
# ---------------------------------------------------------------------------
def _style(fig, n_bars: int, xaxis_title: str):
    fig.update_layout(
        template="plotly_white",
        height=90 + 38 * n_bars,
        margin=dict(l=10, r=10, t=10, b=40),
        font=dict(color=config.COLOR_TEXT),
        xaxis_title=xaxis_title,
        yaxis=dict(autorange="reversed"),   # first item at the top
        showlegend=False,
    )
    return fig


def model_figure(result: TrainResult):
    metric = next(m for m in get_metrics(result.problem_type)
                  if m.key == result.selection_metric)
    ok = [r for r in result.model_results if not r.error]
    colors = [config.CHART_BLUES[0] if r.key == result.best_key else
              config.CHART_BLUES[4] if r.role == "baseline" else config.CHART_BLUES[2]
              for r in ok]
    stds = [r.cv_std[metric.key] for r in ok]
    fig = go.Figure(go.Bar(
        x=[r.cv_mean[metric.key] for r in ok],
        y=[chart_label(r.name) for r in ok],
        orientation="h",
        marker_color=colors,
        error_x=dict(type="data", array=stds, color=config.COLOR_TEXT),
        # Readable hover text: "LightGBM: 0.443 ± 0.011" (not "± 274μ").
        customdata=stds,
        hovertemplate="%{y}: %{x:.3f} ± %{customdata:.3f}<extra></extra>",
    ))
    direction = "higher is better" if metric.higher_is_better else "lower is better"
    return _style(fig, len(ok), f"{metric.name}, cross-validation mean ± std ({direction})")


def reliability_figure(decision: DecisionAnalysis):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfectly reliable",
                             line=dict(dash="dash", color=config.CHART_BLUES[4])))
    colors = {"Original": config.CHART_BLUES[2], "Calibrated": config.CHART_BLUES[0]}
    for name, (predicted, actual) in decision.reliability.items():
        fig.add_trace(go.Scatter(x=predicted, y=actual, mode="lines+markers", name=name,
                                 line=dict(color=colors[name])))
    fig.update_layout(template="plotly_white", height=380, margin=dict(l=10, r=10, t=10, b=40),
                      font=dict(color=config.COLOR_TEXT),
                      xaxis_title="Predicted probability (test rows, grouped)",
                      yaxis_title="Actual share positive")
    return fig


def gains_figure(decision: DecisionAnalysis):
    gains = decision.gains
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", name="Random order",
                             line=dict(dash="dash", color=config.CHART_BLUES[4])))
    fig.add_trace(go.Scatter(x=gains["pct_acted_on"], y=gains["pct_positives_reached"],
                             mode="lines", name="Model order",
                             line=dict(color=config.CHART_BLUES[0])))
    fig.update_layout(template="plotly_white", height=380, margin=dict(l=10, r=10, t=10, b=40),
                      font=dict(color=config.COLOR_TEXT),
                      xaxis_title="% of rows acted on (highest risk first)",
                      yaxis_title=f"% of '{chart_label(decision.positive_class)}' cases reached")
    return fig


def importance_figure(explanation: Explanation, top: int = 10):
    table = explanation.importance.head(top)
    if explanation.shap_available:
        values, title = table["shap_share"] * 100, "Share of the model's reliance (%), SHAP"
        hover = "%{y}: %{x:.1f}%<extra></extra>"
    else:
        values, title = table["perm_mean"], "Score drop when shuffled (permutation)"
        hover = "%{y}: %{x:.3f}<extra></extra>"
    fig = go.Figure(go.Bar(
        x=list(values),
        y=[chart_label(c) for c in table["column"]],
        orientation="h",
        marker_color=config.CHART_BLUES[1],
        hovertemplate=hover,
    ))
    return _style(fig, len(table), title)


# ---------------------------------------------------------------------------
# Sidebar: step tracker and data notes
# ---------------------------------------------------------------------------
def render_steps(current: int):
    """Done steps get a tick, the current one is highlighted, the rest are grey.
    `current` is an index into STEPS (len(STEPS) means everything is done)."""
    st.markdown("#### Progress")
    lines = []
    for i, step in enumerate(STEPS):
        if i < current:
            lines.append(f":blue[:material/check_circle:] {step}")
        elif i == current:
            lines.append(f":blue[:material/radio_button_checked: **{step}**]")
        else:
            lines.append(f":gray[:material/radio_button_unchecked: {step}]")
    st.markdown("  \n".join(lines))


def render_sidebar_notes():
    st.markdown("#### Your data")
    st.caption("The file stays in memory on this computer and is discarded when you close "
               "the page. Nothing leaves this computer unless you switch on the Gemini "
               "summary, which receives aggregated results only, never your data.")
    st.caption(f"Limits: CSV only, up to {config.MAX_FILE_SIZE_MB} MB, "
               f"{config.MAX_ROWS:,} rows and {config.MAX_COLUMNS} columns.")


# ---------------------------------------------------------------------------
# Result sections
# ---------------------------------------------------------------------------
def show_profile(profile):
    rows = []
    for c in profile.columns:
        notes = [FLAG_NAMES[f] for f in c.flags if f in FLAG_NAMES]
        rows.append({
            "Column": c.name,
            "Type": KIND_NAMES.get(c.kind, c.kind),
            "Missing": f"{c.missing_share:.0%}",
            "Distinct values": c.n_unique,
            "Used": "Yes" if c.use else "No",
            "Note": c.reason or ", ".join(notes),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    if profile.class_balance:
        st.markdown("**Class balance** (training rows)")
        st.dataframe(pd.DataFrame({"Class": list(profile.class_balance),
                                   "Share": [f"{v:.1%}" for v in profile.class_balance.values()]}),
                     hide_index=True)
    for warning in profile.warnings:
        st.warning(md_escape(warning))


def leakage_choices(profile) -> list[str]:
    """Checkboxes to keep columns flagged as possible leakage."""
    if not profile.leaky_columns:
        return []
    st.markdown("**Possible target leakage.** These columns predict the target almost "
                "perfectly on their own, so they were left out. Tick one only if you are "
                "sure its value is known *before* the outcome happens.")
    return [name for name in profile.leaky_columns
            if st.checkbox(f"Keep {md_escape(name)} anyway", key=f"keep::{name}")]


def show_decision(decision: DecisionAnalysis):
    st.subheader("Turning scores into decisions")
    text = decision_context(decision, charts=False)
    if not text["applicable"]:
        st.info(md_escape(text["reason"]))
        return

    chosen, default = decision.test_chosen, decision.test_default
    fp, fn = decision.cost_fp, decision.cost_fn
    cols = st.columns(3)
    cols[0].metric("Chosen cut-off", f"{decision.threshold:.2f}",
                   help="Chosen on training data only, to minimise your costs.")
    cols[1].metric("Positive cases caught (test)", f"{chosen.recall:.0%}",
                   help="Share of all positive test rows that the cut-off flags.")
    cols[2].metric("Cost per 1,000 rows (test)", f"{chosen.cost_per_1000(fp, fn):,.0f}",
                   delta=f"{chosen.cost_per_1000(fp, fn) - default.cost_per_1000(fp, fn):,.0f} "
                         "vs cut-off 0.50",
                   delta_color="inverse")
    st.markdown(md_escape(f"{text['costs']} {text['threshold']} {text['comparison']}"))
    st.dataframe(pd.DataFrame([{
        "Strategy (test set)": r["name"], "Rows flagged": r["flagged"],
        "Positive cases caught": r["caught"], "False alarms": r["false_alarms"],
        "Missed": r["missed"], "Cost per 1,000 rows": r["cost"],
    } for r in text["rows"]]), hide_index=True, width="stretch")

    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.markdown("**Can the probabilities be trusted?**")
            st.plotly_chart(reliability_figure(decision), width="stretch")
            st.caption(md_escape(text["calibration"]))
    with right:
        with st.container(border=True):
            st.markdown("**Who to act on first**")
            st.plotly_chart(gains_figure(decision), width="stretch")
            st.markdown("\n".join(f"- {md_escape(line)}" for line in text["gains_lines"]))


def show_results(result: TrainResult, explanation: Explanation,
                 decision: DecisionAnalysis, narrative: Narrative, report_html: str):
    st.divider()
    st.header("Results")
    metrics = get_metrics(result.problem_type)
    overview, models, decisions, drivers, data = st.tabs(
        ["Overview", "Models", "Decisions", "What the model relies on", "Data & caveats"])

    with overview:
        st.info(md_escape(headline(result)))
        for col, m in zip(st.columns(len(metrics)), metrics):
            label = f"{m.name} (test)" + (" ★" if m.key == result.selection_metric else "")
            col.metric(label, fmt(result.test_scores[m.key]), help=m.description)
        st.caption("★ = metric used to pick the best model. The test set was used once, "
                   "for these numbers only.")
        with st.container(border=True):
            st.subheader("Summary")
            st.markdown(md_escape(narrative.text))
            st.caption("Written by Gemini from aggregated results only, and checked against them."
                       if narrative.source == "gemini"
                       else "Template summary, written directly from the results.")
            if narrative.note:
                st.warning(md_escape(narrative.note))
        st.download_button("Download the full report (HTML)", data=report_html,
                           file_name="foresight_report.html", mime="text/html", type="primary",
                           icon=":material/download:")
        st.caption("Need a PDF? Open the downloaded report and press Ctrl+P "
                   "(Cmd+P on a Mac), then choose Save as PDF.")

    with models:
        st.subheader("Model comparison")
        st.caption(f"{config.CV_FOLDS}-fold cross-validation on the training data "
                   "(mean ± standard deviation). The darkest bar is the chosen model.")
        st.plotly_chart(model_figure(result), width="stretch")
        rows = []
        for r in result.model_results:
            row = {"Model": r.name + (" (best)" if r.key == result.best_key else ""),
                   "Role": r.role}
            for m in metrics:
                row[m.name] = (f"{fmt(r.cv_mean[m.key])} ± {fmt(r.cv_std[m.key])}"
                               if not r.error else r.error)
            row["Seconds per fold"] = round(r.fit_seconds, 2)
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    with decisions:
        show_decision(decision)

    with drivers:
        st.subheader("What the model relies on")
        st.caption("These show what the model relies on, not what causes the outcome.")
        st.plotly_chart(importance_figure(explanation), width="stretch")
        table = explanation.importance.head(10)
        st.dataframe(pd.DataFrame({
            "Column": table["column"],
            "Share of reliance (SHAP)": [f"{v:.0%}" if explanation.shap_available else "-"
                                         for v in table["shap_share"]],
            "Score drop when shuffled": [f"{fmt(m)} ± {fmt(s)}"
                                         for m, s in zip(table["perm_mean"], table["perm_std"])],
            "Pattern": table["direction"],
        }), hide_index=True, width="stretch")
        st.caption(md_escape(explanation.agreement_text))
        for note in explanation.notes:
            st.warning(md_escape(note))

    with data:
        st.subheader("Caveats")
        st.markdown("\n".join(f"- {md_escape(c)}" for c in caveats(result, explanation)))
        st.subheader("Data checks")
        show_profile(result.profile)


# ---------------------------------------------------------------------------
# Setup steps (centered) and the workflow after upload
# ---------------------------------------------------------------------------
def setup_steps(df: pd.DataFrame, dataset_name: str, file_key: str) -> int:
    """Target, data checks, options and training. Returns the current step index."""
    state = st.session_state
    st.markdown(f"**{md_escape(dataset_name)}**: {len(df):,} rows × {df.shape[1]} columns")
    with st.expander("Preview the first rows"):
        st.dataframe(df.head(20), width="stretch")

    # --- Step 2: what to predict ---------------------------------------------
    st.subheader("2. What do you want to predict?")
    target = st.selectbox("Target column", options=list(df.columns), index=None,
                          placeholder="Choose the column to predict")
    if target is None:
        return 1
    try:
        clean, n_dropped = prepare_target(df, target)
    except IngestError as err:
        st.error(str(err))
        return 1
    if n_dropped:
        st.warning(f"{n_dropped:,} rows have no value for the target and will be left out.")

    detected = detect_problem_type(clean[target])
    if detected.ambiguous:
        st.warning(f"{detected.reason} Please choose how to treat it.")
        options = ["Categories (classification)", "Amounts (regression)"]
        choice = st.radio("Treat the target as", options,
                          index=0 if detected.suggestion == "multiclass" else 1)
        problem_type = "multiclass" if choice == options[0] else "regression"
    else:
        problem_type = detected.problem_type
        st.markdown(f"Detected a **{PROBLEM_NAMES[problem_type]}** task. {detected.reason}")

    positive_class = None
    if problem_type == "binary":
        labels = sorted(clean[target].unique(), key=str)
        default = default_positive_class(clean[target])
        positive_class = st.selectbox(
            "Which outcome is the event to predict? (positive class)", labels,
            index=labels.index(default), format_func=str,
            help="Usually the rarer, more important outcome, such as 'churned' or "
                 "'defaulted'. Defaults to the rarer class.")

    question = st.text_input("Decision question (optional)", max_chars=300,
                             placeholder="e.g. Which customers are most likely to churn, and why?")

    # Any change above invalidates earlier checks and results.
    setup_key = (file_key, target, problem_type, str(positive_class))
    if state.get("setup_key") != setup_key:
        state.setup_key = setup_key
        state.profile = None
        state.outputs = None

    # --- Step 3: data checks ---------------------------------------------------
    st.subheader("3. Check the data")
    if state.profile is None:
        if not st.button("Run data checks", icon=":material/fact_check:"):
            return 2
        with st.spinner("Checking data quality and possible leakage..."):
            try:
                state.profile = preview_profile(clean, target, problem_type, positive_class)
            except TrainError as err:
                st.error(str(err))
                return 2
            except Exception:
                st.error(GENERIC_ERROR)
                return 2
    show_profile(state.profile)
    include = leakage_choices(state.profile)

    # --- Step 4: options and training ------------------------------------------
    st.subheader("4. Train and compare models")
    st.caption(f"{config.CV_FOLDS}-fold cross-validation on {1 - config.TEST_SIZE:.0%} of "
               f"the rows; the other {config.TEST_SIZE:.0%} is held out for one final test.")

    with st.expander("Options: decision costs, tuning, AI summary, run log",
                     icon=":material/tune:"):
        cost_fp, cost_fn = 1.0, 1.0
        if problem_type == "binary":
            pos = md_escape(positive_class)
            st.markdown("**What do mistakes cost?** Used to choose the cut-off for flagging "
                        "rows. Any unit works (money, hours, points); only the ratio matters.")
            left, right = st.columns(2)
            cost_fn = left.number_input(f"Cost of missing a '{pos}' case", min_value=0.01,
                                        max_value=float(config.MAX_COST), value=1.0, step=1.0)
            cost_fp = right.number_input(
                f"Cost of a false alarm (flagging a row that is not '{pos}')",
                min_value=0.01, max_value=float(config.MAX_COST), value=1.0, step=1.0)
            st.divider()

        tune = st.checkbox(
            "Also tune random forest and LightGBM (slower)",
            help=f"Runs an Optuna search (up to {config.TUNE_TRIALS} trials) inside each "
                 f"cross-validation fold, so tuned scores stay honest. Can take several "
                 f"minutes on large files.")

        has_key = api_key() is not None
        use_gemini = st.checkbox(
            "Write the summary with Gemini (sends aggregated results to Google, never your data)",
            disabled=not has_key,
            help=("Only scores, model names and importance shares are sent. Column names and "
                  "values are replaced by placeholders such as [COLUMN_1]. The answer is "
                  "checked against the real results; if it fails, a template summary is used."
                  if has_key else "Add GEMINI_API_KEY to the .env file in the project folder "
                                  "to enable this."))
        if has_key:
            st.caption("Note: on Google's free tier, requests may be used to improve Google's "
                       "products. Only aggregated, placeholder-masked results are sent.")

        track = st.checkbox(
            "Save this run to the local MLflow log",
            help="Stores settings, scores, the report and the model in mlflow.db and mlruns/ "
                 "in the project folder, so runs can be compared later. Nothing is sent online.")

    if st.button("Train models", type="primary", icon=":material/play_arrow:",
                 width="stretch"):
        bar = st.progress(0.0, text="Starting")
        try:
            result = run_training(clean, target, problem_type, positive_class,
                                  include_columns=include, n_dropped_target=n_dropped,
                                  on_progress=lambda msg, frac: bar.progress(frac, text=msg),
                                  tune=tune)
            bar.progress(1.0, text="Explaining the model and choosing a cut-off")
            explanation = explain_model(result)
            decision = analyze_decision(result, cost_fp=float(cost_fp), cost_fn=float(cost_fn))
            if use_gemini:
                bar.progress(1.0, text="Writing the summary with Gemini")
            narrative = summarize(result, explanation, decision, use_gemini=use_gemini)
            report_html = build_report(result, explanation, dataset_name, question,
                                       decision, narrative)
            state.outputs = (result, explanation, decision, narrative, report_html)
            if track:
                # Imported only when needed: tracking.py switches off MLflow's
                # internet telemetry before MLflow itself is loaded.
                from foresight import tracking
                tracking.log_run(result, explanation, decision, report_html, dataset_name)
                st.toast("Run saved to the local MLflow log.")
        except (TrainError, DecisionError) as err:
            st.error(str(err))
            return 3
        except Exception:
            st.error(GENERIC_ERROR)
            return 3
        finally:
            bar.empty()
    return 3


def workflow(df: pd.DataFrame, dataset_name: str, file_key: str) -> int:
    """Everything after the upload. Setup is centered; results use the full width.
    Returns the current step index (for the progress tracker)."""
    with centered():
        step = setup_steps(df, dataset_name, file_key)
    if step == 3 and st.session_state.get("outputs"):
        show_results(*st.session_state.outputs)
        return len(STEPS)
    return step


# ---------------------------------------------------------------------------
# About tab
# ---------------------------------------------------------------------------
def render_about():
    with centered():
        st.subheader("About Foresight AutoML")
        st.markdown(
            "Foresight AutoML turns any tabular CSV into a **validated predictive model** and "
            "a **decision-ready report** in minutes. It is built for teams who need a reliable, "
            "explained baseline model for a new dataset without writing code.")
        st.markdown("#### How it works")
        st.markdown(
            "1. **Ingest and validate**: the file is checked for size, format and a usable target.\n"
            "2. **Profile**: column types, missing values, ID-like columns and likely target "
            "leakage are flagged.\n"
            "3. **Train and compare**: a baseline, a simple linear model, random forest and "
            "LightGBM are compared with 5-fold cross-validation (optional Optuna tuning uses "
            "nested cross-validation).\n"
            "4. **Test once**: the best model is evaluated a single time on a held-out test set.\n"
            "5. **Explain and decide**: SHAP and permutation importance show what the model "
            "relies on; calibration and a cost-based cut-off turn scores into decisions.\n"
            "6. **Report**: a plain-English HTML report with clear caveats.")
        st.markdown("#### Rigor")
        st.markdown(
            "- Every model is compared against a baseline.\n"
            "- All preprocessing is learned inside the training folds only, so test data never leaks.\n"
            "- Results are reported as mean ± standard deviation, and every number comes from a real run.\n"
            "- Explanations describe what the model relies on, never causes.")
        st.markdown("#### Privacy and safety")
        st.markdown(
            "- Uploaded files stay in memory and are never saved to disk.\n"
            "- Uploaded text is always treated as text, never as code or HTML.\n"
            "- The optional Gemini summary receives aggregated results only, with column names "
            "replaced by placeholders, and its numbers are checked against the real results.\n"
            "- Optional MLflow run tracking stays on this computer.")
        st.markdown("#### Limitations")
        st.markdown(
            "- The leakage check looks at one column at a time; leaks spread across columns or "
            "caused by timing still need a human review.\n"
            "- Date columns are not used as model inputs yet.\n"
            "- Results assume future data looks like the data the model was trained on.")
        st.markdown("#### Built with")
        st.caption("Python · scikit-learn · LightGBM · SHAP · Optuna · MLflow · Streamlit · "
                   "Plotly · Jinja2 · Google Gemini (optional)")
        st.divider()
        st.markdown("Built by **Hrisita Mohapatra** · "
                    "[GitHub repository](https://github.com/hrisitamohapatra/foresight-automl)")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def analyze_page(progress) -> None:
    state = st.session_state
    with centered():
        with st.container(border=True):
            st.subheader("1. Upload your data")
            uploaded = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")
            if uploaded is None and not state.get("demo"):
                left, right = st.columns([3, 2], vertical_alignment="center")
                left.caption("No file to hand? Try a synthetic churn dataset "
                             "(randomly generated, not real customers).")
                if right.button("Try demo data", icon=":material/science:", width="stretch"):
                    state.demo = True
                    st.rerun()   # redraw at once so the card shows "Remove demo data"
            elif uploaded is None and state.get("demo"):
                left, right = st.columns([3, 2], vertical_alignment="center")
                left.caption("Using the synthetic demo dataset.")
                if right.button("Remove demo data", icon=":material/close:", width="stretch"):
                    state.demo = False
                    st.rerun()

    # Decide which data to use: a real upload always wins over the demo.
    if uploaded is not None:
        state.demo = False
        source_key, name, data = uploaded.file_id, uploaded.name, None
    elif state.get("demo"):
        source_key, name, data = "demo", DEMO_NAME, make_demo_csv()
    else:
        for key in ("file_key", "df", "dataset_name", "setup_key", "profile", "outputs"):
            state.pop(key, None)
        with centered():
            st.info("Upload a CSV file, or try the demo data, to begin.")
        with progress:
            render_steps(0)
        return

    if state.get("file_key") != source_key:
        try:
            raw = uploaded.getvalue() if data is None else data
            state.df = load_csv(raw, name)
        except IngestError as err:
            state.pop("df", None)
            with centered():
                st.error(str(err))
            with progress:
                render_steps(0)
            return
        except Exception:
            with centered():
                st.error(GENERIC_ERROR)
            with progress:
                render_steps(0)
            return
        state.file_key = source_key
        state.dataset_name = safe_display_name(name)

    step = workflow(state.df, state.dataset_name, state.file_key)
    with progress:
        render_steps(step)


def main():
    st.set_page_config(page_title="Foresight AutoML", layout="wide",
                       page_icon=":material/insights:")
    st.html(PAGE_STYLE)
    st.html(HEADER_HTML)

    with st.sidebar:
        progress = st.container()        # filled in once we know the current step
        st.divider()
        render_sidebar_notes()

    analyze, about = st.tabs(["Analyze", "About"])
    with analyze:
        analyze_page(progress)
    with about:
        render_about()


if __name__ == "__main__":
    main()
