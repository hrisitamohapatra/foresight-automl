"""Foresight AutoML: Streamlit app.

Start it with:
    venv\\Scripts\\python.exe -m streamlit run app.py

Security notes:
- The uploaded file stays in memory (Streamlit discards it when the session
  ends or the file is removed). It is never saved to disk.
- Uploaded text (column names, values) can contain Markdown or HTML. It is
  shown in tables (st.dataframe, which never interprets it), or escaped with
  md_escape() / chart_label() before being shown anywhere else.
- Errors show friendly messages only. .streamlit/config.toml also hides
  error details in the browser as a backstop.
"""

import html
import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from foresight import config
from foresight.explain import Explanation, explain_model
from foresight.ingest import (
    IngestError,
    detect_problem_type,
    load_csv,
    prepare_target,
    safe_display_name,
)
from foresight.models import get_metrics
from foresight.report import build_report, caveats, fmt, headline
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
    fig = go.Figure(go.Bar(
        x=[r.cv_mean[metric.key] for r in ok],
        y=[chart_label(r.name) for r in ok],
        orientation="h",
        marker_color=colors,
        error_x=dict(type="data", array=[r.cv_std[metric.key] for r in ok],
                     color=config.COLOR_TEXT),
    ))
    direction = "higher is better" if metric.higher_is_better else "lower is better"
    return _style(fig, len(ok), f"{metric.name}, cross-validation mean ± std ({direction})")


def importance_figure(explanation: Explanation, top: int = 10):
    table = explanation.importance.head(top)
    if explanation.shap_available:
        values, title = table["shap_share"] * 100, "Share of the model's reliance (%), SHAP"
    else:
        values, title = table["perm_mean"], "Score drop when shuffled (permutation)"
    fig = go.Figure(go.Bar(
        x=list(values),
        y=[chart_label(c) for c in table["column"]],
        orientation="h",
        marker_color=config.CHART_BLUES[1],
    ))
    return _style(fig, len(table), title)


# ---------------------------------------------------------------------------
# Page sections
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


def show_results(result: TrainResult, explanation: Explanation, report_html: str):
    st.header("Results")
    st.info(md_escape(headline(result)))

    metrics = get_metrics(result.problem_type)
    for col, m in zip(st.columns(len(metrics)), metrics):
        label = f"{m.name} (test)" + (" ★" if m.key == result.selection_metric else "")
        col.metric(label, fmt(result.test_scores[m.key]), help=m.description)
    st.caption("★ = metric used to pick the best model. The test set was used once, "
               "for these numbers only.")

    st.subheader("Model comparison")
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

    st.subheader("Caveats")
    st.markdown("\n".join(f"- {md_escape(c)}" for c in caveats(result, explanation)))

    st.download_button("Download the full report (HTML)", data=report_html,
                       file_name="foresight_report.html", mime="text/html", type="primary")


# ---------------------------------------------------------------------------
# Workflow after upload (separate so it can be tested without a file upload)
# ---------------------------------------------------------------------------
def workflow(df: pd.DataFrame, dataset_name: str, file_key: str):
    state = st.session_state
    st.markdown(f"**{md_escape(dataset_name)}**: {len(df):,} rows × {df.shape[1]} columns")
    with st.expander("Preview the first rows"):
        st.dataframe(df.head(20), width="stretch")

    # --- Step 2: what to predict ---------------------------------------------
    st.subheader("2. What do you want to predict?")
    target = st.selectbox("Target column", options=list(df.columns), index=None,
                          placeholder="Choose the column to predict")
    if target is None:
        return
    try:
        clean, n_dropped = prepare_target(df, target)
    except IngestError as err:
        st.error(str(err))
        return
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
        if not st.button("Run data checks"):
            return
        with st.spinner("Checking data quality and possible leakage..."):
            try:
                state.profile = preview_profile(clean, target, problem_type, positive_class)
            except TrainError as err:
                st.error(str(err))
                return
            except Exception:
                st.error(GENERIC_ERROR)
                return
    show_profile(state.profile)
    include = leakage_choices(state.profile)

    # --- Step 4: train -------------------------------------------------------
    st.subheader("4. Train and compare models")
    st.caption(f"{config.CV_FOLDS}-fold cross-validation on {1 - config.TEST_SIZE:.0%} of "
               f"the rows; the other {config.TEST_SIZE:.0%} is held out for one final test.")
    if st.button("Train models", type="primary"):
        bar = st.progress(0.0, text="Starting")
        try:
            result = run_training(clean, target, problem_type, positive_class,
                                  include_columns=include, n_dropped_target=n_dropped,
                                  on_progress=lambda msg, frac: bar.progress(frac, text=msg))
            bar.progress(1.0, text="Explaining the best model")
            explanation = explain_model(result)
            report_html = build_report(result, explanation, dataset_name, question)
            state.outputs = (result, explanation, report_html)
        except TrainError as err:
            st.error(str(err))
            return
        except Exception:
            st.error(GENERIC_ERROR)
            return
        finally:
            bar.empty()

    if state.get("outputs"):
        show_results(*state.outputs)


def main():
    st.set_page_config(page_title="Foresight AutoML", layout="wide")
    st.title("Foresight AutoML")
    st.caption("Upload a CSV, choose what to predict, and get a validated, "
               "explained baseline model with a plain-English report.")

    with st.sidebar:
        st.markdown("### How it works")
        st.markdown("1. Upload a CSV\n2. Choose the target column\n3. Review the data checks\n"
                    "4. Train and compare models\n5. Read and download the report")
        st.markdown("### Your data")
        st.caption("The file stays in memory on this computer and is discarded when you "
                   "close the page. Nothing is sent to external services.")
        st.caption(f"Limits: CSV only, up to {config.MAX_FILE_SIZE_MB} MB, "
                   f"{config.MAX_ROWS:,} rows and {config.MAX_COLUMNS} columns.")

    st.subheader("1. Upload your data")
    uploaded = st.file_uploader("CSV file", type=["csv"])
    state = st.session_state
    if uploaded is None:
        for key in ("file_key", "df", "dataset_name", "setup_key", "profile", "outputs"):
            state.pop(key, None)
        st.info("Upload a CSV file to begin.")
        return

    if state.get("file_key") != uploaded.file_id:
        try:
            state.df = load_csv(uploaded.getvalue(), uploaded.name)
        except IngestError as err:
            state.pop("df", None)
            st.error(str(err))
            return
        except Exception:
            st.error(GENERIC_ERROR)
            return
        state.file_key = uploaded.file_id
        state.dataset_name = safe_display_name(uploaded.name)

    workflow(state.df, state.dataset_name, state.file_key)


if __name__ == "__main__":
    main()
