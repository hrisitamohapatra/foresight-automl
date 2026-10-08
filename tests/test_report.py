"""Tests for foresight/report.py."""

import numpy as np
import pandas as pd
import pytest
from markupsafe import escape   # the same escaping Jinja2 autoescape uses

from foresight import config, report
from foresight.explain import explain_model
from foresight.report import build_report, fmt, save_report
from foresight.train import run_training

N = 300
XSS_COLUMN = "<script>alert('col')</script>"
XSS_CATEGORY = "<img src=x onerror=alert(1)>"
FORMULA_COLUMN = "=HYPERLINK(\"http://evil\")"


def churn_df(n=N):
    rng = np.random.default_rng(config.RANDOM_SEED)
    df = pd.DataFrame({
        "tenure": rng.normal(24, 10, size=n),
        XSS_COLUMN: rng.normal(size=n),
        FORMULA_COLUMN: rng.normal(size=n),
        "plan": rng.choice(["basic", XSS_CATEGORY], size=n),
    })
    logit = -0.15 * (df["tenure"] - 24) + 1.5 * (df["plan"] == "basic") - 1
    df["churn"] = np.where(logit + rng.logistic(size=n) > 0, "yes", "no")
    return df


@pytest.fixture(scope="module")
def binary_run():
    result = run_training(churn_df(), "churn", "binary", positive_class="yes")
    return result, explain_model(result)


@pytest.fixture(scope="module")
def binary_html(binary_run):
    result, exp = binary_run
    return build_report(result, exp, dataset_name="../../<b>evil</b>.csv",
                        decision_question="Who will churn? <script>steal()</script>")


# ---------------------------------------------------------------------------
# HTML in uploaded text is escaped
# ---------------------------------------------------------------------------
def test_autoescape_is_on():
    assert report._env.autoescape is True


def test_column_names_escaped(binary_html):
    assert XSS_COLUMN not in binary_html
    assert "<script>" not in binary_html
    assert str(escape(XSS_COLUMN)) in binary_html   # shown as harmless text


def test_category_values_escaped(binary_html):
    assert XSS_CATEGORY not in binary_html
    assert "<img src=x" not in binary_html


def test_dataset_name_and_question_escaped(binary_html):
    assert "<b>evil</b>" not in binary_html
    assert "steal()" in binary_html            # shown as text...
    assert "<script>steal()" not in binary_html  # ...not as a script
    assert "../" not in binary_html            # folders stripped from the name


def test_only_our_own_tags(binary_html):
    # The only <script>/<img> tags allowed are none and our two chart images.
    assert binary_html.count("<img ") == 2
    assert binary_html.count('src="data:image/png;base64,') == 2


# ---------------------------------------------------------------------------
# Formula injection guard in report tables
# ---------------------------------------------------------------------------
def test_formula_column_name_neutralized_in_tables(binary_html):
    raw_cell = f"<td>{escape(FORMULA_COLUMN)}</td>"
    safe_cell = f"<td>{escape(chr(39) + FORMULA_COLUMN)}</td>"   # chr(39) is '
    assert raw_cell not in binary_html
    assert safe_cell in binary_html


# ---------------------------------------------------------------------------
# Content comes from the real run
# ---------------------------------------------------------------------------
def test_report_shows_real_numbers(binary_run, binary_html):
    result, _ = binary_run
    assert fmt(result.test_scores["roc_auc"]) in binary_html
    best = result.best
    assert f"{fmt(best.cv_mean['roc_auc'])} ± {fmt(best.cv_std['roc_auc'])}" in binary_html
    assert f"{result.n_test:,} held-out rows" in binary_html


def test_report_sections(binary_html):
    for heading in ("The question", "How well the model does", "Model comparison",
                    "What the model relies on", "Data checks", "Caveats", "Method"):
        assert f"<h2>{heading}</h2>" in binary_html


def test_non_causal_disclaimer(binary_html):
    assert "not what causes the outcome" in binary_html


def test_palette_used(binary_html):
    for color in (config.COLOR_PRIMARY, config.COLOR_TEXT, config.COLOR_WARNING,
                  config.COLOR_SECONDARY_BACKGROUND):
        assert color in binary_html


def test_default_question(binary_run):
    result, exp = binary_run
    page = build_report(result, exp)
    assert "most likely to be &#39;yes&#39;" in page


def test_leaky_column_listed_with_reason():
    df = churn_df()
    df["refund_issued"] = (df["churn"] == "yes").astype(int)
    result = run_training(df, "churn", "binary", positive_class="yes")
    page = build_report(result, explain_model(result))
    assert "refund_issued" in page
    assert "predicts the target almost perfectly" in page


def test_override_creates_caveat():
    df = churn_df()
    df["refund_issued"] = (df["churn"] == "yes").astype(int)
    result = run_training(df, "churn", "binary", positive_class="yes",
                          include_columns=["refund_issued"])
    page = build_report(result, explain_model(result))
    assert "kept at the user&#39;s request" in page


def test_regression_report():
    rng = np.random.default_rng(config.RANDOM_SEED)
    df = pd.DataFrame({"size": rng.normal(100, 20, size=N), "rooms": rng.integers(1, 6, size=N)})
    df["price"] = 3000 * df["size"] + rng.normal(scale=20_000, size=N)
    result = run_training(df, "price", "regression")
    page = build_report(result, explain_model(result))
    assert "(lower is better)" in page
    assert "RMSE" in page and "R2" in page
    assert "higher predicted price" in page


def test_multiclass_report():
    rng = np.random.default_rng(config.RANDOM_SEED)
    df = pd.DataFrame({"x1": rng.normal(size=N), "x2": rng.normal(size=N)})
    df["tier"] = pd.cut(df["x1"] + rng.normal(scale=0.3, size=N), [-np.inf, -0.5, 0.5, np.inf],
                        labels=["bronze", "silver", "gold"]).astype(str)
    result = run_training(df, "tier", "multiclass")
    page = build_report(result, explain_model(result))
    assert "Macro F1" in page and "bronze" in page


def test_dollar_sign_column_does_not_break_charts():
    df = churn_df()
    df["$price$ in $"] = np.arange(N) % 7
    result = run_training(df, "churn", "binary", positive_class="yes")
    assert "<html" in build_report(result, explain_model(result))


# ---------------------------------------------------------------------------
# Decision section
# ---------------------------------------------------------------------------
def test_decision_section(binary_run):
    from foresight.decision import analyze_decision

    result, exp = binary_run
    decision = analyze_decision(result, cost_fp=1, cost_fn=5)
    page = build_report(result, exp, decision=decision)
    assert "<h2>Turning scores into decisions</h2>" in page
    assert f"{decision.threshold:.2f}" in page
    assert page.count('src="data:image/png;base64,') == 4   # + reliability + gains
    chosen = decision.test_chosen
    assert f"{chosen.tp} of {chosen.tp + chosen.fn}" in page
    assert "costing 5" in page and "costing 1" in page


def test_decision_section_absent_without_analysis(binary_html):
    assert "Turning scores into decisions" not in binary_html


def test_decision_not_applicable_shows_reason():
    from foresight.decision import analyze_decision

    rng = np.random.default_rng(config.RANDOM_SEED)
    df = pd.DataFrame({"size": rng.normal(100, 20, size=N)})
    df["price"] = 3 * df["size"] + rng.normal(scale=10, size=N)
    result = run_training(df, "price", "regression")
    page = build_report(result, explain_model(result), decision=analyze_decision(result))
    assert "apply to yes/no predictions only" in page


def test_positive_class_html_escaped_in_decision_section():
    from foresight.decision import analyze_decision

    df = churn_df()
    df["churn"] = df["churn"].map({"yes": "<i>left</i>", "no": "stayed"})
    result = run_training(df, "churn", "binary", positive_class="<i>left</i>")
    page = build_report(result, explain_model(result), decision=analyze_decision(result))
    assert "<i>left</i>" not in page
    assert str(escape("<i>left</i>")) in page


# ---------------------------------------------------------------------------
# Formatting and saving
# ---------------------------------------------------------------------------
def test_fmt():
    assert fmt(0.84213) == "0.842"
    assert fmt(12345.6) == "12,346"
    assert fmt(-0.5) == "-0.500"
    assert fmt(-0.0001) == "0.000"


def test_save_report(binary_html, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    path = save_report(binary_html, "a" * 32)
    assert path == tmp_path / f"report_{'a' * 32}.html"
    assert path.read_text(encoding="utf-8") == binary_html


@pytest.mark.parametrize("bad_id", ["../evil", "", "x" * 32])
def test_save_report_rejects_bad_ids(bad_id, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    with pytest.raises(ValueError):
        save_report("<html></html>", bad_id)
    assert list(tmp_path.iterdir()) == []
