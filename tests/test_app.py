"""Tests for app.py (the Streamlit UI).

Streamlit's AppTest cannot simulate a file upload, so the upload screen gets a
smoke test and the rest of the flow is driven through app.workflow() with a
synthetic DataFrame.
"""

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import app
from foresight import config

APP_PATH = str(config.PROJECT_ROOT / "app.py")
XSS = "![x](http://evil.example/t.png) <script>alert(1)</script> **bold**"


# ---------------------------------------------------------------------------
# Escaping helpers
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text", [
    "![x](http://evil.example/t.png)", "[click](http://evil.example)",
    "**bold**", "<script>alert(1)</script>", "$x^2$", "# heading", "http://evil.example",
])
def test_md_escape_neutralizes_markdown(text):
    escaped = app.md_escape(text)
    # Every special character is preceded by a backslash.
    for i, ch in enumerate(escaped):
        if ch in "![]()*<>$#:" and (i == 0 or escaped[i - 1] != "\\"):
            pytest.fail(f"unescaped {ch!r} in {escaped!r}")


def test_md_escape_keeps_plain_text_readable():
    assert app.md_escape("tenure months") == "tenure months"
    assert app.md_escape("line1\nline2") == "line1 line2"


def test_chart_label_escapes_html_and_truncates():
    assert "<" not in app.chart_label("<b>x</b>")
    assert len(app.chart_label("a" * 200)) == 60


ALLOWED_HTML_CONSTANTS = {"PAGE_STYLE", "HEADER_HTML"}


def test_raw_html_only_from_fixed_constants():
    """st.html may only receive PAGE_STYLE or HEADER_HTML, and those must be
    plain string literals written in app.py (no f-strings, no data)."""
    import ast

    source = (config.PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
    assert "unsafe_allow_html" not in source
    assert "components.html" not in source and "components.v1" not in source
    tree = ast.parse(source)

    html_calls = [node for node in ast.walk(tree)
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                  and node.func.attr == "html"]
    assert len(html_calls) == 2
    for call in html_calls:
        assert len(call.args) == 1 and not call.keywords
        assert isinstance(call.args[0], ast.Name), "st.html must get a named constant"
        assert call.args[0].id in ALLOWED_HTML_CONSTANTS

    # Each constant is assigned exactly once, at module level, to a plain string.
    assignments = [node for node in tree.body
                   if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                   and node.targets[0].id in ALLOWED_HTML_CONSTANTS]
    assert sorted(a.targets[0].id for a in assignments) == sorted(ALLOWED_HTML_CONSTANTS)
    for a in assignments:
        assert isinstance(a.value, ast.Constant) and isinstance(a.value.value, str)


def test_demo_data_is_valid_and_reproducible():
    from foresight.ingest import load_csv

    raw = app.make_demo_csv()
    assert raw == app.make_demo_csv()                  # fixed seed
    df = load_csv(raw, app.DEMO_NAME)                  # passes the real upload checks
    assert len(df) == 1500 and "churned" in df.columns


# ---------------------------------------------------------------------------
# Page: header, menu, upload screen, demo data, About, progress
# ---------------------------------------------------------------------------
@pytest.fixture
def page(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ENV_FILE", tmp_path / "no.env")
    return AppTest.from_file(APP_PATH, default_timeout=180).run()


def sidebar_text(at) -> str:
    return "\n".join(m.value for m in at.sidebar.markdown)


def test_upload_screen_runs(page):
    assert not page.exception
    assert [t.label for t in page.tabs[:2]] == ["Analyze", "About"]
    assert any("Upload a CSV" in i.value for i in page.info)
    assert "**Upload data**" in sidebar_text(page)        # step 1 is current


def test_about_tab(page):
    about = "\n".join(m.value for m in page.tabs[1].markdown)
    assert "Built by **Hrisita Mohapatra**" in about
    assert "github.com/hrisitamohapatra/foresight-automl" in about


def test_demo_data_flow(page):
    next(b for b in page.button if b.label == "Try demo data").click().run()
    assert not page.exception
    assert any("1,500 rows × 9 columns" in m.value for m in page.markdown)
    assert "**Choose target**" in sidebar_text(page)
    page.selectbox[0].select("churned").run()
    next(b for b in page.button if b.label == "Run data checks").click().run()
    assert not page.exception
    # The planted leak is offered as "keep anyway" (label is escaped: refund\_issued).
    assert any(c.label.startswith("Keep refund") for c in page.checkbox)
    assert "**Train models**" in sidebar_text(page)
    next(b for b in page.button if b.label == "Remove demo data").click().run()
    assert any("Upload a CSV" in i.value for i in page.info)


# ---------------------------------------------------------------------------
# Full flow via workflow()
# ---------------------------------------------------------------------------
def workflow_script():
    """Runs inside AppTest: builds synthetic data and calls app.workflow()."""
    import numpy as np
    import pandas as pd

    import app

    rng = np.random.default_rng(42)
    n = 300
    df = pd.DataFrame({
        "tenure": rng.normal(24, 10, n),
        "plan": rng.choice(["basic", "pro"], n),
        "![x](http://evil.example/t.png) <script>alert(1)</script> **bold**": rng.normal(size=n),
    })
    logit = -0.15 * (df["tenure"] - 24) + 1.2 * (df["plan"] == "basic") - 1
    df["churn"] = np.where(logit + rng.logistic(size=n) > 0, "yes", "no")
    df["refund_issued"] = (df["churn"] == "yes").astype(int)
    app.workflow(df, "demo.csv", "file-1")


@pytest.fixture
def at(tmp_path, monkeypatch):
    # Never read the real .env in tests: behave as if no Gemini key exists.
    monkeypatch.setattr(config, "ENV_FILE", tmp_path / "no.env")
    return AppTest.from_function(workflow_script, default_timeout=180).run()


def all_markdown(at) -> str:
    parts = [m.value for m in at.markdown] + [w.value for w in at.warning] + \
            [i.value for i in at.info] + [c.value for c in at.caption]
    return "\n".join(parts)


def test_full_flow(at):
    assert not at.exception
    at.selectbox[0].select("churn").run()
    assert not at.exception
    # Binary: the positive class defaults to the rarer outcome.
    assert at.selectbox[1].value in ("yes", "no")

    at.button[0].click().run()            # Run data checks
    assert not at.exception
    assert any("refund_issued" in str(df.value) for df in at.dataframe)
    assert at.checkbox[0].label.startswith("Keep refund")   # leakage override offered

    assert len(at.number_input) == 2      # cost of a miss / of a false alarm
    at.number_input[0].set_value(5.0)     # misses cost 5x a false alarm
    at.button[1].click().run()            # Train models
    assert not at.exception
    assert at.header[0].value == "Results"
    # 4 test metrics + 3 decision cards (cut-off, caught, cost)
    assert len(at.metric) == 7
    assert any(s.value == "Turning scores into decisions" for s in at.subheader)
    assert any("costing 5" in m.value for m in at.markdown)
    assert at.get("download_button")


def test_gemini_checkbox_disabled_without_key_and_summary_shown(at):
    at.selectbox[0].select("churn").run()
    at.button[0].click().run()
    gemini = next(c for c in at.checkbox if c.label.startswith("Write the summary with Gemini"))
    assert gemini.disabled and gemini.value is False
    next(b for b in at.button if b.label == "Train models").click().run()
    assert not at.exception
    assert any(s.value == "Summary" for s in at.subheader)
    assert any("Template summary" in c.value for c in at.caption)


def test_gemini_checkbox_enabled_with_key(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("GEMINI_API_KEY=test-key-for-ui-only\n", encoding="utf-8")
    monkeypatch.setattr(config, "ENV_FILE", env)
    at = AppTest.from_function(workflow_script, default_timeout=180).run()
    at.selectbox[0].select("churn").run()
    at.button[0].click().run()
    gemini = next(c for c in at.checkbox if c.label.startswith("Write the summary with Gemini"))
    assert not gemini.disabled
    assert any("free tier" in c.value for c in at.caption)


def test_tracking_checkbox_saves_run(at, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MLFLOW_DB", tmp_path / "mlflow.db")
    monkeypatch.setattr(config, "MLFLOW_ARTIFACT_DIR", tmp_path / "mlruns")
    at.selectbox[0].select("churn").run()
    at.button[0].click().run()
    labels = [c.label for c in at.checkbox]
    assert any(label.startswith("Also tune") for label in labels)
    track = next(c for c in at.checkbox if c.label.startswith("Save this run"))
    assert track.value is False                    # off by default
    track.check().run()
    next(b for b in at.button if b.label == "Train models").click().run()
    assert not at.exception
    from foresight import tracking
    assert len(tracking.recent_runs()) == 1


def test_uploaded_text_is_escaped_in_markdown(at):
    at.selectbox[0].select("churn").run()
    at.button[0].click().run()
    at.button[1].click().run()
    text = all_markdown(at)
    # The malicious column name may appear only in escaped form.
    assert XSS not in text
    assert "![x](" not in text and "<script>" not in text


def test_ambiguous_target_asks_user():
    def script():
        import numpy as np
        import pandas as pd

        import app

        rng = np.random.default_rng(42)
        df = pd.DataFrame({"x": rng.normal(size=300),
                           "rating": rng.integers(1, 6, size=300)})
        app.workflow(df, "ratings.csv", "file-2")

    at = AppTest.from_function(script, default_timeout=60).run()
    at.selectbox[0].select("rating").run()
    assert not at.exception
    assert at.radio[0].value == "Categories (classification)"
    assert any("distinct values" in w.value for w in at.warning)


def test_bad_target_shows_friendly_error():
    def script():
        import pandas as pd

        import app

        df = pd.DataFrame({"x": range(100), "const": [1] * 100})
        app.workflow(df, "bad.csv", "file-3")

    at = AppTest.from_function(script, default_timeout=60).run()
    at.selectbox[0].select("const").run()
    assert not at.exception
    assert "only one value" in at.error[0].value
