"""Sanity checks for foresight/config.py and the project setup files."""

import re
import tomllib

from foresight import config

HEX_COLOR = re.compile(r"^#[0-9A-Fa-f]{6}$")


def test_limits_are_positive():
    assert config.MAX_FILE_SIZE_BYTES == config.MAX_FILE_SIZE_MB * 1024 * 1024
    assert config.MAX_ROWS > config.MIN_ROWS > 0
    assert config.MAX_COLUMNS > 1


def test_split_and_cv_settings():
    assert 0 < config.TEST_SIZE < 1
    assert config.CV_FOLDS >= 2
    assert isinstance(config.RANDOM_SEED, int)


def test_thresholds_are_fractions():
    for value in (
        config.IMBALANCE_THRESHOLD,
        config.ID_LIKE_UNIQUE_RATIO,
        config.HIGH_MISSING_THRESHOLD,
        config.LEAKAGE_SCORE_THRESHOLD,
        config.NUMERIC_TEXT_RATIO,
        config.DATETIME_PARSE_RATIO,
    ):
        assert 0 < value < 1


def test_leakage_warning_below_exclusion():
    assert 0 < config.LEAKAGE_WARNING_THRESHOLD < config.LEAKAGE_SCORE_THRESHOLD


def test_only_csv_allowed():
    assert config.ALLOWED_EXTENSIONS == (".csv",)


def test_formula_prefixes_cover_required_characters():
    for char in ("=", "+", "-", "@"):
        assert char in config.FORMULA_PREFIXES


def test_selection_metric_for_every_problem_type():
    assert set(config.SELECTION_METRIC) == {
        "binary", "binary_imbalanced", "multiclass", "regression",
    }


def test_colors_are_valid_hex():
    colors = [
        config.COLOR_PRIMARY, config.COLOR_BACKGROUND,
        config.COLOR_SECONDARY_BACKGROUND, config.COLOR_TEXT,
        config.COLOR_WARNING, *config.CHART_BLUES,
    ]
    for color in colors:
        assert HEX_COLOR.match(color), color


def test_streamlit_theme_matches_config():
    # The UI theme and the Python palette must not drift apart.
    path = config.PROJECT_ROOT / ".streamlit" / "config.toml"
    settings = tomllib.loads(path.read_text(encoding="utf-8"))
    theme = settings["theme"]
    assert theme["base"] == "light"
    assert theme["primaryColor"] == config.COLOR_PRIMARY
    assert theme["backgroundColor"] == config.COLOR_BACKGROUND
    assert theme["secondaryBackgroundColor"] == config.COLOR_SECONDARY_BACKGROUND
    assert theme["textColor"] == config.COLOR_TEXT
    assert theme["yellowColor"] == config.COLOR_WARNING
    assert settings["server"]["maxUploadSize"] == config.MAX_FILE_SIZE_MB


def test_streamlit_safety_settings():
    path = config.PROJECT_ROOT / ".streamlit" / "config.toml"
    settings = tomllib.loads(path.read_text(encoding="utf-8"))
    assert settings["client"]["showErrorDetails"] == "none"   # no stack traces in the UI
    assert settings["browser"]["gatherUsageStats"] is False
    assert settings["server"]["address"] == "localhost"


def test_gitignore_protects_secrets_and_data():
    lines = (config.PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    for entry in (".env", "venv/", "__pycache__/", "uploads/", "outputs/", "*.joblib"):
        assert entry in lines, f"{entry} missing from .gitignore"
