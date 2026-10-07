"""Tests for foresight/ingest.py."""

import numpy as np
import pandas as pd
import pytest

from foresight import config
from foresight.ingest import (
    IngestError,
    check_target_for_problem_type,
    detect_problem_type,
    load_csv,
    neutralize_cell,
    neutralize_formula_injection,
    prepare_target,
    safe_display_name,
)


def make_csv(n_rows=60, n_cols=3) -> bytes:
    """Build a small valid CSV in memory."""
    rng = np.random.default_rng(config.RANDOM_SEED)
    data = {f"x{i}": rng.normal(size=n_rows) for i in range(n_cols - 1)}
    data["target"] = rng.integers(0, 2, size=n_rows)
    return pd.DataFrame(data).to_csv(index=False).encode("utf-8")


# ---------------------------------------------------------------------------
# Loading valid files
# ---------------------------------------------------------------------------
def test_valid_csv_loads():
    df = load_csv(make_csv(), "data.csv")
    assert df.shape == (60, 3)


def test_uppercase_extension_accepted():
    assert len(load_csv(make_csv(), "DATA.CSV")) == 60


def test_latin1_file_loads():
    data = "name,target\ncaf\xe9,1\nna\xefve,0\n".encode("latin-1")
    df = load_csv(data, "data.csv")
    assert df["name"].iloc[0] == "caf\xe9"


# ---------------------------------------------------------------------------
# Oversized or malformed CSVs are rejected
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["data.xlsx", "data.csv.exe", "data", "", "data.txt"])
def test_non_csv_extension_rejected(name):
    with pytest.raises(IngestError, match="CSV"):
        load_csv(make_csv(), name)


def test_oversized_file_rejected(monkeypatch):
    monkeypatch.setattr(config, "MAX_FILE_SIZE_BYTES", 100)
    with pytest.raises(IngestError, match="MB limit"):
        load_csv(make_csv(), "data.csv")


def test_too_many_rows_rejected(monkeypatch):
    monkeypatch.setattr(config, "MAX_ROWS", 10)
    with pytest.raises(IngestError, match="rows"):
        load_csv(make_csv(n_rows=11), "data.csv")


def test_row_limit_is_inclusive(monkeypatch):
    monkeypatch.setattr(config, "MAX_ROWS", 10)
    assert len(load_csv(make_csv(n_rows=10), "data.csv")) == 10


def test_too_many_columns_rejected(monkeypatch):
    monkeypatch.setattr(config, "MAX_COLUMNS", 5)
    with pytest.raises(IngestError, match="columns"):
        load_csv(make_csv(n_cols=6), "data.csv")


def test_empty_file_rejected():
    with pytest.raises(IngestError, match="empty"):
        load_csv(b"", "data.csv")


def test_whitespace_only_file_rejected():
    with pytest.raises(IngestError, match="no data"):
        load_csv(b"\n\n  \n", "data.csv")


def test_header_only_rejected():
    with pytest.raises(IngestError, match="no data rows"):
        load_csv(b"a,b,c\n", "data.csv")


def test_ragged_rows_rejected():
    data = b"a,b\n1,2\n3,4,5,6\n"
    with pytest.raises(IngestError, match="same number of columns"):
        load_csv(data, "data.csv")


def test_binary_file_rejected():
    data = b"PK\x03\x04\x00\x00binary spreadsheet content"
    with pytest.raises(IngestError, match="plain-text"):
        load_csv(data, "data.csv")


def test_single_column_rejected():
    with pytest.raises(IngestError, match="two columns"):
        load_csv(b"a\n1\n2\n", "data.csv")


def test_error_messages_do_not_leak_internals():
    # Friendly messages only: no tracebacks, file paths, or exception class names.
    with pytest.raises(IngestError) as err:
        load_csv(b"a,b\n1,2\n3,4,5,6\n", "data.csv")
    message = str(err.value)
    assert "Traceback" not in message
    assert "\\" not in message and "/" not in message
    assert "ParserError" not in message


# ---------------------------------------------------------------------------
# Path traversal filenames are ignored
# ---------------------------------------------------------------------------
TRAVERSAL_NAMES = [
    "../../secret.csv",
    "..\\..\\Windows\\System32\\evil.csv",
    "/etc/passwd.csv",
    "C:\\Users\\someone\\Desktop\\data.csv",
    "uploads/../../../app.csv",
]


@pytest.mark.parametrize("name", TRAVERSAL_NAMES)
def test_traversal_filename_never_touches_disk(name, tmp_path, monkeypatch):
    # Run inside an empty temp folder and confirm loading creates no files.
    monkeypatch.chdir(tmp_path)
    df = load_csv(make_csv(), name)
    assert len(df) == 60
    assert list(tmp_path.rglob("*")) == []


@pytest.mark.parametrize("name", TRAVERSAL_NAMES)
def test_display_name_strips_folders(name):
    shown = safe_display_name(name)
    assert "/" not in shown and "\\" not in shown and ".." not in shown
    assert shown.endswith(".csv")


def test_display_name_defaults_and_truncates():
    assert safe_display_name("") == "uploaded.csv"
    assert len(safe_display_name("a" * 500 + ".csv")) == 100


# ---------------------------------------------------------------------------
# Target validation
# ---------------------------------------------------------------------------
def test_missing_target_column_rejected():
    df = load_csv(make_csv(), "data.csv")
    with pytest.raises(IngestError, match="not found"):
        prepare_target(df, "does_not_exist")


def test_constant_target_rejected():
    df = pd.DataFrame({"x": range(60), "target": [1] * 60})
    with pytest.raises(IngestError, match="only one value"):
        prepare_target(df, "target")


def test_too_few_rows_rejected():
    df = pd.DataFrame({"x": range(20), "target": [0, 1] * 10})
    with pytest.raises(IngestError, match="At least"):
        prepare_target(df, "target")


def test_rows_with_missing_target_dropped():
    target = [0, 1] * 30 + [np.nan] * 5 + [np.inf]
    df = pd.DataFrame({"x": range(66), "target": target})
    clean, n_dropped = prepare_target(df, "target")
    assert n_dropped == 6
    assert len(clean) == 60
    assert clean["target"].notna().all()


# ---------------------------------------------------------------------------
# Problem type detection
# ---------------------------------------------------------------------------
def test_text_binary():
    result = detect_problem_type(pd.Series(["yes", "no"] * 30))
    assert result.problem_type == "binary" and not result.ambiguous


def test_numeric_binary():
    assert detect_problem_type(pd.Series([0, 1] * 30)).problem_type == "binary"


def test_bool_binary():
    assert detect_problem_type(pd.Series([True, False] * 30)).problem_type == "binary"


def test_text_multiclass():
    result = detect_problem_type(pd.Series(["a", "b", "c"] * 20))
    assert result.problem_type == "multiclass"


def test_continuous_regression():
    rng = np.random.default_rng(config.RANDOM_SEED)
    result = detect_problem_type(pd.Series(rng.normal(size=100)))
    assert result.problem_type == "regression"


def test_many_whole_numbers_is_regression():
    result = detect_problem_type(pd.Series(range(100)))
    assert result.problem_type == "regression"


def test_small_integer_range_is_ambiguous():
    # A 1-5 rating could be categories or an amount: the user must decide.
    result = detect_problem_type(pd.Series([1, 2, 3, 4, 5] * 20))
    assert result.ambiguous
    assert result.problem_type is None
    assert result.suggestion == "multiclass"


def test_whole_number_floats_are_ambiguous():
    result = detect_problem_type(pd.Series([1.0, 2.0, 3.0] * 20))
    assert result.ambiguous


def test_rare_class_rejected():
    y = pd.Series(["a"] * 50 + ["b"] * 50 + ["c"] * 2)
    with pytest.raises(IngestError, match="at least"):
        check_target_for_problem_type(y, "multiclass")


def test_too_many_classes_rejected(monkeypatch):
    monkeypatch.setattr(config, "MAX_CLASSES", 3)
    y = pd.Series(list("abcd") * 10)
    with pytest.raises(IngestError, match="categories"):
        check_target_for_problem_type(y, "multiclass")


def test_text_target_cannot_be_regression():
    with pytest.raises(IngestError, match="numeric"):
        check_target_for_problem_type(pd.Series(["a", "b"] * 30), "regression")


def test_valid_targets_pass():
    check_target_for_problem_type(pd.Series([0, 1] * 30), "binary")
    check_target_for_problem_type(pd.Series([1, 2, 3] * 20), "multiclass")
    check_target_for_problem_type(pd.Series([1, 2, 3] * 20), "regression")


# ---------------------------------------------------------------------------
# Formula injection is neutralized
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cell", ["=SUM(A1:A9)", "+1+1", "-2+3", "@cmd", "\t=1", "\r=1",
                                  '=HYPERLINK("http://evil","click")'])
def test_dangerous_cells_prefixed(cell):
    assert neutralize_cell(cell) == "'" + cell


@pytest.mark.parametrize("value", ["hello", "a=b", "", 5, -5, 1.5, None, np.nan])
def test_safe_values_unchanged(value):
    result = neutralize_cell(value)
    if isinstance(value, float) and np.isnan(value):
        assert np.isnan(result)
    else:
        assert result == value


def test_dataframe_neutralized():
    df = pd.DataFrame({
        "=evil_name": ["=1+1", "fine", None],
        "amount": [-5, 10, 3],
    })
    safe = neutralize_formula_injection(df)
    assert list(safe.columns) == ["'=evil_name", "amount"]
    assert safe.iloc[0, 0] == "'=1+1"
    assert safe.iloc[1, 0] == "fine"
    assert pd.isna(safe.iloc[2, 0])
    assert safe["amount"].tolist() == [-5, 10, 3]   # numbers untouched
    assert df.iloc[0, 0] == "=1+1"                   # original not modified


def test_neutralized_csv_export_is_safe():
    # End to end: what actually gets written to a CSV file is safe.
    df = pd.DataFrame({"note": ["@SUM(1)", "ok"], "target": [0, 1]})
    exported = neutralize_formula_injection(df).to_csv(index=False)
    for line in exported.splitlines()[1:]:
        assert not line.startswith(config.FORMULA_PREFIXES)
