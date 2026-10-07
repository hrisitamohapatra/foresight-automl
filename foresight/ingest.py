"""Safe loading and validation of uploaded CSV files.

Every upload is treated as untrusted:
- The file is read from memory (bytes). It is never written to disk, and the
  user's filename is never used to build a file path (no path traversal).
- Size, row, and column limits from config.py are enforced.
- Parsing problems become friendly IngestError messages (no stack traces).
- Nothing in the data is ever executed.
"""

import io
import math
from dataclasses import dataclass
from pathlib import PurePosixPath, PureWindowsPath

import numpy as np
import pandas as pd

from foresight import config


class IngestError(Exception):
    """A problem with the uploaded data. The message is safe to show users."""


# ---------------------------------------------------------------------------
# Filenames
# ---------------------------------------------------------------------------
def safe_display_name(filename: str, max_length: int = 100) -> str:
    """Return just the last part of a filename, for display only.

    "../../secret.csv" and "C:\\Users\\x\\data.csv" both become a bare name.
    This is only ever shown to users; it is never used to open or save files.
    """
    name = str(filename or "")
    # Strip folders written with either Windows (\) or Unix (/) separators.
    name = PureWindowsPath(PurePosixPath(name).name).name
    name = name.strip() or "uploaded.csv"
    return name[:max_length]


def _check_filename_and_size(filename: str, size_bytes: int) -> None:
    if not str(filename or "").lower().endswith(config.ALLOWED_EXTENSIONS):
        raise IngestError("Please upload a CSV file (ending in .csv).")
    if size_bytes == 0:
        raise IngestError("The file is empty.")
    if size_bytes > config.MAX_FILE_SIZE_BYTES:
        raise IngestError(
            f"The file is larger than the {config.MAX_FILE_SIZE_MB} MB limit."
        )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_csv(data: bytes, filename: str) -> pd.DataFrame:
    """Parse uploaded CSV bytes into a DataFrame, enforcing all limits.

    `filename` is only used to check the extension.
    """
    _check_filename_and_size(filename, len(data))

    # Real CSV files are text. A null byte means binary content
    # (e.g. a spreadsheet or program renamed to .csv).
    if b"\x00" in data[:8192]:
        raise IngestError("This file does not look like a plain-text CSV.")

    # Try UTF-8 first (the "-sig" variant also handles Excel's BOM marker),
    # then Latin-1, which can decode any byte so it never fails.
    df = None
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(
                io.BytesIO(data),
                encoding=encoding,
                # Read one extra row so we can tell if the limit was exceeded
                # without loading a huge file completely.
                nrows=config.MAX_ROWS + 1,
                low_memory=False,
            )
            break
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError:
            raise IngestError("The file has no data.") from None
        except (pd.errors.ParserError, ValueError):
            raise IngestError(
                "The file could not be read as a CSV. Check that every row "
                "has the same number of columns."
            ) from None

    if df is None:  # pragma: no cover - latin-1 decodes everything
        raise IngestError("The file's text encoding is not supported.")

    if len(df) > config.MAX_ROWS:
        raise IngestError(f"The file has more than {config.MAX_ROWS:,} rows.")
    if df.shape[1] > config.MAX_COLUMNS:
        raise IngestError(f"The file has more than {config.MAX_COLUMNS} columns.")
    if df.shape[1] < 2:
        raise IngestError(
            "The file needs at least two columns: a target and one or more inputs. "
            "If it has more, check that it uses commas as separators."
        )
    if len(df) == 0:
        raise IngestError("The file has a header row but no data rows.")

    # Column names are always treated as plain text.
    df.columns = [str(c).strip() for c in df.columns]
    return df


# ---------------------------------------------------------------------------
# Target column
# ---------------------------------------------------------------------------
def prepare_target(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, int]:
    """Check the target column and drop rows where it is missing.

    Returns the cleaned DataFrame and how many rows were dropped.
    """
    if target not in df.columns:
        raise IngestError("The selected target column was not found.")

    y = df[target]
    if pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y):
        # Infinite values cannot be learned from; treat them as missing.
        y = y.replace([np.inf, -np.inf], np.nan)

    keep = y.notna()
    n_dropped = int((~keep).sum())
    clean = df.loc[keep].reset_index(drop=True)
    clean[target] = y[keep].reset_index(drop=True)

    if len(clean) < config.MIN_ROWS:
        raise IngestError(
            f"At least {config.MIN_ROWS} rows with a target value are needed "
            f"(found {len(clean)})."
        )
    if clean[target].nunique() < 2:
        raise IngestError(
            "The target column has only one value, so there is nothing to predict."
        )
    return clean, n_dropped


# ---------------------------------------------------------------------------
# Problem type detection
# ---------------------------------------------------------------------------
@dataclass
class ProblemType:
    """Result of problem type detection.

    problem_type is "binary", "multiclass", "regression", or None when the
    case is ambiguous and the user must confirm. `suggestion` is our best
    guess for ambiguous cases.
    """
    problem_type: str | None
    ambiguous: bool
    suggestion: str
    n_unique: int
    reason: str


def _all_whole_numbers(values: pd.Series) -> bool:
    return bool(np.all(np.mod(values.to_numpy(dtype=float), 1) == 0))


def detect_problem_type(y: pd.Series) -> ProblemType:
    """Guess whether the target means classification or regression."""
    y = y.dropna()
    n_unique = int(y.nunique())
    is_numeric = pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y)

    if n_unique == 2:
        return ProblemType("binary", False, "binary", n_unique,
                           "The target has exactly two values.")

    if not is_numeric:
        return ProblemType("multiclass", False, "multiclass", n_unique,
                           f"The target is text with {n_unique} categories.")

    if _all_whole_numbers(y) and n_unique <= config.AMBIGUOUS_MAX_UNIQUE:
        # e.g. a 1-5 rating or a count of 0-10: could be categories or amounts.
        suggestion = "multiclass" if n_unique <= 10 else "regression"
        return ProblemType(None, True, suggestion, n_unique,
                           f"The target is whole numbers with only {n_unique} "
                           "distinct values. It could be categories or amounts.")

    return ProblemType("regression", False, "regression", n_unique,
                       "The target is a number with many distinct values.")


def check_target_for_problem_type(y: pd.Series, problem_type: str) -> None:
    """Final check once the problem type is decided (detected or confirmed)."""
    if problem_type not in ("binary", "multiclass", "regression"):
        raise IngestError("Unknown problem type.")

    if problem_type == "regression":
        if not (pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y)):
            raise IngestError("Regression needs a numeric target column.")
        return

    counts = y.value_counts()
    if problem_type == "binary" and len(counts) != 2:
        raise IngestError("Binary classification needs exactly two target values.")
    if len(counts) > config.MAX_CLASSES:
        raise IngestError(
            f"The target has {len(counts)} categories; the limit is "
            f"{config.MAX_CLASSES}. Is this an ID or free-text column?"
        )
    if counts.min() < config.MIN_ROWS_PER_CLASS:
        raise IngestError(
            f"Every target category needs at least {config.MIN_ROWS_PER_CLASS} "
            f"rows for cross-validation; the smallest has {int(counts.min())}."
        )


# ---------------------------------------------------------------------------
# Spreadsheet formula injection
# ---------------------------------------------------------------------------
def neutralize_cell(value):
    """Make a text cell safe to open in Excel/Sheets.

    Text starting with = + - @ (or tab / carriage return) could run as a
    formula. Prefixing a single quote makes the spreadsheet show it as text.
    Non-text values (numbers, missing values) are returned unchanged.
    """
    if isinstance(value, str) and value.startswith(config.FORMULA_PREFIXES):
        return "'" + value
    return value


def neutralize_formula_injection(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df that is safe to export as CSV or show in a report.

    Applies to column names and to every text column. Numeric columns are
    left alone (a real number like -5 is not a formula).
    """
    safe = df.copy()
    safe.columns = [neutralize_cell(str(c)) for c in safe.columns]
    for col in safe.columns:
        if not (pd.api.types.is_numeric_dtype(safe[col])
                or pd.api.types.is_bool_dtype(safe[col])):
            safe[col] = safe[col].map(neutralize_cell).astype(object)
    return safe
