"""Central settings for Foresight AutoML.

Every limit, threshold, and color lives here so the rest of the code
never hardcodes values (and nothing is tied to a specific dataset).
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# The project root is the folder that contains this "foresight" package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Saved models and reports go here. Models are only ever loaded from this
# folder, never from files a user provides (loading untrusted joblib/pickle
# files can run arbitrary code).
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Upload limits (every CSV is treated as untrusted input)
# ---------------------------------------------------------------------------
ALLOWED_EXTENSIONS = (".csv",)
MAX_FILE_SIZE_MB = 50          # keep in sync with .streamlit/config.toml
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_ROWS = 500_000
MAX_COLUMNS = 500

# A model needs enough rows to split into train/test and 5 CV folds.
MIN_ROWS = 50

# ---------------------------------------------------------------------------
# Train / test split and cross-validation
# ---------------------------------------------------------------------------
TEST_SIZE = 0.20   # 20% held out, used exactly once at the very end
CV_FOLDS = 5

# Every class needs at least this many rows. After the test split removes
# ~20%, the training data must still have at least one row of each class per
# CV fold (5 folds), so 10 leaves a safety margin.
MIN_ROWS_PER_CLASS = 10

# ---------------------------------------------------------------------------
# Problem type detection
# ---------------------------------------------------------------------------
# Text/boolean targets are always classification. For numeric targets:
#   - 2 unique values                      -> binary classification
#   - whole numbers with 3..AMBIGUOUS_MAX   -> ambiguous, ask the user
#     unique values (e.g. a 1-5 rating could be either)
#   - anything else                        -> regression
AMBIGUOUS_MAX_UNIQUE = 20

# More classes than this is treated as too many for this tool.
MAX_CLASSES = 50

# ---------------------------------------------------------------------------
# Data profiling thresholds
# ---------------------------------------------------------------------------
# Minority class below this share of rows -> "imbalanced".
IMBALANCE_THRESHOLD = 0.20

# A text column with more distinct values than this is "high cardinality"
# (one-hot encoding it would create too many columns).
HIGH_CARDINALITY_THRESHOLD = 50

# A column where (almost) every value is unique looks like an ID.
# IDs carry no general pattern, so they are dropped from modeling.
ID_LIKE_UNIQUE_RATIO = 0.95

# A whole-number column is also treated as an ID if its name looks like one
# (e.g. "customer_id", "ID", "uuid") or its values are a 1, 2, 3... sequence.
ID_NAME_HINTS = ("id", "uuid", "guid", "key", "index", "row")

# Columns missing more than this share of values get a warning.
HIGH_MISSING_THRESHOLD = 0.40

# A text column is treated as numbers if at least this share of its non-blank
# values are valid numbers (e.g. "12.5" stored as text because of a few blanks).
NUMERIC_TEXT_RATIO = 0.95

# A text column is treated as dates if at least this share of a sample of its
# values parse as dates. Dates are not used as model inputs in Phase 1.
DATETIME_PARSE_RATIO = 0.90

# Likely target leakage: if ONE column on its own predicts the target almost
# perfectly (score >= this), it probably contains information that would not
# be available at prediction time (e.g. "refund_issued" when predicting churn).
# Score = ROC-AUC (binary), balanced accuracy (multiclass), or R2 (regression).
LEAKAGE_SCORE_THRESHOLD = 0.97

# Columns scoring at least this on their own (but below the threshold above)
# are kept but get a strong warning: honest columns rarely get this close,
# but some do, so a person should check them. Example: a vendor's own churn
# score (0.94) in a churn dataset.
LEAKAGE_WARNING_THRESHOLD = 0.90

# The leakage check trains one small model per column, so it uses a sample
# of the training rows and fewer folds to stay fast on large files.
LEAKAGE_SAMPLE_ROWS = 5_000
LEAKAGE_CV_FOLDS = 3

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
# One-hot encoding: categories seen fewer than ONEHOT_MIN_FREQUENCY times in
# the training data are grouped into one "(other)" column, and each column
# produces at most ONEHOT_MAX_CATEGORIES output columns. This keeps
# high-cardinality columns from creating thousands of features.
ONEHOT_MIN_FREQUENCY = 5
ONEHOT_MAX_CATEGORIES = 20

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
# The metric used to pick the best model (all metrics are still reported).
# For binary problems, imbalanced data uses PR-AUC because ROC-AUC can look
# good even when the model is poor at finding the rare class.
SELECTION_METRIC = {
    "binary": "roc_auc",
    "binary_imbalanced": "pr_auc",
    "multiclass": "f1_macro",
    "regression": "rmse",
}

# ---------------------------------------------------------------------------
# Hyperparameter tuning (Optuna, optional, off by default)
# ---------------------------------------------------------------------------
# Tuning uses NESTED cross-validation: inside each of the CV_FOLDS outer
# folds, a separate Optuna search runs on that fold's training part only
# (with TUNE_INNER_FOLDS inner folds). The outer folds therefore give an
# honest score for "tune, then train", not an optimistic one.
TUNE_TRIALS = 20             # trials per search
TUNE_TIMEOUT_SECONDS = 30    # safety limit per search; if reached, fewer trials run
TUNE_INNER_FOLDS = 3

# ---------------------------------------------------------------------------
# Experiment tracking (MLflow, optional, local only)
# ---------------------------------------------------------------------------
# Runs are stored in a local SQLite file; logged files (report, model) in mlruns/.
# Both are git-ignored. MLflow's internet telemetry is switched off in
# foresight/tracking.py, the only module allowed to import MLflow.
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"
MLFLOW_ARTIFACT_DIR = PROJECT_ROOT / "mlruns"
MLFLOW_EXPERIMENT = "foresight-automl"

# ---------------------------------------------------------------------------
# Narrative summary (template always; Gemini optional, off by default)
# ---------------------------------------------------------------------------
# Gemini receives ONLY aggregated results, with every piece of uploaded text
# (column names, class labels, target name) replaced by placeholders such as
# [COLUMN_1]. Its answer is checked against the real results before use;
# otherwise the template summary is shown instead.
ENV_FILE = PROJECT_ROOT / ".env"          # holds GEMINI_API_KEY (never committed)
GEMINI_MODEL = "gemini-2.5-flash"         # change if your key uses a different model
GEMINI_TIMEOUT_SECONDS = 30
NARRATIVE_MAX_CHARS = 1500
NARRATIVE_TOP_DRIVERS = 5

# ---------------------------------------------------------------------------
# Explanations
# ---------------------------------------------------------------------------
# SHAP is computed on a random sample of held-out test rows. Exact SHAP for a
# random forest of fully grown trees is much slower, so it gets fewer rows.
SHAP_SAMPLE_ROWS = 500
SHAP_SAMPLE_ROWS_FOREST = 100
SHAP_BACKGROUND_ROWS = 100      # reference rows for linear-model SHAP

# Permutation importance: shuffle each column this many times and measure
# how much the score drops, on up to this many test rows.
PERMUTATION_REPEATS = 5
PERMUTATION_SAMPLE_ROWS = 5_000

# How many top columns to compare between SHAP and permutation importance.
TOP_K_DRIVERS = 5

# A numeric column gets an up/down direction hint only if its values and its
# SHAP values have at least this rank correlation (otherwise "mixed").
DIRECTION_MIN_CORRELATION = 0.3

# Columns with less than this share of total SHAP influence get "Very little
# influence" instead of a direction (linear models give every column a
# direction, even when its effect is negligible).
DIRECTION_MIN_SHARE = 0.02

# ---------------------------------------------------------------------------
# Decisions: calibration, thresholds, lift (binary problems only)
# ---------------------------------------------------------------------------
# Isotonic calibration is more flexible but needs plenty of data; below this
# many training rows, sigmoid (Platt) calibration is used instead.
CALIBRATION_ISOTONIC_MIN_ROWS = 5_000
# Folds used INSIDE calibration (nested within each outer CV fold).
CALIBRATION_CV_FOLDS = 3
# Number of bins in the reliability (calibration) chart.
CALIBRATION_BINS = 10

# Candidate cut-offs searched when choosing a cost-based threshold.
THRESHOLD_GRID = [round(0.01 * i, 2) for i in range(1, 100)]   # 0.01 ... 0.99

# Cost inputs must be positive and below this (guards against typos).
MAX_COST = 1_000_000_000

# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------
# Spreadsheet apps treat cells starting with these characters as formulas.
# Any exported cell starting with one is prefixed with a quote to make it
# plain text. Tab and carriage return are included per OWASP guidance.
FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")

# ---------------------------------------------------------------------------
# Colors (shared by the Streamlit UI, charts, and the HTML report)
# ---------------------------------------------------------------------------
COLOR_PRIMARY = "#1E64D4"
COLOR_BACKGROUND = "#FFFFFF"
COLOR_SECONDARY_BACKGROUND = "#F3F7FD"
COLOR_TEXT = "#1B2A41"
COLOR_WARNING = "#E8A317"   # muted amber, used only for warnings

# Shades of blue for charts, darkest to lightest.
CHART_BLUES = ["#0B3D91", "#1E64D4", "#4A8BEA", "#7FAEF2", "#B5D0F8"]
