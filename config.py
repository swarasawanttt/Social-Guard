import os
import re

# ── Base Paths ────────────────────────────────────────────────────────────────
# Walk up from this file until we find the project root
# (identified by the presence of app.py)
def _find_project_root():
    current = os.path.dirname(os.path.abspath(__file__))
    for _ in range(5):  # max 5 levels up
        if os.path.exists(os.path.join(current, "app.py")):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    # Fallback: use directory of this file
    return os.path.dirname(os.path.abspath(__file__))

BASE_DIR   = _find_project_root()
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads", "temp_files")
REPORT_DIR = os.path.join(BASE_DIR, "static", "reports")
XAI_CHART_DIR = os.path.join(BASE_DIR, "static", "xai_charts")

# ── Dataset ───────────────────────────────────────────────────────────────────
# Input: the unified multi-platform table built by build_unified_dataset.py
UNIFIED_DATASET_PATH = os.path.join(DATA_DIR, "processed", "unified_raw.csv")
# Output of train_model.py: the cleaned feature table the model was trained on.
# Also used by src/xai.py as a fallback reference if background_data.pkl is missing.
FINAL_DATASET_PATH = os.path.join(DATA_DIR, "processed", "model_features.csv")

# ── Model Paths ───────────────────────────────────────────────────────────────
XGBOOST_MODEL_PATH       = os.path.join(MODEL_DIR, "xgboost.pkl")
LIGHTGBM_MODEL_PATH      = os.path.join(MODEL_DIR, "lightgbm.pkl")
RANDOM_FOREST_MODEL_PATH = os.path.join(MODEL_DIR, "random_forest.pkl")
ADABOOST_MODEL_PATH      = os.path.join(MODEL_DIR, "adaboost.pkl")
VOTING_MODEL_PATH        = os.path.join(MODEL_DIR, "voting_classifier.pkl")
SCALER_PATH              = os.path.join(MODEL_DIR, "scaler.pkl")
BEST_MODEL_PATH          = os.path.join(MODEL_DIR, "best_model.pkl")
BEST_MODEL_NAME_PATH     = os.path.join(MODEL_DIR, "best_model_name.txt")

# Explainable AI (SHAP/LIME) — a sample of scaled training data used as the
# "reference distribution" both explainers compare a new prediction against.
BACKGROUND_DATA_PATH     = os.path.join(MODEL_DIR, "background_data.pkl")

# Per-platform percentile reference tables (followers / following / posts).
# Saved at training time and re-used at prediction time so "5,000 followers"
# is judged against the platform it came from, with identical maths.
PERCENTILE_TABLES_PATH   = os.path.join(MODEL_DIR, "percentile_tables.json")

# Human-readable record of how the model was trained: dataset size,
# train/test split, class balance before & after SMOTE, etc.
# Read by report_generator.py so every PDF states exactly what data the
# model was trained/evaluated on.
TRAINING_METADATA_PATH   = os.path.join(MODEL_DIR, "training_metadata.json")

# ── Features ──────────────────────────────────────────────────────────────────
# 10 features, all readable from a public profile page. A field the page or the
# user does not provide is passed to the model as NaN (never as 0): the models
# were trained with whole feature groups randomly blanked, so they handle it.
# Reduced from 16 after an ablation study (see ablation_results.md): the six
# removed features (raw followers / following / posts, has_url, username_len,
# fullname_words) were redundant with the kept ones. The raw counts still feed
# the percentile ranks and the two ratios; only the model inputs were removed.
FEATURE_COLUMNS = [
    # per-platform percentile ranks of the three counts
    "followers_pr", "following_pr", "posts_pr",
    # derived
    "follower_following_ratio", "activity_rate",
    # bio / username / identity
    "bio_len", "username_digit_ratio",
    "name_eq_username", "has_profile_pic", "is_private",
]
# Feature groups blanked together during training (see train_model.py)
FEATURE_GROUPS = {
    "identity": ["name_eq_username", "has_profile_pic", "is_private"],
    "username": ["username_digit_ratio"],
    "bio":      ["bio_len"],
}
BINARY_FEATURES = ["name_eq_username", "has_profile_pic", "is_private"]
PERCENTILE_BASE = ["followers", "following", "posts"]

# Platform name used by the app  ->  platform name in the training table
PLATFORM_TRAIN_NAME = {
    "Instagram": "Instagram",
    "Twitter/X": "Twitter",
    "Facebook":  "Facebook",
    # LinkedIn has no training data: its percentiles use the pooled table "ALL"
}
TARGET_COLUMN = "label"

# ── Labels:  0=Real  1=Fake  2=Bot ───────────────────────────────────────────
LABEL_MAP = {0: "Real", 1: "Fake", 2: "Bot"}

LABEL_COLORS = {
    "Real": "#3a675b",
    "Fake": "#ba1a1a",
    "Bot":  "#46405e",
}

# ── Training ──────────────────────────────────────────────────────────────────
TEST_SIZE    = 0.2
RANDOM_STATE = 42
CV_FOLDS     = 5

# SMOTE oversamples every minority class in the TRAINING split only, up to
# the size of the majority class, so the model sees a balanced class
# distribution during training. The held-out test set is never touched by
# SMOTE — it keeps the real, imbalanced distribution so evaluation numbers
# reflect real-world performance.
USE_SMOTE          = True
SMOTE_K_NEIGHBORS  = 5
# SMOTE is applied PER platform x class cell (never across platforms) and only
# to cells that have at least SMOTE_MIN_CELL real rows, raised up to
# SMOTE_TARGET_CELL. Interpolating a handful of rows invents accounts that
# exist nowhere (measured: it made small cells worse), so tiny cells are
# left alone and handled by sample weights instead.
SMOTE_MIN_CELL     = 300
SMOTE_TARGET_CELL  = 2500
MASK_PROBABILITY   = 0.5      # chance each feature group is blanked in a masked copy

# ── Verdict ───────────────────────────────────────────────────────────────────
# The app always answers Real, Fake or Bot. Below this top-class probability it
# adds a "treat with caution" note to the result (it no longer abstains).
INCONCLUSIVE_THRESHOLD = 0.70

# ── Explainable AI (SHAP / LIME) ─────────────────────────────────────────────
# Both explainers are model-agnostic (work regardless of which model wins
# best_model.pkl — a single tree ensemble or the VotingClassifier).
XAI_BACKGROUND_SAMPLE_SIZE = 100   # rows of training data used as reference
XAI_LIME_NUM_FEATURES      = 10   # = len(FEATURE_COLUMNS)
XAI_SHAP_NSAMPLES          = 150   # KernelExplainer perturbation samples (10 features)

# ── Flask ─────────────────────────────────────────────────────────────────────
# Set SECRET_KEY in the environment for any deployment; the fallback is for local dev only.
SECRET_KEY         = os.environ.get("SECRET_KEY", "dev-only-change-me")
DEBUG              = True
MAX_CONTENT_LENGTH = 16 * 1024 * 1024

# ── Platform URL Patterns  (capture group 1 = username) ──────────────────────
PLATFORM_PATTERNS = {
    "Instagram": re.compile(r"(?:instagram\.com/|@)([\w.]+)", re.I),
    "Twitter/X": re.compile(r"(?:twitter\.com/|x\.com/|@)([\w]+)", re.I),
    "Facebook":  re.compile(r"(?:facebook\.com/|fb\.com/)([\w.]+)", re.I),
    "LinkedIn":  re.compile(r"(?:linkedin\.com/in/)([\w-]+)", re.I),
}

# ── Simulated Data Ranges Per Platform ───────────────────────────────────────
PLATFORM_RANGES = {
    "Instagram": {
        "followers": (0, 50_000),
        "following": (0, 5_000),
        "posts":     (0, 2_000),
        "bio_len":   (0, 150),
        "username_len_max": 30,
    },
    "Twitter/X": {
        "followers": (0, 100_000),
        "following": (0, 10_000),
        "posts":     (0, 50_000),
        "bio_len":   (0, 160),
        "username_len_max": 15,
    },
    "Facebook": {
        "followers": (0, 20_000),
        "following": (0, 5_000),
        "posts":     (0, 1_000),
        "bio_len":   (0, 200),
        "username_len_max": 50,
    },
    "LinkedIn": {
        "followers": (0, 30_000),
        "following": (0, 30_000),
        "posts":     (0, 500),
        "bio_len":   (0, 220),
        "username_len_max": 100,
    },
}

# ── Debug helper ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("BASE_DIR  :", BASE_DIR)
    print("MODEL_DIR :", MODEL_DIR)
    print("DATASET   :", FINAL_DATASET_PATH)
    import os as _os
    print("model exists:", _os.path.exists(BEST_MODEL_PATH))
    print("data  exists:", _os.path.exists(UNIFIED_DATASET_PATH))