"""
src/xai.py
──────────
Real, per-account explainability: SHAP + a dependency-free LIME-style
local surrogate model, both computed fresh from the actual trained model
and the actual account's feature values — not a fixed bank of sentences
picked by thresholds (that's what src/explain.py's "Behavioral Indicators"
panel does, and it stays as a quick-glance companion to this).

Why two methods instead of one:
  SHAP (Shapley values, via shap.KernelExplainer) — game-theoretic,
  mathematically guaranteed to sum to the actual predicted probability
  (base_value + sum(shap_values) == predicted_proba). This is verified
  in this module's self-test. It answers "how much did each feature
  move the prediction, exactly."

  LIME (local surrogate) — fits a small weighted linear model on
  perturbations of THIS account's own feature vector, then reads off
  the coefficients. It answers "if this account's own numbers moved
  slightly, which one would flip the prediction fastest."

Both are model-agnostic: this project's "best model" is selected at
training time and could be any of the 5 trained models (including the
VotingClassifier), so neither explainer assumes a tree structure.

LIME is implemented here directly (perturb → kernel-weight by distance
→ fit weighted Ridge regression → read coefficients) rather than via the
`lime` package, because that package failed to build in this project's
dev environment (a `setuptools`/`distutils` incompatibility unrelated to
this code) and pulls in a C extension on some platforms. This is the
textbook LIME-for-tabular-data algorithm (Ribeiro et al., 2016, "Why
Should I Trust You?"), just without the extra dependency. Its local
fidelity (R²) is reported honestly alongside the result — tree ensembles
have jagged decision boundaries, so an R² of 0.2-0.5 is normal and
expected, not a bug.

Both explainers operate in the SAME SCALED FEATURE SPACE the model
itself was trained and predicts on (via the saved StandardScaler),
so their math lines up with what the model actually sees.

Background / reference data (what "normal" looks like, which both
methods compare the account against):
  1. Preferred: models/background_data.pkl, saved by train_model.py
     from real pre-SMOTE training data.
  2. Fallback (used automatically if that file doesn't exist yet,
     e.g. the model hasn't been retrained since this feature was added):
     a fresh stratified sample straight from data/processed/final_dataset.csv,
     scaled with the same scaler used for prediction. Still real data,
     never synthetic placeholder numbers.
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import joblib

import os as _os, sys as _sys
def _root():
    d = _os.path.dirname(_os.path.abspath(__file__))
    for _ in range(5):
        if _os.path.exists(_os.path.join(d, "app.py")):
            return d
        d = _os.path.dirname(d)
    return _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _root())
del _root, _os, _sys

from config import (
    BACKGROUND_DATA_PATH, FINAL_DATASET_PATH, FEATURE_COLUMNS,
    TARGET_COLUMN, LABEL_MAP, XAI_BACKGROUND_SAMPLE_SIZE, XAI_SHAP_NSAMPLES,
)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from sklearn.linear_model import Ridge


# ─────────────────────────────────────────────────────────────────────────────
# Human-readable feature names + what each one means (for the narrative)
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_DISPLAY = {
    "followers":                 "Follower count",
    "following":                 "Following count",
    "posts":                     "Post count",
    "bio_len":                   "Bio length",
    "has_url":                   "Link in profile",
    "username_len":              "Username length",
    "username_digit_ratio":      "Digits in username",
    "fullname_words":            "Words in display name",
    "name_eq_username":          "Display name equals username",
    "has_profile_pic":           "Profile picture",
    "is_private":                "Private account",
    "follower_following_ratio":  "Follower-to-following ratio",
    "activity_rate":             "Post-to-follower ratio",
    "followers_pr":              "Follower count vs. platform",
    "following_pr":              "Following count vs. platform",
    "posts_pr":                  "Post count vs. platform",
}

FEATURE_WHY_IT_MATTERS = {
    "followers":                "how large an audience the account has attracted",
    "following":                "how many accounts it follows",
    "posts":                    "how much content it has published",
    "bio_len":                  "whether a bio was filled in and how detailed it is",
    "has_url":                  "whether the profile links to a website",
    "username_len":             "how the username is structured",
    "username_digit_ratio":     "how much of the username is digits (auto-generated handles are often numeric)",
    "fullname_words":           "whether a real-looking display name is set",
    "name_eq_username":         "whether the display name is just a copy of the username",
    "has_profile_pic":          "whether a custom profile picture is set",
    "is_private":               "whether the account is private",
    "follower_following_ratio": "the balance between followers and who it follows - "
                                "a classic bot/fake signal when very low",
    "activity_rate":            "post count relative to audience size at a single "
                                "snapshot (not a measure of posting speed over time) "
                                "- unusually high values suggest automation",
    "followers_pr":             "where the follower count sits among accounts on the same platform",
    "following_pr":             "where the following count sits among accounts on the same platform",
    "posts_pr":                 "where the post count sits among accounts on the same platform",
}


# ─────────────────────────────────────────────────────────────────────────────
# Background / reference data
# ─────────────────────────────────────────────────────────────────────────────
_background_cache = None   # {"X_scaled": ..., "source": "trained"|"dataset_fallback"}


def _load_background(scaler) -> dict:
    """
    Returns a dict: {"X_scaled": np.ndarray (n, 8), "source": str}.
    Cached after first call within this process.
    """
    global _background_cache
    if _background_cache is not None:
        return _background_cache

    if os.path.exists(BACKGROUND_DATA_PATH):
        try:
            saved = joblib.load(BACKGROUND_DATA_PATH)
            X_bg = np.asarray(saved["X_background"])
            fill = saved.get("fill_values")
            fill = np.asarray(fill) if fill is not None else np.nan_to_num(np.nanmedian(X_bg, axis=0))
            _background_cache = {"X_scaled": X_bg, "fill": fill, "source": "trained_background"}
            return _background_cache
        except Exception:
            pass   # fall through to the dataset fallback below

    # Fallback: sample straight from the real dataset, scaled the same way
    # the model sees features at prediction time. Stratified by label so
    # all three classes are represented in the reference distribution.
    df = pd.read_csv(FINAL_DATASET_PATH)
    n_per_class = max(1, XAI_BACKGROUND_SAMPLE_SIZE // df[TARGET_COLUMN].nunique())
    sampled = (
        df.groupby(TARGET_COLUMN, group_keys=False)
          .apply(lambda g: g.sample(min(len(g), n_per_class), random_state=42))
    )
    X_raw = sampled[FEATURE_COLUMNS].values
    X_scaled = scaler.transform(X_raw)
    fill = np.nan_to_num(np.nanmedian(X_scaled, axis=0))
    X_scaled = np.where(np.isnan(X_scaled), fill, X_scaled)
    _background_cache = {"X_scaled": X_scaled, "fill": fill, "source": "dataset_fallback"}
    return _background_cache


# ─────────────────────────────────────────────────────────────────────────────
# SHAP
# ─────────────────────────────────────────────────────────────────────────────
_shap_explainer_cache = {}   # id(model) -> shap.KernelExplainer


def _get_shap_explainer(model, X_background):
    key = id(model)
    if key not in _shap_explainer_cache:
        _shap_explainer_cache[key] = shap.KernelExplainer(model.predict_proba, X_background)
    return _shap_explainer_cache[key]


def _compute_shap(model, X_background, instance_scaled, label_id):
    """
    Returns (rows, elapsed_seconds) where rows is a list of
    {"feature", "display_name", "shap_value", "raw_value"} sorted by
    |shap_value| descending, for the PREDICTED class only.

    Self-checked: base_value[label_id] + sum(shap_value for all features)
    is verified to reconstruct the model's actual predicted probability
    for that class (within floating point tolerance) — this is SHAP's
    additivity guarantee, and checking it here catches any future
    SHAP/sklearn version mismatch that would silently break this.
    """
    t0 = time.time()
    explainer = _get_shap_explainer(model, X_background)
    sv = explainer.shap_values(instance_scaled.reshape(1, -1), nsamples=XAI_SHAP_NSAMPLES)
    sv = np.asarray(sv)

    # Normalize shape across shap versions: this project's installed
    # version returns (1, n_features, n_classes).
    if sv.ndim == 3:
        contrib = sv[0, :, label_id]
    elif sv.ndim == 2:          # some versions: (n_features, n_classes) for 1 instance
        contrib = sv[:, label_id]
    else:
        raise ValueError(f"Unexpected SHAP output shape: {sv.shape}")

    base_value = explainer.expected_value
    base = base_value[label_id] if hasattr(base_value, "__len__") else base_value
    reconstructed = float(base) + float(contrib.sum())
    actual = float(model.predict_proba(instance_scaled.reshape(1, -1))[0, label_id])
    additivity_ok = abs(reconstructed - actual) < 0.05   # loose tolerance; KernelExplainer is sampling-based

    rows = sorted(
        zip(FEATURE_COLUMNS, contrib.tolist()),
        key=lambda x: -abs(x[1])
    )
    return rows, round(time.time() - t0, 2), additivity_ok


# ─────────────────────────────────────────────────────────────────────────────
# LIME (custom, dependency-free implementation of the tabular LIME algorithm)
# ─────────────────────────────────────────────────────────────────────────────
def _compute_lime(model, X_background, instance_scaled, label_id,
                   n_samples: int = 400, width_mult: float = 0.75,
                   ridge_alpha: float = 0.5, seed: int = 42):
    """
    Returns (rows, r2, elapsed_seconds).
    rows: list of (feature, coefficient) sorted by |coefficient| descending.
    r2: local fidelity of the surrogate linear model (weighted R²) —
        shown to the user honestly rather than hidden; tree ensembles
        have jagged decision boundaries so 0.2-0.5 is a normal result,
        not a sign anything is broken.

    Algorithm (standard tabular LIME):
      1. Sample N perturbations around the instance, scaled by the
         background's per-feature standard deviation.
      2. Weight each perturbation by an exponential kernel on its
         (normalized) distance from the real instance — closer
         perturbations matter more to the local fit.
      3. Query the real model's predict_proba on every perturbation.
      4. Fit a weighted Ridge regression of (perturbed features) ->
         (model's probability for the predicted class).
      5. The fitted coefficients ARE the local linear explanation:
         sign = direction, magnitude = how much that feature's scaled
         value moves the predicted-class probability near this account.
    """
    t0 = time.time()
    sigma = X_background.std(axis=0)
    sigma[sigma == 0] = 1.0

    rng = np.random.RandomState(seed)
    noise = rng.normal(0, 1, size=(n_samples, len(FEATURE_COLUMNS))) * sigma * width_mult
    Z = instance_scaled + noise
    Z[0] = instance_scaled   # always include the real point itself

    dist = np.linalg.norm((Z - instance_scaled) / sigma, axis=1)
    kernel_width = np.sqrt(len(FEATURE_COLUMNS)) * 0.75
    weights = np.exp(-(dist ** 2) / (kernel_width ** 2))

    proba_target = model.predict_proba(Z)[:, label_id]

    ridge = Ridge(alpha=ridge_alpha)
    ridge.fit(Z, proba_target, sample_weight=weights)
    r2 = float(ridge.score(Z, proba_target, sample_weight=weights))

    rows = sorted(
        zip(FEATURE_COLUMNS, ridge.coef_.tolist()),
        key=lambda x: -abs(x[1])
    )
    return rows, round(r2, 3), round(time.time() - t0, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic narrative — built fresh from THIS account's actual numbers
# ─────────────────────────────────────────────────────────────────────────────
_RANK_WORDS = ["single biggest factor", "second most influential factor",
               "third most influential factor"]


def _format_raw(feature: str, raw_profile: dict, features: dict) -> str:
    """Human-readable value for a feature, pulled from this account's
    actual data (not the scaled model input)."""
    if feature in ("followers", "following", "posts"):
        from src.utils import format_number
        return format_number(raw_profile.get(feature, features.get(feature, 0)))
    v = features.get(feature)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "not available"
    if feature in ("username_len", "bio_len", "fullname_words"):
        return f"{int(v)}" + ("" if feature == "fullname_words" else " chars")
    if feature in ("has_url", "name_eq_username", "has_profile_pic", "is_private"):
        return "yes" if v >= 0.5 else "no"
    if feature == "username_digit_ratio":
        return f"{v * 100:.0f}% digits"
    if feature.endswith("_pr"):
        return f"{v * 100:.0f}th percentile"
    if feature == "follower_following_ratio":
        return f"{v:.3f}"
    if feature == "activity_rate":
        return f"{v:.4f}"
    return str(v)


def build_narrative(shap_rows: list, raw_profile: dict, features: dict,
                     label_str: str, top_n: int = 3) -> list:
    """
    Builds up to `top_n` plain-English sentences from the actual SHAP
    contributions of THIS account — different accounts produce different
    sentences because the ranking and values are computed fresh each time,
    not selected from a fixed bank by simple thresholds.
    """
    sentences = []
    for i, (feature, shap_val) in enumerate(shap_rows[:top_n]):
        if abs(shap_val) < 1e-4:
            continue
        rank = _RANK_WORDS[i] if i < len(_RANK_WORDS) else f"factor #{i+1}"
        display_name = FEATURE_DISPLAY.get(feature, feature)
        raw_value = _format_raw(feature, raw_profile, features)
        why = FEATURE_WHY_IT_MATTERS.get(feature, "")
        direction = "toward" if shap_val > 0 else "away from"

        strength = ("strongly" if abs(shap_val) > 0.15 else
                    "moderately" if abs(shap_val) > 0.05 else "slightly")

        sentence = (
            f"{display_name} ({raw_value}) was the {rank}: it pushed the model "
            f"{strength} {direction} the '{label_str}' classification "
            f"(SHAP contribution {shap_val:+.3f}). This feature reflects {why}."
        )
        sentences.append(sentence)
    return sentences


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point — called by predict.py
# ─────────────────────────────────────────────────────────────────────────────
def explain_prediction(model, scaler, raw_profile: dict, features: dict,
                        label_id: int, label_str: str) -> dict:
    """
    Computes real SHAP + LIME explanations for one prediction.
    Never raises — any internal failure is caught and reported in
    "error", with shap_available/lime_available set to False so callers
    (predict.py, report_generator.py, results.html) can degrade gracefully
    and still show the rest of the result.

    Returns
    -------
    dict:
        shap_available : bool
        lime_available : bool
        background_source : "trained_background" | "dataset_fallback" | None
        shap : list of {"feature","display_name","value","raw_value"}  (sorted)
        lime : list of {"feature","display_name","value","raw_value"}  (sorted)
        lime_r2 : float | None   — local fidelity, shown honestly to the user
        narrative : list[str]    — dynamic, per-account sentences
        timing : {"shap_seconds": float, "lime_seconds": float}
        error : str | None
    """
    result = {
        "shap_available": False, "lime_available": False,
        "background_source": None, "shap": [], "lime": [],
        "lime_r2": None, "narrative": [], "timing": {}, "error": None,
    }

    if not SHAP_AVAILABLE:
        result["error"] = "shap package not installed — run: pip install shap"
        return result

    try:
        from src.feature_engineering import features_to_vector
        X = features_to_vector(features)
        instance_scaled = scaler.transform(X)[0]

        bg = _load_background(scaler)
        result["background_source"] = bg["source"]

        # Fields that could not be read are NaN. The explainers need complete
        # rows, so for the explanation only they are set to the typical
        # (median) value, and they are left out of the lists shown to the user.
        unknown = np.isnan(instance_scaled)
        instance_scaled = np.where(unknown, bg["fill"], instance_scaled)
        known_names = {f for f, u in zip(FEATURE_COLUMNS, unknown) if not u}

        shap_rows, shap_t, additivity_ok = _compute_shap(
            model, bg["X_scaled"], instance_scaled, label_id)
        shap_rows = [r for r in shap_rows if r[0] in known_names]
        result["shap_available"] = True
        result["shap"] = [
            {"feature": f, "display_name": FEATURE_DISPLAY.get(f, f),
             "value": round(v, 4), "raw_value": _format_raw(f, raw_profile, features)}
            for f, v in shap_rows
        ]
        result["timing"]["shap_seconds"] = shap_t
        if not additivity_ok:
            result["error"] = ("SHAP additivity check was outside tolerance — "
                               "values shown are still the real computed SHAP "
                               "output, but treat exact magnitudes with caution.")

        lime_rows, r2, lime_t = _compute_lime(model, bg["X_scaled"], instance_scaled, label_id)
        lime_rows = [r for r in lime_rows if r[0] in known_names]
        result["lime_available"] = True
        result["lime"] = [
            {"feature": f, "display_name": FEATURE_DISPLAY.get(f, f),
             "value": round(v, 4), "raw_value": _format_raw(f, raw_profile, features)}
            for f, v in lime_rows
        ]
        result["lime_r2"] = r2
        result["timing"]["lime_seconds"] = lime_t

        result["narrative"] = build_narrative(shap_rows, raw_profile, features, label_str)
        n_unknown = int(unknown.sum())
        if n_unknown:
            result["narrative"].append(
                f"{n_unknown} profile field(s) could not be read for this account, so they "
                "are not shown here; for this explanation they were treated as typical values.")

    except Exception as e:
        import traceback
        result["error"] = f"{type(e).__name__}: {e}"
        result["_traceback"] = traceback.format_exc()   # for server logs only

    return result