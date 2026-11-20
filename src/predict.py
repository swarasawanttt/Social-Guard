"""
src/predict.py
──────────────
The prediction pipeline entry point called by app.py.

Flow:
    1. data_loader.fetch_profile()        → raw profile dict
    2. feature_engineering.engineer_features() → model features (see config.FEATURE_COLUMNS)
    3. Scale with saved StandardScaler
    4. best_model.predict() + predict_proba()
    5. explain.generate_indicators()      → behavioral explanations
    6. xai.explain_prediction()           → real SHAP + LIME explanation
    7. Return full result dict to app.py
"""

import os
import sys
import joblib
import numpy as np

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

from config              import (BEST_MODEL_PATH, BEST_MODEL_NAME_PATH,
                                  SCALER_PATH, FEATURE_COLUMNS, LABEL_MAP,
                                  INCONCLUSIVE_THRESHOLD, PLATFORM_TRAIN_NAME)
from src.scraper             import fetch_profile_data, parse_manual_value
from src.feature_engineering import (engineer_features, features_to_vector,
                                     get_display_features, missing_features)
from src.explain             import generate_indicators, get_summary_sentence
try:
    # Real per-account SHAP + LIME explanation. If shap is missing or this
    # file is absent, everything else still works and the results page shows
    # "explanation not available".
    from src.xai import explain_prediction
except Exception:
    explain_prediction = None


class InvalidProfileInput(Exception):
    """Raised when the link/username is invalid or the account does not exist
    (the scraper reports fetch_result["invalid"]). app.py shows an error page
    instead of asking the user to type in stats for an account that isn't there.
    """
    def __init__(self, message: str, code: str = "invalid_input",
                 suggestion: str = ""):
        self.message = message
        self.code = code
        self.suggestion = suggestion
        super().__init__(message)


class InsufficientData(Exception):
    """Raised when followers, following and posts are all zero.

    Such a profile carries no usable signal (the model was trained with these
    rows removed), so we say so instead of returning a made-up verdict.
    """
    pass


class ManualInputRequired(Exception):
    """
    Raised by run_prediction() when real profile data could not be
    auto-fetched (private account, blocked scrape, or a platform that
    doesn't allow logged-out scraping at all) and the user needs to
    type in the visible stats themselves.

    `context` carries everything app.py needs to render manual_input.html:
        identifier, platform, username, fields, error
    """
    def __init__(self, context: dict):
        self.context = context
        super().__init__(context.get("error", "Manual input required."))


# ── Lazy-loaded singletons ────────────────────────────────────────────────────
_model      = None
_scaler     = None
_model_name = None


def _load():
    global _model, _scaler, _model_name
    if _model is None:
        if not os.path.exists(BEST_MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {BEST_MODEL_PATH}. "
                "Run: python src/train_model.py"
            )
        _model  = joblib.load(BEST_MODEL_PATH)
        _scaler = joblib.load(SCALER_PATH)
        if os.path.exists(BEST_MODEL_NAME_PATH):
            with open(BEST_MODEL_NAME_PATH) as f:
                _model_name = f.read().strip()
        else:
            _model_name = "Unknown"
    return _model, _scaler, _model_name


# ─────────────────────────────────────────────────────────────────────────────
def _run_prediction_from_raw(raw_profile: dict) -> dict:
    """
    Shared second half of the pipeline: takes a raw profile dict
    (however it was obtained — scraped or manually typed in) and runs
    it through feature engineering + the trained model.

    Parameters
    ----------
    raw_profile : dict
        Must contain: username, platform, followers, following, posts,
        bio_len, username_len. May also contain: profile_url, source,
        simulated, is_verified, full_name, etc. (extra keys are ignored
        by feature engineering, but kept for display/report purposes).

    Returns
    -------
    dict — see run_prediction() docstring for the schema.
    """
    model, scaler, model_name = _load()

    # 1. Engineer features
    features = engineer_features(raw_profile)

    if (features["followers"] == 0 and features["following"] == 0
            and features["posts"] == 0):
        raise InsufficientData(
            "This profile shows 0 followers, 0 following and 0 posts, so there is "
            "not enough information to assess it. If this is a brand-new or private "
            "account, or the numbers were read incorrectly, check them and try again."
        )

    # 2. Build and scale feature vector
    # Fields that could not be read stay NaN (never 0); the models were trained
    # with whole feature groups blanked, so they handle that.
    X    = features_to_vector(features)          # shape (1, len(FEATURE_COLUMNS))
    X_sc = scaler.transform(X)

    # 3. Predict
    label_id = int(model.predict(X_sc)[0])
    proba    = model.predict_proba(X_sc)[0]      # length 3

    # Map probabilities to class names using model.classes_
    classes    = model.classes_
    probs_dict = {
        LABEL_MAP[int(cls)]: round(float(p) * 100, 2)
        for cls, p in zip(classes, proba)
    }

    confidence = round(float(proba.max()) * 100, 2)
    top_class  = LABEL_MAP[label_id]

    # The system always answers Real, Fake or Bot. When the answer is weak (low
    # confidence, or a platform with no training data) the verdict is still
    # given, with a plain-language caution attached instead of an abstention.
    no_platform_data = raw_profile.get("platform") not in PLATFORM_TRAIN_NAME
    low_confidence   = float(proba.max()) < INCONCLUSIVE_THRESHOLD
    caution = None
    if no_platform_data:
        caution = (f"The model was not trained on {raw_profile.get('platform')} data, so treat "
                   "this result with extra caution.")
    elif low_confidence:
        caution = (f"The model is only {confidence:.0f}% sure of this result, so treat it "
                   "with caution.")
    inconclusive = False
    reason       = None
    label_str    = top_class

    # Verified-badge rule. A platform-verified account (is_verified is exactly
    # True; None means "could not tell" and does NOT trigger this) is treated as
    # genuine. This is a documented rule layered on top of the model, not a
    # learned result, and it is shown as such. Paid verification exists, so it is
    # strong evidence rather than proof. The model's own probabilities are left
    # untouched and reported as "model score".
    verified_override = raw_profile.get("is_verified") is True
    if verified_override:
        label_str    = "Real"
        label_id     = {v: k for k, v in LABEL_MAP.items()}["Real"]
        inconclusive = False
        reason       = None
        confidence   = probs_dict["Real"]      # the model's own score for Real
        top_class    = "Real"
    unavailable  = missing_features(features)

    # 4. Generate explanations
    indicators = generate_indicators(features, top_class, probs_dict, model_name=model_name,
                                     inconclusive=inconclusive, unavailable=unavailable,
                                     inconclusive_reason=reason)
    summary    = get_summary_sentence(
        label_str, confidence,
        raw_profile["username"], raw_profile["platform"],
        top_class=top_class,
    )
    if caution and not verified_override:
        indicators = list(indicators) + [{"text": caution, "severity": "warning", "icon": "info"}]
    if verified_override:
        summary = (f"@{raw_profile['username']} carries a verified badge on "
                   f"{raw_profile['platform']}, which this system treats as strong evidence "
                   f"of a genuine account (the statistical model alone scored Real at "
                   f"{probs_dict['Real']:.0f}%). Paid verification exists, so this is a "
                   "safeguard rather than proof.")
        indicators = [{
            "text": (f"Verified badge: {raw_profile['platform']} confirms this account's identity, "
                     "so it is classified Real by rule, overriding the model's statistical score."),
            "severity": "positive", "icon": "verified_user",
        }] + list(indicators)

    # 5. Build display features for results page + PDF
    display_features = get_display_features(raw_profile, features)

    # 6. Real SHAP + LIME explanation for THIS account (src/xai.py never raises)
    xai = None
    if explain_prediction is not None:
        xai = explain_prediction(model, scaler, raw_profile, features,
                                 label_id, top_class)

    return {
        "label":            label_str,          # "Real" | "Fake" | "Bot" | "Inconclusive"
        "label_id":         label_id,           # the model's top class id (also when Inconclusive)
        "top_class":        top_class,
        "inconclusive":     inconclusive,
        "verified_override": verified_override,
        "caution":          caution,
        "unavailable_fields": unavailable,
        "confidence":       confidence,
        "probabilities":    probs_dict,
        "indicators":       indicators,
        "summary":          summary,
        "model_used":       model_name,
        "username":         raw_profile["username"],
        "platform":         raw_profile["platform"],
        "display_features": display_features,
        "raw_profile":      raw_profile,
        "data_source":      raw_profile.get("source", "unknown"),
        "xai":              xai,
    }


def run_prediction(identifier: str, platform: str) -> dict:
    """
    Full prediction pipeline for a fresh URL/handle submitted by the user.

    Tries to fetch REAL public profile data first (src/scraper.py):
        - Instagram public accounts → scraped directly, no login needed.
        - Private Instagram accounts, and Twitter/X, Facebook, LinkedIn
          (which block logged-out scraping) → raises ManualInputRequired
          so app.py can show the manual-entry form instead.

    Parameters
    ----------
    identifier : str   URL or @handle entered by the user
    platform   : str   Platform selected from the form

    Raises
    ------
    InvalidProfileInput
        When the link is malformed or the account does not exist.
    ManualInputRequired
        When the account's stats couldn't be auto-fetched. Catch this in
        app.py and redirect to the manual input page using `.context`.

    Returns
    -------
    dict with keys:
        label            str    "Real" | "Fake" | "Bot" | "Inconclusive"
        label_id         int    0 | 1 | 2   (model's top class)
        top_class        str    the model's closest class, even when Inconclusive
        confidence       float  0–100
        probabilities    dict   {label_str: pct_float}
        indicators       list   of indicator dicts (text, severity, icon)
        summary          str    one-sentence verdict
        model_used       str    name of the best model
        username         str
        platform         str
        display_features dict   human-friendly table for report + results page
        raw_profile      dict   raw profile data actually used for prediction
        data_source      str    "scraped" | "manual"
        xai              dict   SHAP + LIME explanation (see src/xai.py)
    """
    fetch_result = fetch_profile_data(identifier, platform)

    if fetch_result.get("invalid"):
        raise InvalidProfileInput(
            fetch_result.get("error") or fetch_result.get("message")
            or "This profile link or username is not valid.",
            code=fetch_result.get("error_code") or fetch_result.get("code")
            or "invalid_input",
            suggestion=fetch_result.get("suggestion") or "",
        )

    if fetch_result["manual_required"]:
        raise ManualInputRequired({
            "identifier":    identifier,
            "platform":      fetch_result["platform"],
            "username":      fetch_result["username"],
            "fields":        fetch_result["manual_fields"],
            "error":         fetch_result["error"] or "Manual input required.",
        })

    raw_profile = {
        "username":    fetch_result["username"],
        "platform":    fetch_result["platform"],
        "profile_url": identifier,
        "simulated":   False,
        "source":      fetch_result["source"],
        **fetch_result["data"],
    }
    return _run_prediction_from_raw(raw_profile)


def _clean_bio(text) -> str:
    """Normalise pasted bio text: browsers send line breaks as \\r\\n, which
    would be counted twice, so convert to \\n and trim the ends."""
    if not text:
        return ""
    return str(text).replace("\r\n", "\n").replace("\r", "\n").strip()[:2000]


def _bio_length(form) -> int:
    """Bio length, counted on the SERVER from the pasted text (characters,
    so an emoji counts as one). Falls back to a numeric bio_len only when no
    bio_text field was submitted at all."""
    if form.get("bio_text") is not None:
        return len(_clean_bio(form.get("bio_text")))
    try:
        return max(0, int(float(form.get("bio_len") or 0)))
    except (TypeError, ValueError):
        return 0


def run_prediction_manual(identifier: str, platform: str, username: str,
                           form) -> dict:
    """
    Prediction pipeline for stats the user typed in by hand (manual_input.html),
    used when run_prediction() raised ManualInputRequired.

    Parameters
    ----------
    identifier : str   original URL/handle the user first entered
    platform   : str   platform name (e.g. "Twitter/X")
    username   : str   username parsed earlier from the URL
    form       : dict-like (e.g. flask's request.form)
        Expected keys: followers, following, posts, bio_len, username_len.
        followers/following/posts may be human strings like "1.2K" or
        "239.8M" — parsed with scraper.parse_manual_value(). bio_len and
        username_len are plain integers.

    Returns
    -------
    Same schema as run_prediction(), with data_source="manual".
    """
    raw_profile = {
        "username":     username,
        "platform":     platform,
        "profile_url":  identifier,
        "simulated":    False,
        "source":       "manual",
        "followers":    parse_manual_value(form.get("followers")),
        "following":    parse_manual_value(form.get("following")),
        "posts":        parse_manual_value(form.get("posts")),
        "bio_len":      _bio_length(form),
        "bio_text":     _clean_bio(form.get("bio_text")),
        "username_len": int(form.get("username_len") or (len(username) if username else 0)),
        # Optional "Verified badge?" choice on the form: "yes" / "no" / blank (not sure).
        "is_verified":  {"yes": True, "no": False}.get(str(form.get("is_verified") or "").strip().lower()),
    }
    return _run_prediction_from_raw(raw_profile)