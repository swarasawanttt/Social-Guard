"""
src/feature_engineering.py
───────────────────────────
Turns a raw profile dict (scraped, typed in, or read from screenshots) into the
16 model features listed in config.FEATURE_COLUMNS.

Rule: a field that is not available is NaN, never 0. The model was trained with
whole feature groups randomly blanked, so it knows how to cope; inventing a 0
would claim "no profile picture", "no link", "username has no digits" etc.

Optional raw_profile keys (all may be absent):
    username, username_len, bio_len, bio_text,
    external_url / has_url, full_name, has_profile_pic, is_private,
    is_verified (True/False/None; used for display and the verified-badge rule, not a model feature)
"""
import math
import re

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
from config import (FEATURE_COLUMNS, PERCENTILE_BASE, PERCENTILE_TABLES_PATH,
                    PLATFORM_TRAIN_NAME)
from src.model_utils import derived_features, load_tables, percentile_rank

NAN = float("nan")
_URL_RE = re.compile(r"(https?://|www\.|\b[a-z0-9-]+\.(com|in|net|org|io|co|me|ly|link|bio)\b)", re.I)


# ─────────────────────────────────────────────────────────────────────────────
def _clean_username(raw_profile: dict) -> str:
    return str(raw_profile.get("username") or "").strip().lstrip("@")


def _norm_name(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _flag(v):
    """True/False/1/0/'true'/'false' -> 1.0/0.0 ; None/missing -> NaN."""
    if v is None:
        return NAN
    if isinstance(v, str):
        t = v.strip().lower()
        if t in ("1", "true", "yes", "y"):
            return 1.0
        if t in ("0", "false", "no", "n"):
            return 0.0
        return NAN
    return 1.0 if bool(v) else 0.0


def _num_or_nan(v):
    try:
        x = float(v)
        return NAN if math.isnan(x) else x
    except (TypeError, ValueError):
        return NAN


# ─────────────────────────────────────────────────────────────────────────────
def engineer_features(raw_profile: dict) -> dict:
    """Returns a dict with exactly the keys in FEATURE_COLUMNS (NaN = unknown)."""
    followers = float(raw_profile.get("followers", 0) or 0)
    following = float(raw_profile.get("following", 0) or 0)
    posts     = float(raw_profile.get("posts",     0) or 0)

    username = _clean_username(raw_profile)
    ulen = _num_or_nan(raw_profile.get("username_len"))
    if math.isnan(ulen) or ulen <= 0:
        ulen = float(len(username)) if username else NAN
    digit_ratio = (sum(c.isdigit() for c in username) / len(username)) if username else NAN

    bio_len = _num_or_nan(raw_profile.get("bio_len"))
    bio_text = raw_profile.get("bio_text")

    # has_url = the profile has a website/link field. Prefer an explicit field;
    # otherwise look for a URL inside the pasted bio text; otherwise unknown.
    ext = raw_profile.get("external_url")
    if isinstance(ext, str) and ext.strip():
        has_url = 1.0                      # a real link is present
    elif isinstance(ext, str):
        has_url = 0.0                      # the profile has an explicit, empty link field
    else:
        has_url = _flag(raw_profile.get("has_url"))
    if math.isnan(has_url) and isinstance(bio_text, str) and bio_text.strip():
        has_url = 1.0 if _URL_RE.search(bio_text) else NAN   # a URL proves a link; absence proves nothing

    if "full_name" in raw_profile and raw_profile.get("full_name") is not None:
        full = str(raw_profile.get("full_name") or "")
        fullname_words = float(len(full.split()))
        name_eq = 1.0 if (_norm_name(full) and _norm_name(full) == _norm_name(username)) else 0.0
    else:
        fullname_words, name_eq = NAN, NAN

    ratio, activity = derived_features(followers, following, posts)

    # Per-platform percentile ranks of the three counts
    pr = {f"{c}_pr": NAN for c in PERCENTILE_BASE}
    tables = load_tables(PERCENTILE_TABLES_PATH)
    if tables:
        plat = PLATFORM_TRAIN_NAME.get(raw_profile.get("platform"), "ALL")
        tab = tables.get(plat) or tables.get("ALL")
        if tab:
            vals = {"followers": followers, "following": following, "posts": posts}
            for c in PERCENTILE_BASE:
                pr[f"{c}_pr"] = float(percentile_rank(vals[c], tab[c]))

    features = {
        "followers": followers, "following": following, "posts": posts,
        "bio_len": bio_len, "has_url": has_url,
        "username_len": ulen, "username_digit_ratio": digit_ratio,
        "fullname_words": fullname_words, "name_eq_username": name_eq,
        "has_profile_pic": _flag(raw_profile.get("has_profile_pic")),
        "is_private": _flag(raw_profile.get("is_private")),
        "follower_following_ratio": ratio, "activity_rate": activity,
        **pr,
    }
    for col in FEATURE_COLUMNS:
        if col not in features:
            raise ValueError(f"feature_engineering: missing expected column '{col}'")
    return features


def features_to_vector(features: dict) -> np.ndarray:
    """dict -> array in config.FEATURE_COLUMNS order, shape (1, n). NaN preserved."""
    return np.array([[features[c] for c in FEATURE_COLUMNS]], dtype=float)


def missing_features(features: dict) -> list:
    """Names of features that were not available for this account."""
    return [c for c in FEATURE_COLUMNS if isinstance(features[c], float) and math.isnan(features[c])]


# ─────────────────────────────────────────────────────────────────────────────
def get_display_features(raw_profile: dict, features: dict) -> dict:
    """Human-friendly table for the results page and PDF. Only fields we
    actually know are shown, so nothing is displayed as a made-up value."""
    from src.utils import format_number

    def known(k):
        v = features[k]
        return not (isinstance(v, float) and math.isnan(v))

    out = {
        "Username":  raw_profile.get("username", "—"),
        "Platform":  raw_profile.get("platform", "—"),
        "Followers": format_number(raw_profile.get("followers", 0)),
        "Following": format_number(raw_profile.get("following", 0)),
        "Posts":     format_number(raw_profile.get("posts", 0)),
    }
    if known("bio_len"):
        out["Bio Length"] = f"{int(features['bio_len'])} chars"
    if known("username_len"):
        out["Username Length"] = f"{int(features['username_len'])} chars"
    if known("username_digit_ratio"):
        out["Digits in Username"] = f"{features['username_digit_ratio'] * 100:.0f}%"
    if known("has_url"):
        out["Link in Profile"] = "Yes" if features["has_url"] >= 0.5 else "No"
    if known("has_profile_pic"):
        out["Profile Picture"] = "Yes" if features["has_profile_pic"] >= 0.5 else "No"
    if known("is_private"):
        out["Private Account"] = "Yes" if features["is_private"] >= 0.5 else "No"
    if known("fullname_words"):
        out["Full-name Words"] = str(int(features["fullname_words"]))
    if known("name_eq_username"):
        out["Name = Username"] = "Yes" if features["name_eq_username"] >= 0.5 else "No"
    if raw_profile.get("is_verified") is not None:
        out["Verified Badge"] = "Yes" if raw_profile.get("is_verified") else "No"
    out["Follower/Following Ratio"] = f"{features['follower_following_ratio']:.4f}"
    out["Activity Rate"] = f"{features['activity_rate']:.4f}"
    if known("followers_pr"):
        out["Follower Percentile"] = f"{features['followers_pr'] * 100:.0f}th (on this platform)"
    return out