"""
src/data_loader.py
──────────────────
Simulates fetching publicly available profile data for a given
social media account URL.

Approach:
  - Parse the username from the URL
  - Use the username as a deterministic seed (same URL → same values always)
  - Generate realistic profile metrics based on platform-specific ranges
  - The generated values mimic what a real API scrape would return:
    followers, following, posts, bio_len, username_len

This is the demo/prototype data collection layer. When a real API
integration is added later, only this file needs to change — everything
downstream (feature_engineering, predict, app) stays the same.
"""

import os
import sys
import hashlib
import random

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
from config import PLATFORM_RANGES
from src.utils  import parse_url


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _seed_from_username(username: str, platform: str) -> int:
    """
    Convert username + platform into a stable integer seed.
    SHA-256 ensures the same input always gives the same seed,
    making results reproducible (same URL = same result every time).
    """
    key   = f"{platform.lower()}::{username.lower()}"
    digest = hashlib.sha256(key.encode()).hexdigest()
    return int(digest[:8], 16)   # first 8 hex chars → int


def _seeded_int(rng: random.Random, lo: int, hi: int) -> int:
    return rng.randint(lo, hi)


def _seeded_float(rng: random.Random, lo: float, hi: float) -> float:
    return rng.uniform(lo, hi)


# ─────────────────────────────────────────────────────────────────────────────
# Account type profiles
# (weighted probability that a seeded account falls into each bucket)
# ─────────────────────────────────────────────────────────────────────────────

# Each bucket maps to different realistic follower/following distributions
# that produce genuinely different feature patterns for the model to classify.

ACCOUNT_PROFILES = {
    # Real accounts: organic follower growth, reasonable ff ratio, bio present
    "organic_micro": {
        "weight": 0.35,
        "followers_range": (50,   5_000),
        "following_range": (50,   1_000),
        "posts_range":     (10,   500),
        "bio_len_range":   (30,   150),
    },
    "organic_established": {
        "weight": 0.20,
        "followers_range": (5_000, 500_000),
        "following_range": (100,   5_000),
        "posts_range":     (100,   3_000),
        "bio_len_range":   (50,    200),
    },
    # Fake accounts: low followers, high following, little/no bio
    "fake_new": {
        "weight": 0.15,
        "followers_range": (0,    200),
        "following_range": (500,  5_000),
        "posts_range":     (0,    30),
        "bio_len_range":   (0,    20),
    },
    "fake_aged": {
        "weight": 0.10,
        "followers_range": (10,   800),
        "following_range": (1_000, 8_000),
        "posts_range":     (5,    100),
        "bio_len_range":   (0,    40),
    },
    # Bot accounts: very high posts, extreme ff imbalance, minimal bio
    "bot_activity": {
        "weight": 0.12,
        "followers_range": (0,    500),
        "following_range": (2_000, 50_000),
        "posts_range":     (500,  20_000),
        "bio_len_range":   (0,    10),
    },
    "bot_follower_farm": {
        "weight": 0.08,
        "followers_range": (10_000, 500_000),
        "following_range": (10_000, 200_000),
        "posts_range":     (0,      50),
        "bio_len_range":   (0,      5),
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Main function
# ─────────────────────────────────────────────────────────────────────────────

def fetch_profile(identifier: str, platform: str) -> dict:
    """
    Simulate fetching public profile data for the given account.

    Parameters
    ----------
    identifier : str   URL or @handle as entered by the user
    platform   : str   Selected platform from the form

    Returns
    -------
    dict with keys:
        username    : str
        platform    : str
        followers   : int
        following   : int
        posts       : int
        bio_len     : int
        username_len: int
        profile_url : str   (the original input)
        simulated   : bool  (always True in this version)
    """
    # 1. Parse username from URL
    parsed   = parse_url(identifier)
    username = parsed["username"] if parsed["username"] else identifier.strip()

    # Override platform if URL parsing found one and user selected "auto"
    if platform in ("", "auto") and parsed["platform"] != "Unknown":
        platform = parsed["platform"]
    if not platform:
        platform = "Instagram"   # default fallback

    # 2. Seed the RNG deterministically
    seed = _seed_from_username(username, platform)
    rng  = random.Random(seed)

    # 3. Pick an account profile bucket using weighted random selection
    profiles = list(ACCOUNT_PROFILES.keys())
    weights  = [ACCOUNT_PROFILES[p]["weight"] for p in profiles]
    chosen   = rng.choices(profiles, weights=weights, k=1)[0]
    profile  = ACCOUNT_PROFILES[chosen]

    # 4. Generate values within the chosen profile's ranges
    # Apply platform-level caps so Instagram values don't exceed Twitter norms etc.
    p_ranges = PLATFORM_RANGES.get(platform, PLATFORM_RANGES["Instagram"])

    followers = min(
        _seeded_int(rng, *profile["followers_range"]),
        p_ranges["followers"][1]
    )
    following = min(
        _seeded_int(rng, *profile["following_range"]),
        p_ranges["following"][1]
    )
    posts = min(
        _seeded_int(rng, *profile["posts_range"]),
        p_ranges["posts"][1]
    )
    bio_len = min(
        _seeded_int(rng, *profile["bio_len_range"]),
        p_ranges["bio_len"][1]
    )
    username_len = min(len(username), p_ranges["username_len_max"])

    return {
        "username":     username,
        "platform":     platform,
        "followers":    followers,
        "following":    following,
        "posts":        posts,
        "bio_len":      bio_len,
        "username_len": username_len,
        "profile_url":  identifier,
        "simulated":    True,
        "account_profile_type": chosen,   # for internal debugging
    }