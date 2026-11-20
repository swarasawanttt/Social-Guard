"""
src/url_resolver.py
───────────────────
Offline (no network) validation of whatever the user typed into the
"profile URL or handle" box.

It answers four questions BEFORE any request is made:
  1. Which platform is this?                (URL host, or the dropdown choice)
  2. Is the host spelled correctly?         (instagarm.com → "did you mean instagram.com?")
  3. Is it a profile link, not a post/reel/explore/login link?
  4. Is the username legal on that platform? (length, allowed characters)

Whether the account actually EXISTS can only be known by asking the
platform, so that part lives in scraper.py. This module only catches
mistakes that are provably wrong from the text alone.

Public API
----------
resolve_target(identifier, platform_hint="") -> dict
    {
      "ok":        bool,
      "platform":  "Instagram" | "Twitter/X" | "Facebook" | "LinkedIn" | "",
      "username":  str,
      "url":       canonical profile URL (when ok),
      "error_code": None | "empty" | "wrong_domain" | "unsupported_site"
                    | "not_a_profile" | "bad_username" | "no_username",
      "error":     human-readable message (when not ok),
      "suggestion": optional "did you mean ..." text,
    }
"""

import re
import difflib
from urllib.parse import urlparse, parse_qs, unquote


# ── Known hosts → platform ───────────────────────────────────────────────────
HOSTS = {
    "instagram.com":  "Instagram",
    "instagr.am":     "Instagram",
    "twitter.com":    "Twitter/X",
    "x.com":          "Twitter/X",
    "facebook.com":   "Facebook",
    "fb.com":         "Facebook",
    "fb.me":          "Facebook",
    "linkedin.com":   "LinkedIn",
}

# Sub-domains that are just front doors to the same site.
_IGNORED_SUBDOMAINS = {"www", "m", "mobile", "web", "touch", "mbasic", "l", "lm"}

# Username rules per platform (conservative: reject only what is provably illegal).
_USERNAME_RULES = {
    "Instagram": {
        "regex": re.compile(r"^[A-Za-z0-9._]{1,30}$"),
        "extra": lambda u: not (".." in u or u.endswith(".")),
        "hint":  "Instagram usernames are 1-30 characters: letters, numbers, "
                 "periods and underscores only (no spaces, no '..', can't end with '.').",
    },
    "Twitter/X": {
        "regex": re.compile(r"^[A-Za-z0-9_]{1,15}$"),
        "extra": lambda u: True,
        "hint":  "X/Twitter usernames are 1-15 characters: letters, numbers and "
                 "underscores only.",
    },
    "Facebook": {
        # New vanity names need 5+ chars, but older accounts (e.g. /zuck) are
        # shorter, so only reject what is provably illegal.
        "regex": re.compile(r"^[A-Za-z0-9.]{1,50}$"),
        "extra": lambda u: True,
        "hint":  "Facebook usernames use letters, numbers and periods only (max 50 characters).",
    },
    "LinkedIn": {
        "regex": re.compile(r"^[A-Za-z0-9\-_%]{3,100}$"),
        "extra": lambda u: True,
        "hint":  "LinkedIn profile links look like linkedin.com/in/your-name.",
    },
}

# First path segments that are NOT profiles.
_RESERVED = {
    "Instagram": {
        "p", "reel", "reels", "tv", "stories", "explore", "accounts", "direct",
        "about", "legal", "developer", "privacy", "web", "directory", "challenge",
        "oauth", "emails", "session", "static", "api", "graphql", "download",
        "business", "creators", "press", "help", "locations", "tags", "login",
    },
    "Twitter/X": {
        "home", "explore", "search", "notifications", "messages", "i", "intent",
        "settings", "login", "signup", "compose", "hashtag", "share", "tos",
        "privacy", "about", "jobs", "download", "who_to_follow",
    },
    "Facebook": {
        "login", "login.php", "groups", "pages", "events", "marketplace", "watch",
        "gaming", "stories", "share", "sharer", "sharer.php", "photo", "photo.php",
        "video", "videos", "reel", "help", "policies", "privacy", "business",
        "dialog", "plugins", "tr", "ads", "hashtag", "public", "people", "story.php",
    },
    "LinkedIn": set(),   # profiles are always /in/<slug>
}

_POST_HINTS = {"p": "post", "reel": "reel", "reels": "reel", "tv": "video",
               "stories": "story"}


def _err(code, message, suggestion=""):
    return {"ok": False, "platform": "", "username": "", "url": "",
            "error_code": code, "error": message, "suggestion": suggestion}


def _clean_host(host: str) -> str:
    host = host.lower().strip().rstrip(".")
    if host.startswith("www."):
        host = host[4:]
    parts = host.split(".")
    # drop front-door sub-domains (m.facebook.com, mobile.twitter.com ...)
    while len(parts) > 2 and parts[0] in _IGNORED_SUBDOMAINS:
        parts = parts[1:]
    return ".".join(parts)


def _match_host(host: str):
    """(platform, matched_host) for an exact/sub-domain match, else (None, None)."""
    for known, platform in HOSTS.items():
        if host == known or host.endswith("." + known):
            return platform, known
    return None, None


def _suggest_host(host: str) -> str:
    """Closest known domain to a mistyped one, or ''."""
    # compare on the registrable part first: 'instagarm.com' vs 'instagram.com'
    candidates = difflib.get_close_matches(host, list(HOSTS), n=1, cutoff=0.78)
    if candidates:
        return candidates[0]
    base = host.split(".")[0]
    bases = {k.split(".")[0]: k for k in HOSTS}
    close = difflib.get_close_matches(base, list(bases), n=1, cutoff=0.75)
    return bases[close[0]] if close else ""


def _looks_like_url(s: str) -> bool:
    if re.match(r"^[a-z][a-z0-9+.\-]*://", s, re.I):
        return True
    # "instagram.com/foo", "www.instagarm.com/foo", "foo.com"
    return bool(re.match(r"^(www\.)?[A-Za-z0-9\-]+(\.[A-Za-z0-9\-]+)+(/|$|\?)", s)) \
        and not re.match(r"^@", s)


# ─────────────────────────────────────────────────────────────────────────────
def _validate_username(platform: str, username: str):
    rule = _USERNAME_RULES.get(platform)
    if not rule:
        return None
    if not (rule["regex"].match(username) and rule["extra"](username)):
        return _err(
            "bad_username",
            f"“{username}” can't be a valid {platform} username.",
            rule["hint"],
        )
    return None


def _canonical_url(platform: str, username: str) -> str:
    return {
        "Instagram": f"https://www.instagram.com/{username}/",
        "Twitter/X": f"https://x.com/{username}",
        "Facebook":  f"https://www.facebook.com/{username}",
        "LinkedIn":  f"https://www.linkedin.com/in/{username}/",
    }[platform]


def _ok(platform, username):
    return {"ok": True, "platform": platform, "username": username,
            "url": _canonical_url(platform, username),
            "error_code": None, "error": "", "suggestion": ""}


# ─────────────────────────────────────────────────────────────────────────────
def resolve_target(identifier: str, platform_hint: str = "") -> dict:
    raw = (identifier or "").strip().strip("<>\"'")
    if not raw:
        return _err("empty", "No profile URL or username was provided.")

    hint = platform_hint if platform_hint in _USERNAME_RULES else ""

    # ── Case A: looks like a URL ─────────────────────────────────────────────
    if _looks_like_url(raw):
        to_parse = raw if "://" in raw else "https://" + raw
        try:
            u = urlparse(to_parse)
        except ValueError:
            return _err("wrong_domain", "That doesn't look like a valid web address.")
        host = _clean_host(u.hostname or "")
        platform, _ = _match_host(host)

        if not platform:
            close = _suggest_host(host)
            if close:
                fixed = re.sub(re.escape(u.hostname or host), close, to_parse, count=1,
                               flags=re.I)
                return _err(
                    "wrong_domain",
                    f"“{host}” isn't a supported site — did you misspell it?",
                    f"Did you mean {close}?  Try: {fixed}",
                )
            return _err(
                "unsupported_site",
                f"“{host}” isn't a supported platform.",
                "Supported: instagram.com, x.com / twitter.com, facebook.com, linkedin.com.",
            )

        segs = [unquote(s) for s in u.path.split("/") if s]
        qs = parse_qs(u.query)

        # LinkedIn: /in/<slug>
        if platform == "LinkedIn":
            if len(segs) >= 2 and segs[0].lower() == "in":
                slug = segs[1]
                bad = _validate_username(platform, slug)
                return bad or _ok(platform, slug)
            if segs and segs[0].lower() in {"company", "school", "pub", "posts", "feed"}:
                return _err("not_a_profile",
                            "That's a LinkedIn company/page/post link, not a personal profile.",
                            "Personal profiles look like linkedin.com/in/your-name.")
            return _err("no_username", "No LinkedIn profile found in that link.",
                        "Personal profiles look like linkedin.com/in/your-name.")

        # Facebook: profile.php?id=123
        if platform == "Facebook" and segs and segs[0].lower() == "profile.php":
            ids = qs.get("id", [])
            if ids and ids[0].isdigit():
                return _ok(platform, ids[0])
            return _err("no_username", "That Facebook link has no profile id.")

        if not segs:
            return _err("no_username",
                        f"That's the {platform} home page — there's no username in it.",
                        "Paste the full profile link, e.g. " + _canonical_url(platform, "username"))

        first = segs[0]
        # X/Twitter profile links sometimes carry '@'
        first = first.lstrip("@")

        if first.lower() in _RESERVED.get(platform, set()):
            kind = _POST_HINTS.get(first.lower())
            if kind:
                return _err("not_a_profile",
                            f"That's a link to a {kind}, not to a profile.",
                            "Open the account's profile page and copy that address instead.")
            return _err("not_a_profile",
                        f"“{first}” is a {platform} page, not a user profile.",
                        "Paste a profile link like " + _canonical_url(platform, "username"))

        bad = _validate_username(platform, first)
        return bad or _ok(platform, first)

    # ── Case B: bare handle, with or without '@' ─────────────────────────────
    handle = raw.lstrip("@").strip()
    if not handle:
        return _err("no_username", "Only '@' was entered — add the username after it.")

    if not hint:
        return _err("no_username",
                    "Choose a platform, or paste the full profile link.",
                    "")

    if re.search(r"\s", handle):
        return _err("bad_username",
                    f"“{handle}” contains spaces, so it can't be a {hint} username.",
                    _USERNAME_RULES[hint]["hint"])

    if handle.lower() in _RESERVED.get(hint, set()):
        return _err("not_a_profile", f"“{handle}” is a reserved {hint} page name, not a user.")

    bad = _validate_username(hint, handle)
    return bad or _ok(hint, handle)