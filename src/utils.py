"""
src/utils.py
────────────
Utility helpers:
  - parse_url()        extract platform + username from any social URL / @handle
  - validate_input()   basic sanity checks before prediction
  - format_number()    pretty-print large integers  (1_200_000 → "1.2M")
  - timestamp()        current datetime string for report filenames
"""

import re
import os
import sys
from datetime import datetime

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
from config import PLATFORM_PATTERNS
from src.platform_check import check_platform_matches


# ─────────────────────────────────────────────────────────────────────────────
def parse_url(raw: str) -> dict:
    """
    Parse a social media URL or @handle and return platform + username.

    Accepts any of:
        https://www.instagram.com/cristiano/
        instagram.com/cristiano
        @cristiano
        cristiano          (bare username — platform must be selected separately)

    Returns
    -------
    dict:
        platform : str   e.g. "Instagram"  or "Unknown"
        username : str   e.g. "cristiano"
        raw      : str   original input
        valid    : bool
        error    : str | None
    """
    raw = raw.strip()

    if not raw:
        return {"platform": "Unknown", "username": "", "raw": raw,
                "valid": False, "error": "Empty input."}

    # Try each platform pattern
    for platform, pattern in PLATFORM_PATTERNS.items():
        m = pattern.search(raw)
        if m:
            username = m.group(1).rstrip("/")
            return {"platform": platform, "username": username,
                    "raw": raw, "valid": True, "error": None}

    # Bare @handle with no URL
    at_match = re.match(r"@?([\w.]+)$", raw)
    if at_match:
        username = at_match.group(1)
        return {"platform": "Unknown", "username": username,
                "raw": raw, "valid": True, "error": None}

    return {"platform": "Unknown", "username": raw,
            "raw": raw, "valid": False,
            "error": "Could not extract a username from the input."}


# ─────────────────────────────────────────────────────────────────────────────
def validate_input(identifier: str, platform: str) -> tuple[bool, str]:
    """
    Returns (ok: bool, error_message: str).
    error_message is empty string when ok=True.
    """
    if not identifier or not identifier.strip():
        return False, "No profile URL or username was provided."

    if not platform or platform.strip() == "":
        return False, "Please select a platform."

    # The platform chosen in the dropdown must match the website in the link
    # (e.g. Instagram selected + a twitter.com link is an error).
    ok, msg = check_platform_matches(identifier, platform)
    if not ok:
        return False, msg

    # Offline check: typo'd domains, post links, illegal usernames. Whether the
    # account really exists is checked later by the scraper.
    from src.url_resolver import resolve_target
    target = resolve_target(identifier, platform)
    if not target["ok"]:
        msg = target["error"]
        if target["suggestion"]:
            msg += " " + target["suggestion"]
        return False, msg

    return True, ""


# ─────────────────────────────────────────────────────────────────────────────
def format_number(n) -> str:
    """
    Pretty-print a large integer.
      1_234         → "1,234"
      1_234_567     → "1.2M"
      1_234_567_890 → "1.2B"
    """
    try:
        n = float(n)
    except (TypeError, ValueError):
        return str(n)

    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n:,.0f}"
    return str(int(n))


# ─────────────────────────────────────────────────────────────────────────────
def timestamp() -> str:
    """Return a filesystem-safe timestamp string e.g. '20240421_153022'"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ─────────────────────────────────────────────────────────────────────────────
def safe_filename(username: str, platform: str) -> str:
    """
    Build a safe filename for the PDF report.
    e.g. "report_instagram_cristiano_20240421_153022.pdf"
    """
    uname = re.sub(r"[^\w]", "_", username.lower())
    plat  = re.sub(r"[^\w]", "_", platform.lower())
    return f"report_{plat}_{uname}_{timestamp()}.pdf"