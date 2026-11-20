"""
src/platform_check.py
─────────────────────
Checks that the platform the user SELECTED matches the website in the link
they pasted (e.g. Instagram selected + a twitter.com link = error).

    check_platform_matches(identifier, selected_platform) -> (ok, message)

Only links are checked. A bare username or @handle has no website in it, so
it is always accepted here. Hosts we do not recognise are also accepted and
left to the other validators (typo / lookalike-host checks).
"""

# website -> platform name used by the app
_HOSTS = {
    "Instagram": ("instagram.com", "instagr.am"),
    "Twitter/X": ("twitter.com", "x.com", "t.co"),
    "Facebook":  ("facebook.com", "fb.com", "fb.me"),
    "LinkedIn":  ("linkedin.com", "lnkd.in"),
}


def normalise_platform(selected: str):
    """Map the dropdown text ('LinkedIn (Academic)', 'Twitter/X' ...) to a key."""
    s = (selected or "").strip().lower()
    if s.startswith("insta"):
        return "Instagram"
    if s.startswith("twitter") or s in ("x", "x.com", "twitter/x"):
        return "Twitter/X"
    if s.startswith("face") or s == "fb":
        return "Facebook"
    if s.startswith("linked"):
        return "LinkedIn"
    return None


def _host_of(identifier: str) -> str:
    """Website part of a pasted link, lower-case, without port or user-info."""
    s = (identifier or "").strip().lower()
    if "://" in s:
        s = s.split("://", 1)[1]
    s = s.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
    s = s.rsplit("@", 1)[-1]          # drop user-info (user@host)
    return s.split(":", 1)[0]         # drop port


def detect_platform_from_link(identifier: str):
    """Platform whose website the link points to, or None."""
    host = _host_of(identifier)
    if not host or identifier.strip().startswith("@"):
        return None
    for platform, domains in _HOSTS.items():
        for d in domains:
            if host == d or host.endswith("." + d):
                return platform
    return None


def check_platform_matches(identifier: str, selected_platform: str):
    """Return (ok, message). message is '' when ok."""
    chosen = normalise_platform(selected_platform)
    found = detect_platform_from_link(identifier)
    if chosen and found and chosen != found:
        return False, (
            f"You selected {chosen}, but this link is for {found}. "
            f"Choose {found} as the platform, or paste "
            f"{'an' if chosen[0] in 'AEIOU' else 'a'} {chosen} profile link."
        )
    return True, ""