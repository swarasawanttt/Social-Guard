"""
src/scraper.py
──────────────
Fetches REAL, publicly visible profile statistics for a profile URL / handle.

Design goals
────────────
1. Never trust a single page layout.
   Social sites change their HTML constantly. Instead of one hard-coded path
   (e.g. data["graphql"]["user"]["edge_followed_by"]["count"]) this module runs
   several INDEPENDENT extraction layers on every response and merges what they
   find:

     Layer A  <meta> tags (og:description / description)   "1.2M Followers, 560 Following, 3,800 Posts"
     Layer B  embedded JSON / JSON-LD / JSON-inside-strings, matched by the
              MEANING of a key (follower_count, edge_followed_by, followersCount,
              friends_count, ...) and by owner (the object whose username equals
              the one we asked for), NOT by its position in the document
     Layer C  visible page text ("3,800 posts")
     Layer D  loose regex over raw script text (last resort, lowest trust)

   If the site renames a key or moves a blob, other layers still work. If ALL
   layers find nothing on a page that loaded fine, that is reported as
   `layout_changed` and the raw HTML is saved to data/scrape_debug/ so the
   extractor can be updated in minutes instead of hours.

2. Say WHY it failed, precisely.
   error_code is one of:
     invalid_input   typo'd URL/domain, post link, illegal username (offline check)
     not_found       the platform says the account does not exist (HTTP 404 /
                     "page isn't available"), so manual entry would be pointless
     private         account is private and its numbers are hidden
     blocked         platform demands a login / rate-limited us
     layout_changed  page loaded but nothing recognisable was in it
     network_error   couldn't reach the platform at all
     incomplete      some numbers found, others missing (e.g. bio)

3. Never fabricate and never trust silently.
   Anything not found is left blank; anything found with only weak evidence is
   returned as a PRE-FILLED manual form so the user confirms it.

Public API (unchanged contract used by predict.py / app.py)
───────────────────────────────────────────────────────────
    fetch_profile_data(identifier, platform) -> dict
    parse_manual_value(raw) -> int
    MANUAL_FIELDS, build_manual_fields()
    scraper_health() -> dict           (which strategies have been working)

Self-test from a terminal on a machine with internet access:
    python src/scraper.py cristiano instagram
    python src/scraper.py https://www.instagram.com/some_user/
    python src/scraper.py --html saved_page.html --user cristiano --platform Instagram
"""

import os
import re
import sys
import json
import gzip
import zlib
import copy
import time
import html as _htmllib
import urllib.request
import urllib.error
from html.parser import HTMLParser
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.url_resolver import resolve_target


# ─────────────────────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────────────────────
TIMEOUT          = 8          # seconds per request
MAX_BODY_BYTES   = 4_000_000  # never read more than this from one response
CACHE_TTL_OK     = 300        # seconds a good / not-found result is reused
CACHE_TTL_BLOCK  = 60         # seconds a blocked / failed result is reused
MIN_HOST_GAP     = 0.8        # seconds between requests to the same host
DEBUG_DIR        = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "scrape_debug")

UA_BROWSER = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
# Honest, self-identifying fallback agent (some sites serve link-preview
# metadata to unknown bots even when they wall off browsers).
UA_BOT = "Mozilla/5.0 (compatible; SocialGuardBot/1.0; academic profile-metadata reader)"

_BASE_HEADERS = {
    "Accept":          "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    # NOTE: only encodings urllib can decode here. The old scraper advertised
    # "br" (Brotli) but could only unzip gzip, so Brotli replies came back
    # as unreadable bytes and every parse silently failed.
    "Accept-Encoding": "gzip, deflate",
}

_STAT_FIELDS = ("followers", "following", "posts")
SAVE_ALL_HTML = False      # set by `--save-html`: keep every raw response for diagnosis


# ─────────────────────────────────────────────────────────────────────────────
# Counts: "1.2M" / "3,800" / "239.8K" -> int
# ─────────────────────────────────────────────────────────────────────────────
def parse_count(raw):
    """
    Returns (value:int|None, rounded:bool).  rounded=True when the text used a
    K/M/B abbreviation, i.e. the true figure is only known approximately.
    """
    if raw is None:
        return None, False
    t = str(raw).strip().replace(" ", "").replace(" ", "").replace(" ", "")
    m = re.fullmatch(r"(\d[\d.,]*)([KkMmBb])?", t)
    if not m:
        return None, False
    num, suf = m.group(1), m.group(2)
    if suf:
        n = num.replace(",", ".")
        if n.count(".") > 1:
            n = n.replace(".", "")
        try:
            val = float(n)
        except ValueError:
            return None, False
        return int(round(val * {"K": 1e3, "M": 1e6, "B": 1e9}[suf.upper()])), True
    digits = re.sub(r"[.,]", "", num)
    return (int(digits), False) if digits else (None, False)


def parse_manual_value(raw: str) -> int:
    """User-typed count ("239.8M", "1.2K", "500+", "1,234") -> int; 0 if unreadable."""
    if raw is None:
        return 0
    s = str(raw).strip().replace("+", "")
    if not s:
        return 0
    val, _ = parse_count(s)
    if val is not None:
        return val
    try:
        return int(float(s.replace(",", "")))
    except ValueError:
        return 0


def _sane(v):
    return v is not None and 0 <= v <= 5_000_000_000


# ─────────────────────────────────────────────────────────────────────────────
# HTTP
# ─────────────────────────────────────────────────────────────────────────────
_last_hit = {}


def _http_get(url, headers=None, timeout=TIMEOUT, retries=1):
    """
    Returns {"status": int, "headers": {...lowercase}, "body": str,
             "final_url": str, "error": str|None}.
    HTTP errors (404, 403 ...) are returned WITH their body — a 404 page tells
    us the account doesn't exist. status 0 means we never got a response.
    """
    host = urlparse(url).hostname or ""
    hdrs = dict(_BASE_HEADERS)
    hdrs["User-Agent"] = UA_BROWSER
    hdrs.update(headers or {})

    for attempt in range(retries + 1):
        gap = MIN_HOST_GAP - (time.time() - _last_hit.get(host, 0))
        if gap > 0:
            time.sleep(min(gap, MIN_HOST_GAP))
        _last_hit[host] = time.time()

        req = urllib.request.Request(url, headers=hdrs)
        resp_obj, status, err = None, 0, None
        try:
            resp_obj = urllib.request.urlopen(req, timeout=timeout)
            status = resp_obj.status
        except urllib.error.HTTPError as e:
            resp_obj, status = e, e.code
        except Exception as e:                      # DNS, timeout, TLS ...
            err = f"{type(e).__name__}: {e}"

        if resp_obj is not None:
            try:
                raw = resp_obj.read(MAX_BODY_BYTES)
                enc = (resp_obj.headers.get("Content-Encoding") or "").lower()
                if "gzip" in enc:
                    raw = gzip.decompress(raw)
                elif "deflate" in enc:
                    try:
                        raw = zlib.decompress(raw)
                    except zlib.error:
                        raw = zlib.decompress(raw, -zlib.MAX_WBITS)
                charset = resp_obj.headers.get_content_charset() or "utf-8"
                body = raw.decode(charset, errors="replace")
                out = {"status": status,
                       "headers": {k.lower(): v for k, v in resp_obj.headers.items()},
                       "body": body,
                       "final_url": getattr(resp_obj, "url", None) or
                                    getattr(resp_obj, "geturl", lambda: url)(),
                       "error": None}
                if status in (429, 500, 502, 503, 504) and attempt < retries:
                    time.sleep(1.2)
                    continue
                return out
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
            finally:
                try:
                    resp_obj.close()
                except Exception:
                    pass

        if attempt < retries:
            time.sleep(1.0)
            continue
        return {"status": 0, "headers": {}, "body": "", "final_url": url, "error": err}


# ─────────────────────────────────────────────────────────────────────────────
# HTML parsing helpers
# ─────────────────────────────────────────────────────────────────────────────
class _PageParser(HTMLParser):
    """Collects meta tags, <title>, script bodies and visible text in one pass."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.meta = {}
        self.title = ""
        self.canonical = ""
        self.scripts = []          # (type, text)
        self.text = []
        self._script = None
        self._in_title = False
        self._in_style = False

    def handle_starttag(self, tag, attrs):
        a = {k.lower(): (v or "") for k, v in attrs}
        if tag == "meta":
            key = (a.get("property") or a.get("name") or a.get("itemprop") or "").lower()
            if key and "content" in a:
                self.meta.setdefault(key, a["content"])
        elif tag == "link" and a.get("rel", "").lower() == "canonical":
            self.canonical = a.get("href", "")
        elif tag == "title":
            self._in_title = True
        elif tag == "script":
            self._script = [a.get("type", "").lower(), []]
        elif tag == "style":
            self._in_style = True

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        elif tag == "script" and self._script is not None:
            self.scripts.append((self._script[0], "".join(self._script[1])))
            self._script = None
        elif tag == "style":
            self._in_style = False

    def handle_data(self, data):
        if self._script is not None:
            self._script[1].append(data)
        elif self._in_title:
            self.title += data
        elif not self._in_style:
            self.text.append(data)

    @property
    def visible_text(self):
        return re.sub(r"\s+", " ", " ".join(self.text)).strip()


def _parse_page(body: str) -> _PageParser:
    p = _PageParser()
    try:
        p.feed(body)
        p.close()
    except Exception:
        pass            # malformed HTML: keep whatever was collected
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Layer A / C — statistics written in plain text
# ─────────────────────────────────────────────────────────────────────────────
_NUM = r"(\d[\d.,  ]*\d|\d)"
_SUF = r"(?:\s?([KkMmBb])(?![A-Za-z]))?"
_WORDS = {
    "followers": r"followers?|abonn[ée]s|seguidores|abonnenten",
    "following": r"following|connections|abonnements|seguidos|folgt",
    "posts":     r"posts?|tweets?|publicaciones|publications|beitr[äa]ge",
}


def _stats_from_text(text: str) -> dict:
    """{'followers': (1200000, True), ...} for every statistic phrase found."""
    out = {}
    if not text:
        return out
    for field, words in _WORDS.items():
        m = re.search(rf"(?<![\w.,]){_NUM}{_SUF}\s*(?:{words})\b", text, re.I)
        if not m:
            m = re.search(rf"\b(?:{words})\s*[:\-]?\s*{_NUM}{_SUF}(?![\w])", text, re.I)
        if m:
            val, rounded = parse_count(m.group(1) + (m.group(2) or ""))
            if _sane(val):
                out[field] = (val, rounded)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Layer B — structured data matched by meaning, not position
# ─────────────────────────────────────────────────────────────────────────────
def _norm(k: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(k).lower())


_KEYSETS = {
    "followers": {"edgefollowedby", "followercount", "followerscount", "followers",
                  "followedbycount", "subscribercount", "subscriberscount",
                  "followerstotal", "numfollowers"},
    "following": {"edgefollow", "followingcount", "followscount", "friendscount",
                  "following", "followeecount", "numfollowing"},
    "posts":     {"edgeownertotimelinemedia", "mediacount", "postcount", "postscount",
                  "statusescount", "tweetcount", "posts", "timelinemediacount",
                  "nummedia"},
}
_KEY_TO_FIELD = {k: f for f, ks in _KEYSETS.items() for k in ks}
_BIO_KEYS      = {"biography", "bio", "description", "about", "summary", "tagline"}
_BIO_PRIORITY  = ("biography", "bio", "about", "description", "summary", "tagline")
_USERNAME_KEYS = {"username", "screenname", "handle", "alternatename", "login"}
_PRIVATE_KEYS  = {"isprivate", "protected", "private"}
_NAME_KEYS     = {"fullname", "displayname", "name"}
_VERIFIED_KEYS = {"isverified", "verified"}


def _coerce_count(v, depth=0):
    """number-ish JSON value (int, '1,234', '1.2M', {'count': 5}) -> (int, rounded)."""
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, int):
        return (v, False) if _sane(v) else None
    if isinstance(v, float):
        return (int(v), False) if v.is_integer() and _sane(int(v)) else None
    if isinstance(v, str):
        val, rounded = parse_count(v)
        return (val, rounded) if _sane(val) else None
    if isinstance(v, dict) and depth < 2:
        for k in ("count", "value", "total", "total_count", "number"):
            if k in v:
                return _coerce_count(v[k], depth + 1)
    return None


def _walk(obj, depth=0):
    """Yield every dict inside an arbitrarily nested JSON value (including
    JSON that was serialised into a string, which modern pages love to do)."""
    if depth > 40:
        return
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk(v, depth + 1)
    elif isinstance(obj, list):
        for v in obj[:600]:
            yield from _walk(v, depth + 1)
    elif isinstance(obj, str) and 20 < len(obj) < 3_000_000 and obj[0] in "{[":
        low = obj[:4000].lower()
        if "follow" in low or "biography" in low or "media" in low:
            try:
                yield from _walk(json.loads(obj), depth + 1)
            except ValueError:
                pass


_HIT_RE = re.compile(
    r'"[A-Za-z_]*(?:ollow|edia_?[Cc]ount|tatuses_?[Cc]ount|riends_?[Cc]ount|'
    r'iography|ostCount|weetCount)[A-Za-z_]*"\s*:')


def _objects_near_hits(text: str, max_hits=60):
    """
    For script text that is JavaScript rather than pure JSON
    (`window.__data = {...};`), find every keyword hit and decode the nearest
    enclosing {...} object. Independent of variable names and script ids.
    """
    dec = json.JSONDecoder()
    found, accepted = [], set()
    for n, m in enumerate(_HIT_RE.finditer(text)):
        if n >= max_hits:
            break
        h, pos, lo, tries = m.start(), m.start(), max(0, m.start() - 30000), 0
        while tries < 200:
            i = text.rfind("{", lo, pos)
            if i < 0:
                break
            pos, tries = i, tries + 1
            if i in accepted:
                break
            try:
                obj, end = dec.raw_decode(text, i)
            except ValueError:
                continue
            if end > h:
                accepted.add(i)
                found.append(obj)
                break
    return found


def _json_values_from_scripts(scripts):
    for stype, text in scripts:
        t = text.strip()
        if not t:
            continue
        parsed = False
        if "json" in stype:
            try:
                yield json.loads(t)
                parsed = True
            except ValueError:
                pass
        if parsed:
            continue
        variants = [t]
        if '\\"' in t:      # JSON that was escaped inside a JS string
            variants.append(t.replace('\\"', '"').replace("\\\\", "\\"))
        for v in variants:
            if _HIT_RE.search(v) or "interactionStatistic" in v:
                yield from _objects_near_hits(v)


def _name_matches(candidate, target) -> bool:
    return (str(candidate).lstrip("@").strip().lower() == str(target).lstrip("@").strip().lower())


def _candidate_from_dict(d: dict, target: str):
    """Describe one JSON object as a possible profile record (or None)."""
    fields, bios, private, fname, verified, uname, ext = {}, {}, None, None, None, None, None
    for k, v in d.items():
        nk = _norm(k)
        f = _KEY_TO_FIELD.get(nk)
        if f and f not in fields:
            c = _coerce_count(v)
            if c:
                fields[f] = c
        if nk in _USERNAME_KEYS and isinstance(v, str) and uname is None:
            uname = v
        elif nk in _BIO_KEYS and isinstance(v, str):
            bios.setdefault(nk, v)
        elif nk in _PRIVATE_KEYS and isinstance(v, bool):
            private = v
        elif nk in _NAME_KEYS and isinstance(v, str) and fname is None:
            fname = v
        elif nk in _VERIFIED_KEYS and isinstance(v, bool):
            verified = v
        elif nk == "externalurl":
            ext = v.strip() if isinstance(v, str) else ""
        elif nk == "interactionstatistic":                 # schema.org JSON-LD
            for item in (v if isinstance(v, list) else [v]):
                if isinstance(item, dict) and "userInteractionCount" in item:
                    kind = str(item.get("interactionType", "")).lower()
                    c = _coerce_count(item["userInteractionCount"])
                    if c and "follow" in kind:
                        fields.setdefault("followers", c)
                    elif c and ("write" in kind or "post" in kind):
                        fields.setdefault("posts", c)
    if not fields:
        return None

    # Pick the bio by KEY PRIORITY, and only if it has text. Several keys can
    # hold prose ("description", "summary", ...) and many are legitimately
    # empty; taking the first one seen made a profile with a real bio look
    # like it had a 0-character bio.
    bio = next((bios[k] for k in _BIO_PRIORITY if bios.get(k, "").strip()), None)
    bio_empty = bio is None and "biography" in bios      # explicit empty biography field

    matched = uname is not None and _name_matches(uname, target)
    if uname is not None and not matched and not str(target).isdigit():
        return None            # a DIFFERENT user's record (related / suggested profile)
    if len(fields) < 2 and not matched:
        return None
    preview = {k: (v if isinstance(v, (int, float, bool)) or v is None
                   else v[:60] if isinstance(v, str) else f"<{type(v).__name__}>")
               for k, v in list(d.items())[:60]}
    return {"fields": fields, "bio": bio, "bio_empty": bio_empty, "private": private,
            "full_name": fname, "verified": verified, "external_url": ext,
            "matched": matched, "preview": preview,
            "score": len(fields) + (3 if matched else 0) + (1 if bio else 0)}


def _best_profile_record(json_values, target):
    best = None
    for root in json_values:
        for d in _walk(root):
            cand = _candidate_from_dict(d, target)
            if cand and (best is None or cand["score"] > best["score"]):
                best = cand
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Layer D — loose regex over raw text (lowest trust)
# ─────────────────────────────────────────────────────────────────────────────
_LOOSE = {
    "followers": [r'"edge_followed_by"\s*:\s*\{\s*"count"\s*:\s*(\d+)',
                  r'"follower_count"\s*:\s*(\d+)', r'"followers_count"\s*:\s*(\d+)'],
    "following": [r'"edge_follow"\s*:\s*\{\s*"count"\s*:\s*(\d+)',
                  r'"following_count"\s*:\s*(\d+)', r'"friends_count"\s*:\s*(\d+)'],
    "posts":     [r'"edge_owner_to_timeline_media"\s*:\s*\{\s*"count"\s*:\s*(\d+)',
                  r'"media_count"\s*:\s*(\d+)', r'"statuses_count"\s*:\s*(\d+)'],
}
_LOOSE_BIO = r'"biography"\s*:\s*"((?:[^"\\]|\\.)*)"'


def _loose_extract(raw: str, target: str):
    flat = raw.replace('\\"', '"')
    if target.lower() not in flat.lower():
        return {}, None          # page isn't even about this account
    out = {}
    for field, pats in _LOOSE.items():
        for pat in pats:
            m = re.search(pat, flat)
            if m and _sane(int(m.group(1))):
                out[field] = (int(m.group(1)), False)
                break
    bio = None
    m = re.search(_LOOSE_BIO, flat)
    if m:
        try:
            bio = json.loads('"' + m.group(1) + '"')
        except ValueError:
            bio = m.group(1)
    return out, bio


# ─────────────────────────────────────────────────────────────────────────────
# Page classification (does it exist? private? walled?)
# ─────────────────────────────────────────────────────────────────────────────
_NOT_FOUND = re.compile(
    r"(page (isn.?t|is not) available|page not found|sorry, this page|"
    r"this account doesn.?t exist|account doesn.?t exist|user not found|"
    r"couldn.?t find this account|the link you followed may be broken|"
    r"this content isn.?t available|profile (is )?(not found|not available)|"
    r"no such user|account (has been )?suspended|hmm\.\.\.\s*this page)", re.I)
_LOGIN_WALL = re.compile(
    r"(log in to (see|continue|instagram)|create an account or log in|"
    r"sign in to (continue|x\b|twitter)|you must (log|sign) in|login required|"
    r"join linkedin|sign up to see|see photos and videos from your friends)", re.I)
_PRIVATE = re.compile(
    r"(this account is private|this profile is private|account.s tweets are protected|"
    r"these (tweets|posts) are protected)", re.I)
_LOGIN_PATH = re.compile(r"/(accounts/login|login|i/flow/login|authwall|checkpoint|uas/login)", re.I)


def _classify(resp, page, username, got_stats):
    st = resp["status"]
    if st == 0:
        return "network"
    if st in (404, 410):
        return "not_found"
    if st == 429:
        return "rate_limited"
    final_path = urlparse(resp["final_url"] or "").path
    if _LOGIN_PATH.search(final_path) and username.lower() not in final_path.lower():
        return "login_wall"
    if st in (401, 403):
        return "blocked"
    if got_stats:
        return "ok"
    probe = " ".join([page.title, page.meta.get("og:title", ""), page.visible_text[:4000]])
    if _NOT_FOUND.search(probe):
        return "not_found"
    if _PRIVATE.search(probe):
        return "private"
    if _LOGIN_WALL.search(probe) or page.title.strip().lower() in ("instagram", "x", "facebook", "linkedin", "log in"):
        return "login_wall"
    if st >= 500:
        return "blocked"
    return "empty"


# ─────────────────────────────────────────────────────────────────────────────
# One response -> observations from every layer
# ─────────────────────────────────────────────────────────────────────────────
TRUST = {"json_owner": 5, "meta": 4, "json_unowned": 3, "text": 2, "loose": 1}


def _extract_all(resp, username):
    """
    Run every layer on one HTTP response.
    Returns (observations, page_class, private_flag).
    Each observation: {source, trust, fields{name:(val,rounded)}, bio, ...}
    """
    body = resp["body"]
    obs = []
    page = _parse_page(body) if body else _PageParser()
    private = None

    # JSON endpoint answered with pure JSON (e.g. an API URL)
    json_roots = []
    stripped = body.lstrip()
    if stripped[:1] in "{[":
        try:
            json_roots.append(json.loads(stripped))
        except ValueError:
            pass

    # Layer A — meta tags
    meta_fields, owner_ok = {}, False
    for key in ("og:description", "description", "twitter:description"):
        for f, v in _stats_from_text(page.meta.get(key, "")).items():
            meta_fields.setdefault(f, v)
    ident = " ".join([page.meta.get("og:title", ""), page.meta.get("og:description", ""),
                      page.meta.get("og:url", ""), page.canonical, page.title])
    owner_ok = username.lower() in ident.lower()
    if meta_fields:
        obs.append({"source": "meta", "trust": TRUST["meta"] if owner_ok else TRUST["text"],
                    "fields": meta_fields, "bio": None, "owner": owner_ok})

    # Layer B — structured data
    values = list(json_roots) + list(_json_values_from_scripts(page.scripts))
    if not values and body and not page.scripts:
        values = list(_objects_near_hits(body))
    rec = _best_profile_record(values, username)
    if rec:
        private = rec["private"]
        obs.append({"source": "json_owner" if rec["matched"] else "json_unowned",
                    "trust": TRUST["json_owner"] if rec["matched"] else TRUST["json_unowned"],
                    "fields": rec["fields"], "bio": rec["bio"], "owner": rec["matched"],
                    "bio_empty": rec["bio_empty"], "preview": rec["preview"],
                    "full_name": rec["full_name"], "verified": rec["verified"],
                    "external_url": rec["external_url"]})

    # Layer C — visible text
    text_fields = _stats_from_text(page.visible_text[:20000])
    if text_fields:
        obs.append({"source": "text", "trust": TRUST["text"], "fields": text_fields,
                    "bio": None, "owner": False})

    # Layer D — loose regex, only if nothing structured was found
    if not any(o["source"].startswith("json") or o["source"] == "meta" for o in obs):
        lf, lbio = _loose_extract(body, username)
        if lf:
            obs.append({"source": "loose", "trust": TRUST["loose"], "fields": lf,
                        "bio": lbio, "owner": False})

    got_stats = any(o["fields"] for o in obs)
    if private is None and _PRIVATE.search(" ".join([page.title, page.visible_text[:4000]])):
        private = True
    return obs, _classify(resp, page, username, got_stats), private, page


# ─────────────────────────────────────────────────────────────────────────────
# Merge observations from all layers / strategies
# ─────────────────────────────────────────────────────────────────────────────
def _close(a, b):
    (va, ra), (vb, rb) = a, b
    if ra or rb:
        return abs(va - vb) <= 0.06 * max(va, vb) + 1
    return abs(va - vb) <= 0.01 * max(va, vb) + 1


def _merge(observations):
    """
    Combine what every layer read.

    Layers that agree (within rounding) form a cluster. The cluster backed by
    the MOST INDEPENDENT layers wins. Link-preview <meta> text is often cached
    and can lag behind the live page, so when it alone disagrees with two
    other layers it is outvoted (and the result is capped at "medium"
    confidence). Only a genuine tie between trusted layers is a conflict.
    """
    merged, conf, approx, conflicts, disputes = {}, {}, [], [], {}
    for f in _STAT_FIELDS:
        cands = [(o["fields"][f], o["trust"], o["source"]) for o in observations if f in o["fields"]]
        if not cands:
            continue
        clusters = []
        for c in sorted(cands, key=lambda c: (c[0][1], -c[1])):      # exact values first
            for cl in clusters:
                if _close(c[0], cl[0][0]):
                    cl.append(c)
                    break
            else:
                clusters.append([c])

        def support(cl):
            return len({m[2] for m in cl})

        ranked = sorted(clusters, key=lambda cl: (support(cl), max(m[1] for m in cl)), reverse=True)
        top = ranked[0]
        exact = [m for m in top if not m[0][1]]
        best = max(exact or top, key=lambda m: m[1])
        (val, rounded), trust, src = best
        sup = support(top)

        disputed = len(ranked) > 1
        tie = disputed and support(ranked[1]) == sup and max(m[1] for m in ranked[1]) >= 3 \
            and max(m[1] for m in top) >= 3
        if disputed:
            disputes[f] = {"chosen": val, "outvoted": [m[0][0] for cl in ranked[1:] for m in cl],
                           "outvoted_layers": [m[2] for cl in ranked[1:] for m in cl]}
        if tie:
            conflicts.append(f)
            level = "low"
        elif trust >= 4 and (sup >= 2 or src == "json_owner"):
            level = "high"
        elif trust >= 3 or (trust >= 2 and sup >= 2):
            level = "medium"
        else:
            level = "low"
        if disputed and not tie and level == "high":
            level = "medium"
        merged[f], conf[f] = val, level
        if rounded:
            approx.append(f)

    bio, full_name, verified, bio_empty = None, None, None, False
    for o in sorted(observations, key=lambda o: -o["trust"]):
        if bio is None and o.get("bio") and o["trust"] >= 3:
            bio = o["bio"]
        if o.get("bio_empty"):
            bio_empty = True
        if full_name is None and o.get("full_name"):
            full_name = o["full_name"]
        if verified is None and o.get("verified") is not None:
            verified = o["verified"]
    if bio is None:                                  # loose-layer bio, lowest trust
        for o in observations:
            if o.get("bio"):
                bio = o["bio"]
                break
    return merged, conf, approx, conflicts, bio, full_name, verified, disputes, bio_empty


# ─────────────────────────────────────────────────────────────────────────────
# Per-platform fetch plans (ordered; later steps run only if needed)
# ─────────────────────────────────────────────────────────────────────────────
def _plan(platform, username, url):
    if platform == "Instagram":
        # Order matters. In live testing a browser-style request got Instagram's
        # login wall and the API call got HTTP 429, while the self-identifying
        # link-preview style request returned the profile metadata. Try the
        # one that works first; the rest only run if it didn't give everything.
        return [
            ("ig_page_botua", url, {"User-Agent": UA_BOT}),
            ("ig_page", url, {}),
            ("ig_api", f"https://www.instagram.com/api/v1/users/web_profile_info/?username={username}",
             {"X-IG-App-ID": "936619743392459", "Accept": "application/json",
              "Referer": url, "X-Requested-With": "XMLHttpRequest"}),
        ]
    if platform == "Twitter/X":
        return [
            ("x_syndication",
             f"https://syndication.twitter.com/srv/timeline-profile/screen-name/{username}", {}),
            ("x_page", f"https://x.com/{username}", {}),
            ("x_page_botua", f"https://x.com/{username}", {"User-Agent": UA_BOT}),
        ]
    return [(f"{platform.lower()}_page", url, {}),
            (f"{platform.lower()}_page_botua", url, {"User-Agent": UA_BOT})]


# ─────────────────────────────────────────────────────────────────────────────
# Health / drift monitoring
# ─────────────────────────────────────────────────────────────────────────────
_STATS = {}


def _record(strategy, produced_data: bool):
    s = _STATS.setdefault(strategy, {"ok": 0, "no_data": 0, "last_ok": None, "last_try": None})
    s["last_try"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if produced_data:
        s["ok"] += 1
        s["last_ok"] = s["last_try"]
    else:
        s["no_data"] += 1


def scraper_health() -> dict:
    """Which strategies have produced data lately. A strategy whose `ok` count
    stops growing while `no_data` climbs means that site changed its layout."""
    return copy.deepcopy(_STATS)


def _dump_debug(platform, username, strategy, resp):
    try:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        files = sorted(os.listdir(DEBUG_DIR))
        for old in files[:-19]:
            os.remove(os.path.join(DEBUG_DIR, old))
        name = f"{time.strftime('%Y%m%d_%H%M%S')}_{platform.replace('/', '')}_{username}_{strategy}.html"
        with open(os.path.join(DEBUG_DIR, name), "w", encoding="utf-8") as fh:
            fh.write(resp["body"])
        return name
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Manual-entry fields
# ─────────────────────────────────────────────────────────────────────────────
MANUAL_FIELDS = {
    "Twitter/X": [
        {"name": "followers",    "label": "Followers",          "type": "text",   "placeholder": "e.g. 1.2K or 239.8M"},
        {"name": "following",    "label": "Following",          "type": "text",   "placeholder": "e.g. 340 or 1.3K"},
        {"name": "posts",        "label": "Tweets / Posts",     "type": "text",   "placeholder": "e.g. 4.5K or 102"},
        {"name": "bio_len",      "label": "Bio length (chars)", "type": "number", "placeholder": "e.g. 120"},
        {"name": "username_len", "label": "Username length",    "type": "number", "placeholder": "e.g. 8"},
    ],
    "Facebook": [
        {"name": "followers",    "label": "Followers",           "type": "text",   "placeholder": "e.g. 500 or 12.3K"},
        {"name": "following",    "label": "Following / Friends", "type": "text",   "placeholder": "e.g. 200 or 1.1K"},
        {"name": "posts",        "label": "Total Posts",         "type": "text",   "placeholder": "e.g. 150"},
        {"name": "bio_len",      "label": "Bio length (chars)",  "type": "number", "placeholder": "e.g. 80"},
        {"name": "username_len", "label": "Username length",     "type": "number", "placeholder": "e.g. 10"},
    ],
    "LinkedIn": [
        {"name": "followers",    "label": "Followers",               "type": "text",   "placeholder": "e.g. 800 or 2.4K"},
        {"name": "following",    "label": "Connections",             "type": "text",   "placeholder": "e.g. 500+"},
        {"name": "posts",        "label": "Posts / Articles",        "type": "text",   "placeholder": "e.g. 40"},
        {"name": "bio_len",      "label": "About section (chars)",   "type": "number", "placeholder": "e.g. 200"},
        {"name": "username_len", "label": "Profile URL length",      "type": "number", "placeholder": "e.g. 12"},
    ],
    "Instagram": [
        {"name": "followers",    "label": "Followers",          "type": "text",   "placeholder": "e.g. 2.5K or 1.4M"},
        {"name": "following",    "label": "Following",          "type": "text",   "placeholder": "e.g. 400 or 1.3K"},
        {"name": "posts",        "label": "Posts",              "type": "text",   "placeholder": "e.g. 180"},
        {"name": "bio_len",      "label": "Bio length (chars)", "type": "number", "placeholder": "e.g. 90"},
        {"name": "username_len", "label": "Username length",    "type": "number", "placeholder": "e.g. 11"},
    ],
}


def build_manual_fields(platform, prefill=None, username=""):
    """MANUAL_FIELDS copy; anything already found is pre-filled (and flagged)
    so the user only has to type what is genuinely missing."""
    fields = copy.deepcopy(MANUAL_FIELDS.get(platform, MANUAL_FIELDS["Instagram"]))
    prefill = dict(prefill or {})
    if username and "username_len" not in prefill:
        prefill["username_len"] = len(username)
    for f in fields:
        v = prefill.get(f["name"])
        f["value"] = "" if v is None else v
        f["prefilled"] = v is not None
    return fields


# ─────────────────────────────────────────────────────────────────────────────
# Messages
# ─────────────────────────────────────────────────────────────────────────────
def _message(code, platform, username):
    return {
        "not_found": (f"@{username} doesn't appear to exist on {platform} — {platform} reports "
                      "that page as unavailable. Check the spelling of the username, or open "
                      "the link in your browser to confirm."),
        "private": (f"@{username} is a private {platform} account, so its numbers are hidden "
                    "from us. Enter the visible stats below, or upload a screenshot."),
        "blocked": (f"{platform} wouldn't show this profile without logging in, so it couldn't "
                    "be read automatically. Enter the stats below, or upload a screenshot."),
        "layout_changed": (f"{platform} returned a page we couldn't read — its layout may have "
                           "changed. Enter the stats below, or upload a screenshot."),
        "network_error": (f"Couldn't reach {platform} (network problem). Check your connection, "
                          "or enter the stats below / upload a screenshot."),
        "incomplete": ("Some of this profile's numbers were found automatically; please fill in "
                       "or confirm the rest."),
    }.get(code, "Please enter the profile stats manually.")


_CACHE = {}


def _result(identifier, platform, username, **kw):
    base = {
        "success": False, "source": "manual_required", "manual_required": True,
        "invalid": False, "manual_fields": [], "data": {}, "error": None,
        "error_code": None, "suggestion": "", "username": username,
        "platform": platform, "report": {}, "confidence": None,
    }
    base.update(kw)
    return base


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────
def fetch_profile_data(identifier: str, platform: str, _http=None) -> dict:
    """
    Called by predict.run_prediction().

    Returns dict:
        success, manual_required, invalid : bool
        source          "scraped" | "manual_required" | "invalid"
        manual_fields   list (pre-filled where data was found)
        data            profile fields when success
        error, error_code, suggestion
        username, platform
        report          per-strategy diagnostics
    """
    platform = {"instagram": "Instagram", "ig": "Instagram", "twitter": "Twitter/X",
                "twitter/x": "Twitter/X", "x": "Twitter/X", "facebook": "Facebook",
                "fb": "Facebook", "linkedin": "LinkedIn"}.get((platform or "").strip().lower(),
                                                              platform or "")
    target = resolve_target(identifier, platform)
    if not target["ok"]:
        return _result(identifier, platform or "", (identifier or "").strip(),
                       source="invalid", manual_required=False, invalid=True,
                       error=target["error"], error_code="invalid_input",
                       suggestion=target["suggestion"],
                       report={"input_check": target["error_code"]})

    platform, username, url = target["platform"], target["username"], target["url"]
    use_cache = _http is None
    key = (platform, username.lower())
    if use_cache and key in _CACHE and _CACHE[key][0] > time.time():
        return copy.deepcopy(_CACHE[key][1])

    http = _http or _http_get
    observations, attempts, classes, private_flag = [], [], [], None
    debug_file = None

    for name, u, extra in _plan(platform, username, url):
        resp = http(u, extra) if _http else http(u, headers=extra)
        obs, cls, priv, page = _extract_all(resp, username)
        for o in obs:
            o["strategy"] = name
        observations += obs
        if SAVE_ALL_HTML:
            _dump_debug(platform, username, name, resp)
        classes.append(cls)
        if priv:
            private_flag = True
        elif priv is False and private_flag is None:
            private_flag = False        # an explicit "not private" is evidence; silence is not
        _record(name, bool(obs))
        attempts.append({"strategy": name, "status": resp["status"], "class": cls,
                         "found": sorted({f for o in obs for f in o["fields"]}),
                         "layers": [o["source"] for o in obs],
                         "error": resp.get("error")})
        if cls == "empty" and debug_file is None:
            debug_file = _dump_debug(platform, username, name, resp)

        merged, *_ = _merge(observations)
        complete = all(f in merged for f in _STAT_FIELDS) and any(
            o.get("bio") is not None for o in observations)
        if complete or cls == "not_found":
            break

    merged, conf, approx, conflicts, bio, full_name, verified, disputes, bio_empty = _merge(observations)
    report = {"attempts": attempts, "confidence": conf, "approximate_fields": approx,
              "conflicts": conflicts, "disputes": disputes, "bio_empty_unverified": bio_empty,
              "debug_html": debug_file,
              # what EACH layer read, so a disagreement can be diagnosed at a glance
              "layer_values": [
                  {"strategy": o.get("strategy"), "layer": o["source"], "trust": o["trust"],
                   "fields": {k: {"value": v[0], "rounded": v[1]} for k, v in o["fields"].items()},
                   "bio_chars": None if o.get("bio") is None else len(o["bio"]),
                   "record_preview": o.get("preview")}
                  for o in observations]}
    have_counts = all(f in merged for f in _STAT_FIELDS)
    weakest = ("low" if "low" in conf.values() else
               "medium" if "medium" in conf.values() else "high") if conf else None

    # ── Real data ────────────────────────────────────────────────────────────
    if merged:
        if have_counts and bio is not None and weakest in ("high", "medium"):
            data = {
                "username": username, "platform": platform,
                "followers": merged["followers"], "following": merged["following"],
                "posts": merged["posts"], "bio_len": len(bio),
                "username_len": len(username), "full_name": full_name or "",
                "is_verified": verified, "is_private": private_flag,   # None = could not tell
                "simulated": False, "approximate_fields": approx, "data_confidence": weakest,
            }
            # Link in bio: "" means the profile explicitly has none; absent = unknown.
            ext = next((o["external_url"] for o in sorted(observations, key=lambda o: -o["trust"])
                        if o.get("external_url") is not None), None)
            if ext is not None:
                data["external_url"] = ext
            res = _result(identifier, platform, username, success=True, source="scraped",
                          manual_required=False, data=data, report=report, confidence=weakest)
        else:
            # Found something, but not everything / not confidently → the user
            # confirms a pre-filled form rather than starting from nothing.
            prefill = dict(merged)
            if bio is not None and weakest in ("high", "medium"):
                prefill["bio_len"] = len(bio)
            res = _result(identifier, platform, username,
                          manual_fields=build_manual_fields(platform, prefill, username),
                          error=_message("incomplete", platform, username)
                                + (" Instagram returned an EMPTY bio field for this profile — that can mean "
                                   "the account has no bio, or that it is hidden from logged-out visitors, "
                                   "so please enter the bio length yourself (0 if there is none)."
                                   if bio_empty and bio is None else ""),
                          error_code="incomplete", report=report, confidence=weakest)
    # ── Nothing found: say why ───────────────────────────────────────────────
    else:
        if "not_found" in classes:
            code = "not_found"
        elif private_flag or "private" in classes:
            code = "private"
        elif all(c == "network" for c in classes):
            code = "network_error"
        elif any(c in ("login_wall", "blocked", "rate_limited") for c in classes):
            code = "blocked"
        else:
            code = "layout_changed"
        if code == "not_found":
            res = _result(identifier, platform, username, source="invalid",
                          manual_required=False, invalid=True,
                          error=_message(code, platform, username), error_code=code,
                          report=report)
        else:
            res = _result(identifier, platform, username,
                          manual_fields=build_manual_fields(platform, None, username),
                          error=_message(code, platform, username),
                          error_code=code, report=report)

    if use_cache:
        ttl = CACHE_TTL_OK if (res["success"] or res["error_code"] == "not_found") else CACHE_TTL_BLOCK
        _CACHE[key] = (time.time() + ttl, copy.deepcopy(res))
    return res


# ─────────────────────────────────────────────────────────────────────────────
# Command-line self-test
# ─────────────────────────────────────────────────────────────────────────────
def _cli(argv):
    global SAVE_ALL_HTML
    if "--save-html" in argv:
        SAVE_ALL_HTML = True
        argv = [a for a in argv if a != "--save-html"]
    if "--html" in argv:                       # offline: run extractors on a saved page
        def arg(flag, default=""):
            return argv[argv.index(flag) + 1] if flag in argv else default
        with open(arg("--html"), encoding="utf-8", errors="replace") as fh:
            body = fh.read()
        user = arg("--user")
        fake = lambda u, headers=None: {"status": 200, "headers": {}, "body": body,
                                        "final_url": u, "error": None}
        out = fetch_profile_data(user, arg("--platform", "Instagram"), _http=fake)
    else:
        args = [a for a in argv if not a.startswith("-")]
        if not args:
            print(__doc__)
            return
        out = fetch_profile_data(args[0], args[1] if len(args) > 1 else "Instagram")
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    _cli(sys.argv[1:])