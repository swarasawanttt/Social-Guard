"""
src/image_reader.py
────────────────────
Reads profile stats (followers, following, posts, bio, username) out of
1-4 screenshots the user uploads, when URL scraping fails or the account
is private.

Engine: Tesseract OCR (via pytesseract). Swapping in a different OCR engine
later only means replacing _ocr_words(); everything downstream works on
plain word boxes.

HOW IT READS A SCREENSHOT (and why it is not just "regex over OCR text")
─────────────────────────────────────────────────────────────────────────
Earlier versions ran a regex over the flat OCR text, line by line. That
failed on real Instagram screenshots in three ways:

  1. Mobile layout stacks the number ABOVE its label ("48.2K" on one line,
     "followers" on the next), so "<number> followers" never matched.
  2. Dark-mode screenshots (light text on black) are read much worse by
     Tesseract unless inverted first.
  3. The "bio" was just "the longest line that isn't a stats line", which
     happily picked up the display name, the link, a "Followed by ..."
     line or even a post caption, and presented it as the bio.

This version works from word POSITIONS instead:

  - OCR returns every word with a bounding box and confidence.
  - Words are regrouped into lines by geometry (not by Tesseract's own
    line splitting), so stacked layouts and multi-column rows work.
  - Stats are found by locating the labels (posts / followers / following)
    and then looking for the nearest number just before, directly above,
    or just after each label.
  - The bio is the block of text BELOW the stats row and ABOVE the first
    "terminator" (action buttons, "Followed by", tab row, or a big
    vertical gap). The first line of that block is the display name and is
    excluded, a muted line right under it is the category and is excluded,
    and a lone URL line is the profile link and is excluded -- none of
    those are part of Instagram's `biography` field.
  - If the stats row cannot be located, no bio is guessed at all. The field
    is reported as Not Found so the review screen asks the person for it.
    Nothing here ever invents a value.

Every extracted field carries a confidence level and its source image, and
app.py always shows an editable review table before anything is used.
"""

import os
import re
import difflib
from statistics import median

try:
    import pytesseract
    from PIL import Image, ImageOps
    import numpy as np
    OCR_AVAILABLE = True
except ImportError:                                   # pragma: no cover
    OCR_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Vocabulary
# ─────────────────────────────────────────────────────────────────────────────
NUM_RE = re.compile(r"^\d[\d.,]*[KkMmBb]?$")

# word (lowercased, punctuation stripped) -> stat it labels
STAT_LABELS = {
    "posts": "posts", "post": "posts", "tweets": "posts", "tweet": "posts",
    "followers": "followers", "follower": "followers",
    "following": "following", "connections": "following",
}

# Words that appear on action-button rows. A line made (almost) entirely of
# these ends the bio block.
BUTTON_WORDS = {
    "follow", "following", "followed", "message", "contact", "email", "call",
    "directions", "edit", "profile", "share", "view", "archive", "subscribe",
    "subscribed", "requested", "back", "dashboard", "professional", "insights",
    "promotions", "shop", "book", "gift", "order", "text", "add", "unfollow",
    "notify", "mute", "restrict", "report", "mutual", "more", "options",
}
TAB_WORDS = {"posts", "reels", "tagged", "videos", "guides", "photos",
             "grid", "igtv", "shop", "collab", "collabs", "replies", "media"}

URL_RE = re.compile(
    r"^(https?://\S+|www\.\S+|[\w-]+(\.[\w-]+)*\.(com|in|org|net|io|co|me|ly|"
    r"ee|link|app|dev|xyz|shop|store|site|page|bio|uk|us|ca|au|de|fr|jp|"
    r"info|tv|fm|gg|to)(/\S*)?)$", re.I)

USERNAME_TOKEN_RE = re.compile(r"^[A-Za-z0-9._]{2,30}$")
TIME_RE           = re.compile(r"^\d{1,2}[:.]\d{2}$")
STATUS_WORDS      = {"5g", "4g", "lte", "wifi", "wi-fi", "am", "pm", "airtel",
                     "jio", "vi", "verizon", "t-mobile", "att"}


def _norm(word: str) -> str:
    return re.sub(r"[^a-z0-9]", "", word.lower())


def _parse_count(raw: str) -> int:
    """Same suffix logic as scraper.parse_manual_value, kept local so this
    module can be tested without importing the rest of the app."""
    if not raw:
        return 0
    s = raw.strip().upper().replace(",", "").replace("+", "").replace(" ", "")
    for suf, mult in (("K", 1_000), ("M", 1_000_000), ("B", 1_000_000_000)):
        if s.endswith(suf):
            try:
                return int(float(s[:-1]) * mult)
            except ValueError:
                return 0
    # "1.284" with exactly 3 decimals and no suffix is almost certainly a
    # thousands separator written the European way, not a fraction.
    if re.fullmatch(r"\d{1,3}(\.\d{3})+", s):
        s = s.replace(".", "")
    try:
        return int(float(s))
    except ValueError:
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# Geometry containers
# ─────────────────────────────────────────────────────────────────────────────
class Word:
    __slots__ = ("text", "conf", "x0", "y0", "x1", "y1")

    def __init__(self, text, conf, x0, y0, x1, y1):
        self.text, self.conf = text, conf
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1

    @property
    def cx(self): return (self.x0 + self.x1) / 2

    @property
    def cy(self): return (self.y0 + self.y1) / 2

    @property
    def h(self): return max(1, self.y1 - self.y0)


class Line:
    def __init__(self, words):
        self.words = sorted(words, key=lambda w: w.x0)
        self.role = "other"

    @property
    def text(self): return " ".join(w.text for w in self.words)

    @property
    def x0(self): return min(w.x0 for w in self.words)

    @property
    def x1(self): return max(w.x1 for w in self.words)

    @property
    def y0(self): return min(w.y0 for w in self.words)

    @property
    def y1(self): return max(w.y1 for w in self.words)

    @property
    def cy(self): return sum(w.cy for w in self.words) / len(self.words)

    @property
    def h(self): return median(w.h for w in self.words)

    @property
    def conf(self): return sum(w.conf for w in self.words) / len(self.words)


# ─────────────────────────────────────────────────────────────────────────────
# Image preparation + OCR
# ─────────────────────────────────────────────────────────────────────────────
TARGET_WIDTH = 1600


def _prepare(path: str):
    """
    Returns (ocr_ready_image, scale, original_gray_ndarray, is_dark).

    - Respects phone EXIF rotation.
    - Detects dark mode (most pixels are dark) and inverts, because
      Tesseract is trained mostly on dark-on-light text.
    - Upscales narrow screenshots; small UI text OCRs far better enlarged.
    """
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    gray = ImageOps.grayscale(img)
    arr = np.asarray(gray)
    is_dark = float(np.median(arr)) < 110

    work = ImageOps.invert(gray) if is_dark else gray
    scale = 1.0
    if work.width < TARGET_WIDTH:
        scale = TARGET_WIDTH / work.width
        work = work.resize((int(work.width * scale), int(work.height * scale)),
                           Image.LANCZOS)
    work = ImageOps.autocontrast(work)
    return work, scale, arr, is_dark


def _ocr_words(img, scale: float, psm: int = 11) -> list:
    """Run Tesseract and return Word objects in ORIGINAL image coordinates."""
    data = pytesseract.image_to_data(
        img, config=f"--oem 3 --psm {psm}", output_type=pytesseract.Output.DICT)
    words = []
    for i, raw in enumerate(data["text"]):
        text = (raw or "").strip()
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            continue
        if not text or conf < 0:
            continue
        x, y, w, h = (data[k][i] for k in ("left", "top", "width", "height"))
        words.append(Word(text, conf, x / scale, y / scale,
                          (x + w) / scale, (y + h) / scale))
    return words


def _group_lines(words: list) -> list:
    """Cluster words into lines by vertical position (not by Tesseract's
    own line ids), so stacked / multi-column layouts come out right."""
    lines, current = [], []
    for w in sorted(words, key=lambda w: w.cy):
        if current:
            ref_cy = sum(x.cy for x in current) / len(current)
            ref_h = median(x.h for x in current)
            if abs(w.cy - ref_cy) <= 0.55 * max(w.h, ref_h):
                current.append(w)
                continue
            lines.append(Line(current))
        current = [w]
    if current:
        lines.append(Line(current))
    return lines


def _ink_strength(arr, line, bg: float) -> float:
    """How strongly the text in this line stands out from the background.
    Muted text (category, 'Followed by', placeholder grey) scores lower
    than body text, which lets us tell a category line from a bio line."""
    h, w = arr.shape
    x0, x1 = max(0, int(line.x0)), min(w, int(line.x1) + 1)
    y0, y1 = max(0, int(line.y0)), min(h, int(line.y1) + 1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    crop = np.abs(arr[y0:y1, x0:x1].astype(float) - bg)
    return float(np.percentile(crop, 97))


def _ink_coverage(arr, line, bg: float) -> float:
    """Fraction of the line's box covered by ink. Bold text (the display
    name) covers noticeably more of its box than regular text (the bio)."""
    h, w = arr.shape
    x0, x1 = max(0, int(line.x0)), min(w, int(line.x1) + 1)
    y0, y1 = max(0, int(line.y0)), min(h, int(line.y1) + 1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    crop = np.abs(arr[y0:y1, x0:x1].astype(float) - bg)
    peak = crop.max()
    return float((crop > 0.5 * peak).mean()) if peak > 0 else 0.0


def _suspect_symbol_gaps(line) -> int:
    """Unusually wide gaps between words inside a line usually mean an emoji
    or icon sat there that OCR could not read (it drops them silently)."""
    gaps = 0
    for prev, w in zip(line.words, line.words[1:]):
        if (w.x0 - prev.x1) > 0.85 * max(prev.h, w.h):
            gaps += 1
    return gaps


# ─────────────────────────────────────────────────────────────────────────────
# Stats (followers / following / posts)
# ─────────────────────────────────────────────────────────────────────────────
def _clean_num(text: str):
    t = text.strip(".,:;|()[]")
    return t if NUM_RE.match(t) else None


def _find_stats(words: list, lines: list) -> dict:
    """
    Strategy 1 (layout): for each label word find the nearest number
      (a) just before it on the same line   -> "327 posts"
      (b) directly above it                 -> stacked mobile layout
      (c) just after it                     -> "Followers: 1,200"
    Strategy 2 (text): regex over the joined line text, for the case where
      OCR glued the number and label into one token ("12.9Kfollowers").
    Returns {stat: {"display", "conf", "label_word"}}.
    """
    nums = [(w, _clean_num(w.text)) for w in words]
    nums = [(w, n) for w, n in nums if n]
    used = set()
    found = {}

    labels = [(w, STAT_LABELS[_norm(w.text)]) for w in words
              if _norm(w.text) in STAT_LABELS]

    def best(label_w, mode):
        cands = []
        for w, n in nums:
            if id(w) in used:
                continue
            same_line = abs(w.cy - label_w.cy) <= 0.6 * max(w.h, label_w.h)
            if mode == "before" and same_line and w.x1 <= label_w.x0 + label_w.h * 0.5:
                gap = label_w.x0 - w.x1
                if gap <= 4 * label_w.h:
                    cands.append((gap, w, n))
            elif mode == "above" and w.y1 <= label_w.y0 + label_w.h * 0.3:
                vgap = label_w.y0 - w.y1
                overlap = abs(w.cx - label_w.cx)
                if vgap <= 2.6 * label_w.h and overlap <= max(w.x1 - w.x0, label_w.x1 - label_w.x0) * 0.9:
                    cands.append((vgap + overlap * 0.5, w, n))
            elif mode == "after" and same_line and w.x0 >= label_w.x1 - label_w.h * 0.3:
                gap = w.x0 - label_w.x1
                if gap <= 3 * label_w.h:
                    cands.append((gap, w, n))
        return min(cands, key=lambda c: c[0]) if cands else None

    # Resolve most-constrained labels first so one number is never used twice.
    for label_w, stat in sorted(labels, key=lambda p: (p[0].cy, p[0].x0)):
        if stat in found:
            continue
        for mode in ("before", "above", "after"):
            hit = best(label_w, mode)
            if hit:
                _, num_w, display = hit
                used.add(id(num_w))
                found[stat] = {"display": display, "conf": (num_w.conf + label_w.conf) / 2,
                               "label_word": label_w}
                break

    # Strategy 2: glued tokens / regex over line text
    for stat, pats in (
        ("followers", (r"(\d[\d.,]*\s?[KkMmBb]?)\s*followers",)),
        ("following", (r"(\d[\d.,]*\s?[KkMmBb]?)\s*following",)),
        ("posts",     (r"(\d[\d.,]*\s?[KkMmBb]?)\s*(?:posts|tweets)",)),
    ):
        if stat in found:
            continue
        for ln in lines:
            for pat in pats:
                m = re.search(pat, ln.text, re.I)
                if m:
                    found[stat] = {"display": m.group(1).replace(" ", ""),
                                   "conf": ln.conf, "label_word": ln.words[-1]}
                    break
            if stat in found:
                break
    return found


# ─────────────────────────────────────────────────────────────────────────────
# Line roles + bio block
# ─────────────────────────────────────────────────────────────────────────────
def _tokens(line: Line) -> list:
    return [_norm(w.text) for w in line.words if _norm(w.text)]


def _word_tokens(line: Line) -> list:
    """Tokens that can be real words. Icons (dropdown chevrons, person+ glyphs,
    grid/reel icons) come out of OCR as 1-2 character junk such as 'v', 'Vv',
    '~', 'oO'; counting them as words made 'Follow v Message' look only 67%
    button-like, so the button row was mistaken for bio text."""
    return [t for t in _tokens(line) if len(t) >= 3]


def _is_button_row(line: Line) -> bool:
    toks = _word_tokens(line)
    return bool(toks) and sum(t in BUTTON_WORDS for t in toks) / len(toks) >= 0.8


def _is_tab_row(line: Line) -> bool:
    toks = _word_tokens(line)
    return bool(toks) and sum(t in TAB_WORDS for t in toks) / len(toks) >= 0.8


def _is_link_line(line: Line) -> bool:
    words = [w for w in line.words if re.search(r"[A-Za-z0-9]", w.text)]
    # a link line is ONE url-looking token (an emoji/icon token in front of
    # it has already been filtered out as junk)
    return len(words) == 1 and bool(URL_RE.match(words[0].text.strip(".,;")))


_NOISE_LINE = re.compile(r"^(see translation|translate|\.\.\.? ?more|… ?more|more)$", re.I)


def _is_terminator(line: Line) -> bool:
    t = line.text.strip().lower()
    return (t.startswith("followed by") or t.startswith("suggested for you")
            or t.startswith("no posts") or t.startswith("this account is private")
            or _is_button_row(line) or _is_tab_row(line))


def _drop_junk_words(line: Line):
    """Emoji / icons come out of OCR as short low-confidence garbage
    ('©', 'Q', '@', '|'). Remove them so they don't pollute bio text."""
    kept = [w for w in line.words
            if not (w.conf < 40 and len(w.text) <= 2)
            and not (len(w.text) == 1 and not w.text.isalnum() and w.text not in "&+|/-")]
    return Line(kept) if kept else None


def _extract_header_username(lines, stats_top, anchor_known=None):
    """Best-effort guess at the username printed in the screenshot header
    (plain text, no '@'). Used only to WARN when a screenshot appears to
    belong to a different account than the one entered -- never as a
    replacement for the username the person typed.

    Tesseract often splits 'alex_travels' into 'alex' + 'travels' (it reads
    underscores poorly), so neighbouring tokens that sit close together are
    joined back into one candidate."""
    best, best_h = None, 0
    for ln in lines:
        if ln.y1 > stats_top:
            break
        if TIME_RE.match(ln.words[0].text) or _norm(ln.words[0].text) in STATUS_WORDS:
            continue
        # group neighbouring words into visual segments
        segments, cur = [], [ln.words[0]]
        for prev, w in zip(ln.words, ln.words[1:]):
            if w.x0 - prev.x1 <= 1.6 * max(prev.h, w.h):
                cur.append(w)
            else:
                segments.append(cur); cur = [w]
        segments.append(cur)
        for seg in segments:
            text = "".join(w.text.strip() for w in seg)
            if not USERNAME_TOKEN_RE.match(text) or not re.search(r"[A-Za-z]", text):
                continue
            if _norm(text) in BUTTON_WORDS or _norm(text) in TAB_WORDS:
                continue
            h = median(w.h for w in seg)
            if h > best_h:
                best, best_h = text, h
    return best.strip(".,") if best else None


def _analyze_words(words, arr, is_dark):
    lines = _group_lines(words)
    stats = _find_stats(words, lines)

    result = {"stats": stats, "bio": None, "name": None, "header_username": None,
              "transcript": [], "ocr_conf": (sum(w.conf for w in words) / len(words)) if words else 0.0}

    # Role-label every line for the transcript shown on the review page.
    for ln in lines:
        t = ln.text.strip().lower()
        ln.role = ("buttons" if _is_button_row(ln) else
                   "tabs" if _is_tab_row(ln) else
                   "followed_by" if t.startswith("followed by") else
                   "status" if TIME_RE.match(ln.words[0].text) else "other")

    if not stats:
        result["transcript"] = [{"text": l.text, "role": l.role, "conf": round(l.conf)} for l in lines]
        return result

    stat_label_words = [s["label_word"] for s in stats.values()]
    anchor_bottom = max(w.y1 for w in stat_label_words)
    stats_top = min(w.y0 for w in stat_label_words) - 2 * median(w.h for w in stat_label_words)
    stats_top = max(stats_top, 0)
    stat_ids = {id(w) for s in stats.values() for w in (s["label_word"],)}

    for ln in lines:
        if ln.y1 <= anchor_bottom + 2 and any(abs(w.cy - ln.cy) < ln.h for w in stat_label_words) \
           or any(id(w) in stat_ids for w in ln.words):
            ln.role = "stats"

    bg = float(np.median(arr))
    result["header_username"] = _extract_header_username(lines, stats_top)

    # Lines below the stats row, in order, until a terminator / big gap.
    below = [l for l in lines if l.cy > anchor_bottom + 1 and l.role not in ("stats",)]
    block, prev = [], None
    median_h = median(l.h for l in below) if below else 20
    terminated = False
    for ln in below:
        if _NOISE_LINE.match(ln.text.strip()):
            # "... more" / "See translation" belong to the bio box itself. They
            # must NOT end the block ('more' is also a button word), otherwise
            # the "bio is cut off" marker is never seen and a truncated bio is
            # reported as complete.
            ln.role = "other"
            block.append(ln)
            prev = ln
            continue
        if _is_terminator(ln):
            terminated = True
            break
        if prev is not None and (ln.y0 - prev.y1) > 2.2 * median_h and block:
            terminated = True
            break
        cleaned = _drop_junk_words(ln)
        if cleaned is not None:
            cleaned.role = "other"
            block.append(cleaned)
            prev = ln
    # (a line that was entirely junk is simply skipped)

    # name / category / link / noise / bio
    strengths = [_ink_strength(arr, l, bg) for l in block]
    top_strength = max(strengths) if strengths else 0
    coverages = [_ink_coverage(arr, l, bg) for l in block]
    notes = []

    # Is the first line a (bold) display name? Compare its ink coverage with
    # the regular-weight lines below it. If it is NOT visibly bolder, the
    # profile probably has no display name and that line is bio text.
    has_name, name_ambiguous = True, False
    rest_cov = [c for c, l in zip(coverages[1:], block[1:]) if not _is_link_line(l)]
    if len(block) >= 2 and rest_cov:
        ratio = coverages[0] / max(median(rest_cov), 1e-6)
        if ratio <= 1.0:
            has_name = False
            notes.append("No bold display name was detected, so the first line was counted as part of the bio.")
        elif ratio < 1.4:
            name_ambiguous = True
            notes.append("The first line looks like a display name but it is hard to be sure; if it is actually part of the bio, add its length.")

    name_line, category_line, link_line, bio_lines, truncated = None, None, None, [], False
    suspect_gaps = 0

    for i, ln in enumerate(block):
        text = ln.text.strip()
        if i == 0 and has_name:
            name_line = ln; ln.role = "name"; continue
        if i == 1 and has_name and top_strength and strengths[i] < 0.72 * top_strength and len(text) <= 45:
            category_line = ln; ln.role = "category"; continue
        if _is_link_line(ln):
            link_line = ln; ln.role = "link"; continue
        if _NOISE_LINE.match(text):
            truncated = truncated or "more" in text.lower()
            ln.role = "noise"; continue
        m = re.search(r"(\.\.\.|…)\s*more$", text, re.I)
        ln.role = "bio"
        suspect_gaps += _suspect_symbol_gaps(ln)
        if m:
            truncated = True
            clean = text[:m.start()].rstrip()
            if clean:
                bio_lines.append(clean)
            continue
        bio_lines.append(text)

    bio_text = "\n".join(bio_lines)
    bio_conf_words = [w.conf for l in block if l.role == "bio" for w in l.words]
    avg = sum(bio_conf_words) / len(bio_conf_words) if bio_conf_words else result["ocr_conf"]

    if truncated:
        tier = "low"                     # screenshot shows a cut-off bio -> length is a minimum
        notes.append("The bio is cut off with '... more' in this screenshot, so its length is a minimum. Upload a screenshot with the bio expanded for an exact count.")
    elif not terminated:
        tier = "medium"                  # never saw where the bio ends
        notes.append("The end of the bio was not visible in this screenshot (no buttons or tab row below it), so lines may be missing.")
    elif avg >= 75:
        tier = "high" if bio_lines else "medium"
    else:
        tier = "medium" if avg >= 50 else "low"
    if not bio_lines and tier != "low":
        notes.append("No bio text was found. If the profile really has no bio, 0 is correct.")
    if name_ambiguous and tier == "high":
        tier = "medium"
    if suspect_gaps and tier == "high":
        tier = "medium"
    if suspect_gaps:
        notes.append(f"About {suspect_gaps} emoji/symbol(s) in the bio could not be read, so the length may be that many characters too low.")

    result["bio"] = {"text": bio_text, "lines": bio_lines, "len": len(bio_text),
                     "tier": tier, "truncated": truncated, "terminated": terminated,
                     "notes": notes, "has_name": has_name,
                     "name_line": name_line.text.strip() if name_line else None,
                     "category_line": category_line.text.strip() if category_line else None,
                     "link_line": link_line.text.strip() if link_line else None}
    result["name"] = result["bio"]["name_line"]
    result["transcript"] = [{"text": l.text, "role": l.role, "conf": round(l.conf)} for l in lines]
    return result


def analyze_image(path: str) -> dict:
    """OCR + layout analysis of ONE screenshot."""
    work, scale, arr, is_dark = _prepare(path)

    best = None
    # Try the sparse-text mode first (best for scattered UI text); fall back
    # to the block mode if no stats row could be located.
    for psm in (11, 6):
        words = _ocr_words(work, scale, psm)
        res = _analyze_words(words, arr, is_dark)
        if best is None or len(res["stats"]) > len(best["stats"]):
            best = res
        if len(best["stats"]) >= 3:
            break
    best["filename"] = os.path.basename(path)
    best["dark_mode"] = bool(is_dark)
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Merge across images
# ─────────────────────────────────────────────────────────────────────────────
def _confidence_label(ocr_conf: float, matched: bool, n_candidates: int) -> str:
    if not matched:
        return "none"
    if n_candidates > 1:
        return "low"        # conflicting candidates — needs human eyes
    if ocr_conf >= 75:
        return "high"
    if ocr_conf >= 45:
        return "medium"
    return "low"


_TIER_RANK = {"high": 3, "medium": 2, "low": 1, "none": 0}


def merge_analyses(analyses: list) -> dict:
    """Combine per-image analyses into the review-table `fields` dict."""
    fields = {}

    for stat in ("followers", "following", "posts"):
        cands = []   # (display, parsed, conf, filename)
        for a in analyses:
            s = a["stats"].get(stat)
            if s:
                cands.append((s["display"], _parse_count(s["display"]), s["conf"], a["filename"]))
        if not cands:
            fields[stat] = {"value": None, "display": "", "confidence": "none",
                            "source_image": None, "candidates": []}
            continue
        best = max(cands, key=lambda c: c[2])
        unique = {c[1] for c in cands}
        fields[stat] = {
            "value": best[1], "display": best[0].strip(),
            "confidence": _confidence_label(best[2], True, len(unique)),
            "source_image": best[3],
            "candidates": sorted({f"{c[0].strip()} ({c[3]})" for c in cands}),
        }

    # Bio: prefer the most trustworthy reading; among equals prefer the
    # fuller one (a screenshot with the bio expanded beats a truncated one).
    bios = [(a["bio"], a["filename"]) for a in analyses if a.get("bio")]
    if bios:
        b, src = max(bios, key=lambda p: (not p[0]["truncated"], _TIER_RANK[p[0]["tier"]], p[0]["len"]))
        fields["bio_text"] = {
            "value": b["text"], "display": b["text"], "confidence": b["tier"],
            "source_image": src, "candidates": [],
            "name_line": b["name_line"], "category_line": b["category_line"],
            "link_line": b["link_line"], "truncated": b["truncated"],
            "notes": b["notes"],
        }
        fields["bio_len"] = {
            "value": b["len"], "display": str(b["len"]), "confidence": b["tier"],
            "source_image": src, "candidates": [],
        }
    else:
        # Stats row not found in any image -> we cannot tell where the bio is.
        # Report Not Found rather than guess.
        fields["bio_text"] = {"value": "", "display": "", "confidence": "none",
                              "source_image": None, "candidates": []}
        fields["bio_len"] = {"value": None, "display": "", "confidence": "none",
                             "source_image": None, "candidates": []}

    # Username: never invented. Header guess is kept ONLY as a hint.
    guesses = [(a["header_username"], a["filename"]) for a in analyses if a.get("header_username")]
    uname, usrc = guesses[0] if guesses else ("", None)
    fields["username"] = {"value": uname, "display": uname,
                          "confidence": "low" if uname else "none",
                          "source_image": usrc, "candidates": []}
    fields["username_len"] = {"value": len(uname) if uname else None,
                              "display": str(len(uname)) if uname else "",
                              "confidence": "low" if uname else "none",
                              "source_image": usrc, "candidates": []}
    return fields


def username_matches(a: str, b: str, threshold: float = 0.75) -> bool:
    """Tolerant comparison. OCR drops or swaps characters and cannot read
    underscores/periods reliably, so those are ignored on both sides."""
    strip = lambda x: re.sub(r"[^a-z0-9]", "", (x or "").lower())
    a, b = strip(a), strip(b)
    if not a or not b:
        return False
    if a in b or b in a:
        return True
    return difflib.SequenceMatcher(None, a, b).ratio() >= threshold


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point — called by app.py
# ─────────────────────────────────────────────────────────────────────────────
def read_profile_images(image_paths: list) -> dict:
    """
    OCRs every uploaded screenshot and extracts profile fields.

    Returns
    -------
    dict:
        ocr_available : bool   False => Tesseract missing; caller falls back
                               to manual entry.
        images        : list of {"filename", "ocr_conf", "dark_mode"}
        fields        : dict   per-field {value, display, confidence, ...}
        transcripts   : dict   filename -> [{"text", "role", "conf"}, ...]
                               every line the reader saw, with the role it
                               assigned (name / category / bio / link /
                               buttons / ...), so the review page can show
                               exactly what was understood.
        header_username : str | None   username read from a screenshot header
    """
    if not OCR_AVAILABLE:
        return {"ocr_available": False, "images": [], "fields": {},
                "transcripts": {}, "header_username": None}

    analyses = []
    for path in image_paths:
        try:
            analyses.append(analyze_image(path))
        except Exception as exc:                       # one bad file must not sink the rest
            analyses.append({"filename": os.path.basename(path), "stats": {}, "bio": None,
                             "name": None, "header_username": None, "transcript": [],
                             "ocr_conf": 0.0, "dark_mode": False, "error": str(exc)})

    fields = merge_analyses(analyses)
    return {
        "ocr_available": True,
        "images": [{"filename": a["filename"], "ocr_conf": round(a["ocr_conf"], 1),
                    "dark_mode": a.get("dark_mode", False)} for a in analyses],
        "fields": fields,
        "transcripts": {a["filename"]: a["transcript"] for a in analyses},
        "header_username": fields["username"]["value"] or None,
    }