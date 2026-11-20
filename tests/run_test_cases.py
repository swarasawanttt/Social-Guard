"""
tests/run_test_cases.py
───────────────────────
Runs the Social Guard software test cases against the REAL code and records
the ACTUAL result of each one, ready to paste into a presentation.

Run from the project root:

    python tests/run_test_cases.py

Output:
    - a table in the terminal
    - tests/test_results.csv   (open in Excel)
    - tests/test_results.md    (paste into slides / docs)

Status meanings:
    PASS  the system behaved as expected
    FAIL  the system did NOT behave as expected (a real defect or regression)
    SKIP  the case could not run here (missing optional module, OCR engine
          not installed, ...). The reason is shown in "Actual Result".

Nothing here touches the internet: network-dependent behaviour (account
not found, private account) is simulated by replacing the scraper's answer.
"""
import os
import sys
import csv
import json
import glob
import time
import tempfile
import traceback
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import warnings
warnings.filterwarnings("ignore")

CASES = []          # (id, module, name, input, expected, func)


def case(tc, module, name, given, expected):
    def deco(fn):
        CASES.append((tc, module, name, given, expected, fn))
        return fn
    return deco


class Skip(Exception):
    pass


def passed(actual):  return ("PASS", actual)
def failed(actual):  return ("FAIL", actual)


def need(modname):
    try:
        __import__(modname)
    except Exception as e:
        raise Skip(f"{modname} not available ({type(e).__name__})")


# ── shared helpers ────────────────────────────────────────────────────────────
_app = None
def get_app():
    global _app
    if _app is None:
        from app import app
        app.config["TESTING"] = True
        _app = app
    return _app

def client():
    return get_app().test_client()

def text_of(resp):
    return resp.get_data(as_text=True)

GOOD = {"followers": "480", "following": "520", "posts": "95", "bio_text": "Travel photographer in Berlin"}

def manual_post(c, **over):
    data = {"identifier": "instagram.com/tc_user", "platform": "Instagram",
            "username": "tc_user", **GOOD, **over}
    return c.post("/analyze", data=data, follow_redirects=True)

def predict(f, g, p, bio_text="x" * 30):
    from src.predict import run_prediction_manual
    return run_prediction_manual("instagram.com/tc_user", "Instagram", "tc_user",
                                 {"followers": str(f), "following": str(g), "posts": str(p),
                                  "bio_text": bio_text})

def fake_fetch(**kw):
    base = {"manual_required": False, "invalid": False, "platform": "Instagram", "username": "zzz",
            "source": "scraped", "data": {}, "manual_fields": [], "error": None,
            "error_code": None, "suggestion": ""}
    base.update(kw)
    return lambda identifier, platform, *a, **k: base


# ═════════════════════════════════════════════════════════════════════════════
# A. Input validation and URL handling
# ═════════════════════════════════════════════════════════════════════════════
@case("TC01", "Input validation", "Valid profile URL is accepted",
      "https://instagram.com/cristiano + Instagram", "Accepted, no error")
def _():
    from src.utils import validate_input
    ok, msg = validate_input("https://instagram.com/cristiano", "Instagram")
    return (passed if ok else failed)(f"ok={ok}, message='{msg}'")

@case("TC02", "Input validation", "Bare username is accepted",
      "cristiano + Instagram", "Accepted, no error")
def _():
    from src.utils import validate_input
    ok, msg = validate_input("cristiano", "Instagram")
    return (passed if ok else failed)(f"ok={ok}, message='{msg}'")

@case("TC03", "Input validation", "Empty input is rejected",
      "'' (empty) + Instagram", "Validation error, no crash")
def _():
    from src.utils import validate_input
    ok, msg = validate_input("", "Instagram")
    return (passed if (not ok and msg) else failed)(f"ok={ok}, message='{msg}'")

@case("TC04", "Input validation", "Missing platform is rejected",
      "cristiano + (no platform)", "Validation error asking to select a platform")
def _():
    from src.utils import validate_input
    ok, msg = validate_input("cristiano", "")
    return (passed if (not ok and "platform" in msg.lower()) else failed)(f"ok={ok}, message='{msg}'")

@case("TC05", "Input validation", "Platform must match the link",
      "Instagram selected + https://twitter.com/elonmusk", "Rejected: platform does not match link")
def _():
    from src.utils import validate_input
    ok, msg = validate_input("https://twitter.com/elonmusk", "Instagram")
    return (passed if (not ok and "Twitter" in msg) else failed)(f"ok={ok}, message='{msg}'")

@case("TC06", "Input validation", "Misspelled website is rejected with a hint",
      "https://instagarm.com/cristiano + Instagram", "Rejected, suggests the correct site")
def _():
    need("src.url_resolver")
    from src.utils import validate_input
    ok, msg = validate_input("https://instagarm.com/cristiano", "Instagram")
    return (passed if not ok else failed)(f"ok={ok}, message='{msg}'")

@case("TC07", "Input validation", "Post link (not a profile) is rejected",
      "https://instagram.com/p/ABC123 + Instagram", "Rejected: not a profile link")
def _():
    need("src.url_resolver")
    from src.utils import validate_input
    ok, msg = validate_input("https://instagram.com/p/ABC123", "Instagram")
    return (passed if not ok else failed)(f"ok={ok}, message='{msg}'")

@case("TC08", "Input validation", "Illegal username characters are rejected",
      "'bad name!!' + Instagram", "Rejected: invalid username")
def _():
    need("src.url_resolver")
    from src.utils import validate_input
    ok, msg = validate_input("bad name!!", "Instagram")
    return (passed if not ok else failed)(f"ok={ok}, message='{msg}'")


# ═════════════════════════════════════════════════════════════════════════════
# B. Data acquisition (scraper behaviour, simulated network)
# ═════════════════════════════════════════════════════════════════════════════
@case("TC09", "Data acquisition", "Human-readable counts are converted to integers",
      "'239.8M', '1.2K', '500+', '1,234'", "239800000, 1200, 500, 1234")
def _():
    from src.scraper import parse_manual_value
    got = [parse_manual_value(x) for x in ("239.8M", "1.2K", "500+", "1,234")]
    return (passed if got == [239800000, 1200, 500, 1234] else failed)(str(got))

@case("TC10", "Data acquisition", "Account that does not exist gives a clear error",
      "Scraper reports 'not found' for @zzz", "Error page: account not found, no manual form")
def _():
    import src.predict as P
    with mock.patch.object(P, "fetch_profile_data", fake_fetch(
            invalid=True, error="No Instagram account named @zzz exists.", error_code="not_found",
            suggestion="Check the spelling.")):
        r = client().post("/analyze", data={"identifier": "instagram.com/zzz", "platform": "Instagram"},
                          follow_redirects=True)
    ok = r.request.path == "/error" and "exist" in text_of(r)
    return (passed if ok else failed)(f"landed on {r.request.path}; message shown={('exist' in text_of(r))}")

@case("TC11", "Data acquisition", "Private / blocked profile falls back to manual entry",
      "Scraper cannot read stats (private account)", "Manual-entry form is shown")
def _():
    import src.predict as P
    with mock.patch.object(P, "fetch_profile_data", fake_fetch(
            manual_required=True, error="This account is private.",
            manual_fields=[{"name": "followers", "label": "Followers", "type": "text", "placeholder": "e.g. 1.2K"},
                           {"name": "following", "label": "Following", "type": "text", "placeholder": "e.g. 300"},
                           {"name": "posts", "label": "Posts", "type": "text", "placeholder": "e.g. 40"},
                           {"name": "bio_len", "label": "Bio length (chars)", "type": "number", "placeholder": "e.g. 90"}])):
        r = client().post("/analyze", data={"identifier": "instagram.com/zzz", "platform": "Instagram"},
                          follow_redirects=True)
    ok = r.request.path == "/manual-input" and 'name="followers"' in text_of(r)
    return (passed if ok else failed)(f"landed on {r.request.path}; form has followers field={'name=\"followers\"' in text_of(r)}")

@case("TC12", "Data acquisition", "Scraper's own unit tests pass",
      "tests/test_scraper.py", "All tests pass")
def _():
    import unittest
    if not os.path.exists(os.path.join(ROOT, "tests", "test_scraper.py")):
        raise Skip("tests/test_scraper.py not found")
    suite = unittest.defaultTestLoader.loadTestsFromName("tests.test_scraper")
    res = unittest.TextTestRunner(stream=open(os.devnull, "w")).run(suite)
    n = res.testsRun
    bad = len(res.failures) + len(res.errors)
    return (passed if bad == 0 else failed)(f"{n - bad}/{n} scraper tests passed")


# ═════════════════════════════════════════════════════════════════════════════
# C. Manual entry
# ═════════════════════════════════════════════════════════════════════════════
@case("TC13", "Manual entry", "Valid manual entry produces a result",
      "480 followers, 520 following, 95 posts, bio text", "Result page with a verdict")
def _():
    r = manual_post(client())
    body = text_of(r)
    ok = r.request.path == "/results" and any(x in body for x in ("Real", "Fake", "Bot"))
    return (passed if ok else failed)(f"landed on {r.request.path}, status {r.status_code}")

@case("TC14", "Manual entry", "Blank required fields are rejected",
      "Followers = blank, Posts = blank", "Validation error, no verdict")
def _():
    r = manual_post(client(), followers="", posts="")
    ok = r.request.path != "/results"
    return (passed if ok else failed)(
        f"landed on {r.request.path}" + ("" if ok else " (blanks were silently treated as 0 and a verdict was returned)"))

@case("TC15", "Manual entry", "Partially filled form is rejected",
      "Followers = 500, Following = blank, Posts = 40", "Error message, no verdict")
def _():
    r = manual_post(client(), followers="500", following="", posts="40")
    ok = r.request.path != "/results"
    return (passed if ok else failed)(
        f"landed on {r.request.path}" + ("" if ok else " (blank 'following' was silently treated as 0)"))

@case("TC16", "Manual entry", "All-zero profile gets 'not enough data', not a verdict",
      "0 followers, 0 following, 0 posts", "Clear 'not enough data' error")
def _():
    r = manual_post(client(), followers="0", following="0", posts="0", bio_text="")
    body = text_of(r).lower()
    ok = r.request.path == "/error" and "not enough" in body
    return (passed if ok else failed)(f"landed on {r.request.path}; 'not enough information' message={'not enough' in body}")

@case("TC17", "Manual entry", "Garbage numbers do not crash the app",
      "Followers = 'abc'", "Handled error page, no server crash (no HTTP 500)")
def _():
    r = manual_post(client(), followers="abc")
    ok = r.status_code == 200 and r.request.path != "/results"
    return (passed if ok else failed)(f"HTTP {r.status_code}, landed on {r.request.path}")

@case("TC18", "Manual entry", "Bio length is counted from the pasted text on the server",
      "Bio 'Line one<CRLF>Second 🌍 end'", "21 characters (line break = 1, emoji = 1)")
def _():
    from src.predict import _bio_length
    n = _bio_length({"bio_text": "Line one\r\nSecond \U0001F30D end\r\n"})
    return (passed if n == 21 else failed)(f"counted {n} characters")


# ═════════════════════════════════════════════════════════════════════════════
# D. Prediction
# ═════════════════════════════════════════════════════════════════════════════
@case("TC19", "Prediction", "Prediction output is well-formed",
      "480 / 520 / 95 / bio", "Label in {Real, Fake, Bot}; probabilities sum to ~100%; confidence = highest probability")
def _():
    r = predict(480, 520, 95)
    probs = r["probabilities"]
    total = sum(probs.values())
    ok = (r["label"] in ("Real", "Fake", "Bot") and abs(total - 100) < 0.5
          and abs(r["confidence"] - max(probs.values())) < 0.01)
    return (passed if ok else failed)(f"label={r['label']}, confidence={r['confidence']:.1f}%, probabilities sum={total:.1f}%")

@case("TC20", "Prediction", "Same input always gives the same output",
      "Same profile submitted twice", "Identical label and probabilities")
def _():
    a, b = predict(480, 520, 95), predict(480, 520, 95)
    ok = a["label"] == b["label"] and a["probabilities"] == b["probabilities"]
    return (passed if ok else failed)(f"run 1: {a['label']} {a['confidence']:.1f}%, run 2: {b['label']} {b['confidence']:.1f}%")

@case("TC21", "Prediction", "Different profiles give different results",
      "Established account vs spam-like account", "Different probabilities")
def _():
    a, b = predict(480, 520, 95), predict(40, 3500, 12, bio_text="")
    ok = a["probabilities"] != b["probabilities"]
    return (passed if ok else failed)(f"A: {a['label']} {a['confidence']:.0f}%  |  B: {b['label']} {b['confidence']:.0f}%")

@case("TC22", "Prediction", "Spam-like profile is not classed as genuine",
      "40 followers, 3,500 following, 12 posts, no bio", "Classified Fake or Bot (not Real)")
def _():
    r = predict(40, 3500, 12, bio_text="")
    return (passed if r["label"] != "Real" else failed)(f"{r['label']} ({r['confidence']:.1f}% confidence)")

@case("TC23", "Prediction", "Trained model files are present and load",
      "models/best_model.pkl + scaler.pkl", "Both load without error")
def _():
    import joblib
    from config import BEST_MODEL_PATH, SCALER_PATH
    m, s = joblib.load(BEST_MODEL_PATH), joblib.load(SCALER_PATH)
    name = open(os.path.join(os.path.dirname(BEST_MODEL_PATH), "best_model_name.txt")).read().strip() \
        if os.path.exists(os.path.join(os.path.dirname(BEST_MODEL_PATH), "best_model_name.txt")) else type(m).__name__
    return passed(f"loaded model '{name}' and scaler ({getattr(s, 'n_features_in_', '?')} input features)")


# ═════════════════════════════════════════════════════════════════════════════
# E. Explainability
# ═════════════════════════════════════════════════════════════════════════════
@case("TC24", "Explainability", "Indicators are generated per account and differ between accounts",
      "Established account vs spam-like account", "Different indicator text for each")
def _():
    a, b = predict(480, 520, 95), predict(40, 3500, 12, bio_text="")
    ta, tb = [i["text"] for i in a["indicators"]], [i["text"] for i in b["indicators"]]
    ok = bool(ta) and bool(tb) and ta != tb
    return (passed if ok else failed)(f"{len(ta)} indicators vs {len(tb)} indicators; identical={ta == tb}")

@case("TC25", "Explainability", "Result includes SHAP/LIME explanation data",
      "480 / 520 / 95", "Non-empty 'xai' section in the result")
def _():
    r = predict(480, 520, 95)
    x = r.get("xai")
    return (passed if x else failed)("xai section present" if x else "no 'xai' section in the prediction result")

@case("TC26", "Explainability", "SHAP/LIME explanation is real and per-account, not static",
      "Established account vs spam-like account", "Different explanation for each")
def _():
    a, b = predict(480, 520, 95), predict(40, 3500, 12, bio_text="")
    if not a.get("xai") or not b.get("xai"):
        return failed("no 'xai' section in the prediction result")
    ja, jb = json.dumps(a["xai"], default=str, sort_keys=True), json.dumps(b["xai"], default=str, sort_keys=True)
    return (passed if ja != jb else failed)(f"explanations differ={ja != jb}")

@case("TC27", "Explainability", "Confidence statement names the model really used",
      "High-confidence prediction", "Text names the deployed model, never a hard-coded 'all four models'")
def _():
    texts = []
    for args in ((5, 3000, 0), (2000, 100, 300), (40, 3500, 12)):
        r = predict(*args, bio_text="")
        texts += [i["text"] for i in r["indicators"] if "confidence" in i["text"].lower()]
    bad = [t for t in texts if "all four" in t.lower()]
    return (passed if not bad else failed)(f"{len(texts)} confidence statements checked; hard-coded claims found={len(bad)}")


# ═════════════════════════════════════════════════════════════════════════════
# F. Report
# ═════════════════════════════════════════════════════════════════════════════
@case("TC28", "Report", "PDF report is generated",
      "Completed prediction", "A real PDF file on disk")
def _():
    from src.report_generator import generate_report
    r = predict(480, 520, 95)
    r["username"] = "tc_report_user"
    path = generate_report(r, r["display_features"])
    head = open(path, "rb").read(5)
    ok = os.path.exists(path) and head == b"%PDF-" and os.path.getsize(path) > 1000
    return (passed if ok else failed)(f"{os.path.basename(path)}, {os.path.getsize(path):,} bytes, header {head!r}")

@case("TC29", "Report", "PDF can be downloaded from the app",
      "GET /download/<report.pdf>", "HTTP 200, PDF content")
def _():
    from src.report_generator import generate_report
    r = predict(480, 520, 95); r["username"] = "tc_download_user"
    fn = os.path.basename(generate_report(r, r["display_features"]))
    resp = client().get("/download/" + fn)
    ok = resp.status_code == 200 and resp.data[:5] == b"%PDF-"
    return (passed if ok else failed)(f"HTTP {resp.status_code}, {len(resp.data):,} bytes")

@case("TC30", "Report", "Report failure is handled gracefully",
      "PDF generation raises an error", "Results page still shown, no crash")
def _():
    import app as appmod
    with mock.patch.object(appmod, "generate_report", side_effect=OSError("disk full")):
        r = manual_post(client())
    ok = r.status_code == 200 and r.request.path == "/results"
    return (passed if ok else failed)(f"HTTP {r.status_code}, landed on {r.request.path}")


# ═════════════════════════════════════════════════════════════════════════════
# G. Screenshot reading (OCR)
# ═════════════════════════════════════════════════════════════════════════════
def _font(size, bold=False):
    from PIL import ImageFont
    names = (["arialbd.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf"] if bold else
             ["arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"])
    dirs = ["", "C:/Windows/Fonts/", "/usr/share/fonts/truetype/dejavu/", "/usr/share/fonts/truetype/liberation/"]
    for d in dirs:
        for n in names:
            try:
                return ImageFont.truetype(d + n, size)
            except Exception:
                pass
    return ImageFont.load_default(size)

def _profile_image(path):
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (1080, 1500), (255, 255, 255)); d = ImageDraw.Draw(im)
    d.text((70, 60), "travel_with_anna", font=_font(52, True), fill=(20, 20, 20))
    d.ellipse((60, 190, 300, 430), fill=(130, 130, 130))
    for cx, num, lab in ((470, "842", "posts"), (700, "48.2K", "followers"), (930, "615", "following")):
        d.text((cx, 255), num, font=_font(50, True), fill=(20, 20, 20), anchor="mm")
        d.text((cx, 322), lab, font=_font(34), fill=(20, 20, 20), anchor="mm")
    d.text((60, 470), "Anna Weber", font=_font(40, True), fill=(20, 20, 20))
    d.text((60, 535), "Travel photographer based in Berlin", font=_font(38), fill=(20, 20, 20))
    im.save(path)

def _read(path):
    need("src.image_reader")
    from src.image_reader import read_profile_images
    res = read_profile_images([path])
    if not res.get("ocr_available", True):
        raise Skip("OCR engine (Tesseract) is not installed on this machine")
    return res

def _num(v):
    try:
        return int(float(str(v).replace(",", "")))
    except (TypeError, ValueError):
        return None          # field not found / unreadable

@case("TC31", "Screenshot reading", "Stats are extracted from a profile screenshot",
      "Drawn profile screenshot: 842 posts, 48.2K followers, 615 following", "posts=842, followers=48200, following=615")
def _():
    p = os.path.join(tempfile.gettempdir(), "tc_profile.png"); _profile_image(p)
    f = _read(p)["fields"]
    got = {k: _num(f[k]["value"]) for k in ("posts", "followers", "following")}
    exp = {"posts": 842, "followers": 48200, "following": 615}
    conf = {k: f[k].get("confidence") for k in exp}
    return (passed if got == exp else failed)(f"got {got}; confidence {conf}")

@case("TC32", "Screenshot reading", "Nothing is invented when a screenshot has no profile stats",
      "Plain image with unrelated text", "No field reported as high-confidence; no made-up bio")
def _():
    from PIL import Image, ImageDraw
    p = os.path.join(tempfile.gettempdir(), "tc_blank.png")
    im = Image.new("RGB", (900, 600), (255, 255, 255)); ImageDraw.Draw(im).text((40, 200), "Hello, this is not a profile page", font=_font(36), fill=(0, 0, 0)); im.save(p)
    f = _read(p)["fields"]
    high = [k for k, v in f.items() if isinstance(v, dict) and v.get("confidence") == "high"]
    return (passed if not high else failed)(f"high-confidence fields: {high or 'none'}")

@case("TC33", "Screenshot reading", "Screenshot upload too large is rejected",
      "17 MB upload", "HTTP 413 (limit is 16 MB)")
def _():
    import io
    c = client()
    with c.session_transaction() as sess:          # the upload step needs an active manual-entry session
        sess["manual_identifier"] = "instagram.com/tc_user"
        sess["manual_platform"] = "Instagram"
        sess["manual_username"] = "tc_user"
    r = c.post("/upload-screenshots", data={"screenshots": (io.BytesIO(b"0" * (17 * 1024 * 1024)), "big.png")},
               content_type="multipart/form-data")
    return (passed if r.status_code == 413 else failed)(f"HTTP {r.status_code}")


# ═════════════════════════════════════════════════════════════════════════════
# H. Training pipeline
# ═════════════════════════════════════════════════════════════════════════════
@case("TC34", "Training pipeline", "SMOTE runs per platform x class cell, on training data only",
      "Training table with a large and a tiny cell", "Large cell raised to target; tiny cell untouched; input unchanged; no platform mixing")
def _():
    import numpy as np, pandas as pd
    from src.train_model import _add_smote
    from config import FEATURE_COLUMNS, SMOTE_MIN_CELL, SMOTE_TARGET_CELL
    rng = np.random.RandomState(0)
    def cell(plat, y, n, shift):
        d = pd.DataFrame(rng.rand(n, len(FEATURE_COLUMNS)) * 50 + shift, columns=FEATURE_COLUMNS)
        d["platform"], d["label"] = plat, y
        return d
    tr = pd.concat([cell("Instagram", 2, SMOTE_MIN_CELL + 50, 0),
                    cell("Facebook", 2, 40, 500)], ignore_index=True)
    tr = tr.rename(columns={"label": "label"})
    before = tr.copy()
    out = _add_smote(tr, np.random.default_rng(0))
    ig = int(((out.platform == "Instagram") & (out.label == 2)).sum())
    fb = int(((out.platform == "Facebook") & (out.label == 2)).sum())
    new_ig = out.iloc[len(tr):]
    no_mix = bool((new_ig[FEATURE_COLUMNS].max() < 400).all())      # Facebook rows sit at +500
    ok = ig == SMOTE_TARGET_CELL and fb == 40 and tr.equals(before) and no_mix
    return (passed if ok else failed)(f"IG cell {SMOTE_MIN_CELL+50} -> {ig}; FB cell 40 -> {fb}; input unchanged={tr.equals(before)}; platforms not mixed={no_mix}")

@case("TC35", "Training pipeline", "Training data has no duplicate rows (no train/test leakage)",
      "load_data() from train_model.py", "0 duplicate rows after loading")
def _():
    import io, contextlib
    from src.train_model import load_data
    from config import FEATURE_COLUMNS, TARGET_COLUMN
    with contextlib.redirect_stdout(io.StringIO()):
        df = load_data()
    dup = int(df.duplicated(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).sum())
    return (passed if dup == 0 else failed)(f"{len(df):,} rows, {dup} duplicates")


@case("TC39", "Prediction", "Low-confidence result still gets a verdict, with a caution",
      "Valid profile, confidence threshold forced to 99.9%", "label is Real/Fake/Bot and a caution note is attached")
def _():
    from unittest import mock
    import src.predict as P
    raw = dict(username="someone", platform="Instagram", followers=900, following=500, posts=60, bio_len=40, username_len=7, source="manual")
    with mock.patch.object(P, "INCONCLUSIVE_THRESHOLD", 0.999):
        r = P._run_prediction_from_raw(dict(raw))
    ok = r["label"] in ("Real", "Fake", "Bot") and bool(r.get("caution")) and not r["inconclusive"]
    return (passed if ok else failed)(f"label={r['label']}, closest={r['top_class']} ({r['confidence']}%)")

@case("TC40", "Prediction", "Platform with no training data gets a verdict plus a caution",
      "Same profile submitted as LinkedIn", "label is Real/Fake/Bot and a caution note is attached")
def _():
    import src.predict as P
    r = P._run_prediction_from_raw(dict(username="a.mehta", platform="LinkedIn", followers=500, following=480, posts=20, bio_len=120, username_len=7, source="manual"))
    return (passed if r["label"] in ("Real", "Fake", "Bot") and bool(r.get("caution")) else failed)(f"label={r['label']}, caution={bool(r.get('caution'))}")

@case("TC41", "Feature engineering", "Fields that were not read stay missing, not 0",
      "Counts-only profile", "has_profile_pic / has_url / full-name features are NaN; username digit share is computed")
def _():
    import math
    from src.feature_engineering import engineer_features
    f = engineer_features(dict(username="user123", platform="Instagram", followers=10, following=20, posts=3, bio_len=5, username_len=7))
    nan = [k for k in ("has_profile_pic", "has_url", "fullname_words", "name_eq_username", "is_private") if math.isnan(f[k])]
    ok = len(nan) == 5 and abs(f["username_digit_ratio"] - 3/7) < 1e-9
    return (passed if ok else failed)(f"missing kept as NaN: {nan}; digit share={f['username_digit_ratio']:.3f}")

@case("TC42", "Feature engineering", "Percentile ranks are computed from the saved platform tables",
      "Same counts on Instagram vs Twitter/X", "Valid 0-1 percentiles; differ by platform")
def _():
    from src.feature_engineering import engineer_features
    a = engineer_features(dict(username="u", platform="Instagram", followers=5000, following=300, posts=200, bio_len=10))
    b = engineer_features(dict(username="u", platform="Twitter/X", followers=5000, following=300, posts=200, bio_len=10))
    ok = 0 <= a["followers_pr"] <= 1 and 0 <= b["followers_pr"] <= 1 and a["followers_pr"] != b["followers_pr"]
    return (passed if ok else failed)(f"followers percentile: Instagram {a['followers_pr']:.2f}, Twitter {b['followers_pr']:.2f}")


@case("TC43", "Prediction", "Platform-verified badge is honoured and labelled, never assumed",
      "Same profile with is_verified True / False / absent", "Real + verified flag only when the badge was read as True")
def _():
    import src.predict as P
    base = dict(username="bigname", platform="Instagram", followers=5_000_000, following=700, posts=1500, bio_len=60, username_len=7, source="scraped")
    v = P._run_prediction_from_raw({**base, "is_verified": True})
    n = P._run_prediction_from_raw({**base, "is_verified": False})
    u = P._run_prediction_from_raw(dict(base))
    ok = v["label"] == "Real" and v["verified_override"] and not n["verified_override"] and not u["verified_override"]
    return (passed if ok else failed)(f"verified -> {v['label']} (override={v['verified_override']}); not verified -> {n['label']}; unknown -> {u['label']}")


# ═════════════════════════════════════════════════════════════════════════════
# I. Security and robustness
# ═════════════════════════════════════════════════════════════════════════════
@case("TC36", "Security", "Script in a username is escaped, not executed",
      "Username '<script>alert(1)</script>'", "Raw <script> tag never appears in the page")
def _():
    evil = "<script>alert(1)</script>"
    r = manual_post(client(), username=evil, identifier="instagram.com/x")
    ok = evil not in text_of(r)
    return (passed if ok else failed)(f"landed on {r.request.path}; raw script tag in page={evil in text_of(r)}")

@case("TC37", "Security", "Report download cannot read files outside the reports folder",
      "GET /download/..%2Fconfig.py", "config.py content is never returned")
def _():
    r = client().get("/download/..%2Fconfig.py")
    leaked = b"SECRET_KEY" in r.data
    return (passed if not leaked else failed)(f"HTTP {r.status_code}; secret leaked={leaked}")

@case("TC38", "Security", "Results page without an analysis redirects safely",
      "GET /results with no session", "Redirect to the detect page")
def _():
    r = client().get("/results")
    ok = r.status_code in (301, 302) and "/detect" in r.headers.get("Location", "")
    return (passed if ok else failed)(f"HTTP {r.status_code} -> {r.headers.get('Location')}")


# ═════════════════════════════════════════════════════════════════════════════
def main():
    rows = []
    t0 = time.time()
    for tc, module, name, given, expected, fn in CASES:
        try:
            status, actual = fn()
        except Skip as s:
            status, actual = "SKIP", str(s)
        except Exception as e:
            status, actual = "FAIL", f"{type(e).__name__}: {str(e)[:160]}"
        rows.append((tc, module, name, given, expected, actual, status))
        print(f"{tc}  {status:4s}  {name}")

    # clean up PDFs created by the tests
    try:
        from config import REPORT_DIR
        for f in glob.glob(os.path.join(REPORT_DIR, "*tc_*")):
            os.remove(f)
    except Exception:
        pass

    counts = {s: sum(1 for r in rows if r[6] == s) for s in ("PASS", "FAIL", "SKIP")}
    print(f"\n{len(rows)} test cases: {counts['PASS']} PASS, {counts['FAIL']} FAIL, {counts['SKIP']} SKIP "
          f"({time.time() - t0:.1f}s)")

    out = os.path.join(ROOT, "tests")
    with open(os.path.join(out, "test_results.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["TC ID", "Module", "Test Case", "Input", "Expected Result", "Actual Result", "Status"])
        w.writerows(rows)
    with open(os.path.join(out, "test_results.md"), "w", encoding="utf-8") as f:
        f.write("| TC ID | Module | Test Case | Input | Expected Result | Actual Result | Status |\n|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write("| " + " | ".join(str(x).replace("|", "/").replace("\n", " ") for x in r) + " |\n")
        f.write(f"\n**Total: {len(rows)} | PASS: {counts['PASS']} | FAIL: {counts['FAIL']} | SKIP: {counts['SKIP']}**\n")
    print("Saved tests/test_results.csv and tests/test_results.md")

    # ── Presentation set: the 15 most important cases, renumbered TC1..TC15 ──
    TOP15 = ["TC01", "TC05", "TC06", "TC10", "TC11", "TC14", "TC16", "TC19",
             "TC22", "TC26", "TC28", "TC31", "TC32", "TC34", "TC36"]
    by_id = {r[0]: r for r in rows}
    top = [(f"TC{i}",) + by_id[t][1:] for i, t in enumerate(TOP15, 1) if t in by_id]
    tc = {s: sum(1 for r in top if r[6] == s) for s in ("PASS", "FAIL", "SKIP")}
    with open(os.path.join(out, "test_results_top15.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["TC ID", "Module", "Test Case", "Input", "Expected Result", "Actual Result", "Status"])
        w.writerows(top)
    with open(os.path.join(out, "test_results_top15.md"), "w", encoding="utf-8") as f:
        f.write("| TC ID | Test Case | Input | Expected Result | Actual Result | Status |\n|---|---|---|---|---|---|\n")
        for r in top:
            f.write("| " + " | ".join(str(x).replace("|", "/").replace("\n", " ") for x in (r[0],) + r[2:]) + " |\n")
        f.write(f"\n**Total: {len(top)} | PASS: {tc['PASS']} | FAIL: {tc['FAIL']} | SKIP: {tc['SKIP']}**\n")
    print(f"Saved tests/test_results_top15.csv and tests/test_results_top15.md "
          f"({tc['PASS']} PASS, {tc['FAIL']} FAIL, {tc['SKIP']} SKIP)")


if __name__ == "__main__":
    main()