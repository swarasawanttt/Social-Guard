"""
app.py
──────
Flask application for Social Guard.

Routes:
    GET  /                → homepage
    GET  /detect          → detection form
    POST /analyze         → runs full ML pipeline → redirects to /results
    GET  /results         → shows prediction result
    GET  /download/<fname>→ downloads PDF report
    GET  /about           → about page
    GET  /error           → error page
"""

import os
import sys
import secrets
from collections import OrderedDict
from flask import (Flask, render_template, request, redirect,
                   url_for, session, send_from_directory)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # app.py is at project root

from config         import SECRET_KEY, DEBUG, UPLOAD_DIR, REPORT_DIR
from src.utils          import validate_input
from src.predict        import (run_prediction, run_prediction_manual,
                                ManualInputRequired, InsufficientData,
                                InvalidProfileInput)
from src.report_generator import generate_report

app = Flask(__name__)
app.secret_key          = SECRET_KEY
app.debug               = DEBUG
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# In-memory result store
# ─────────────────────────────────────────────────────────────────────────────
# Flask's default session lives in a signed browser COOKIE, capped at ~4KB.
# The full result (SHAP + LIME rows for all 8 features, narrative text,
# indicators, display table...) comfortably exceeds that — measured at
# ~4.2KB just for the XAI section alone before anything else is added, so
# putting it straight in the cookie would silently corrupt or drop the
# session on some browsers. Instead, the full result is kept here, in this
# process's memory, keyed by a short random token; only that token goes in
# the cookie.
#
# This is a single-process dev-server pattern (python app.py / flask run),
# which is exactly how this project runs. It would need a shared store
# (e.g. Redis) instead of this dict if ever deployed behind multiple
# worker processes — noted here so that isn't a surprise later.
_RESULTS_STORE: "OrderedDict[str, dict]" = OrderedDict()
_RESULTS_STORE_MAX = 50   # evict oldest once more than this many results are held


def _store_result(result: dict) -> str:
    token = secrets.token_urlsafe(16)
    _RESULTS_STORE[token] = result
    while len(_RESULTS_STORE) > _RESULTS_STORE_MAX:
        _RESULTS_STORE.popitem(last=False)   # evict oldest
    return token


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def homepage():
    return render_template("homepage.html")


@app.route("/detect")
def detect():
    return render_template("detect.html")




@app.route("/loading", methods=["POST"])
def loading():
    """
    Shows the animated loading screen.
    Stores form data in session so the JS on the page can POST to /analyze.
    """
    identifier = request.form.get("identifier", "").strip()
    platform   = request.form.get("platform",   "").strip()

    # Quick validation before showing loader
    ok, err = validate_input(identifier, platform)
    if not ok:
        session["error_reason"] = "invalid_url"
        session["error_detail"] = err
        return redirect(url_for("error"))

    # Store inputs so the loading page can submit them via JS
    session["pending_identifier"] = identifier
    session["pending_platform"]   = platform
    return render_template("analysis.html")

def _finish_analysis(result):
    """
    Shared tail end of both the auto-scraped and manual-entry flows:
    generate the PDF report (gets the FULL result, including all 8
    SHAP/LIME features — no size constraint there), store the full
    result server-side keyed by a token, and put only that small token
    in the browser session before redirecting to /results.
    """
    # Generate PDF report — uses the full result, untrimmed
    try:
        pdf_path = generate_report(result, result["display_features"])
        pdf_filename = os.path.basename(pdf_path)
    except Exception:
        # PDF failure is non-fatal — results page still shows without download
        pdf_filename = None

    full_result = {
        "label":            result["label"],
        "label_id":         result["label_id"],
        "confidence":       result["confidence"],
        "probabilities":    result["probabilities"],
        "indicators":       result["indicators"],
        "summary":          result["summary"],
        "model_used":       result["model_used"],
        "username":         result["username"],
        "platform":         result["platform"],
        "display_features": result["display_features"],
        "data_source":      result.get("data_source", "unknown"),
        "pdf_filename":     pdf_filename,
        "xai":              result.get("xai"),
    }
    token = _store_result(full_result)
    session["result_id"] = token
    return redirect(url_for("results"))


import re as _re
_COUNT_RE = _re.compile(r"^\d+(?:\.\d+)?[KMB]?$")


def _check_manual_counts(form) -> str:
    """Return a user-facing problem description, or '' when followers,
    following and posts are all present and valid numbers (digits, optionally
    with a K / M / B suffix, e.g. 480 or 1.2K). Blank or garbage values must
    never be silently treated as 0 - that would produce a made-up verdict."""
    labels = {"followers": "Followers", "following": "Following", "posts": "Posts"}
    missing, invalid = [], []
    for key, label in labels.items():
        raw = (form.get(key) or "").strip()
        if not raw:
            missing.append(label)
            continue
        cleaned = raw.upper().replace(",", "").replace("+", "").replace(" ", "")
        if not _COUNT_RE.match(cleaned):
            invalid.append(f'{label} ("{raw}")')
    parts = []
    if missing:
        parts.append("Please fill in: " + ", ".join(missing) + ".")
    if invalid:
        parts.append("These are not valid numbers: " + ", ".join(invalid) +
                     ". Use digits, optionally with K, M or B (for example 480 or 1.2K).")
    return " ".join(parts)


@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    """
    Main prediction route.

    Three ways in:
      1. GET  from the loading page — reads pending identifier/platform
         from session, attempts a REAL data fetch via src/scraper.py.
      2. POST from the detect form directly (identifier + platform only)
         — same real-fetch attempt as above.
      3. POST from manual_input.html (identifier + platform + username +
         followers/following/posts/bio_len/username_len) — skips fetching
         and runs the prediction on the stats the user typed in.
    """
    # ── Case 3: manual stats submitted from manual_input.html ───────────────
    if request.method == "POST" and "followers" in request.form:
        identifier = request.form.get("identifier", "").strip()
        platform   = request.form.get("platform",   "").strip()
        username   = request.form.get("username",   "").strip()

        problem = _check_manual_counts(request.form)
        if problem:
            session["error_reason"] = "invalid_input"
            session["error_detail"] = problem
            return redirect(url_for("error"))

        try:
            result = run_prediction_manual(identifier, platform, username, request.form)
        except InsufficientData as e:
            session["error_reason"] = "insufficient_data"
            session["error_detail"] = str(e)
            return redirect(url_for("error"))
        except FileNotFoundError:
            session["error_reason"] = "server_error"
            session["error_detail"] = (
                "ML models not found. Please run this command in your terminal first: "
                "python src/train_model.py — this trains the models and takes about 2-3 minutes."
            )
            return redirect(url_for("error"))
        except Exception as e:
            import traceback
            session["error_reason"] = "server_error"
            session["error_detail"] = f"Prediction failed: {str(e)}"
            app.logger.error(traceback.format_exc())
            return redirect(url_for("error"))

        return _finish_analysis(result)

    # ── Cases 1 & 2: fresh identifier/platform, attempt a real fetch ────────
    if request.method == "POST":
        identifier = request.form.get("identifier", "").strip()
        platform   = request.form.get("platform",   "").strip()
    else:
        # Called via GET from loading page after animation completes
        identifier = session.pop("pending_identifier", "").strip()
        platform   = session.pop("pending_platform",   "").strip()

    # 1. Validate
    ok, err = validate_input(identifier, platform)
    if not ok:
        session["error_reason"] = "invalid_url"
        session["error_detail"] = err
        return redirect(url_for("error"))

    # 2. Run prediction pipeline (tries real scraping first)
    try:
        result = run_prediction(identifier, platform)
    except InvalidProfileInput as e:
        # Bad link, or the account does not exist: say so, don't ask for stats.
        session["error_reason"] = "not_found" if e.code == "not_found" else "invalid_url"
        session["error_detail"] = (e.message + (" " + e.suggestion if e.suggestion else "")).strip()
        return redirect(url_for("error"))
    except InsufficientData as e:
        session["error_reason"] = "insufficient_data"
        session["error_detail"] = str(e)
        return redirect(url_for("error"))
    except ManualInputRequired as m:
        # Couldn't auto-fetch — stash context and send the user to the
        # manual-entry form (templates/manual_input.html).
        ctx = m.context
        session["manual_identifier"] = ctx["identifier"]
        session["manual_platform"]   = ctx["platform"]
        session["manual_username"]   = ctx["username"]
        session["manual_fields"]     = ctx["fields"]
        session["manual_error"]      = ctx["error"]
        return redirect(url_for("manual_input"))
    except FileNotFoundError:
        session["error_reason"] = "server_error"
        session["error_detail"] = (
            "ML models not found. Please run this command in your terminal first: "
            "python src/train_model.py — this trains the models and takes about 2-3 minutes."
        )
        return redirect(url_for("error"))
    except Exception as e:
        import traceback
        session["error_reason"] = "server_error"
        session["error_detail"] = f"Prediction failed: {str(e)}"
        app.logger.error(traceback.format_exc())
        return redirect(url_for("error"))

    return _finish_analysis(result)


@app.route("/manual-input")
def manual_input():
    """
    Shown when a profile's stats couldn't be auto-fetched (private
    Instagram account, or a platform that blocks logged-out scraping
    like Twitter/X, Facebook, LinkedIn). User types in the numbers they
    can see on the public profile page themselves.
    """
    identifier = session.get("manual_identifier")
    if not identifier:
        return redirect(url_for("detect"))

    return render_template(
        "manual_input.html",
        identifier=identifier,
        platform=session.get("manual_platform"),
        username=session.get("manual_username"),
        fields=session.get("manual_fields", []),
        error_message=session.get("manual_error", ""),
    )


@app.route("/upload-screenshots", methods=["GET", "POST"])
def upload_screenshots():
    """
    Alternative to typing stats by hand: user uploads 1-4 screenshots of
    the profile (profile page, posts grid, etc). OCR reads what it can,
    and the result is shown as an editable review table (/review-screenshots)
    rather than trusted blindly — every field can be corrected before
    prediction runs.
    """
    identifier = session.get("manual_identifier")
    if not identifier:
        return redirect(url_for("detect"))

    if request.method == "GET":
        return render_template(
            "upload_screenshots.html",
            identifier=identifier,
            platform=session.get("manual_platform"),
            username=session.get("manual_username"),
        )

    files = [f for f in request.files.getlist("screenshots") if f and f.filename]
    if not files:
        session["manual_error"] = "No images were uploaded. Please choose at least one screenshot."
        return redirect(url_for("upload_screenshots"))
    if len(files) > 4:
        files = files[:4]

    from werkzeug.utils import secure_filename
    from src.image_reader import read_profile_images

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    saved_paths = []
    for f in files:
        safe = secure_filename(f.filename) or "upload.png"
        path = os.path.join(UPLOAD_DIR, f"{session.get('manual_username','user')}_{safe}")
        f.save(path)
        saved_paths.append(path)

    try:
        ocr_result = read_profile_images(saved_paths)
    finally:
        # Screenshots are deleted right after OCR — never kept beyond this request.
        for p in saved_paths:
            try:
                os.remove(p)
            except OSError:
                pass

    if not ocr_result["ocr_available"]:
        session["manual_error"] = ("Screenshot reading isn't available on this server "
                                   "(OCR engine not installed). Please enter the stats manually.")
        return redirect(url_for("manual_input"))

    # The username is already known for certain — it's whatever the user
    # typed into the detect form, parsed earlier by src/scraper.py and
    # stashed in session["manual_username"] before this screenshot step
    # ever started. That beats OCR every time: Instagram's profile header
    # shows the username as plain text with no "@" sign at all (the "@"
    # only shows up in bio links/mentions), so the regex-based OCR field
    # extractor can only ever match a username BY ACCIDENT — usually by
    # grabbing some unrelated @mention or link out of the bio text, which
    # is a real value but the WRONG one, not an empty one. That means a
    # check like "only use the known username if OCR found nothing" is
    # not safe: OCR frequently finds *something*, just something wrong,
    # and that wrong value would then win. So this always overrides with
    # the known value when we have it, full stop — OCR's username guess
    # is never trusted over it.
    known_username = (session.get("manual_username") or "").strip()
    fields = ocr_result["fields"]
    if known_username:
        fields["username"] = {
            "value": known_username, "display": known_username,
            "confidence": "high", "source_image": None, "candidates": [],
        }
        fields["username_len"] = {
            "value": len(known_username), "display": str(len(known_username)),
            "confidence": "high", "source_image": None, "candidates": [],
        }

    session["ocr_fields"] = fields
    session["ocr_images"] = ocr_result["images"]
    return redirect(url_for("review_screenshots"))


@app.route("/review-screenshots")
def review_screenshots():
    """
    Editable review table of what OCR extracted from the uploaded
    screenshots, with a confidence flag per field, before prediction runs.
    """
    fields = session.get("ocr_fields")
    if not fields:
        return redirect(url_for("detect"))

    return render_template(
        "review_screenshots.html",
        identifier=session.get("manual_identifier"),
        platform=session.get("manual_platform"),
        username=session.get("manual_username"),
        fields=fields,
        images=session.get("ocr_images", []),
    )


@app.route("/results")
def results():
    # The full result (including SHAP/LIME data, which is too big for a
    # browser cookie) lives server-side in _RESULTS_STORE, keyed by a short
    # token kept in the session. If the token is missing, or has aged out
    # of the in-memory store (server restarted, or evicted under the LRU
    # cap), send the person back to start a fresh analysis instead of
    # crashing on a KeyError.
    token = session.get("result_id")
    result = _RESULTS_STORE.get(token) if token else None
    if not result:
        return redirect(url_for("detect"))
    return render_template("results.html", result=result)


@app.route("/download/<filename>")
def download_report(filename):
    """Serve the generated PDF report as a file download."""
    # Security: only allow files that exist in REPORT_DIR
    safe = os.path.basename(filename)   # strip any path traversal
    filepath = os.path.join(REPORT_DIR, safe)
    if not os.path.exists(filepath):
        session["error_reason"] = "server_error"
        session["error_detail"] = "Report file not found. Please run the analysis again."
        return redirect(url_for("error"))
    return send_from_directory(
        REPORT_DIR, safe,
        as_attachment=True,
        download_name=safe,
    )



@app.route("/debug")
def debug_info():
    """
    Debug page — visit http://localhost:5000/debug to diagnose issues.
    Shows paths, model status, dataset status.
    Remove this route before deploying to production.
    """
    import os
    from config import (BASE_DIR, MODEL_DIR, FINAL_DATASET_PATH,
                        BEST_MODEL_PATH, SCALER_PATH)
    info = {
        "BASE_DIR":           BASE_DIR,
        "MODEL_DIR":          MODEL_DIR,
        "model_exists":       os.path.exists(BEST_MODEL_PATH),
        "scaler_exists":      os.path.exists(SCALER_PATH),
        "dataset_exists":     os.path.exists(FINAL_DATASET_PATH),
        "BEST_MODEL_PATH":    BEST_MODEL_PATH,
        "SCALER_PATH":        SCALER_PATH,
        "FINAL_DATASET_PATH": FINAL_DATASET_PATH,
    }
    if os.path.exists(MODEL_DIR):
        info["models_folder_contents"] = os.listdir(MODEL_DIR)
    else:
        info["models_folder_contents"] = "FOLDER DOES NOT EXIST"

    rows = "".join(
        f"<tr><td style='padding:8px;font-weight:bold;color:#2f4561'>{k}</td>"
        f"<td style='padding:8px;color:{'green' if v is True else ('red' if v is False else '#333')}'>{v}</td></tr>"
        for k, v in info.items()
    )
    html = f"""<!DOCTYPE html><html><body style='font-family:monospace;padding:40px;background:#fbf9f6'>
    <h2 style='color:#2f4561'>Social Guard — Debug Info</h2>
    <table border='1' cellspacing='0' style='border-collapse:collapse;width:100%;max-width:900px'>{rows}</table>
    <br>
    <p style='color:#666'>If model_exists=False: run <code>python src/train_model.py</code> from your project root.</p>
    <p style='color:#666'>If dataset_exists=False: place final_dataset.csv in <code>data/processed/</code></p>
    <a href='/detect' style='color:#2f4561'>← Back to Detect</a>
    </body></html>"""
    return html


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/error")
def error():
    reason = session.pop("error_reason", "unknown")
    detail = session.pop("error_detail", "An unexpected error occurred.")
    return render_template("error.html", reason=reason, detail=detail)



# ── Startup check ─────────────────────────────────────────────────────────────
import logging
def _startup_check():
    from config import BEST_MODEL_PATH, SCALER_PATH
    import os
    if not os.path.exists(BEST_MODEL_PATH):
        logging.warning(
            "\n" + "="*60 +
            "\n  WARNING: No trained model found." +
            "\n  Run this before starting the app:" +
            "\n    python src/train_model.py" +
            "\n" + "="*60
        )

_startup_check()

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=DEBUG, host="0.0.0.0", port=5000)