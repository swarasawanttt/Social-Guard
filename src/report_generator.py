"""
src/report_generator.py
────────────────────────
Generates a downloadable PDF analysis report for a given prediction result.
Uses reportlab (Platypus) for layout.

Called by app.py after a successful prediction.
Returns the absolute path to the saved PDF file.
"""

import os
import sys
from datetime import datetime

from reportlab.lib.pagesizes  import letter
from reportlab.lib.units       import inch
from reportlab.lib             import colors
from reportlab.lib.styles      import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums       import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus        import (SimpleDocTemplate, Paragraph, Spacer,
                                        Table, TableStyle, HRFlowable,
                                        KeepTogether)

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
from config import REPORT_DIR, LABEL_COLORS
from src.utils  import safe_filename, timestamp

# ── Colour palette (matches frontend theme) ───────────────────────────────────
C_PRIMARY   = colors.HexColor("#2f4561")
C_SECONDARY = colors.HexColor("#3a675b")
C_TERTIARY  = colors.HexColor("#46405e")
C_ERROR     = colors.HexColor("#ba1a1a")
C_SURFACE   = colors.HexColor("#fbf9f6")
C_OUTLINE   = colors.HexColor("#e4e2df")
C_MUTED     = colors.HexColor("#74777e")
C_WHITE     = colors.white

LABEL_RL_COLORS = {
    "Real": C_SECONDARY,
    "Fake": C_ERROR,
    "Bot":  C_TERTIARY,
    "Inconclusive": C_MUTED,
}


# ─────────────────────────────────────────────────────────────────────────────
# Style sheet
# ─────────────────────────────────────────────────────────────────────────────
def _build_styles():
    base = getSampleStyleSheet()

    styles = {
        "title": ParagraphStyle(
            "title", parent=base["Normal"],
            fontSize=22, fontName="Helvetica-Bold",
            textColor=C_PRIMARY, alignment=TA_CENTER,
            spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "subtitle", parent=base["Normal"],
            fontSize=11, fontName="Helvetica",
            textColor=C_MUTED, alignment=TA_CENTER,
            spaceAfter=2,
        ),
        "section": ParagraphStyle(
            "section", parent=base["Normal"],
            fontSize=13, fontName="Helvetica-Bold",
            textColor=C_PRIMARY, spaceBefore=14, spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "body", parent=base["Normal"],
            fontSize=10, fontName="Helvetica",
            textColor=colors.HexColor("#1b1c1a"),
            spaceAfter=4, leading=14,
        ),
        "body_muted": ParagraphStyle(
            "body_muted", parent=base["Normal"],
            fontSize=9, fontName="Helvetica",
            textColor=C_MUTED, spaceAfter=3, leading=13,
        ),
        "indicator": ParagraphStyle(
            "indicator", parent=base["Normal"],
            fontSize=9.5, fontName="Helvetica",
            textColor=colors.HexColor("#1b1c1a"),
            spaceAfter=3, leading=14,
            leftIndent=10,
        ),
        "footer": ParagraphStyle(
            "footer", parent=base["Normal"],
            fontSize=8, fontName="Helvetica",
            textColor=C_MUTED, alignment=TA_CENTER,
        ),
        "label_genuine": ParagraphStyle(
            "label_genuine", parent=base["Normal"],
            fontSize=26, fontName="Helvetica-Bold",
            textColor=C_SECONDARY, alignment=TA_CENTER,
        ),
        "label_fake": ParagraphStyle(
            "label_fake", parent=base["Normal"],
            fontSize=26, fontName="Helvetica-Bold",
            textColor=C_ERROR, alignment=TA_CENTER,
        ),
        "label_inconclusive": ParagraphStyle(
            "label_inconclusive", parent=base["Normal"],
            fontSize=22, fontName="Helvetica-Bold",
            textColor=C_MUTED, alignment=TA_CENTER,
        ),
        "label_bot": ParagraphStyle(
            "label_bot", parent=base["Normal"],
            fontSize=26, fontName="Helvetica-Bold",
            textColor=C_TERTIARY, alignment=TA_CENTER,
        ),
    }
    return styles


# ─────────────────────────────────────────────────────────────────────────────
# Section builders
# ─────────────────────────────────────────────────────────────────────────────

def _header(styles, result: dict) -> list:
    """Top banner: project name, report title, date."""
    elements = []
    elements.append(Paragraph("Social Guard", styles["title"]))
    elements.append(Paragraph(
        "AI-Based Social Media Account Analysis Report", styles["subtitle"]))
    elements.append(Paragraph(
        f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M:%S')}  |  "
        f"Model: {result.get('model_used', 'Unknown')}",
        styles["subtitle"]))
    elements.append(Spacer(1, 8))
    elements.append(HRFlowable(width="100%", thickness=1.5,
                               color=C_PRIMARY, spaceAfter=10))
    return elements


def _verdict_box(styles, result: dict) -> list:
    """Large verdict + confidence badge."""
    label      = result["label"]
    confidence = result["confidence"]
    username   = result.get("username", "—")
    platform   = result.get("platform", "—")

    label_style_key = {
        "Real": "label_genuine",
        "Fake": "label_fake",
        "Bot":  "label_bot",
        "Inconclusive": "label_inconclusive",
    }.get(label, "label_genuine")

    bg_color = LABEL_RL_COLORS.get(label, C_PRIMARY)

    # Verdict table: label on left, confidence on right
    data = [[
        Paragraph(label.upper(), styles[label_style_key]),
        Paragraph(
            ("<b>Verified</b><br/><font size=8 color='#74777e'>Confirmed by the platform; model score for Real "
             f"{confidence:.1f}%</font>") if result.get("verified_override") else
            f"<b>{confidence:.1f}%</b><br/><font size=8 color='#74777e'>Model Confidence Score</font>",
            ParagraphStyle("conf", parent=styles["body"],
                           alignment=TA_CENTER, fontSize=20,
                           fontName="Helvetica-Bold",
                           textColor=bg_color)),
    ]]
    t = Table(data, colWidths=[3.5 * inch, 3.5 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, -1), colors.HexColor("#f5f3f0")),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.HexColor("#f5f3f0")]),
        ("BOX",         (0, 0), (-1, -1), 1, C_OUTLINE),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",  (0, 0), (-1, -1), 16),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 16),
        ("LEFTPADDING", (0, 0), (-1, -1), 20),
        ("RIGHTPADDING", (0, 0), (-1, -1), 20),
    ]))

    elements = [t, Spacer(1, 6)]

    # Summary sentence
    summary = result.get("summary", "")
    if summary:
        elements.append(Paragraph(summary, styles["body"]))
        elements.append(Spacer(1, 4))

    elements.append(Paragraph(
        "This score is the trained model's own output probability for its "
        "top predicted class, not a calibrated real-world likelihood. "
        "Treat it as a relative strength-of-signal indicator rather than "
        "a statistical certainty.",
        ParagraphStyle("conf_caveat", parent=styles["body"],
                       fontSize=7.5, textColor=colors.HexColor("#74777e"))))
    elements.append(Spacer(1, 4))

    return elements


def _profile_table(styles, display_features: dict) -> list:
    """Feature breakdown table."""
    elements = [Paragraph("Collected Profile Metrics", styles["section"])]

    rows = [["Metric", "Value"]]
    for k, v in display_features.items():
        rows.append([k, str(v)])

    col_w = [3.5 * inch, 3.5 * inch]
    t = Table(rows, colWidths=col_w, repeatRows=1)
    t.setStyle(TableStyle([
        # Header row
        ("BACKGROUND",    (0, 0), (-1, 0), C_PRIMARY),
        ("TEXTCOLOR",     (0, 0), (-1, 0), C_WHITE),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0), 10),
        ("TOPPADDING",    (0, 0), (-1, 0), 8),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
        # Data rows alternating
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#ffffff"), colors.HexColor("#f5f3f0")]),
        ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",      (0, 1), (-1, -1), 9.5),
        ("FONTNAME",      (0, 1), (0, -1), "Helvetica-Bold"),   # bold label col
        ("TEXTCOLOR",     (0, 1), (0, -1), C_PRIMARY),
        ("TOPPADDING",    (0, 1), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ("GRID",          (0, 0), (-1, -1), 0.5, C_OUTLINE),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 8))
    return elements


def _probability_table(styles, probabilities: dict) -> list:
    """Probability breakdown for all three classes."""
    elements = [Paragraph("Classification Probabilities", styles["section"])]

    rows = [["Class", "Model Output Score", "Interpretation"]]
    interpretations = {
        "Real": "Genuine human-operated account",
        "Fake": "Human-operated inauthentic account",
        "Bot":  "Automated / scripted account",
    }
    for cls, pct in sorted(probabilities.items(),
                            key=lambda x: x[1], reverse=True):
        rows.append([cls, f"{pct:.2f}%", interpretations.get(cls, "")])

    col_w = [1.5 * inch, 1.5 * inch, 4.0 * inch]
    t = Table(rows, colWidths=col_w, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), C_SECONDARY),
        ("TEXTCOLOR",     (0, 0), (-1, 0), C_WHITE),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0), 10),
        ("TOPPADDING",    (0, 0), (-1, 0), 8),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#ffffff"), colors.HexColor("#f5f3f0")]),
        ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",      (0, 1), (-1, -1), 9.5),
        ("TOPPADDING",    (0, 1), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ("GRID",          (0, 0), (-1, -1), 0.5, C_OUTLINE),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 8))
    return elements


def _indicators_section(styles, indicators: list) -> list:
    """Behavioral indicators with severity markers."""
    elements = [Paragraph("Behavioral Indicators", styles["section"])]

    severity_prefix = {
        "critical": "[CRITICAL]",
        "warning":  "[WARNING] ",
        "positive": "[OK]     ",
    }
    severity_colors = {
        "critical": C_ERROR,
        "warning":  colors.HexColor("#b45309"),
        "positive": C_SECONDARY,
    }

    for ind in indicators:
        sev    = ind.get("severity", "positive")
        prefix = severity_prefix.get(sev, "[ ]")
        color  = severity_colors.get(sev, C_PRIMARY)

        # Build colored prefix + body text
        para = Paragraph(
            f'<font name="Helvetica-Bold" color="{color.hexval()}">'
            f'{prefix}</font>  {ind["text"]}',
            styles["indicator"]
        )
        elements.append(para)
        elements.append(Spacer(1, 3))

    elements.append(Spacer(1, 6))
    return elements


def _xai_section(styles, xai: dict) -> list:
    """
    Real, per-account explanation of WHY this verdict was reached, built
    from this account's own SHAP values and a local LIME fit (src/xai.py).
    Nothing here is static boilerplate — it's computed fresh for every
    account from its own feature values and the actual trained model.
    If SHAP/LIME couldn't run (package missing, or something failed),
    this says so honestly instead of pretending an explanation exists.
    """
    elements = [Paragraph("Why This Verdict — Model Explanation", styles["section"])]

    if not xai or not xai.get("shap_available"):
        msg = (xai or {}).get("error") or "Explanation data is not available for this result."
        elements.append(Paragraph(msg, styles["body_muted"]))
        elements.append(Spacer(1, 8))
        return elements

    # Dynamic narrative sentences (per-account, not canned text)
    for sentence in xai.get("narrative", []):
        elements.append(Paragraph(f"&bull;&nbsp; {sentence}", styles["body"]))
    elements.append(Spacer(1, 6))

    # SHAP table — global, additive feature attributions for this prediction
    shap_rows = xai.get("shap", [])
    if shap_rows:
        elements.append(Paragraph(
            "SHAP Feature Contributions (signed — positive pushes toward the "
            "predicted class, negative pushes away from it):",
            styles["body_muted"]))
        rows = [["Feature", "This Account's Value", "SHAP Contribution"]]
        for r in shap_rows[:8]:
            rows.append([r["display_name"], r["raw_value"], f"{r['value']:+.4f}"])
        t = Table(rows, colWidths=[2.3 * inch, 2.3 * inch, 2.4 * inch], repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), C_TERTIARY),
            ("TEXTCOLOR",     (0, 0), (-1, 0), C_WHITE),
            ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, 0), 9.5),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.HexColor("#ffffff"), colors.HexColor("#f5f3f0")]),
            ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE",      (0, 1), (-1, -1), 9),
            ("TOPPADDING",    (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING",   (0, 0), (-1, -1), 8),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
            ("GRID",          (0, 0), (-1, -1), 0.5, C_OUTLINE),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 8))

    # LIME table — local linear approximation around this instance
    lime_rows = xai.get("lime", [])
    if lime_rows and xai.get("lime_available"):
        r2 = xai.get("lime_r2")
        r2_txt = f" (local fit R&sup2; = {r2:.2f})" if r2 is not None else ""
        elements.append(Paragraph(
            f"LIME Local Approximation{r2_txt} — a simple linear model fit "
            "around this specific account's values:",
            styles["body_muted"]))
        rows = [["Feature", "This Account's Value", "LIME Weight"]]
        for r in lime_rows[:8]:
            rows.append([r["display_name"], r["raw_value"], f"{r['value']:+.4f}"])
        t = Table(rows, colWidths=[2.3 * inch, 2.3 * inch, 2.4 * inch], repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), C_SECONDARY),
            ("TEXTCOLOR",     (0, 0), (-1, 0), C_WHITE),
            ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, 0), 9.5),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.HexColor("#ffffff"), colors.HexColor("#f5f3f0")]),
            ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE",      (0, 1), (-1, -1), 9),
            ("TOPPADDING",    (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING",   (0, 0), (-1, -1), 8),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
            ("GRID",          (0, 0), (-1, -1), 0.5, C_OUTLINE),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 4))

    source_note = {
        "trained_background": "Background reference: a saved sample of the model's own training data.",
        "dataset_fallback":   "Background reference: sampled directly from the processed dataset (no saved "
                               "training background found).",
    }.get(xai.get("background_source"))
    if source_note:
        elements.append(Paragraph(source_note, styles["body_muted"]))
    if xai.get("error"):
        elements.append(Paragraph(f"Note: {xai['error']}", styles["body_muted"]))

    elements.append(Spacer(1, 8))
    return elements


def _limitations_section(styles) -> list:
    """
    Known-limitations note. Included so the report never implies more
    certainty or generality than the underlying model actually supports —
    these are genuine, acknowledged constraints of the current system, not
    boilerplate legal language.
    """
    elements = [Paragraph("Known Limitations", styles["section"])]
    points = [
        "The model uses 10 profile-page features: per-platform percentile ranks "
        "of the follower, following and post counts, two ratios (followers per "
        "following, posts per follower), bio length, digit share of the username, "
        "whether name equals username, profile picture and private flag. A field "
        "that could not be read is left blank, not guessed. It does not analyze "
        "posting history over time, post content, the follower network or "
        "engagement authenticity, so accounts that look statistically typical "
        "but behave suspiciously in ways outside these features may be "
        "misclassified.",

        "The Real / Fake / Bot categories are not necessarily mutually "
        "exclusive in the real world (for example, a human-run account can "
        "use automation tools). The classification reflects the category "
        "definitions used to label this system's training data: Real = verified "
        "genuine accounts, Fake = purchased or inactive fake profiles, Bot = "
        "automated or spam accounts.",

        "One model is trained on data from Instagram, Twitter/X and Facebook "
        "together; the platform is not a model input. When a platform was left "
        "out of training and tested, accuracy was much lower than on the random "
        "test split, so scores on a new platform or a new kind of account should "
        "be read with extra caution. LinkedIn has no training data and is judged "
        "against pooled statistics.",

        "When the model's top probability is low, or the platform was not in the "
        "training data, the result is still given but carries a caution note and "
        "should be reviewed before any decision is made.",

        "This tool is a proof-of-concept academic system. It has not been "
        "validated for use in making consequential decisions about real "
        "individuals and should not be treated as a sole or final basis "
        "for such decisions.",
    ]
    for p in points:
        elements.append(Paragraph(f"&bull;&nbsp; {p}", styles["body"]))
        elements.append(Spacer(1, 3))
    elements.append(Spacer(1, 6))
    return elements


def _disclaimer(styles) -> list:
    """Footer disclaimer."""
    text = (
        "DISCLAIMER: This report was generated by Social Guard, an academic "
        "AI-based analysis tool. Results are based on publicly observable "
        "profile metrics and machine learning inference. This report does not "
        "constitute legal evidence. Profile data used in this analysis was "
        "collected from publicly available sources only, in accordance with "
        "applicable platform terms of service."
    )
    return [
        HRFlowable(width="100%", thickness=0.5, color=C_OUTLINE, spaceAfter=6),
        Paragraph(text, styles["footer"]),
        Paragraph(
            "Social Guard  |  Somaiya Vidyavihar University  |  "
            "Department of Information Technology",
            styles["footer"]),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(result: dict, display_features: dict) -> str:
    """
    Build and save the PDF report.

    Parameters
    ----------
    result : dict
        Full prediction result from predict.py, plus:
            username, platform, summary
    display_features : dict
        Human-friendly feature table from feature_engineering.get_display_features()

    Returns
    -------
    str : absolute path to the saved PDF file
    """
    os.makedirs(REPORT_DIR, exist_ok=True)

    filename = safe_filename(
        result.get("username", "unknown"),
        result.get("platform", "unknown")
    )
    filepath = os.path.join(REPORT_DIR, filename)

    doc = SimpleDocTemplate(
        filepath,
        pagesize=letter,
        leftMargin=0.85 * inch,
        rightMargin=0.85 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        title="Social Guard Analysis Report",
        author="Social Guard",
    )

    styles   = _build_styles()
    story    = []

    story += _header(styles, result)
    story += _verdict_box(styles, result)
    story.append(Spacer(1, 6))
    ds = result.get("data_source")
    source_label = (f"Read live from {result.get('platform', 'the platform')}" if ds == "scraped"
                    else "Entered or confirmed by the user" if ds == "manual" else "Unknown")
    story += _profile_table(styles, {**display_features, "Data Source": source_label})
    story += _probability_table(styles, result.get("probabilities", {}))
    story += _indicators_section(styles, result.get("indicators", []))
    story += _xai_section(styles, result.get("xai"))
    story += _limitations_section(styles)
    story += _disclaimer(styles)

    doc.build(story)
    return filepath