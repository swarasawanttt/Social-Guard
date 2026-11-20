"""
src/explain.py
──────────────
Generates human-readable explanation indicators shown on the results
page and included in the PDF report.

Each indicator is a plain-English sentence describing one behavioral
signal that contributed to the classification decision.
"""

import os
import sys

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
from config import LABEL_MAP


# ─────────────────────────────────────────────────────────────────────────────
def generate_indicators(features: dict, label: str,
                         probabilities: dict,
                         model_name: str = "the trained model",
                         inconclusive: bool = False,
                         unavailable: list = None,
                         inconclusive_reason: str = None) -> list[dict]:
    """
    Analyse feature values and return a list of indicator dicts.

    Each dict has:
        text     : str   — the explanation sentence
        severity : str   — "positive" | "warning" | "critical"
        icon     : str   — material symbol name for the frontend

    Parameters
    ----------
    features      : dict   output of feature_engineering.engineer_features()
    label         : str    the model's top class: "Real" | "Fake" | "Bot"
    probabilities : dict   {label_name: percentage_float}
    inconclusive  : bool   True when the top-class probability is below the
                           verdict threshold (config.INCONCLUSIVE_THRESHOLD)
    unavailable   : list   feature names that could not be read for this account
    """
    import math
    indicators = []
    unavailable = unavailable or []
    udr  = features.get("username_digit_ratio", float("nan"))
    neq  = features.get("name_eq_username", float("nan"))
    pic  = features.get("has_profile_pic", float("nan"))

    ff    = features["follower_following_ratio"]
    act   = features["activity_rate"]
    bio   = features["bio_len"]
    posts = features["posts"]
    fol   = features["followers"]
    fing  = features["following"]

    confidence = max(probabilities.values())

    # ── Follower / Following Ratio ────────────────────────────────────────────
    if ff < 0.05:
        indicators.append({
            "text": f"Critically low follower-to-following ratio ({ff:.3f}). "
                    "The account follows a large number of profiles while having "
                    "very few followers — a hallmark of bot and fake accounts.",
            "severity": "critical",
            "icon": "group_off",
        })
    elif ff < 0.3:
        indicators.append({
            "text": f"Low follower-to-following ratio ({ff:.3f}). "
                    "The account follows significantly more profiles than it "
                    "attracts, which is a common pattern in inauthentic accounts.",
            "severity": "warning",
            "icon": "person_off",
        })
    elif ff > 10:
        indicators.append({
            "text": f"Strong follower-to-following ratio ({ff:.2f}). "
                    "The account has a large follower base relative to who it "
                    "follows, consistent with an established authentic profile.",
            "severity": "positive",
            "icon": "verified_user",
        })
    else:
        indicators.append({
            "text": f"Moderate follower-to-following ratio ({ff:.3f}) — "
                    "within the range seen in typical genuine accounts.",
            "severity": "positive",
            "icon": "people",
        })

    # ── Activity Rate ─────────────────────────────────────────────────────────
    if act > 100:
        indicators.append({
            "text": f"Extremely high activity rate ({act:.1f} posts per follower). "
                    "The posting volume is disproportionate to the follower count — "
                    "a strong indicator of automated (bot) behaviour.",
            "severity": "critical",
            "icon": "warning",
        })
    elif act > 10:
        indicators.append({
            "text": f"Elevated activity rate ({act:.2f}). "
                    "Post volume is high relative to audience size, which may "
                    "suggest coordinated or automated posting behaviour.",
            "severity": "warning",
            "icon": "pending",
        })
    elif act < 0.001 and posts == 0:
        indicators.append({
            "text": "No posts detected. Accounts with zero posts are frequently "
                    "newly created fake profiles or dormant bot accounts.",
            "severity": "warning",
            "icon": "block",
        })
    else:
        indicators.append({
            "text": f"Normal posting activity ({act:.4f} activity rate). "
                    "Post frequency is proportionate to audience size.",
            "severity": "positive",
            "icon": "check_circle",
        })

    # ── Bio ───────────────────────────────────────────────────────────────────
    if bio == 0:
        indicators.append({
            "text": "No profile bio detected. Genuine accounts typically have "
                    "a bio; its absence is a consistent signal in fake and bot "
                    "account datasets.",
            "severity": "warning",
            "icon": "info",
        })
    elif bio < 15:
        indicators.append({
            "text": f"Very short bio ({bio} characters). "
                    "Minimal profile descriptions are more common in inauthentic "
                    "accounts than in genuine ones.",
            "severity": "warning",
            "icon": "short_text",
        })
    else:
        indicators.append({
            "text": f"Profile bio present ({bio} characters). "
                    "A descriptive bio is associated with genuine account "
                    "ownership and active community presence.",
            "severity": "positive",
            "icon": "description",
        })

    # ── Follower count ────────────────────────────────────────────────────────
    if fol == 0:
        indicators.append({
            "text": "Zero followers recorded. The account has not attracted any "
                    "organic followers, which is highly unusual for a legitimate "
                    "active account.",
            "severity": "critical",
            "icon": "person_remove",
        })
    elif fol < 20 and fing > 1000:
        indicators.append({
            "text": f"Very few followers ({int(fol)}) but following many "
                    f"({int(fing)}). This pattern is a classic indicator of "
                    "mass-follow bot behaviour.",
            "severity": "critical",
            "icon": "manage_accounts",
        })

    # ── Username / identity signals (only when the field was actually read) ──
    if isinstance(udr, float) and not math.isnan(udr) and udr >= 0.4:
        indicators.append({
            "text": f"{udr*100:.0f}% of the username is digits. Heavily numeric "
                    "handles are far more common in auto-generated accounts.",
            "severity": "warning",
            "icon": "pin",
        })
    if isinstance(pic, float) and not math.isnan(pic) and pic < 0.5:
        indicators.append({
            "text": "No custom profile picture. Default avatars are more common "
                    "in fake and throwaway accounts.",
            "severity": "warning",
            "icon": "no_photography",
        })

    # ── Model confidence ──────────────────────────────────────────────────────
    if inconclusive:
        indicators.insert(0, {
            "text": (inconclusive_reason + " Treat the result as 'needs manual review'.")
                    if inconclusive_reason else
                    f"Inconclusive: the model's best guess ({label}, {confidence:.1f}%) "
                    "is below the confidence needed to call this account. Real "
                    "accounts with sparse or unusual statistics can look like this, "
                    "so treat the result as 'needs manual review', not as a finding.",
            "severity": "warning",
            "icon": "help",
        })
    elif confidence >= 90:
        agree = (" The ensemble's constituent models agree on this result."
                 if "voting" in model_name.lower() else "")
        indicators.append({
            "text": f"High model confidence ({confidence:.1f}%) from {model_name}. "
                    "This is the model's own score, not proof about the account." + agree,
            "severity": "positive" if label == "Real" else "critical",
            "icon": "analytics",
        })
    elif confidence < 80:
        indicators.append({
            "text": f"Moderate model confidence ({confidence:.1f}%). The account "
                    "shows mixed signals. Manual review of the profile is "
                    "recommended before drawing conclusions.",
            "severity": "warning",
            "icon": "help",
        })

    if len(unavailable) >= 4:
        indicators.append({
            "text": f"{len(unavailable)} profile fields could not be read for this "
                    "account (for example profile picture, link or full name). The "
                    "model still gives a result, but with less information to go on.",
            "severity": "warning",
            "icon": "info",
        })

    # Ensure we always return at least one indicator
    if not indicators:
        indicators.append({
            "text": "Standard profile patterns detected. No strong anomalies "
                    "found in the collected metrics.",
            "severity": "positive",
            "icon": "check",
        })

    return indicators


# ─────────────────────────────────────────────────────────────────────────────
def get_summary_sentence(label: str, confidence: float,
                          username: str, platform: str,
                          top_class: str = None) -> str:
    """
    One-sentence executive summary for the top of the results page and
    the PDF report.
    """
    templates = {
        "Inconclusive": (
            f"The model could not reach a confident verdict on @{username} on "
            f"{platform}. Its closest match was {top_class or 'unknown'} "
            f"({confidence:.1f}%), but that is not reliable enough to call the "
            f"account. Please review the profile manually."
        ),
        "Real": (
            f"The account @{username} on {platform} shows behavioral patterns "
            f"consistent with a genuine human-operated profile "
            f"(confidence: {confidence:.1f}%)."
        ),
        "Fake": (
            f"The account @{username} on {platform} displays multiple indicators "
            f"of inauthenticity. Our model classifies it as a Fake account "
            f"(confidence: {confidence:.1f}%)."
        ),
        "Bot": (
            f"The account @{username} on {platform} exhibits automated behavioural "
            f"patterns strongly associated with bot accounts "
            f"(confidence: {confidence:.1f}%)."
        ),
    }
    return templates.get(label, f"Classification: {label} ({confidence:.1f}% confidence).")