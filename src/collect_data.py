"""
src/collect_data.py
───────────────────
Builds NEW labelled training rows from real public profiles, using the same
scraper the app uses, and writes them in the schema of data/processed/unified_raw.csv.

WHY: the model's weak spot is data coverage (few real mid-sized / large Instagram
accounts). This script lets you add them.

FASTEST WAY (no list needed):
     python src/collect_data.py auto Instagram 40
  finds 40 public accounts by itself (see 'discover' below), scrapes them, and
  saves what it could read. Run it again later, or for Twitter/X and Facebook.

  python src/collect_data.py discover Instagram 40   -> only builds the seeds list
     Source: Wikidata (public, free). It lists notable accounts (people, organisations) whose Instagram /
     Twitter / Facebook account is recorded there. They are labelled Real because
     a notable person's own account, recorded by Wikidata editors, is independent
     evidence of a genuine account (not this model's opinion). Limits: this gives
     Real only, skews to well-known people, and a recorded handle can be stale
     (those simply fail and go to needs_manual.csv).

HOW IT WORKS
  1. Or prepare a seeds file yourself:  platform,handle,label[,source]
        Instagram,someone_public,Real
        Twitter/X,another_public,Real
     The label is YOUR judgement, made independently of the model
     (never label an account with this project's own prediction: that is circular).
  2. python src/collect_data.py scrape  data/collect/seeds.csv
        - scrapes each handle slowly and politely,
        - writes rows that were read well to  data/collect/collected_raw.csv,
        - writes everything else (blocked, private, not found, partial) to
          data/collect/needs_manual.csv together with the reason,
        - keeps a log and RESUMES: re-running skips handles already done.
  3. For handles the scraper could not read, open the profile yourself, type the
     numbers into a manual file (run: python src/collect_data.py template) and run
        python src/collect_data.py manual data/collect/manual.csv
  4. python src/collect_data.py report            (coverage by follower size)
  5. python src/collect_data.py merge             (appends to unified_raw.csv,
        keeps a timestamped backup, drops exact duplicates)
  6. python src/train_model.py                    (retrain)

Only counts and simple flags are stored in the training rows, never handles,
names or bios. Collect only PUBLIC accounts (public figures, organisations, or
people who agreed). Respect each platform's terms; keep the delay on.
"""
import csv
import os
import random
import re
import shutil
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

COLLECT_DIR = os.path.join(ROOT, "data", "collect")
COLLECTED = os.path.join(COLLECT_DIR, "collected_raw.csv")
NEEDS_MANUAL = os.path.join(COLLECT_DIR, "needs_manual.csv")
LOG = os.path.join(COLLECT_DIR, "collect_log.csv")
UNIFIED = os.path.join(ROOT, "data", "processed", "unified_raw.csv")

# Same columns, same order as build_unified_dataset.py
SCHEMA = [
    "platform", "source", "label",
    "followers", "following", "posts",
    "bio_len", "has_url",
    "username_len", "username_digit_ratio",
    "fullname_words", "name_eq_username",
    "has_profile_pic", "verified", "is_private",
    "account_age_days",
]
TRAIN_PLATFORM = {"instagram": "Instagram", "ig": "Instagram",
                  "twitter": "Twitter", "twitter/x": "Twitter", "x": "Twitter",
                  "facebook": "Facebook", "fb": "Facebook"}
LABELS = {"real": "Real", "genuine": "Real", "fake": "Fake", "bot": "Bot"}
NAN = float("nan")
SIZE_BINS = [(0, 1_000, "<1K"), (1_000, 10_000, "1K-10K"), (10_000, 100_000, "10K-100K"),
             (100_000, 1_000_000, "100K-1M"), (1_000_000, float("inf"), ">1M")]


# ───────────────────────────── row building ────────────────────────────────
def _norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _flag(v):
    """True/False/'yes'/'no'/1/0 -> 1.0/0.0 ; anything else (None, '', '?') -> NaN."""
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if v is None:
        return NAN
    s = str(v).strip().lower()
    if s in ("1", "yes", "y", "true", "t"):
        return 1.0
    if s in ("0", "no", "n", "false", "f"):
        return 0.0
    return NAN


def _num(v):
    try:
        if v is None or str(v).strip() == "":
            return NAN
        return float(str(v).replace(",", "").strip())
    except ValueError:
        return NAN


def row_from_profile(platform, label, source, username, d):
    """
    Turn one profile into one training row.
    d: dict with followers, following, posts, bio_len and optionally
       full_name, external_url (str, '' = none, None = unknown), has_url,
       is_verified, is_private, has_profile_pic.
    Anything that is not known stays NaN, never 0.
    """
    ext = d.get("external_url")
    if isinstance(ext, str):
        has_url = 1.0 if ext.strip() else 0.0
    else:
        has_url = _flag(d.get("has_url"))

    full = d.get("full_name")
    if full is None or str(full).strip() == "":
        fullname_words, name_eq = NAN, NAN          # unknown, never 0
    else:
        full = str(full)
        fullname_words = float(len(full.split()))
        name_eq = 1.0 if (_norm(full) and _norm(full) == _norm(username)) else 0.0

    uname = str(username)
    return {
        "platform": platform, "source": source, "label": label,
        "followers": _num(d.get("followers")), "following": _num(d.get("following")),
        "posts": _num(d.get("posts")), "bio_len": _num(d.get("bio_len")),
        "has_url": has_url,
        "username_len": float(len(uname)),
        "username_digit_ratio": (sum(c.isdigit() for c in uname) / len(uname)) if uname else NAN,
        "fullname_words": fullname_words, "name_eq_username": name_eq,
        "has_profile_pic": _flag(d.get("has_profile_pic")),
        "verified": _flag(d.get("is_verified")),
        "is_private": _flag(d.get("is_private")),
        "account_age_days": NAN,
    }


def _clean_seed(raw):
    plat = TRAIN_PLATFORM.get(str(raw.get("platform", "")).strip().lower())
    label = LABELS.get(str(raw.get("label", "")).strip().lower())
    handle = str(raw.get("handle", "")).strip().lstrip("@")
    if not (plat and label and handle):
        return None
    batch = str(raw.get("source") or "").strip() or datetime.now().strftime("%Y%m%d")
    return plat, label, handle, "collected_" + re.sub(r"[^A-Za-z0-9_]+", "_", batch)


# ───────────────────────────── file helpers ────────────────────────────────
def _append(path, rows, fields):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerows(rows)


def _done_keys(retry_failed=False):
    """Handles not to scrape again. Failed ones are skipped too (retrying a blocked
    profile only adds to the blocking) unless retry_failed is set."""
    if not os.path.exists(LOG):
        return set()
    df = pd.read_csv(LOG, dtype=str).fillna("")
    good = ("ok", "manual_ok") if retry_failed else ("ok", "manual_ok", "failed")
    return {(r.platform.lower(), r.handle.lower()) for r in df.itertuples() if r.status in good}



# ───────────────────────────── discovery ───────────────────────────────────
WIKIDATA_URL = "https://query.wikidata.org/sparql"
WD_PROPERTY = {"Instagram": "P2003", "Twitter": "P2002", "Facebook": "P2013"}
HANDLE_OK = {"Instagram": re.compile(r"^[A-Za-z0-9._]{2,30}$"),
             "Twitter": re.compile(r"^[A-Za-z0-9_]{1,15}$"),
             "Facebook": re.compile(r"^[A-Za-z0-9.]{5,50}$")}


def _wikidata_get(query):
    """One SPARQL request. Standard library only (no extra install)."""
    import json
    import urllib.parse
    import urllib.request
    url = WIKIDATA_URL + "?" + urllib.parse.urlencode({"query": query, "format": "json"})
    req = urllib.request.Request(url, headers={
        "User-Agent": "SocialGuardStudentProject/1.0 (academic research; public data only; contact: "
                      + os.environ.get("COLLECT_CONTACT", "see project") + ")",
        "Accept": "application/sparql-results+json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def discover_handles(platform, n, _get=None):
    """
    Random sample of n handles whose account on `platform` is recorded on Wikidata
    (people, organisations, brands: public, notable accounts).
    The query is deliberately the cheapest possible (one property, LIMIT + a random
    OFFSET); heavier queries are refused by Wikimedia with HTTP 403.
    """
    get = _get or _wikidata_get
    prop = WD_PROPERTY[platform]
    found, seen = [], set()
    top = 20000
    for attempt in range(6):
        off = random.randrange(0, top, 50)
        q = f"SELECT ?handle WHERE {{ ?item wdt:{prop} ?handle . }} LIMIT {max(n * 2, 50)} OFFSET {off}"
        rows = get(q)["results"]["bindings"]
        if not rows:
            top = max(100, top // 4)             # offset past the end: look nearer the start
            continue
        for r in rows:
            h = r["handle"]["value"].strip().lstrip("@")
            if h.lower() in seen or not HANDLE_OK[platform].match(h):
                continue
            seen.add(h.lower())
            found.append(h)
        if len(found) >= n:
            break
        if not _get:
            time.sleep(3)                         # be gentle with the public server
    random.shuffle(found)
    return found[:n]


def cmd_discover(platform_arg, n, _get=None):
    plat = TRAIN_PLATFORM.get(platform_arg.strip().lower())
    if plat not in WD_PROPERTY:
        sys.exit("platform must be Instagram, Twitter/X or Facebook")
    try:
        handles = discover_handles(plat, n, _get)
    except Exception as e:                            # network down, 403, timeout ...
        sys.exit(f"Could not get accounts from Wikidata: {e}\n"
                 "Wait a few minutes and retry with a smaller COUNT. If it keeps failing, "
                 "use the seeds/manual route instead.")
    seeds = os.path.join(COLLECT_DIR, "seeds.csv")
    have = set()
    if os.path.exists(seeds):
        d = pd.read_csv(seeds, dtype=str).fillna("")
        have = {(a.lower(), b.lower()) for a, b in zip(d["platform"], d["handle"])}
    site = "Twitter/X" if plat == "Twitter" else plat
    src = f"wd_{plat.lower()}_{datetime.now():%Y%m%d}"
    rows = [dict(platform=site, handle=h, label="Real", source=src)
            for h in handles if (site.lower(), h.lower()) not in have]
    _append(seeds, rows, ["platform", "handle", "label", "source"])
    print(f"Found {len(handles)} {site} handles, added {len(rows)} new ones to {seeds} (label Real).")
    return seeds


def cmd_auto(platform_arg, n, fetch=None, _get=None, delay=(5.0, 10.0)):
    seeds = cmd_discover(platform_arg, n, _get)
    cmd_scrape(seeds, delay=delay, limit=None, fetch=fetch,
               only_platform=TRAIN_PLATFORM[platform_arg.strip().lower()])

# ─────────────────────────────── commands ──────────────────────────────────
def cmd_scrape(seeds_path, delay=(5.0, 10.0), limit=None, fetch=None, retry_failed=False, only_platform=None):
    if fetch is None:
        from src.scraper import fetch_profile_data as fetch
    seeds = pd.read_csv(seeds_path, dtype=str).fillna("").to_dict("records")
    done = _done_keys(retry_failed)
    todo = []
    for raw in seeds:
        s = _clean_seed(raw)
        if s is None:
            print(f"  skip (bad row, need platform/handle/label): {raw}")
            continue
        if only_platform and s[0] != only_platform:
            continue                       # e.g. `auto Instagram` must not touch Twitter rows
        if (s[0].lower(), s[2].lower()) in done:
            continue
        todo.append(s)
    if limit:
        todo = todo[:limit]
    print(f"{len(todo)} profiles to scrape ({len(seeds) - len(todo)} skipped or already done)")

    ok = fail = 0
    for i, (plat, label, handle, source) in enumerate(todo, 1):
        site = "Twitter/X" if plat == "Twitter" else plat
        try:
            # A handle with a dot (patrick.harvie) can look like a website name to the
            # input checker, so such handles are passed as a full profile link instead.
            ident = handle
            if "." in handle:
                ident = {"Instagram": "https://www.instagram.com/%s/",
                         "Twitter": "https://x.com/%s",
                         "Facebook": "https://www.facebook.com/%s"}[plat] % handle
            res = fetch(ident, site)
        except Exception as e:                       # never let one profile stop the run
            res = {"success": False, "error_code": "exception", "error": str(e)}
        good = (res.get("success")
                and res.get("data", {}).get("data_confidence") in ("high", "medium"))
        stamp = datetime.now().isoformat(timespec="seconds")
        if good:
            _append(COLLECTED, [row_from_profile(plat, label, source, handle, res["data"])], SCHEMA)
            _append(LOG, [dict(time=stamp, platform=plat, handle=handle, label=label,
                               status="ok", detail="")], ["time", "platform", "handle", "label", "status", "detail"])
            ok += 1
            tag = "ok"
        else:
            code = res.get("error_code") or "low_confidence"
            _append(NEEDS_MANUAL, [dict(platform=plat, handle=handle, label=label, source=source,
                                        reason=code, followers="", following="", posts="", bio_len="",
                                        full_name="", external_url="", has_profile_pic="",
                                        is_verified="", is_private="")],
                    ["platform", "handle", "label", "source", "reason", "followers", "following",
                     "posts", "bio_len", "full_name", "external_url", "has_profile_pic",
                     "is_verified", "is_private"])
            _append(LOG, [dict(time=stamp, platform=plat, handle=handle, label=label,
                               status="failed", detail=code)],
                    ["time", "platform", "handle", "label", "status", "detail"])
            fail += 1
            tag = f"FAILED ({code})"
        print(f"  [{i}/{len(todo)}] {plat:9s} {handle:25s} {label:4s} {tag}")
        # Back off if the platform keeps refusing: a run of failures means we are blocked.
        if fail >= 5 and ok == 0:
            print("\nFive failures and no successes: the platform is blocking logged-out requests.\n"
                  "Stop and use the manual route (needs_manual.csv -> fill -> 'manual').")
            break
        if i < len(todo):
            time.sleep(random.uniform(*delay))
    print(f"\nDone: {ok} saved to {COLLECTED}, {fail} need manual entry ({NEEDS_MANUAL}).")


def cmd_manual(path):
    df = pd.read_csv(path, dtype=str).fillna("")
    rows, skipped = [], 0
    for raw in df.to_dict("records"):
        s = _clean_seed(raw)
        counts = [_num(raw.get(k)) for k in ("followers", "following", "posts")]
        if s is None or any(np.isnan(c) for c in counts) or _num(raw.get("bio_len")) != _num(raw.get("bio_len")):
            skipped += 1
            continue
        plat, label, handle, source = s
        d = dict(raw)
        d["external_url"] = raw["external_url"].strip() or None   # empty cell = unknown
        rows.append(row_from_profile(plat, label, source, handle, d))
        _append(LOG, [dict(time=datetime.now().isoformat(timespec="seconds"), platform=plat,
                           handle=handle, label=label, status="manual_ok", detail="")],
                ["time", "platform", "handle", "label", "status", "detail"])
    _append(COLLECTED, rows, SCHEMA)
    print(f"Added {len(rows)} manual rows; skipped {skipped} incomplete rows "
          f"(need platform, handle, label, followers, following, posts, bio_len).")


def cmd_template():
    os.makedirs(COLLECT_DIR, exist_ok=True)
    seeds = os.path.join(COLLECT_DIR, "seeds_template.csv")
    manual = os.path.join(COLLECT_DIR, "manual_template.csv")
    with open(seeds, "w", newline="", encoding="utf-8") as f:
        f.write("platform,handle,label,source\nInstagram,example_handle,Real,ig_real_midsize\n")
    with open(manual, "w", newline="", encoding="utf-8") as f:
        f.write("platform,handle,label,source,followers,following,posts,bio_len,full_name,"
                "external_url,has_profile_pic,is_verified,is_private\n"
                "Instagram,example_handle,Real,ig_real_midsize,5200,310,640,48,Example Person,,yes,no,no\n")
    print(f"Wrote {seeds}\nWrote {manual}\n"
          "In the manual file: leave a cell EMPTY if you did not check it (it stays 'unknown'); "
          "external_url: put the link, or leave empty if unknown; has_profile_pic / is_verified / "
          "is_private: yes or no.")


def cmd_report():
    if not os.path.exists(COLLECTED):
        sys.exit("Nothing collected yet.")
    df = pd.read_csv(COLLECTED)
    print(f"{len(df)} collected rows\n")
    print(df.groupby(["platform", "label"]).size().to_string(), "\n")
    labels = [b[2] for b in SIZE_BINS]
    df["size"] = pd.cut(df["followers"], [b[0] for b in SIZE_BINS] + [float("inf")],
                        labels=labels, right=False)
    print("Followers by size band (aim for spread across ALL bands for Real):")
    print(pd.crosstab([df["platform"], df["label"]], df["size"]).reindex(columns=labels).to_string())


def cmd_merge():
    if not os.path.exists(COLLECTED):
        sys.exit("Nothing collected yet.")
    if not os.path.exists(UNIFIED):
        sys.exit(f"{UNIFIED} not found.")
    new = pd.read_csv(COLLECTED)
    old = pd.read_csv(UNIFIED)
    backup = UNIFIED.replace(".csv", f"_backup_{datetime.now():%Y%m%d_%H%M%S}.csv")
    shutil.copy2(UNIFIED, backup)
    keep = [c for c in old.columns]
    add = new.reindex(columns=[c for c in SCHEMA])
    for c in keep:
        if c not in add.columns:
            add[c] = np.nan                       # derived columns are recomputed by train_model.py
    merged = pd.concat([old, add[keep]], ignore_index=True)
    before = len(merged)
    merged = merged.drop_duplicates(subset=[c for c in SCHEMA if c != "source"])
    merged.to_csv(UNIFIED, index=False)
    print(f"Backup: {backup}\nAdded {len(new)} rows ({before - len(merged)} duplicates dropped). "
          f"unified_raw.csv now has {len(merged)} rows.\nNow retrain:  python src/train_model.py")


def main(argv):
    if len(argv) < 2 or argv[1] not in ("scrape", "manual", "template", "report", "merge", "discover", "auto"):
        print(__doc__)
        return 1
    cmd = argv[1]
    if cmd == "scrape":
        if len(argv) < 3:
            sys.exit("usage: collect_data.py scrape seeds.csv [--limit N] [--retry]")
        limit = int(argv[argv.index("--limit") + 1]) if "--limit" in argv else None
        cmd_scrape(argv[2], limit=limit, retry_failed="--retry" in argv)
    elif cmd in ("discover", "auto"):
        if len(argv) < 4 or not argv[3].isdigit():
            sys.exit(f"usage: collect_data.py {cmd} Instagram|Twitter/X|Facebook COUNT")
        (cmd_discover if cmd == "discover" else cmd_auto)(argv[2], int(argv[3]))
    elif cmd == "manual":
        if len(argv) < 3:
            sys.exit("usage: collect_data.py manual manual.csv")
        cmd_manual(argv[2])
    elif cmd == "template":
        cmd_template()
    elif cmd == "report":
        cmd_report()
    elif cmd == "merge":
        cmd_merge()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))