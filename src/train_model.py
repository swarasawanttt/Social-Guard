"""
src/train_model.py
──────────────────
Trains four models + a VotingClassifier ensemble on ONE unified, multi-platform
table (data/processed/unified_raw.csv, built by build_unified_dataset.py).

Pipeline (all of it applied to the TRAINING split only; the test split and the
held-out-platform checks always contain real accounts only):

    1. load + clean      drop constant / crawl-artefact columns, round counts,
                         recompute derived ratios, per-platform percentile ranks
                         (reference tables are saved for prediction time)
    2. split             80/20, stratified by platform x class
    3. scale             StandardScaler (NaN-tolerant; fit on real rows only)
    4. augment           SMOTE per platform x class cell (large-enough cells
                         only) -> random feature-group masking -> equal weight
                         for every platform x class cell
    5. train + evaluate  XGBoost, LightGBM, RandomForest, AdaBoost, soft Voting
    6. select best       by F1-weighted (tie-break ROC-AUC), as before
    7. honest check      leave-one-platform-out run of the selected model type,
                         plus how often it would answer "Inconclusive"

Platform is NEVER a model input. Missing fields stay NaN.

Run:  python src/train_model.py
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection    import train_test_split
from sklearn.preprocessing      import StandardScaler
from sklearn.metrics            import (accuracy_score, precision_score,
                                        recall_score, f1_score,
                                        classification_report, roc_auc_score,
                                        confusion_matrix)
from sklearn.ensemble           import (RandomForestClassifier,
                                        AdaBoostClassifier,
                                        VotingClassifier)
from sklearn.tree               import DecisionTreeClassifier
from sklearn.neighbors          import NearestNeighbors
from sklearn.base               import clone

from xgboost  import XGBClassifier
from lightgbm import LGBMClassifier

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
from config import (
    UNIFIED_DATASET_PATH, FINAL_DATASET_PATH, PERCENTILE_TABLES_PATH,
    XGBOOST_MODEL_PATH, LIGHTGBM_MODEL_PATH,
    RANDOM_FOREST_MODEL_PATH, ADABOOST_MODEL_PATH,
    VOTING_MODEL_PATH, SCALER_PATH,
    BEST_MODEL_PATH, BEST_MODEL_NAME_PATH,
    BACKGROUND_DATA_PATH, TRAINING_METADATA_PATH,
    FEATURE_COLUMNS, FEATURE_GROUPS, BINARY_FEATURES, PERCENTILE_BASE,
    TARGET_COLUMN, LABEL_MAP,
    TEST_SIZE, RANDOM_STATE, MODEL_DIR,
    USE_SMOTE, SMOTE_K_NEIGHBORS, SMOTE_MIN_CELL, SMOTE_TARGET_CELL,
    MASK_PROBABILITY, INCONCLUSIVE_THRESHOLD, XAI_BACKGROUND_SAMPLE_SIZE,
)
from src.model_utils import (ImputingClassifier, derived_features, build_table,
                             percentile_rank, save_tables, reset_table_cache)

CLASS_NAMES = [LABEL_MAP[i] for i in sorted(LABEL_MAP)]          # Real, Fake, Bot
DROP_COLUMNS = ["verified", "account_age_days"]


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Load + clean
# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    print("\n[1/7] Loading and cleaning the unified dataset ...")
    df = pd.read_csv(UNIFIED_DATASET_PATH)
    print(f"      Shape (raw)  : {df.shape}")

    # 'verified' is 0 in every row of every class (carries nothing) and
    # 'account_age_days' is measured against each source's own crawl date, so it
    # can encode WHICH file a row came from; neither is a profile-page feature
    # the app can read reliably. Both are dropped.
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])

    df[TARGET_COLUMN] = df["label"].map({v: k for k, v in LABEL_MAP.items()})
    df = df.dropna(subset=[TARGET_COLUMN, "platform", "followers", "following", "posts"])
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    for c in PERCENTILE_BASE:
        df[c] = df[c].round().clip(lower=0)
    for c in ("bio_len", "username_len", "fullname_words"):
        if c in df:
            df[c] = df[c].round()

    zero = (df.followers == 0) & (df.following == 0) & (df.posts == 0)
    df = df[~zero]
    df["follower_following_ratio"], df["activity_rate"] = derived_features(
        df.followers, df.following, df.posts)

    # Per-platform percentile reference tables (+ a pooled "ALL" table, used for
    # a platform that has no training data, e.g. LinkedIn).
    tables = {"ALL": {c: build_table(df[c]) for c in PERCENTILE_BASE}}
    for plat, g in df.groupby("platform"):
        tables[plat] = {c: build_table(g[c]) for c in PERCENTILE_BASE}
    save_tables(tables, PERCENTILE_TABLES_PATH)
    reset_table_cache()
    for c in PERCENTILE_BASE:
        df[c + "_pr"] = np.nan
        for plat, g in df.groupby("platform"):
            df.loc[g.index, c + "_pr"] = percentile_rank(g[c].values, tables[plat][c])
    print(f"      Percentile tables saved → {PERCENTILE_TABLES_PATH}")

    # Leakage guard: exact duplicates (same features + label) would otherwise
    # land on both sides of the split.
    before = len(df)
    df = df.drop_duplicates(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).reset_index(drop=True)
    if before - len(df):
        print(f"      Removed {before - len(df)} exact duplicate rows — leakage guard.")

    os.makedirs(os.path.dirname(FINAL_DATASET_PATH), exist_ok=True)
    df[["platform", "source", "label", TARGET_COLUMN] + FEATURE_COLUMNS].to_csv(
        FINAL_DATASET_PATH, index=False)
    print(f"      Shape (clean): {df.shape}  → saved {FINAL_DATASET_PATH}")
    print("      Rows per platform x class:")
    print(df.groupby(["platform", "label"]).size().unstack(fill_value=0).to_string())
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Split (stratified by platform x class so every cell is in both sides)
# ─────────────────────────────────────────────────────────────────────────────
def split_data(df):
    print("\n[2/7] Train / test split (80 / 20, stratified by platform x class) ...")
    strat = df["platform"] + "_" + df[TARGET_COLUMN].astype(str)
    tr, te = train_test_split(df, test_size=TEST_SIZE, stratify=strat,
                              random_state=RANDOM_STATE)
    print(f"      Train : {len(tr):,}  |  Test : {len(te):,}")
    return tr.reset_index(drop=True), te.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Scale (fit on the real training rows only)
# ─────────────────────────────────────────────────────────────────────────────
def scale_data(tr):
    print("\n[3/7] Scaling features (StandardScaler, NaN-tolerant) ...")
    scaler = StandardScaler().fit(tr[FEATURE_COLUMNS].values)
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    print(f"      Scaler saved → {SCALER_PATH}")
    return scaler


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Augment (SMOTE per cell -> masking -> cell weights)
# ─────────────────────────────────────────────────────────────────────────────
def _smote_cell(X, n_new, rng):
    """SMOTE inside ONE platform x class cell: interpolate between a row and one
    of its K nearest neighbours. Columns missing in the whole cell stay NaN; a
    synthetic value is NaN only where both parents were NaN. Binary columns take
    one parent's value (SMOTENC-style)."""
    cols_ok = list(X.columns[X.notna().any()])
    Z = X[cols_ok]
    Zf = Z.fillna(Z.median())
    sd = Zf.std().replace(0, 1)
    A = ((Zf - Zf.mean()) / sd).values
    k = min(SMOTE_K_NEIGHBORS, len(A) - 1)
    nbr = NearestNeighbors(n_neighbors=k + 1).fit(A).kneighbors(A, return_distance=False)[:, 1:]
    i = rng.integers(0, len(A), n_new)
    j = nbr[i, rng.integers(0, k, n_new)]
    lam = rng.random((n_new, 1))
    Zv, Zn = Z.values, Zf.values
    new = Zn[i] + lam * (Zn[j] - Zn[i])
    both_nan = np.isnan(Zv[i]) & np.isnan(Zv[j])
    new[both_nan] = np.nan
    out = pd.DataFrame(new, columns=cols_ok)
    for c in BINARY_FEATURES:
        if c in out:
            ci = cols_ok.index(c)
            out[c] = np.where(rng.random(n_new) < 0.5, Zn[i][:, ci], Zn[j][:, ci])
            out.loc[both_nan[:, ci], c] = np.nan
    for c in ("followers", "following", "posts", "bio_len", "fullname_words", "username_len"):
        if c in out:
            out[c] = out[c].round()
    if {"followers", "following", "posts"} <= set(out.columns):
        out["follower_following_ratio"], out["activity_rate"] = derived_features(
            out["followers"], out["following"], out["posts"])
    return out.reindex(columns=X.columns)


def _add_smote(tr, rng):
    parts = [tr]
    for (plat, y), cell in tr.groupby(["platform", TARGET_COLUMN]):
        n = len(cell)
        if n < SMOTE_MIN_CELL or n >= SMOTE_TARGET_CELL:
            continue
        syn = _smote_cell(cell[FEATURE_COLUMNS], SMOTE_TARGET_CELL - n, rng)
        syn["platform"], syn[TARGET_COLUMN] = plat, y
        parts.append(syn)
    return pd.concat(parts, ignore_index=True)


def _add_masking(tr, rng):
    m = tr.copy()
    for cols in FEATURE_GROUPS.values():
        hide = rng.random(len(m)) < MASK_PROBABILITY
        m.loc[hide, cols] = np.nan
    return pd.concat([tr, m], ignore_index=True)


def _cell_weights(tr):
    """Every platform x class cell gets equal total weight, and every class
    equal total weight overall, so a big platform (Twitter) cannot dominate."""
    n_cell = tr.groupby(["platform", TARGET_COLUMN])[TARGET_COLUMN].transform("size")
    cells_in_class = tr.groupby(TARGET_COLUMN)["platform"].transform("nunique")
    w = 1.0 / n_cell / cells_in_class
    return (w / w.mean()).values


def augment(tr, scaler, seed=RANDOM_STATE):
    """Returns (X_scaled, y, sample_weight, info) for the training split."""
    rng = np.random.default_rng(seed)
    info = {"real_rows": int(len(tr))}
    if USE_SMOTE:
        tr = _add_smote(tr, rng)
        info["after_smote"] = int(len(tr))
    tr = _add_masking(tr, rng)
    info["after_masking"] = int(len(tr))
    w = _cell_weights(tr)
    X = scaler.transform(tr[FEATURE_COLUMNS].values)
    return X, tr[TARGET_COLUMN].astype(int).values, w, info


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Models
# ─────────────────────────────────────────────────────────────────────────────
def build_models():
    """name -> model. XGBoost/LightGBM take NaN natively; RandomForest and
    AdaBoost are wrapped so they median-impute first. Sample weights flow to all."""
    def _xgb():
        return XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="mlogloss",
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)

    def _lgbm():
        return LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)

    def _rf():
        return ImputingClassifier(RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1))

    def _ada():
        return ImputingClassifier(AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=3),
            n_estimators=200, learning_rate=0.5,
            random_state=RANDOM_STATE))

    voting = VotingClassifier(
        estimators=[("xgb", _xgb()), ("lgbm", _lgbm()),
                    ("rf", _rf()),   ("ada", _ada())],
        voting="soft", n_jobs=1,
    )
    return {
        "XGBoost":          _xgb(),
        "LightGBM":         _lgbm(),
        "RandomForest":     _rf(),
        "AdaBoost":         _ada(),
        "VotingClassifier": voting,
    }


def _eval(model, X_te, y_te, name):
    y_pred  = model.predict(X_te)
    y_proba = model.predict_proba(X_te)
    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_te, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_te, y_pred, average="weighted", zero_division=0)
    auc  = roc_auc_score(y_te, y_proba, multi_class="ovr", average="weighted")

    order = sorted(LABEL_MAP.keys())
    pc_p  = precision_score(y_te, y_pred, average=None, labels=order, zero_division=0)
    pc_r  = recall_score(y_te, y_pred, average=None, labels=order, zero_division=0)
    pc_f  = f1_score(y_te, y_pred, average=None, labels=order, zero_division=0)
    macro = f1_score(y_te, y_pred, average="macro", zero_division=0)
    cm    = confusion_matrix(y_te, y_pred, labels=order)
    per_class = {CLASS_NAMES[i]: {"precision": round(float(pc_p[i]), 4),
                                  "recall":    round(float(pc_r[i]), 4),
                                  "f1":        round(float(pc_f[i]), 4),
                                  "support":   int((y_te == order[i]).sum())}
                 for i in range(len(order))}

    print(f"\n  ── {name} ──")
    print(f"     Accuracy {acc:.4f} | F1-Weighted {f1:.4f} (selection) | F1-Macro {macro:.4f} | ROC-AUC {auc:.4f}")
    print(f"  {classification_report(y_te, y_pred, target_names=CLASS_NAMES)}")
    print(f"  Confusion matrix (rows=actual, cols=predicted, order={CLASS_NAMES}):\n  {cm}")
    return {"accuracy": acc, "precision": prec, "recall": rec,
            "f1_weighted": f1, "f1_macro": macro, "roc_auc": auc,
            "per_class": per_class, "confusion_matrix": cm.tolist(),
            "confusion_matrix_labels": CLASS_NAMES}


def train_all(models, X_tr, y_tr, w_tr, X_te, y_te):
    print("\n[5/7] Training all models (sample-weighted) ...")
    save_map = {"XGBoost": XGBOOST_MODEL_PATH, "LightGBM": LIGHTGBM_MODEL_PATH,
                "RandomForest": RANDOM_FOREST_MODEL_PATH, "AdaBoost": ADABOOST_MODEL_PATH,
                "VotingClassifier": VOTING_MODEL_PATH}
    results, trained = {}, {}
    for name, model in models.items():
        print(f"\n  Training {name} ...")
        model.fit(X_tr, y_tr, sample_weight=w_tr)
        results[name] = _eval(model, X_te, y_te, name)
        trained[name] = model
        joblib.dump(model, save_map[name])
        print(f"  Saved → {save_map[name]}")
    return results, trained


def select_best(results, trained):
    """Best = highest F1-weighted, tie-break ROC-AUC (unchanged from before).
    NOTE: this test split is a random split of the same sources the model
    trained on, so these scores are optimistic; see the held-out-platform
    check below for the cross-platform picture."""
    print("\n[6/7] Model comparison and selection ...")
    print("\n  {:18}  {:>9}  {:>9}  {:>9}".format("Model", "Accuracy", "F1-Wtd", "ROC-AUC"))
    print("  " + "-" * 52)
    best = max(results, key=lambda k: (results[k]["f1_weighted"], results[k]["roc_auc"]))
    for n, m in results.items():
        print("  {:18}  {:>9.4f}  {:>9.4f}  {:>9.4f}{}".format(
            n, m["accuracy"], m["f1_weighted"], m["roc_auc"], "  ← BEST" if n == best else ""))
    joblib.dump(trained[best], BEST_MODEL_PATH)
    with open(BEST_MODEL_NAME_PATH, "w") as f:
        f.write(best)
    print(f"\n  Selected : {best}   → {BEST_MODEL_PATH}")
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Step 7 — Honest checks
# ─────────────────────────────────────────────────────────────────────────────
def inconclusive_stats(model, X_te, y_te):
    """How the Inconclusive band behaves on the random test split."""
    proba = model.predict_proba(X_te)
    pred = model.classes_[proba.argmax(1)]
    conf = proba.max(1)
    sure = conf >= INCONCLUSIVE_THRESHOLD
    real = y_te == 0
    return {
        "threshold": INCONCLUSIVE_THRESHOLD,
        "share_inconclusive": round(float(1 - sure.mean()), 4),
        "accuracy_when_confident": round(float((pred[sure] == y_te[sure]).mean()), 4) if sure.any() else None,
        "real_called_bot_without_band": round(float(((pred == 2) & real).sum() / max(1, real.sum())), 4),
        "real_called_bot_with_band": round(float(((pred == 2) & real & sure).sum() / max(1, real.sum())), 4),
    }


def held_out_platform_check(df, best_name, scaler):
    """Train the selected model type WITHOUT one platform, test only on it."""
    print("\n[7/7] Leave-one-platform-out check of the selected model type ...")
    out = {}
    for plat in sorted(df["platform"].unique()):
        te = df[df.platform == plat]
        tr = df[df.platform != plat]
        sc = StandardScaler().fit(tr[FEATURE_COLUMNS].values)
        X, y, w, _ = augment(tr.reset_index(drop=True), sc)
        model = build_models()[best_name]
        model.fit(X, y, sample_weight=w)
        Xt = sc.transform(te[FEATURE_COLUMNS].values)
        yt = te[TARGET_COLUMN].values
        proba = model.predict_proba(Xt)
        pred = model.classes_[proba.argmax(1)]
        sure = proba.max(1) >= INCONCLUSIVE_THRESHOLD
        row = {"n": int(len(te))}
        for c in sorted(np.unique(yt)):
            m = yt == c
            row[f"{LABEL_MAP[int(c)]}_recall"] = round(float((pred[m] == c).mean()), 3)
        real = yt == 0
        if real.any():
            row["real_called_bot"] = round(float(((pred == 2) & real).sum() / real.sum()), 3)
            row["real_called_bot_with_inconclusive_band"] = round(float(((pred == 2) & real & sure).sum() / real.sum()), 3)
        row["share_inconclusive"] = round(float(1 - sure.mean()), 3)
        out[plat] = row
        print(f"      held-out {plat:10s} {row}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
def save_background_data(tr, scaler):
    """Reference sample for SHAP/LIME: REAL (pre-SMOTE, pre-masking) training
    rows, scaled. Missing values are filled with the column median so the
    explainers always get complete rows; the same medians are saved so a
    prediction with unknown fields can be explained consistently."""
    rng = np.random.RandomState(RANDOM_STATE)
    n = min(XAI_BACKGROUND_SAMPLE_SIZE, len(tr))
    X = scaler.transform(tr[FEATURE_COLUMNS].values)
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    Xb = X[rng.choice(len(X), size=n, replace=False)]
    Xb = np.where(np.isnan(Xb), med, Xb)
    joblib.dump({
        "X_background":  Xb,
        "fill_values":   med,
        "feature_names": FEATURE_COLUMNS,
        "class_names":   CLASS_NAMES,
    }, BACKGROUND_DATA_PATH)
    print(f"\nSaved SHAP/LIME background data ({n} rows) → {BACKGROUND_DATA_PATH}")


def save_training_metadata(df, tr, te, info, best_name, results,
                           inc_stats, held_out):
    cell = lambda d: {f"{p}|{CLASS_NAMES[int(c)]}": int(n)
                      for (p, c), n in d.groupby(["platform", TARGET_COLUMN]).size().items()}
    metadata = {
        "trained_at":      datetime.now().isoformat(timespec="seconds"),
        "dataset_file":    os.path.basename(UNIFIED_DATASET_PATH),
        "total_samples":   int(len(df)),
        "feature_columns": FEATURE_COLUMNS,
        "dropped_columns": DROP_COLUMNS,
        "random_state":    RANDOM_STATE,
        "platform_is_model_input": False,
        "split": {"method": "stratified train_test_split by platform x class",
                  "test_size_fraction": TEST_SIZE,
                  "train_samples_original": int(len(tr)),
                  "test_samples": int(len(te))},
        "class_balance": {"train_cells_real_rows": cell(tr), "test_cells": cell(te)},
        "augmentation": {
            "smote": {"used": USE_SMOTE, "scope": "per platform x class cell, training split only",
                      "min_cell_rows": SMOTE_MIN_CELL, "target_cell_rows": SMOTE_TARGET_CELL,
                      "k_neighbors": SMOTE_K_NEIGHBORS},
            "feature_group_masking": {"groups": FEATURE_GROUPS, "probability": MASK_PROBABILITY},
            "sample_weights": "equal total weight per platform x class cell, then per class",
            "rows": info,
        },
        "deployed_model": {
            "name": best_name,
            "note": ("The app serves predictions from this ONE model "
                     "(models/best_model.pkl). The other rows under "
                     "'all_models_compared' are evaluation-time comparisons."),
        },
        "verdict_rule": {"inconclusive_below_top_class_probability": INCONCLUSIVE_THRESHOLD,
                         "behaviour_on_test_split": inc_stats},
        "held_out_platform_check": {
            "what": ("Selected model type retrained WITHOUT one platform and tested only on it. "
                     "This is the honest cross-platform number; the random-split scores above "
                     "are optimistic because test rows come from the same sources as training rows."),
            "results": held_out,
        },
        "all_models_compared": {
            n: {"accuracy": round(float(m["accuracy"]), 4),
                "precision_weighted": round(float(m["precision"]), 4),
                "recall_weighted":    round(float(m["recall"]), 4),
                "f1_weighted": round(float(m["f1_weighted"]), 4),
                "f1_macro":    round(float(m["f1_macro"]), 4),
                "roc_auc":     round(float(m["roc_auc"]), 4),
                "per_class":   m["per_class"],
                "confusion_matrix": m["confusion_matrix"],
                "confusion_matrix_labels": m["confusion_matrix_labels"],
                "confusion_matrix_note": "rows = actual class, columns = predicted class"}
            for n, m in results.items()},
    }
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(TRAINING_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved training metadata → {TRAINING_METADATA_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df          = load_data()
    tr, te      = split_data(df)
    scaler      = scale_data(tr)

    print("\n[4/7] Augmenting the training split (SMOTE per cell, masking, weights) ...")
    X_tr, y_tr, w_tr, info = augment(tr, scaler)
    print(f"      Rows: {info}")
    X_te = scaler.transform(te[FEATURE_COLUMNS].values)
    y_te = te[TARGET_COLUMN].astype(int).values

    results, trained = train_all(build_models(), X_tr, y_tr, w_tr, X_te, y_te)
    best             = select_best(results, trained)

    inc_stats = inconclusive_stats(trained[best], X_te, y_te)
    print(f"\n  Inconclusive band (<{INCONCLUSIVE_THRESHOLD:.0%}) on test split: {inc_stats}")
    held_out = held_out_platform_check(df, best, scaler)

    save_background_data(tr, scaler)
    save_training_metadata(df, tr, te, info, best, results, inc_stats, held_out)

    print(f"\n✅  Done. Best model: {best}")
    print("    All models saved to /models/")