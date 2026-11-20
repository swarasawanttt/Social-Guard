"""
src/model_utils.py
──────────────────
Small helpers shared by TRAINING (train_model.py) and PREDICTION
(feature_engineering.py), so the maths is identical on both sides:

  * derived_features()   follower/following ratio and activity rate
  * percentile tables    per-platform reference distributions + percentile_rank()
  * ImputingClassifier   wraps a classifier that cannot handle missing values
                         (RandomForest, AdaBoost) so it can sit in the same
                         VotingClassifier as XGBoost / LightGBM, which can.

This module must stay importable as `src.model_utils`: the saved model files
refer to ImputingClassifier by that path.
"""
import json
import os

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.impute import SimpleImputer

N_QUANTILES = 201   # 0%, 0.5%, ... 100%


# ─────────────────────────────────────────────────────────────────────────────
def derived_features(followers, following, posts):
    """Identical formulas to the original project. Accepts floats or arrays."""
    ratio = followers / (following + 1)
    activity = posts / (followers + 1)
    return ratio, activity


# ─────────────────────────────────────────────────────────────────────────────
# Per-platform percentile tables
# ─────────────────────────────────────────────────────────────────────────────
def build_table(values) -> list:
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    return np.quantile(v, np.linspace(0, 1, N_QUANTILES)).tolist()


def percentile_rank(x, table):
    """Mid-rank percentile (0..1) of x within the stored quantile table.
    Ties (e.g. the many accounts with 0 posts) get the middle of their range,
    so 0 followers is 'low' but not literally rank 0. Works on scalars/arrays;
    NaN in -> NaN out."""
    q = np.asarray(table, dtype=float)
    x = np.asarray(x, dtype=float)
    lo = np.searchsorted(q, x, side="left")
    hi = np.searchsorted(q, x, side="right")
    out = ((lo + hi) / 2.0) / len(q)
    return np.where(np.isnan(x), np.nan, np.clip(out, 0.0, 1.0))


def save_tables(tables: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tables, f)


_tables_cache = None


def load_tables(path: str):
    """Returns the dict saved by save_tables(), or None if training has not run."""
    global _tables_cache
    if _tables_cache is None and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            _tables_cache = json.load(f)
    return _tables_cache


def reset_table_cache():
    global _tables_cache
    _tables_cache = None


# ─────────────────────────────────────────────────────────────────────────────
class ImputingClassifier(ClassifierMixin, BaseEstimator):   # mixin first: scikit-learn >=1.6 reads the estimator type from it
    """Median-imputes missing values, then fits `estimator`.

    RandomForest/AdaBoost reject NaN in many scikit-learn versions; XGBoost and
    LightGBM accept it. Wrapping lets all four sit in one VotingClassifier and
    lets sample_weight pass straight through to the inner model."""

    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y, sample_weight=None):
        self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True)
        Xi = self.imputer_.fit_transform(X)
        self.model_ = clone(self.estimator)
        if sample_weight is not None:
            self.model_.fit(Xi, y, sample_weight=sample_weight)
        else:
            self.model_.fit(Xi, y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(self.imputer_.transform(X))

    def predict(self, X):
        return self.model_.predict(self.imputer_.transform(X))

    @property
    def feature_importances_(self):
        return self.model_.feature_importances_