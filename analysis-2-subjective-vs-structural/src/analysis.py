"""Core analysis.

Question: does the subjective layer add predictive value for turnover beyond
the structural layer, and which subjective factors stay predictive even after
structural facts are controlled?

1. Incremental value: AUC for structural-only vs. structural plus subjective,
   with a paired t-test and 95% CIs.
2. Interpretation: odds ratios (with CIs) of the subjective factors in the
   full model, i.e. their effect holding structure constant.
3. Stability: sign-consistency of each coefficient across folds.

All preprocessing is inside the pipeline, so there is no leakage.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .config import RANDOM_STATE
from .model import build_pipeline, get_feature_names


def _cv(n):
    return StratifiedKFold(n, shuffle=True, random_state=RANDOM_STATE)


def incremental_value(df, structural, subjective, n_splits=5):
    """Per-fold AUC for both models, the gain, its 95% CI, and a t-test."""
    y = df["y"].values
    cv = _cv(n_splits)

    def per_fold(cols):
        out = []
        for tr, te in cv.split(df[cols], y):
            p = build_pipeline(df, cols)
            p.fit(df[cols].iloc[tr], y[tr])
            out.append(roc_auc_score(
                y[te], p.predict_proba(df[cols].iloc[te])[:, 1]))
        return np.array(out)

    base = per_fold(structural)
    full = per_fold(structural + subjective)
    diff = full - base
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    t, p = stats.ttest_rel(full, base)

    def ci(x):
        s = x.std(ddof=1) / np.sqrt(len(x))
        return (x.mean() - 1.96 * s, x.mean() + 1.96 * s)

    return {
        "structural_auc": (base.mean(), ci(base)),
        "full_auc": (full.mean(), ci(full)),
        "gain": diff.mean(),
        "gain_ci": (diff.mean() - 1.96 * se, diff.mean() + 1.96 * se),
        "t": t, "p": p,
    }


def subjective_effects(df, structural, subjective, n_splits=5):
    """Odds ratios (with CI and sign-consistency) of the subjective factors
    in the full model, their effect holding structure constant."""
    y = df["y"].values
    full = structural + subjective
    cv = _cv(n_splits)

    rows, names = [], None
    for tr, _ in cv.split(df[full], y):
        p = build_pipeline(df, full)
        p.fit(df[full].iloc[tr], y[tr])
        if names is None:
            names = get_feature_names(p)
        rows.append(p.named_steps["clf"].coef_[0])

    coefs = np.vstack(rows)
    mean = coefs.mean(0)
    se = coefs.std(0, ddof=1) / np.sqrt(coefs.shape[0])
    sign = (np.sign(coefs) == np.sign(mean)).mean(0)

    tbl = pd.DataFrame({
        "feature": names,
        "odds_ratio": np.exp(mean),
        "or_ci_low": np.exp(mean - 1.96 * se),
        "or_ci_high": np.exp(mean + 1.96 * se),
        "coef_mean": mean,
        "coef_std": coefs.std(0, ddof=1),
        "sign_consistency": sign,
    })
    tbl = tbl[tbl["feature"].isin(subjective)].copy()
    tbl["abs"] = tbl["coef_mean"].abs()
    return tbl.sort_values("abs", ascending=False).drop(columns="abs"
                                                        ).reset_index(drop=True)
