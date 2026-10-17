"""Core analysis.

Question: does an employee's work experience (tenure, progression, overtime,
travel) predict turnover beyond demographics (age, gender, marital status,
education), and which experience factors still matter after demographics are
controlled?

1. Incremental value: AUC for demographic-only vs. demographic plus
   experience, with a paired t-test and 95% CIs.
2. Interpretation: odds ratios (with CIs) of the experience factors in the
   full model, i.e. their effect holding demographics constant.
3. Stability: sign-consistency of each coefficient across folds.
4. Calibration: are the full model's predicted probabilities trustworthy as
   probabilities, not just as a ranking? Brier score and a reliability curve.

All preprocessing is inside the pipeline, so there is no leakage.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .config import RANDOM_STATE
from .model import build_pipeline, get_feature_names


def _cv(n):
    return StratifiedKFold(n, shuffle=True, random_state=RANDOM_STATE)


def incremental_value(df, demographic, experience, n_splits=5):
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

    base = per_fold(demographic)
    full = per_fold(demographic + experience)
    diff = full - base
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    t, p = stats.ttest_rel(full, base)

    def ci(x):
        s = x.std(ddof=1) / np.sqrt(len(x))
        return (x.mean() - 1.96 * s, x.mean() + 1.96 * s)

    return {
        "demographic_auc": (base.mean(), ci(base)),
        "full_auc": (full.mean(), ci(full)),
        "gain": diff.mean(),
        "gain_ci": (diff.mean() - 1.96 * se, diff.mean() + 1.96 * se),
        "t": t, "p": p,
    }


def experience_effects(df, demographic, experience, n_splits=5):
    """Odds ratios (with CI and sign-consistency) of the experience factors
    in the full model, their effect holding demographics constant."""
    y = df["y"].values
    full = demographic + experience
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
    # Keep experience-layer features. One-hot names carry a category suffix,
    # so match on prefix rather than exact equality.
    keep = [any(f.startswith(e) for e in experience) for f in tbl["feature"]]
    tbl = tbl[keep].copy()
    tbl["abs"] = tbl["coef_mean"].abs()
    return tbl.sort_values("abs", ascending=False).drop(columns="abs"
                                                        ).reset_index(drop=True)


def calibration(df, demographic, experience, n_splits=5, n_bins=10):
    """Out-of-fold predicted probabilities for the full model, then the Brier
    score and a reliability curve. Predictions are always from a fold that did
    not see the employee, so the calibration is honest."""
    y = df["y"].values
    full = demographic + experience
    cv = _cv(n_splits)

    oof = np.zeros(len(df))
    for tr, te in cv.split(df[full], y):
        p = build_pipeline(df, full)
        p.fit(df[full].iloc[tr], y[tr])
        oof[te] = p.predict_proba(df[full].iloc[te])[:, 1]

    brier = brier_score_loss(y, oof)
    frac_pos, mean_pred = calibration_curve(y, oof, n_bins=n_bins,
                                             strategy="quantile")
    return {"brier": brier, "frac_pos": frac_pos, "mean_pred": mean_pred,
            "oof": oof}
