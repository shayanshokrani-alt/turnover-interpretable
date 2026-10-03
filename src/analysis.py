"""Core analysis: incremental value, interpretation, stability, calibration.

Questions, in order:
  1. Does the experience layer add predictive value over the baseline, and is
     that gain statistically significant (not just numerically larger)?
  2. Which features drive attrition, in which direction, and how large each
     with a confidence interval, not a bare point estimate?
  3. Are those interpretations stable across folds?
  4. Are the model's predicted probabilities well calibrated i.e. can the
     numbers be read as real probabilities, not just rankings?

All preprocessing lives inside the pipeline, so every number is computed
without leakage from the evaluation fold.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .config import RANDOM_STATE
from .model import build_pipeline, get_feature_names


def _cv(n_splits: int):
    return StratifiedKFold(n_splits, shuffle=True, random_state=RANDOM_STATE)


def incremental_value(df: pd.DataFrame, baseline: list[str],
                      added: list[str], n_splits: int = 5) -> dict:
    """Per-fold AUC and PR-AUC for both models, the paired-fold difference,
    a 95% CI for that difference, and a paired t-test.

    Both AUC and PR-AUC are reported because the classes are imbalanced
    (~16% attrition); PR-AUC is the more informative view of the minority
    class, and its no-skill baseline is the positive rate, not 0.5.
    """
    y = df["y"].values
    cv = _cv(n_splits)

    def per_fold(cols):
        aucs, aps = [], []
        for tr, te in cv.split(df[cols], y):
            pipe = build_pipeline(df, cols)
            pipe.fit(df[cols].iloc[tr], y[tr])
            p = pipe.predict_proba(df[cols].iloc[te])[:, 1]
            aucs.append(roc_auc_score(y[te], p))
            aps.append(average_precision_score(y[te], p))
        return np.array(aucs), np.array(aps)

    b_auc, b_ap = per_fold(baseline)
    a_auc, a_ap = per_fold(baseline + added)

    diff = a_auc - b_auc
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    ci = (diff.mean() - 1.96 * se, diff.mean() + 1.96 * se)
    t, pval = stats.ttest_rel(a_auc, b_auc)

    def ci95(x):
        s = x.std(ddof=1) / np.sqrt(len(x))
        return (x.mean() - 1.96 * s, x.mean() + 1.96 * s)

    return {
        "baseline_auc": (b_auc.mean(), ci95(b_auc)),
        "augmented_auc": (a_auc.mean(), ci95(a_auc)),
        "baseline_prauc": (b_ap.mean(), ci95(b_ap)),
        "augmented_prauc": (a_ap.mean(), ci95(a_ap)),
        "positive_rate": y.mean(),
        "auc_gain": diff.mean(),
        "auc_gain_ci": ci,
        "ttest_t": t,
        "ttest_p": pval,
    }


def coefficient_stability(df: pd.DataFrame, cols: list[str],
                          n_splits: int = 5) -> pd.DataFrame:
    """Per-fold coefficients, summarized as odds ratios with a 95% CI and a
    sign-consistency score.

    The CI is computed across folds (a cross-validation estimate of
    variability), which also serves as the uncertainty on each odds ratio.
    Numeric features are standardized, so their odds ratios are per one
    standard deviation of change, not per raw unit -- stated so the numbers
    are not misread.
    """
    y = df["y"].values
    cv = _cv(n_splits)

    rows, names = [], None
    for tr, _ in cv.split(df[cols], y):
        pipe = build_pipeline(df, cols)
        pipe.fit(df[cols].iloc[tr], y[tr])
        if names is None:
            names = [n.split("__", 1)[-1] for n in get_feature_names(pipe)]
        rows.append(pipe.named_steps["clf"].coef_[0])

    coefs = np.vstack(rows)
    mean = coefs.mean(axis=0)
    se = coefs.std(axis=0, ddof=1) / np.sqrt(coefs.shape[0])
    lo, hi = mean - 1.96 * se, mean + 1.96 * se
    sign_agree = (np.sign(coefs) == np.sign(mean)).mean(axis=0)

    out = pd.DataFrame({
        "feature": names,
        "odds_ratio": np.exp(mean),
        "or_ci_low": np.exp(lo),
        "or_ci_high": np.exp(hi),
        "coef_mean": mean,
        "coef_std": coefs.std(axis=0, ddof=1),
        "sign_consistency": sign_agree,
    })
    out["abs_coef"] = out["coef_mean"].abs()
    out = out.sort_values("abs_coef", ascending=False).drop(columns="abs_coef")
    return out.reset_index(drop=True)


def calibration(df: pd.DataFrame, cols: list[str], n_splits: int = 5,
                n_bins: int = 10) -> dict:
    """Out-of-fold calibration curve and Brier score.

    Predictions are produced by cross_val_predict so every probability is
    made on data the model did not train on. The Brier score summarizes
    calibration (lower is better); the curve shows predicted vs. observed
    frequency across probability bins.
    """
    y = df["y"].values
    pipe = build_pipeline(df, cols)
    proba = cross_val_predict(pipe, df[cols], y, cv=_cv(n_splits),
                              method="predict_proba")[:, 1]
    frac_pos, mean_pred = calibration_curve(y, proba, n_bins=n_bins,
                                            strategy="quantile")
    brier = np.mean((proba - y) ** 2)
    return {"mean_pred": mean_pred, "frac_pos": frac_pos, "brier": brier}
