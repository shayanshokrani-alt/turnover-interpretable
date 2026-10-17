"""Core analysis.

Question: a logistic model predicts nurse burnout reasonably well overall, but
does it perform equally across groups of nurses, and what does making it fairer
cost in overall accuracy?

1. Overall: out-of-fold AUC, PR-AUC, and calibration (Brier), as in the two
   companion analyses, plus odds ratios with CIs and sign-stability.
2. Fairness audit: per-group AUC, selection rate, true-positive rate (TPR) and
   false-positive rate (FPR) at a shared threshold, and calibration by group.
   The equal-opportunity gap is the spread in TPR across groups.
3. Trade-off: group-aware thresholds chosen to equalize TPR, and the overall
   accuracy given up to close the gap.

All preprocessing is inside the pipeline, refit per fold, so nothing leaks.
Predictions used for the audit are out-of-fold, so no nurse is scored by a
model that trained on them.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (brier_score_loss, roc_auc_score,
                             average_precision_score)
from sklearn.model_selection import StratifiedKFold

from .config import RANDOM_STATE
from .model import build_pipeline, get_feature_names, as_str_categories


def _cv(n):
    return StratifiedKFold(n, shuffle=True, random_state=RANDOM_STATE)


def oof_predictions(df, cols, n_splits=5):
    """Out-of-fold predicted probabilities for the full model."""
    y = df["y"].values
    X = as_str_categories(df, cols)
    cv = _cv(n_splits)
    oof = np.zeros(len(df))
    for tr, te in cv.split(X, y):
        p = build_pipeline(df, cols)
        p.fit(X.iloc[tr], y[tr])
        oof[te] = p.predict_proba(X.iloc[te])[:, 1]
    return oof


def overall_performance(df, cols, oof):
    """Headline numbers: AUC, PR-AUC, Brier, and the base rate."""
    y = df["y"].values
    return {
        "n": len(df),
        "base_rate": y.mean(),
        "auc": roc_auc_score(y, oof),
        "pr_auc": average_precision_score(y, oof),
        "brier": brier_score_loss(y, oof),
    }


def coefficient_effects(df, cols, n_splits=5):
    """Odds ratios (with CI and sign-consistency) across folds, for readability
    consistent with the companion analyses."""
    y = df["y"].values
    X = as_str_categories(df, cols)
    cv = _cv(n_splits)
    rows, names = [], None
    for tr, _ in cv.split(X, y):
        p = build_pipeline(df, cols)
        p.fit(X.iloc[tr], y[tr])
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
        "sign_consistency": sign,
    })
    tbl["abs"] = np.abs(mean)
    return tbl.sort_values("abs", ascending=False).drop(columns="abs"
                                                        ).reset_index(drop=True)


def _rates(y, pred):
    """TPR, FPR, selection rate for binary predictions."""
    pos = y == 1
    neg = y == 0
    tpr = pred[pos].mean() if pos.any() else np.nan
    fpr = pred[neg].mean() if neg.any() else np.nan
    sel = pred.mean()
    return tpr, fpr, sel


def _boot_auc_ci(y, score, n_boot=1000, seed=RANDOM_STATE):
    """Bootstrap 95% CI for AUC within a single group, resampling its rows."""
    rng = np.random.default_rng(seed)
    n = len(y)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        aucs.append(roc_auc_score(yb, score[idx]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return lo, hi


def fairness_audit(df, oof, attr, labels, threshold=0.5, min_n=150,
                   n_boot=1000):
    """Per-group performance at a shared threshold, with a bootstrap 95% CI on
    each group's AUC so a gap can be read against sampling noise. Groups
    smaller than min_n are dropped as too small to measure reliably."""
    y = df["y"].values
    g = df[attr].values
    pred = (oof >= threshold).astype(int)
    rows = []
    for code, name in labels.items():
        m = g == code
        if m.sum() < min_n:
            continue
        yy, pp, oo = y[m], pred[m], oof[m]
        tpr, fpr, sel = _rates(yy, pp)
        if len(np.unique(yy)) > 1:
            auc = roc_auc_score(yy, oo)
            auc_lo, auc_hi = _boot_auc_ci(yy, oo, n_boot)
        else:
            auc = auc_lo = auc_hi = np.nan
        rows.append({
            "group": name, "n": int(m.sum()), "base_rate": yy.mean(),
            "auc": auc, "auc_lo": auc_lo, "auc_hi": auc_hi,
            "selection_rate": sel, "tpr": tpr, "fpr": fpr,
            "brier": brier_score_loss(yy, oo),
        })
    tbl = pd.DataFrame(rows)
    gaps = {
        "tpr_gap": tbl["tpr"].max() - tbl["tpr"].min(),
        "fpr_gap": tbl["fpr"].max() - tbl["fpr"].min(),
        "auc_gap": tbl["auc"].max() - tbl["auc"].min(),
        "selection_gap": tbl["selection_rate"].max() - tbl["selection_rate"].min(),
    }
    return tbl, gaps


def auc_gap_significance(df, oof, attr, labels, hi_group, lo_group,
                        min_n=150, n_boot=2000, seed=RANDOM_STATE):
    """Bootstrap test of whether one group's AUC really exceeds another's.
    Resamples each group independently and reports the AUC-difference CI and
    the share of resamples where the gap is positive (a one-sided p-value)."""
    y = df["y"].values
    g = df[attr].values
    codes = {v: k for k, v in labels.items()}
    mh = g == codes[hi_group]
    ml = g == codes[lo_group]
    yh, sh = y[mh], oof[mh]
    yl, sl = y[ml], oof[ml]
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        ih = rng.integers(0, len(yh), len(yh))
        il = rng.integers(0, len(yl), len(yl))
        if len(np.unique(yh[ih])) < 2 or len(np.unique(yl[il])) < 2:
            continue
        diffs.append(roc_auc_score(yh[ih], sh[ih])
                     - roc_auc_score(yl[il], sl[il]))
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "hi_group": hi_group, "lo_group": lo_group,
        "mean_diff": float(diffs.mean()),
        "ci_low": float(lo), "ci_high": float(hi),
        "p_gap_le_0": float((diffs <= 0).mean()),
    }


GRID = np.linspace(0.05, 0.95, 181)


def _fit_thresholds(y, g, groups, oof, target_tpr):
    """On one split: pick the single accuracy-optimal threshold and the
    per-group thresholds that hit target_tpr. Returns (t_single, {group:thr})."""
    t_single = max(GRID, key=lambda t: ((oof >= t).astype(int) == y).mean())
    chosen = {}
    for c in groups:
        m = g == c
        yy, oo = y[m], oof[m]
        best_t, best_d = t_single, 1e9
        for t in GRID:
            tpr, _, _ = _rates(yy, (oo >= t).astype(int))
            d = abs(tpr - target_tpr)
            if d < best_d:
                best_d, best_t = d, t
        chosen[c] = best_t
    return t_single, chosen


def _target_tpr(y, g, groups, oof):
    """The equalized TPR target: the lowest TPR any group reaches at that
    split's single accuracy-optimal threshold."""
    t_single = max(GRID, key=lambda t: ((oof >= t).astype(int) == y).mean())
    tprs = []
    for c in groups:
        m = g == c
        tpr, _, _ = _rates(y[m], (oof[m] >= t_single).astype(int))
        tprs.append(tpr)
    return min(tprs)


def equal_opportunity_thresholds(df, oof, attr, labels, min_n=150,
                                 n_repeats=20, random_state=RANDOM_STATE):
    """Group-aware thresholds that equalize TPR (equal opportunity), and the
    HONEST overall accuracy cost versus a single shared threshold.

    The thresholds are *tuned* on one half of the out-of-fold predictions and
    the accuracy is *measured* on the other half, so the cost is not computed
    on the same rows that chose the thresholds. This is repeated over random
    splits and averaged, so a single lucky split does not drive the number.
    Reported thresholds are those fit on the full data, for transparency.
    """
    y = df["y"].values
    g = df[attr].values
    groups = [c for c in labels if (g == c).sum() >= min_n]
    rng = np.random.default_rng(random_state)
    idx = np.arange(len(y))

    def tpr_gap(yv, gv, pred):
        """Spread in per-group TPR for a given prediction vector."""
        tprs = []
        for c in groups:
            m = gv == c
            pos = m & (yv == 1)
            if pos.sum() == 0:
                continue
            tprs.append(pred[pos].mean())
        return max(tprs) - min(tprs)

    costs, acc_s, acc_f = [], [], []
    gap_before, gap_after = [], []
    for _ in range(n_repeats):
        # Groups are large, so a plain random half keeps each well represented.
        rng.shuffle(idx)
        half = len(idx) // 2
        tune, test = idx[:half], idx[half:]

        tt = _target_tpr(y[tune], g[tune], groups, oof[tune])
        t_single, thr = _fit_thresholds(y[tune], g[tune], groups, oof[tune], tt)

        # Everything below is measured on the held-out half only.
        in_g = np.isin(g[test], groups)
        yt, gt, ot = y[test][in_g], g[test][in_g], oof[test][in_g]

        pred_single = (ot >= t_single).astype(int)
        pred_fair = np.zeros(len(yt), dtype=int)
        for c in groups:
            m = gt == c
            pred_fair[m] = (ot[m] >= thr[c]).astype(int)

        acc_s.append((pred_single == yt).mean())
        acc_f.append((pred_fair == yt).mean())
        costs.append((pred_single == yt).mean() - (pred_fair == yt).mean())
        # The equal-opportunity gap, measured on held-out data, before and
        # after applying the group thresholds that were tuned elsewhere.
        gap_before.append(tpr_gap(yt, gt, pred_single))
        gap_after.append(tpr_gap(yt, gt, pred_fair))

    tt_full = _target_tpr(y, g, groups, oof)
    t_single_full, thr_full = _fit_thresholds(y, g, groups, oof, tt_full)

    return {
        "single_threshold": t_single_full,
        "target_tpr": tt_full,
        "group_thresholds": {labels[c]: thr_full[c] for c in groups},
        "accuracy_single": float(np.mean(acc_s)),
        "accuracy_fair": float(np.mean(acc_f)),
        "accuracy_cost": float(np.mean(costs)),
        "accuracy_cost_sd": float(np.std(costs, ddof=1)),
        "tpr_gap_before": float(np.mean(gap_before)),
        "tpr_gap_after": float(np.mean(gap_after)),
        "tpr_gap_after_sd": float(np.std(gap_after, ddof=1)),
        "n_repeats": n_repeats,
    }


def before_after_table(df, oof, attr, labels, min_n=150,
                       n_repeats=20, random_state=RANDOM_STATE):
    """Per-group TPR, FPR and accuracy on held-out data, before mitigation (a
    single shared threshold) and after it (group-aware thresholds), each with a
    95% interval from the spread across the random tune/test splits.

    Thresholds are tuned on one half and every metric is read on the other
    half, so the whole table is out-of-sample. Reporting all three metrics,
    not just TPR, is what shows whether equalizing detection cost too much
    elsewhere (a closed TPR gap bought with a worse FPR, say)."""
    y = df["y"].values
    g = df[attr].values
    groups = [c for c in labels if (g == c).sum() >= min_n]
    rng = np.random.default_rng(random_state)
    idx = np.arange(len(y))

    # collect per-split metrics: rec[group][metric] -> list over splits
    metrics = ["tpr", "fpr", "acc"]
    rec = {c: {f"{m}_{w}": [] for m in metrics for w in ("before", "after")}
           for c in groups}

    for _ in range(n_repeats):
        rng.shuffle(idx)
        half = len(idx) // 2
        tune, test = idx[:half], idx[half:]
        tt = _target_tpr(y[tune], g[tune], groups, oof[tune])
        t_single, thr = _fit_thresholds(y[tune], g[tune], groups, oof[tune], tt)
        for c in groups:
            m = (g[test] == c)
            yy, oo = y[test][m], oof[test][m]
            for w, t_or_map in (("before", t_single), ("after", thr[c])):
                pred = (oo >= t_or_map).astype(int)
                tpr, fpr, _ = _rates(yy, pred)
                acc = (pred == yy).mean()
                rec[c][f"tpr_{w}"].append(tpr)
                rec[c][f"fpr_{w}"].append(fpr)
                rec[c][f"acc_{w}"].append(acc)

    def ci(vals):
        v = np.array([x for x in vals if not np.isnan(x)])
        lo, hi = np.percentile(v, [2.5, 97.5])
        return v.mean(), lo, hi

    rows = []
    for c in groups:
        row = {"group": labels[c], "n": int((g == c).sum())}
        for key, vals in rec[c].items():
            mean, lo, hi = ci(vals)
            row[key] = mean
            row[key + "_lo"] = lo
            row[key + "_hi"] = hi
        rows.append(row)
    return pd.DataFrame(rows)
