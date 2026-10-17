"""Console report and figures."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def print_overall(r):
    print("\n" + "=" * 70)
    print("Overall model: predicting nurse burnout (employed RNs)")
    print("=" * 70)
    print(f"  Nurses analysed         {r['n']:,}")
    print(f"  Burnout base rate       {r['base_rate']:.1%}")
    print(f"  AUC                     {r['auc']:.3f}")
    print(f"  PR-AUC                  {r['pr_auc']:.3f}  (base {r['base_rate']:.3f})")
    print(f"  Brier score             {r['brier']:.4f}  (out-of-fold)")
    print("=" * 70)


def print_audit(name, tbl, gaps):
    print(f"\nFairness audit by {name} (shared threshold)")
    print("-" * 70)
    print(f"{'group':<22}{'n':>7}{'base':>7}{'AUC':>7}{'TPR':>7}{'FPR':>7}")
    print("-" * 70)
    for _, r in tbl.iterrows():
        print(f"{r['group']:<22}{int(r['n']):>7}{r['base_rate']:>7.2f}"
              f"{r['auc']:>7.3f}{r['tpr']:>7.2f}{r['fpr']:>7.2f}")
    print("-" * 70)
    print(f"  TPR gap (equal-opportunity gap)  {gaps['tpr_gap']:.3f}")
    print(f"  FPR gap                          {gaps['fpr_gap']:.3f}")
    print(f"  AUC gap                          {gaps['auc_gap']:.3f}")


def print_tradeoff(name, t):
    print(f"\nFairness and its cost, by {name}")
    print("-" * 70)
    print(f"  Single best threshold            {t['single_threshold']:.2f}")
    print(f"  Target TPR (equalized)           {t['target_tpr']:.2f}")
    print("  Group-specific thresholds:")
    for g, thr in t["group_thresholds"].items():
        print(f"      {g:<24} {thr:.2f}")
    print(f"  TPR gap on held-out, before      {t['tpr_gap_before']:.3f}")
    print(f"  TPR gap on held-out, after       {t['tpr_gap_after']:.3f}"
          f"  (SD {t['tpr_gap_after_sd']:.3f})")
    print(f"  Accuracy, single threshold       {t['accuracy_single']:.3f}  (held-out)")
    print(f"  Accuracy, equal-opportunity      {t['accuracy_fair']:.3f}  (held-out)")
    print(f"  Cost of fairness (accuracy)      {t['accuracy_cost']:+.3f}"
          f"  (SD {t['accuracy_cost_sd']:.3f}, {t['n_repeats']} splits)")
    print("  Thresholds tuned on one half; gap and cost measured on the other.")
    print("-" * 70)


def print_before_after(name, tbl):
    print(f"\nBefore vs. after mitigation, by {name} (held-out, 95% interval)")
    print("-" * 78)
    print(f"{'group':<20}{'metric':<6}{'before':>16}{'after':>16}")
    print("-" * 78)
    for _, r in tbl.iterrows():
        for m, lab in [("tpr", "TPR"), ("fpr", "FPR"), ("acc", "Acc")]:
            b = f"{r[m+'_before']:.2f} [{r[m+'_before_lo']:.2f},{r[m+'_before_hi']:.2f}]"
            a = f"{r[m+'_after']:.2f} [{r[m+'_after_lo']:.2f},{r[m+'_after_hi']:.2f}]"
            head = f"{r['group']} (n={int(r['n'])})" if m == "tpr" else ""
            print(f"{head:<20}{lab:<6}{b:>16}{a:>16}")
        print()


def plot_before_after(name, tbl, out_path):
    """Grouped bars: TPR and FPR per group, before (faded) and after (solid)."""
    groups = list(tbl["group"])
    x = np.arange(len(groups))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, metric, title, col in [
        (ax1, "tpr", "True-positive rate", "#2d6cdf"),
        (ax2, "fpr", "False-positive rate", "#c0392b"),
    ]:
        b = tbl[f"{metric}_before"].values
        a = tbl[f"{metric}_after"].values
        b_err = [b - tbl[f"{metric}_before_lo"].values,
                 tbl[f"{metric}_before_hi"].values - b]
        a_err = [a - tbl[f"{metric}_after_lo"].values,
                 tbl[f"{metric}_after_hi"].values - a]
        w = 0.38
        ax.bar(x - w / 2, b, w, yerr=b_err, label="Before", color=col,
               alpha=0.4, error_kw={"ecolor": "#888", "capsize": 3})
        ax.bar(x + w / 2, a, w, yerr=a_err, label="After", color=col,
               error_kw={"ecolor": "#444", "capsize": 3})
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=20, ha="right")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="lower right")
    fig.suptitle(f"Detection and error rates before and after mitigation, "
                 f"by {name}\n(held-out data, bars = 95% interval across splits)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] Saved: {out_path}")


def plot_audit(name, tbl, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(tbl))
    w = 0.38
    ax.bar(x - w / 2, tbl["tpr"], w, label="True-positive rate",
           color="#2d6cdf")
    ax.bar(x + w / 2, tbl["fpr"], w, label="False-positive rate",
           color="#c0392b", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(tbl["group"], rotation=20, ha="right")
    ax.set_ylabel("Rate at the shared threshold")
    ax.set_title(f"Burnout-model error rates differ across groups ({name})\n"
                 "a gap in the blue bars is unequal opportunity")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] Saved: {out_path}")


def print_group_auc(name, tbl):
    print(f"\nPer-group AUC with bootstrap 95% CI, by {name}")
    print("-" * 70)
    print(f"{'group':<22}{'n':>7}{'AUC':>8}{'95% CI':>18}")
    print("-" * 70)
    for _, r in tbl.sort_values("auc", ascending=False).iterrows():
        ci = f"[{r['auc_lo']:.3f}, {r['auc_hi']:.3f}]"
        print(f"{r['group']:<22}{int(r['n']):>7}{r['auc']:>8.3f}{ci:>18}")
    print("-" * 70)


def print_auc_gap_test(sig):
    print(f"  Is {sig['hi_group']} AUC > {sig['lo_group']} AUC? "
          f"diff {sig['mean_diff']:+.3f}, "
          f"95% CI [{sig['ci_low']:.3f}, {sig['ci_high']:.3f}], "
          f"one-sided p = {sig['p_gap_le_0']:.3f}")


def plot_group_auc(name, tbl, overall_auc, out_path):
    t = tbl.sort_values("auc")
    err = [t["auc"].values - t["auc_lo"].values,
           t["auc_hi"].values - t["auc"].values]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(t["group"], t["auc"], color="#2d6cdf", alpha=0.85,
            xerr=err, error_kw={"ecolor": "#444", "capsize": 4})
    ax.axvline(overall_auc, color="#333", linestyle="--",
               label=f"Overall AUC {overall_auc:.3f}")
    ax.set_xlabel("AUC within group (bars = bootstrap 95% CI)")
    ax.set_xlim(0.5, max(0.8, t["auc_hi"].max() + 0.05))
    ax.set_title(f"How well the model separates burnout, by {name}")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] Saved: {out_path}")
