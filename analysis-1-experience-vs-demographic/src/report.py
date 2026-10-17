"""Console report and figures."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ci(c):
    return f"[{c[0]:.3f}, {c[1]:.3f}]"


def print_incremental(r):
    print("\n" + "=" * 68)
    print("Does work experience add value beyond demographics?")
    print("=" * 68)
    print(f"  Demographic only          AUC {r['demographic_auc'][0]:.3f}  "
          f"95% CI {_ci(r['demographic_auc'][1])}")
    print(f"  Demographic + experience  AUC {r['full_auc'][0]:.3f}  "
          f"95% CI {_ci(r['full_auc'][1])}")
    print(f"  Added value               {r['gain']:+.3f}  "
          f"95% CI {_ci(r['gain_ci'])}")
    print(f"  Paired t-test             t={r['t']:.2f}, p={r['p']:.4f}")
    print("=" * 68)


def print_effects(tbl):
    print("\nExperience factors, holding demographics constant (odds ratios)")
    print("-" * 68)
    print(f"{'factor':<30}{'OR':>7}{'95% CI':>18}{'sign':>8}")
    print("-" * 68)
    for _, r in tbl.iterrows():
        ci = f"[{r['or_ci_low']:.2f},{r['or_ci_high']:.2f}]"
        print(f"{r['feature']:<30}{r['odds_ratio']:>7.2f}{ci:>18}"
              f"{r['sign_consistency']:>7.0%}")
    print("-" * 68)
    print("OR < 1 lowers attrition risk, OR > 1 raises it. Numeric effects are")
    print("per one standard deviation, holding the demographic layer fixed.")


def print_calibration(cal):
    print("\nCalibration of the full model (are the probabilities honest?)")
    print("-" * 68)
    print(f"  Brier score  {cal['brier']:.4f}   (lower is better; out-of-fold)")
    print("-" * 68)


def plot_incremental(r, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    vals = [r["demographic_auc"][0], r["full_auc"][0]]
    cis = [r["demographic_auc"][1], r["full_auc"][1]]
    err = [[v - c[0] for v, c in zip(vals, cis)],
           [c[1] - v for v, c in zip(vals, cis)]]
    ax.bar(["Demographic\nonly", "Demographic +\nexperience"], vals, yerr=err,
           color=["#9aa5b1", "#2d6cdf"], width=0.55,
           error_kw={"ecolor": "#333", "capsize": 5})
    ax.set_ylabel("AUC")
    ax.set_ylim(0.5, max(vals) + 0.1)
    ax.set_title("Work experience adds value beyond demographics\n"
                 "(error bars = 95% CI across folds)")
    ax.grid(axis="y", alpha=0.3)
    ax.annotate(f"+{r['gain']:.3f}\n(p={r['p']:.3f})",
                xy=(1, vals[1] + 0.02), ha="center",
                fontsize=10, fontweight="bold", color="#2d6cdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[figure] Saved: {out_path}")


def plot_effects(tbl, out_path):
    t = tbl.iloc[::-1]
    colors = ["#c0392b" if c > 0 else "#2d6cdf" for c in t["coef_mean"]]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(t["feature"], t["coef_mean"], xerr=t["coef_std"],
            color=colors, alpha=0.85,
            error_kw={"ecolor": "#555", "capsize": 3})
    ax.axvline(0, color="#333", linewidth=0.8)
    ax.set_xlabel("Coefficient (log-odds), holding demographics constant")
    ax.set_title("Which experience factors predict attrition\n"
                 "after controlling for demographics (bars = SD across folds)")
    ax.grid(axis="x", alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] Saved: {out_path}")


def plot_calibration(cal, out_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="#999", label="Perfect calibration")
    ax.plot(cal["mean_pred"], cal["frac_pos"], "o-", color="#2d6cdf",
            label=f"Model (Brier {cal['brier']:.3f})")
    ax.set_xlabel("Mean predicted probability of leaving")
    ax.set_ylabel("Observed fraction who left")
    ax.set_title("Reliability curve (out-of-fold)\n"
                 "close to the diagonal means the probabilities can be trusted")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] Saved: {out_path}")
