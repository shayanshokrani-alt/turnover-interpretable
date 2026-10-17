"""Console report and figures."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ci(c):
    return f"[{c[0]:.3f}, {c[1]:.3f}]"


def print_incremental(r):
    print("\n" + "=" * 68)
    print("Does subjective experience add value beyond structural facts?")
    print("=" * 68)
    print(f"  Structural only         AUC {r['structural_auc'][0]:.3f}  "
          f"95% CI {_ci(r['structural_auc'][1])}")
    print(f"  Structural + subjective AUC {r['full_auc'][0]:.3f}  "
          f"95% CI {_ci(r['full_auc'][1])}")
    print(f"  Added value             {r['gain']:+.3f}  "
          f"95% CI {_ci(r['gain_ci'])}")
    print(f"  Paired t-test           t={r['t']:.2f}, p={r['p']:.4f}")
    print("=" * 68)


def print_effects(tbl):
    print("\nSubjective factors, holding structure constant (odds ratios)")
    print("-" * 68)
    print(f"{'factor':<28}{'OR':>7}{'95% CI':>18}{'sign':>8}")
    print("-" * 68)
    for _, r in tbl.iterrows():
        ci = f"[{r['or_ci_low']:.2f},{r['or_ci_high']:.2f}]"
        print(f"{r['feature']:<28}{r['odds_ratio']:>7.2f}{ci:>18}"
              f"{r['sign_consistency']:>7.0%}")
    print("-" * 68)
    print("OR < 1 means higher on that scale lowers attrition risk. All effects")
    print("are per one standard deviation, holding the structural layer fixed.")


def plot_incremental(r, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    vals = [r["structural_auc"][0], r["full_auc"][0]]
    cis = [r["structural_auc"][1], r["full_auc"][1]]
    err = [[v - c[0] for v, c in zip(vals, cis)],
           [c[1] - v for v, c in zip(vals, cis)]]
    ax.bar(["Structural\nonly", "Structural +\nsubjective"], vals, yerr=err,
           color=["#9aa5b1", "#2d6cdf"], width=0.55,
           error_kw={"ecolor": "#333", "capsize": 5})
    ax.set_ylabel("AUC")
    ax.set_ylim(0.5, max(vals) + 0.1)
    ax.set_title("Subjective experience adds value beyond structural facts\n"
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
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(t["feature"], t["coef_mean"], xerr=t["coef_std"],
            color="#2d6cdf", alpha=0.85,
            error_kw={"ecolor": "#555", "capsize": 3})
    ax.axvline(0, color="#333", linewidth=0.8)
    ax.set_xlabel("Coefficient (log-odds), holding structure constant")
    ax.set_title("Subjective factors still predict attrition\n"
                 "after controlling for job structure (bars = SD across folds)")
    ax.grid(axis="x", alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] Saved: {out_path}")
