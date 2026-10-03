"""Console reporting and figures."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _fmt_ci(ci):
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def print_incremental(r: dict) -> None:
    print("\n" + "=" * 70)
    print("Incremental value of the experience layer")
    print("=" * 70)
    print(f"  Baseline   AUC  : {r['baseline_auc'][0]:.3f}  "
          f"95% CI {_fmt_ci(r['baseline_auc'][1])}")
    print(f"  Augmented  AUC  : {r['augmented_auc'][0]:.3f}  "
          f"95% CI {_fmt_ci(r['augmented_auc'][1])}")
    print(f"  AUC gain        : {r['auc_gain']:+.3f}  "
          f"95% CI {_fmt_ci(r['auc_gain_ci'])}")
    print(f"  Paired t-test   : t={r['ttest_t']:.2f}, p={r['ttest_p']:.4f}")
    print("-" * 70)
    print(f"  Baseline   PR-AUC: {r['baseline_prauc'][0]:.3f}  "
          f"95% CI {_fmt_ci(r['baseline_prauc'][1])}")
    print(f"  Augmented  PR-AUC: {r['augmented_prauc'][0]:.3f}  "
          f"95% CI {_fmt_ci(r['augmented_prauc'][1])}")
    print(f"  (no-skill PR-AUC = positive rate = {r['positive_rate']:.3f})")
    print("=" * 70)


def print_top_features(table: pd.DataFrame, n: int = 10) -> None:
    print(f"\nTop {n} features by effect size (odds ratio, 95% CI)")
    print("-" * 70)
    print(f"{'feature':<32}{'OR':>7}{'95% CI':>18}{'sign':>8}")
    print("-" * 70)
    for _, r in table.head(n).iterrows():
        ci = f"[{r['or_ci_low']:.2f},{r['or_ci_high']:.2f}]"
        arrow = "up" if r["odds_ratio"] > 1 else "dn"
        print(f"{r['feature']:<32}{r['odds_ratio']:>7.2f}{ci:>18}"
              f"{r['sign_consistency']:>7.0%} {arrow}")
    print("-" * 70)
    print("OR > 1 raises attrition risk. Numeric features are standardized,")
    print("so their OR is per one standard deviation of change.")


def print_calibration(c: dict) -> None:
    print(f"\nCalibration: Brier score = {c['brier']:.4f} "
          f"(lower is better; 0 = perfect)")


def plot_incremental(r: dict, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    labels = ["Baseline", "Baseline +\nexperience"]
    vals = [r["baseline_auc"][0], r["augmented_auc"][0]]
    cis = [r["baseline_auc"][1], r["augmented_auc"][1]]
    err = [[v - c[0] for v, c in zip(vals, cis)],
           [c[1] - v for v, c in zip(vals, cis)]]
    ax.bar(labels, vals, yerr=err, color=["#9aa5b1", "#2d6cdf"], width=0.55,
           error_kw={"ecolor": "#333", "capsize": 5})
    ax.set_ylabel("AUC")
    ax.set_ylim(0.5, max(vals) + 0.1)
    ax.set_title("Incremental value of the experience layer\n"
                 "(error bars = 95% CI across folds)")
    ax.grid(axis="y", alpha=0.3)
    ax.annotate(f"+{r['auc_gain']:.3f}\n(p={r['ttest_p']:.3f})",
                xy=(1, vals[1] + 0.02), ha="center",
                fontsize=10, fontweight="bold", color="#2d6cdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[figure] Saved: {out_path}")
    return out_path


def plot_feature_effects(table: pd.DataFrame, out_path: Path,
                         n: int = 10) -> Path:
    top = table.head(n).iloc[::-1]
    colors = ["#c0392b" if c > 0 else "#2d6cdf" for c in top["coef_mean"]]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["coef_mean"], xerr=top["coef_std"],
            color=colors, alpha=0.85,
            error_kw={"ecolor": "#555", "capsize": 3})
    ax.axvline(0, color="#333", linewidth=0.8)
    ax.set_xlabel("Logistic-regression coefficient (log-odds)")
    ax.set_title("What drives attrition, and how stable each effect is\n"
                 "(red = raises risk, blue = lowers; bars = SD across folds)")
    ax.grid(axis="x", alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] Saved: {out_path}")
    return out_path


def plot_calibration(c: dict, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="#999", label="Perfect calibration")
    ax.plot(c["mean_pred"], c["frac_pos"], "o-", color="#2d6cdf",
            label=f"Model (Brier={c['brier']:.3f})")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration curve (out-of-fold)")
    ax.legend()
    ax.grid(alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] Saved: {out_path}")
    return out_path
