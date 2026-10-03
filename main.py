"""Interpretable turnover prediction on the IBM HR Attrition dataset.

Question: does how an employee *experiences* the job add predictive value
over who they are on paper -- which factors drive attrition, how stable are
they, and are the model's probabilities trustworthy?

Run with:
    python main.py
"""

from src.config import BASELINE_FEATURES, EXPERIENCE_FEATURES, OUTPUT_DIR
from src.data import load_data
from src.analysis import incremental_value, coefficient_stability, calibration
from src.report import (
    print_incremental, print_top_features, print_calibration,
    plot_incremental, plot_feature_effects, plot_calibration,
)


def main() -> None:
    print("Interpretable Turnover Prediction - IBM HR Attrition")
    print("-" * 70)

    df = load_data()
    print(f"[data] {df.shape[0]} employees, "
          f"attrition rate {df['y'].mean():.1%}")

    full = BASELINE_FEATURES + EXPERIENCE_FEATURES

    # 1. Incremental value, with significance test and CIs.
    inc = incremental_value(df, BASELINE_FEATURES, EXPERIENCE_FEATURES)
    print_incremental(inc)

    # 2. & 3. Feature effects (odds ratios + CI) and stability.
    table = coefficient_stability(df, full)
    print_top_features(table, n=10)

    # 4. Calibration.
    cal = calibration(df, full)
    print_calibration(cal)

    # Figures.
    plot_incremental(inc, OUTPUT_DIR / "incremental_value.png")
    plot_feature_effects(table, OUTPUT_DIR / "feature_effects.png", n=10)
    plot_calibration(cal, OUTPUT_DIR / "calibration.png")

    # Full coefficient table for the write-up.
    out_csv = OUTPUT_DIR / "coefficient_stability.csv"
    table.to_csv(out_csv, index=False)
    print(f"[table] Saved: {out_csv}")


if __name__ == "__main__":
    main()
