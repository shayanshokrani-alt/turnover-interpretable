"""Experience vs. demographics as predictors of employee turnover.

Does what an employee has lived through at work add predictive value beyond
who they are demographically, which experience factors still matter once
demographics are held constant, and can the model's probabilities be trusted?

Run with:  python main.py
"""

from src.config import DEMOGRAPHIC_FEATURES, EXPERIENCE_FEATURES, OUTPUT_DIR
from src.data import load_data
from src.analysis import incremental_value, experience_effects, calibration
from src.report import (print_incremental, print_effects, print_calibration,
                        plot_incremental, plot_effects, plot_calibration)


def main():
    print("Experience vs. Demographics in Turnover - IBM HR Attrition")
    print("-" * 68)

    df = load_data()
    print(f"[data] {df.shape[0]} employees, attrition rate {df['y'].mean():.1%}")

    inc = incremental_value(df, DEMOGRAPHIC_FEATURES, EXPERIENCE_FEATURES)
    print_incremental(inc)

    eff = experience_effects(df, DEMOGRAPHIC_FEATURES, EXPERIENCE_FEATURES)
    print_effects(eff)

    cal = calibration(df, DEMOGRAPHIC_FEATURES, EXPERIENCE_FEATURES)
    print_calibration(cal)

    plot_incremental(inc, OUTPUT_DIR / "incremental_value.png")
    plot_effects(eff, OUTPUT_DIR / "experience_effects.png")
    plot_calibration(cal, OUTPUT_DIR / "calibration.png")

    eff.to_csv(OUTPUT_DIR / "experience_effects.csv", index=False)
    print(f"[table] Saved: {OUTPUT_DIR / 'experience_effects.csv'}")


if __name__ == "__main__":
    main()
