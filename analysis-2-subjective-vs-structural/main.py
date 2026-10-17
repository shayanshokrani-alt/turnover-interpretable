"""Subjective vs. structural predictors of employee turnover.

Does how an employee feels about work add predictive value beyond the
objective facts of their job, and which feelings still matter once those
facts are held constant?

Run with:  python main.py
"""

from src.config import STRUCTURAL_FEATURES, SUBJECTIVE_FEATURES, OUTPUT_DIR
from src.data import load_data
from src.analysis import incremental_value, subjective_effects
from src.report import (print_incremental, print_effects,
                        plot_incremental, plot_effects)


def main():
    print("Subjective vs. Structural Predictors of Turnover - IBM HR Attrition")
    print("-" * 68)

    df = load_data()
    print(f"[data] {df.shape[0]} employees, attrition rate {df['y'].mean():.1%}")

    inc = incremental_value(df, STRUCTURAL_FEATURES, SUBJECTIVE_FEATURES)
    print_incremental(inc)

    eff = subjective_effects(df, STRUCTURAL_FEATURES, SUBJECTIVE_FEATURES)
    print_effects(eff)

    plot_incremental(inc, OUTPUT_DIR / "incremental_value.png")
    plot_effects(eff, OUTPUT_DIR / "subjective_effects.png")

    eff.to_csv(OUTPUT_DIR / "subjective_effects.csv", index=False)
    print(f"[table] Saved: {OUTPUT_DIR / 'subjective_effects.csv'}")


if __name__ == "__main__":
    main()
