"""Fairness of a nurse-burnout prediction model.

A logistic model predicts burnout among employed registered nurses. It is
accurate overall, but this analysis asks whether it is equally accurate across
groups of nurses (sex, race/ethnicity, age), and what equalizing it costs.

Run with:  python main.py
"""

from src.config import MODEL_FEATURES, PROTECTED, OUTPUT_DIR
from src.data import load_data
from src.analysis import (oof_predictions, overall_performance,
                          coefficient_effects, fairness_audit,
                          equal_opportunity_thresholds, before_after_table,
                          auc_gap_significance)
from src.report import (print_overall, print_audit, print_tradeoff,
                        print_before_after, print_group_auc,
                        print_auc_gap_test, plot_audit, plot_group_auc,
                        plot_before_after)


def main():
    print("Fairness in Nurse Burnout Prediction - NSSRN 2022")
    print("-" * 70)

    df = load_data()
    oof = oof_predictions(df, MODEL_FEATURES)

    overall = overall_performance(df, MODEL_FEATURES, oof)
    print_overall(overall)

    eff = coefficient_effects(df, MODEL_FEATURES)
    eff.to_csv(OUTPUT_DIR / "coefficient_effects.csv", index=False)

    for attr, labels in PROTECTED.items():
        name = {"sex": "sex", "rac_ethn_puf": "race/ethnicity",
                "age_band": "age"}[attr]
        tbl, gaps = fairness_audit(df, oof, attr, labels)
        print_audit(name, tbl, gaps)
        print_group_auc(name, tbl)

        # Is the widest AUC gap real, or sampling noise in a small group?
        ranked = tbl.sort_values("auc")
        if len(ranked) >= 2:
            sig = auc_gap_significance(df, oof, attr, labels,
                                       ranked.iloc[-1]["group"],
                                       ranked.iloc[0]["group"])
            print_auc_gap_test(sig)

        trade = equal_opportunity_thresholds(df, oof, attr, labels)
        print_tradeoff(name, trade)

        ba = before_after_table(df, oof, attr, labels)
        print_before_after(name, ba)

        safe = attr.replace("_puf", "")
        plot_audit(name, tbl, OUTPUT_DIR / f"audit_{safe}.png")
        plot_group_auc(name, tbl, overall["auc"],
                       OUTPUT_DIR / f"groupauc_{safe}.png")
        plot_before_after(name, ba, OUTPUT_DIR / f"beforeafter_{safe}.png")
        tbl.to_csv(OUTPUT_DIR / f"audit_{safe}.csv", index=False)
        ba.to_csv(OUTPUT_DIR / f"beforeafter_{safe}.csv", index=False)


if __name__ == "__main__":
    main()
