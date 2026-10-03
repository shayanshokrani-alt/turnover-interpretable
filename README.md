# Interpretable Turnover Prediction — IBM HR Attrition

A small, reproducible people-analytics project asking a focused question on
real data:

> Does how an employee *experiences* the job (satisfaction, work–life balance,
> overtime, involvement) add predictive value for turnover over who they are on
> paper (age, role, pay, distance)  and which of those experience factors are
> the **stable, trustworthy** drivers of attrition?

The emphasis is interpretability, not leaderboard accuracy. The model is a
penalized logistic regression whose coefficients map directly to odds ratios a
manager can read, and the project reports not just which features matter but how
**stable** each effect is across cross-validation folds because a coefficient
you can't reproduce is not something an organization should act on.

## What it does

1. **Incremental value** compares a baseline (demographic/structural) model
   against baseline + experience layer, by cross-validated AUC **and PR-AUC**
   (the classes are imbalanced at 16%), with a **paired t-test and 95%
   confidence intervals** so the gain is shown to be real, not just larger.
2. **Interpretation** fits the interpretable model and ranks features by
   effect size, as odds ratios **with 95% confidence intervals** and direction
   (raises/lowers risk). Numeric features are standardized, so their odds ratios
   are per one standard deviation.
3. **Stability**  refits on every fold and reports how often each coefficient
   keeps its sign, so the interpretation itself is quality-checked.
4. **Calibration**  checks (out-of-fold) whether predicted probabilities match
   observed frequencies, with a Brier score and calibration curve, so the
   outputs can be read as real probabilities.

All preprocessing lives inside the sklearn Pipeline, so scaling and encoding are
refit on training folds only no leakage into the evaluation. The categorical
reference category is the first level (alphabetical), dropped by the encoder.

## Method note

This mirrors the structure of interpretable-ML work in psychiatric genomics
(asking whether an added layer of information improves prediction of a human
outcome), transferred here to an organizational outcome. It connects to the
people-analytics line of work that uses interpretable ML to understand turnover.

## Project structure

```
turnover-interpretable/
├── main.py              # entry point run this
├── src/
│   ├── config.py        # paths, seed, the two feature layers
│   ├── data.py          # loading + numeric/categorical typing
│   ├── model.py         # interpretable logistic-regression pipeline
│   ├── analysis.py      # incremental value + coefficient stability
│   └── report.py        # console tables + figures
├── data/                # the IBM CSV lives here
├── outputs/             # figures + coefficient table land here
├── requirements.txt
└── .vscode/             # F5 runs main.py
```

## Running it (VS Code, Windows)

1. `File → Open Folder` → select this folder (the whole folder).
2. Terminal (`` Ctrl+` ``):
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   python main.py
   ```
   (or just press `F5`).

Figures are written to `outputs/`; the console prints the results tables.

## Data

The dataset is the public [IBM HR Analytics Employee Attrition](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
set (1,470 employees). It ships in `data/` here. It has no genuine time axis, so
this project does not attempt a time-aware split; that limitation is stated
rather than worked around.

## Outputs

- `incremental_value.png` — baseline vs. augmented AUC with 95% CI error bars.
- `feature_effects.png` — top drivers with cross-fold error bars.
- `calibration.png` — out-of-fold calibration curve with Brier score.
- `coefficient_stability.csv` — full table (odds ratio, CI, sign-consistency).
