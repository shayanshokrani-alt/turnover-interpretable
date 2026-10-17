# Experience vs. Demographics in Predicting Turnover

A people-analytics mini-project on the IBM HR Attrition dataset. One of two
companion analyses in this repository.

**Question:** Does what an employee has lived through at work (tenure,
progression, overtime, travel) predict turnover beyond who they are
demographically (age, gender, marital status, education), and which experience
factors still matter once demographics are held constant?

**Headline finding:** Experience dominates. Adding the experience layer raises
cross-validated AUC from 0.655 to 0.781 (gain 0.126, 95% CI 0.088 to 0.164,
p = 0.003). Holding demographics constant, overtime and frequent travel sharply
raise the odds of leaving, while longer tenure in role and more total working
years lower them, each with 100% sign-stability across folds. The model is also
well calibrated (out-of-fold Brier score 0.110), so its predicted probabilities
can be read as probabilities, not just as a ranking. Turnover here is driven by
the conditions of the job, which an employer can act on, far more than by
demographics, which it cannot.

## What it does

1. Incremental value: demographic-only vs. demographic plus experience, by
   cross-validated AUC, with a paired t-test and 95% confidence intervals.
2. Interpretation: odds ratios (with CIs) of the experience factors in the
   full model, i.e. their effect holding demographics constant.
3. Stability: sign-consistency of every coefficient across folds.
4. Calibration: a reliability curve and Brier score on out-of-fold
   predictions, testing whether the probabilities are trustworthy.

All preprocessing lives inside the pipeline, refit on each fold, so there is no
leakage into the evaluation.

## Running it

```
python -m venv .venv
.venv\Scripts\activate          # Windows (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
python main.py
```

Place the public [IBM HR Analytics Employee Attrition](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
CSV in the data/ folder first.

## Outputs

- `incremental_value.png`, bar chart with 95% CI error bars.
- `experience_effects.png`, coefficients of experience factors holding demographics constant.
- `calibration.png`, reliability curve for the full model.
- `experience_effects.csv`, full table (odds ratio, CI, sign-consistency).

## Limitations

A methods demonstration on public data, not a novel finding. No time axis
(cross-sectional); a single public dataset; association, not causation. All
stated plainly rather than worked around.
