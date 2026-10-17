# Subjective Experience vs. Structural Facts in Predicting Turnover

A people-analytics mini-project on the IBM HR Attrition dataset. Companion to
the main interpretable-turnover analysis in this repository.

**Question:** Does how an employee feels about work (satisfaction, work-life
balance, involvement) add predictive value for turnover beyond the objective
facts of their job (role, level, pay, overtime), and which feelings still
matter once those facts are held constant?

**Headline finding:** Yes. Adding the subjective layer raises cross-validated
AUC from 0.748 to 0.792 (gain 0.044, 95% CI 0.024 to 0.064, p = 0.01). Holding
the full structural layer constant, all five subjective factors still lower the
odds of leaving, each with 100% sign-stability across folds. Subjective
experience is an independent layer of information, not a restatement of job
structure.

## What it does

1. Incremental value: structural-only vs. structural plus subjective, by
   cross-validated AUC, with a paired t-test and 95% confidence intervals.
2. Interpretation: odds ratios (with CIs) of the subjective factors in the
   full model, i.e. their effect holding structure constant.
3. Stability: sign-consistency of every coefficient across folds.

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
- `subjective_effects.png`, coefficients of subjective factors holding structure constant.
- `subjective_effects.csv`, full table (odds ratio, CI, sign-consistency).

## Limitations

A methods demonstration on public data, not a novel finding. No time axis
(cross-sectional); self-reported scales share a source with the outcome; a
single public dataset; association, not causation. All stated plainly rather
than worked around.
