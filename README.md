# Interpretable Turnover and Workforce Analysis

Author: Shayan Shokrani

Three small, self-contained people-analytics studies. The first two use the
public [IBM HR Analytics Employee Attrition](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
dataset (1,470 employees, 16.1% attrition); the third uses real national
nursing-workforce data (NSSRN 2022, 32,647 nurses). All favour an interpretable
model (penalized logistic regression) and honest evaluation (preprocessing
refit inside every cross-validation fold, so nothing leaks into the score) over
raw predictive performance, and all report effect sizes, confidence intervals,
calibration, and stability rather than a single headline number.

## The three analyses

### [1. Experience vs. demographics](analysis-1-experience-vs-demographic/)

Does what an employee has lived through at work (tenure, progression, overtime,
travel) predict turnover beyond who they are demographically?

Experience dominates: AUC rises from 0.655 to 0.781 (gain 0.126, p = 0.003).
Holding demographics constant, overtime and frequent travel sharply raise the
odds of leaving while longer tenure lowers them, each 100% sign-stable across
folds. The model is well calibrated (out-of-fold Brier 0.110). Turnover is
driven by job conditions the employer can change far more than by demographics
it cannot.

### [2. Subjective experience vs. structural facts](analysis-2-subjective-vs-structural/)

Does how an employee feels about work (satisfaction, work-life balance,
involvement) add predictive value beyond the objective facts of their job?

Yes: AUC rises from 0.748 to 0.792 (gain 0.044, p = 0.01). Holding the full
structural layer constant, all five subjective factors still lower the odds of
leaving, each 100% sign-stable across folds. Subjective experience is an
independent layer of information, not a restatement of job structure.

### [3. Is a nurse-burnout model fair across groups?](analysis-3-fairness-nurse-burnout/)

Moving from prediction to fairness, on a real health-workforce crisis. A model
predicts burnout among registered nurses at AUC 0.705 (base rate 82%), but it
is not equally accurate for everyone: it separates burnout worse for Black
(0.640) and Asian (0.648) nurses than White (0.711), and its error rates
diverge most by age (false-positive-rate gap 0.19). Equalizing opportunity
across age groups costs only 1.7 points of accuracy. An accurate-on-average
model can still be unequal, and here the inequality is measurable and cheap to
reduce.

## Shared method

- Penalized logistic regression inside a leakage-safe scikit-learn pipeline
  (scaling and one-hot encoding refit on each fold).
- 5-fold stratified cross-validation; out-of-fold predictions for every
  evaluation, so no record is scored by a model trained on it.
- Interpretation by odds ratios with CIs and sign-consistency across folds;
  analyses 1 and 2 add an incremental-value t-test, analysis 3 adds a
  per-group fairness audit and a fairness-accuracy trade-off.

## Running any analysis

Each folder is independent. From inside it:

```
python -m venv .venv
.venv\Scripts\activate          # Windows (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
python main.py
```

Place that analysis's dataset in its data/ folder first (see its README). No
dataset is committed here, due to licence and size.

## Limitations

Methods demonstrations on public, cross-sectional data, not novel findings.
Self-reported scales share a source with the outcome; association, not
causation; each uses a single public dataset. Stated plainly rather than worked
around.
