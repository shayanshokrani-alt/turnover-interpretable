# Is a Nurse-Burnout Model Fair Across Groups of Nurses?

A people-analytics mini-project on real national nursing-workforce data. The
third of three companion analyses in this repository, and the one that moves
from "can we predict?" to "is the prediction fair, and what does fairness cost?"

**Data:** the 2022 [National Sample Survey of Registered Nurses](https://data.hrsa.gov/topics/health-workforce/nursing-workforce-survey-data)
(NSSRN), a public-use file from the US Health Resources and Services
Administration. 32,647 employed registered nurses after keeping those who
answered the burnout item.

**Question:** a logistic model predicts nurse burnout reasonably well on
average. Does it predict equally well for every group of nurses (sex,
race/ethnicity, age), and if we force it to be fair, how much accuracy is given
up?

**Headline finding:** the overall model reaches AUC 0.705 (burnout base rate
81.5%, out-of-fold Brier 0.136), but it is not equally good for everyone. It
separates burnout noticeably worse for Black (AUC 0.640) and Asian (0.648)
nurses than for White nurses (0.711). This gap is confirmed real, not an
artefact of the smaller group sizes: the bootstrap 95% intervals do not overlap,
and a direct test puts the White-minus-Black gap at 0.071 (95% CI 0.038 to
0.103, one-sided p < 0.001). By age the error rates diverge most: for nurses
under 55 the model flags almost everyone (false-positive rate near 1.0), while
for nurses 55 and over it discriminates far better, a false-positive-rate gap
of 0.19.

Group-aware thresholds then help unevenly, and the analysis reports that rather
than claiming a general fix. For age, they narrow the held-out
true-positive-rate gap from 0.053 to 0.009 as a fair trade: the younger groups'
false-positive rate improves too (from about 1.0 to 0.85), at a cost of 2 to 3
points of within-group accuracy, while the oldest group is left unchanged
because it set the target. For race the same thresholds barely move the gap,
and for sex the gap was small to begin with. So the honest claim is narrow:
threshold adjustment closes the age gap out of sample and cheaply, but the
racial gap is a different problem this tool does not solve.

## What it does

1. **Overall model:** penalized logistic regression in a leakage-safe pipeline,
   5-fold stratified cross-validation, with AUC, PR-AUC, calibration (Brier),
   and odds ratios with confidence intervals and sign-stability, matching the
   two companion analyses.
2. **Fairness audit:** for each protected group, per-group AUC (with a bootstrap
   95% confidence interval), selection rate, true-positive rate, false-positive
   rate, and calibration, all on out-of-fold predictions so no nurse is scored
   by a model trained on them. A bootstrap test then checks whether the widest
   AUC gap is real or sampling noise. The spread in true-positive rate across
   groups is the equal-opportunity gap.
3. **The fairness-accuracy trade-off:** group-aware thresholds that equalize the
   true-positive rate, and the overall accuracy given up to close the gap. The
   thresholds are tuned on one half of the held-out predictions, and both the
   remaining true-positive-rate gap and the accuracy cost are measured on the
   other, untouched half, averaged over 20 random splits. So the reported
   fairness is a genuine out-of-sample result, not a condition that only held
   while the thresholds were being chosen.

## Why this matters

Tools that predict burnout or attrition are sold on overall accuracy. This
analysis shows that an accurate model can quietly work less well for some
groups of nurses, measures that hidden cost, and quantifies the small price of
correcting it. That is the value beyond a leaderboard number.

## Running it

```
python -m venv .venv
.venv\Scripts\activate          # Windows (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
python main.py
```

Download the NSSRN 2022 public use file in Stata format from the HRSA link
above and place `nssrn_2022_puf.dta` in the data/ folder first.

## Outputs

- `audit_<group>.png`, true- and false-positive rates by group.
- `groupauc_<group>.png`, AUC within each group against the overall AUC.
- `audit_<group>.csv` and `coefficient_effects.csv`, the full tables.

## Limitations

A methods demonstration on public, cross-sectional, self-reported data from US
nurses, not a causal claim and not a statement about any individual. The high
burnout base rate (82%) makes separation hard, which is itself part of the
story. Group definitions follow the survey's categories; small groups are
omitted. Fairness has many definitions; this uses equal opportunity and
calibration, and does not claim they are the only ones that matter. Stated
plainly rather than worked around.
