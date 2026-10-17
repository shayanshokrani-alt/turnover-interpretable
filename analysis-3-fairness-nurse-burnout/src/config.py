"""Configuration: paths, seed, target, features, and protected groups.

The question: a model that predicts nurse burnout well on average, does it
predict equally well across groups of nurses, and what does fairness cost?
"""

from pathlib import Path

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "nssrn_2022_puf.dta"
OUTPUT_DIR = ROOT / "outputs"

# Target: "felt burned out" in the primary nursing position (employed RNs).
# Coded 1 = Yes, 2 = No in the source; we model P(burned out).
TARGET = "pn_burnout"

# Structural / job layer: objective facts of the job.
STRUCTURAL_FEATURES = [
    "pn_hrs_wrk_puf",       # weekly hours worked
    "pn_empset_comb_puf",   # employment setting (hospital / inpatient / ...)
    "pn_remote",            # worked 100% remotely
    "pn_travel",            # traveling nurse
]

# Demographic layer: used BOTH as model controls and, below, as the protected
# groups the fairness audit compares across.
DEMOGRAPHIC_FEATURES = [
    "age_puf",              # age in years
    "sex",                  # 1 male, 2 female
    "rac_ethn_puf",         # race/ethnicity (detailed)
]

MODEL_FEATURES = STRUCTURAL_FEATURES + DEMOGRAPHIC_FEATURES

# Protected attributes for the fairness audit, with readable group labels.
PROTECTED = {
    "sex": {1: "Male", 2: "Female"},
    "rac_ethn_puf": {
        1: "Hispanic", 2: "White", 3: "Black", 4: "Asian",
        5: "Am. Indian", 8: "Pacific Is. / Multiple",
    },
    "age_band": {0: "Under 40", 1: "40 to 54", 2: "55 and over"},
}

# Codes that mean "not applicable / legitimate skip" and should be dropped.
SKIP_CODES = {".L", ".l", ""}
