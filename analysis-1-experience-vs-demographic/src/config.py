"""Configuration: paths, seed, and the two feature layers.

The question: does what an employee has lived through at work (tenure,
progression, overtime, travel) predict turnover beyond who they are
demographically (age, gender, marital status, education)?
"""

from pathlib import Path

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
OUTPUT_DIR = ROOT / "outputs"

# Demographic layer: who the person is, largely fixed and outside the
# employer's control.
DEMOGRAPHIC_FEATURES = [
    "Age", "Gender", "MaritalStatus", "Education", "EducationField",
]

# Experience layer: what the job has involved, things the employer can act on.
# This is the layer whose added value we test.
EXPERIENCE_FEATURES = [
    "TotalWorkingYears", "YearsAtCompany", "YearsInCurrentRole",
    "YearsSinceLastPromotion", "NumCompaniesWorked",
    "OverTime", "BusinessTravel", "TrainingTimesLastYear",
]

TARGET = "Attrition"
