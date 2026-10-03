"""Project-wide configuration and constants.

One place for paths, the random seed, and the two feature layers whose
incremental value and interpretation this project studies.
"""

from pathlib import Path

RANDOM_STATE = 42

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
OUTPUT_DIR = ROOT / "outputs"

# Baseline layer: demographic and structural attributes an organization
# knows about an employee regardless of how they experience the job.
BASELINE_FEATURES = [
    "Age", "Gender", "MaritalStatus", "DistanceFromHome",
    "MonthlyIncome", "Education", "Department", "JobRole",
]

# Experience layer: how the employee experiences the job. This is the
# added layer whose incremental value we test and then interpret.
EXPERIENCE_FEATURES = [
    "JobSatisfaction", "EnvironmentSatisfaction", "WorkLifeBalance",
    "JobInvolvement", "OverTime", "RelationshipSatisfaction",
    "TrainingTimesLastYear", "StockOptionLevel",
]

TARGET = "Attrition"
