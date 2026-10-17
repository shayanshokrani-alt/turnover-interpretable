"""Configuration: paths, seed, and the two feature layers.

The question: does an employee's subjective experience of work add predictive
value for turnover beyond the structural facts of their job?
"""

from pathlib import Path

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
OUTPUT_DIR = ROOT / "outputs"

# Structural layer: objective facts of the job, knowable from HR records alone.
STRUCTURAL_FEATURES = [
    "JobRole", "JobLevel", "Department",
    "OverTime", "BusinessTravel", "MonthlyIncome",
]

# Subjective layer: how the employee experiences the job (self-reported survey
# scales, 1-4). This is the layer whose added value we test.
SUBJECTIVE_FEATURES = [
    "JobSatisfaction", "EnvironmentSatisfaction", "RelationshipSatisfaction",
    "JobInvolvement", "WorkLifeBalance",
]

TARGET = "Attrition"
