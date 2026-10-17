"""Data loading and typing for the NSSRN 2022 public use file.

We keep nurses who answered the burnout item (employed RNs), build the binary
target, and derive a three-level age band used only for group reporting.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from .config import DATA_PATH, TARGET, MODEL_FEATURES


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the NSSRN .dta, build target y, and return the analytic sample."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download the NSSRN 2022 public use "
            f"file (Stata format) and place it in the data/ folder (see README)."
        )
    df = pd.read_stata(path, convert_categoricals=False)
    df.columns = [c.lower() for c in df.columns]

    # Target: 1 = Yes (burned out), 2 = No. Keep only those two, drop skips.
    df = df[df[TARGET].isin([1, 2])].copy()
    df["y"] = (df[TARGET] == 1).astype(int)

    # Age band for group reporting (the model still uses continuous age).
    age = pd.to_numeric(df["age_puf"], errors="coerce")
    df["age_band"] = np.select(
        [age < 40, (age >= 40) & (age < 55), age >= 55],
        [0, 1, 2], default=np.nan,
    )

    # Drop rows with missing model features so every analysis uses one sample.
    df = df.dropna(subset=[c for c in MODEL_FEATURES if c in df.columns])
    return df.reset_index(drop=True)


def split_feature_types(df: pd.DataFrame, cols: list[str]
                        ) -> tuple[list[str], list[str]]:
    """Numeric vs. categorical split. Here age and hours are numeric; the
    coded categorical columns are treated as categories, not numbers."""
    numeric = [c for c in cols if c in ("age_puf", "pn_hrs_wrk_puf")]
    categorical = [c for c in cols if c not in numeric]
    return numeric, categorical
