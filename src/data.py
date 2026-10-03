"""Data loading and feature typing for the real IBM HR Attrition dataset."""

from pathlib import Path

import pandas as pd

from .config import DATA_PATH, TARGET


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the dataset and add a binary target column ``y`` (1 = left)."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place the IBM HR Attrition CSV "
            f"in the data/ folder (see README)."
        )
    df = pd.read_csv(path)
    df["y"] = (df[TARGET] == "Yes").astype(int)
    return df


def split_feature_types(df: pd.DataFrame, cols: list[str]
                        ) -> tuple[list[str], list[str]]:
    """Split a column list into (numeric, categorical).

    Uses pandas' own numeric-dtype check rather than comparing to the
    string ``"object"``, which is not reliable across pandas versions.
    """
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
    return numeric, categorical
