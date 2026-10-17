"""Interpretable model: penalized logistic regression over a leakage-safe
column transformer (scaling plus one-hot encoding, refit inside each fold)."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import RANDOM_STATE
from .data import split_feature_types


def build_pipeline(df: pd.DataFrame, cols: list[str]) -> Pipeline:
    numeric, categorical = split_feature_types(df, cols)
    # Coded categoricals are strings so the encoder treats each code as a level.
    pre = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"),
         categorical),
    ])
    return Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)),
    ])


def as_str_categories(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Return a copy where categorical codes are strings (so 1 and 1.0 do not
    split into two levels), numeric columns left as numbers."""
    numeric, categorical = split_feature_types(df, cols)
    out = df[cols].copy()
    for c in categorical:
        out[c] = out[c].astype("Int64").astype(str)
    for c in numeric:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def get_feature_names(pipe: Pipeline) -> list[str]:
    return [n.split("__", 1)[-1]
            for n in pipe.named_steps["pre"].get_feature_names_out()]
