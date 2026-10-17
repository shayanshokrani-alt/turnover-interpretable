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
    pre = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"),
         categorical),
    ])
    return Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)),
    ])


def get_feature_names(pipe: Pipeline) -> list[str]:
    return [n.split("__", 1)[-1]
            for n in pipe.named_steps["pre"].get_feature_names_out()]
