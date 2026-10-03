"""Interpretable model definition.

A penalized logistic regression over a column transformer that scales
numeric features and one-hot-encodes categorical ones. The whole thing is a
single sklearn Pipeline, so every fold refits preprocessing on training data
only -- no leakage from the test fold into scaling or encoding.

Logistic regression is chosen deliberately: its coefficients are directly
interpretable as log-odds, which is the point of this project. A black-box
model would predict as well or better but would defeat the purpose.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import RANDOM_STATE
from .data import split_feature_types


def build_pipeline(df: pd.DataFrame, cols: list[str]) -> Pipeline:
    """Build an interpretable logistic-regression pipeline for ``cols``."""
    numeric, categorical = split_feature_types(df, cols)
    pre = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"),
         categorical),
    ])
    return Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ])


def get_feature_names(pipe: Pipeline) -> list[str]:
    """Return readable feature names after preprocessing, for interpretation."""
    return list(pipe.named_steps["pre"].get_feature_names_out())
