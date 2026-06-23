"""
Preprocessing helpers for the Salary Prediction app.

NOTE: The heavy preprocessing (imputation, scaling, one-hot encoding) is
already embedded inside the exported scikit-learn Pipeline (`salary_model.pkl`),
which was fit during training in the notebook. This module is responsible for:

1. Loading feature metadata (valid categories, numeric ranges) used to build
   the Streamlit UI dynamically.
2. Validating and shaping raw user input into the exact DataFrame format the
   pipeline expects before calling `.predict()`.

Keeping this separate from `predict.py` mirrors the suggested project
structure and makes it easy to unit test input validation independently of
model inference.
"""

import json
import os

import pandas as pd

_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
_METADATA_PATH = os.path.join(_MODEL_DIR, "metadata.json")


def load_metadata() -> dict:
    """Load feature metadata (expected columns, categorical options, numeric ranges)."""
    with open(_METADATA_PATH, "r") as f:
        return json.load(f)


def build_input_dataframe(raw_input: dict, metadata: dict) -> pd.DataFrame:
    """
    Convert a dict of raw user input (e.g., from Streamlit widgets) into a
    single-row pandas DataFrame with the exact column names and order the
    trained pipeline expects.

    Parameters
    ----------
    raw_input : dict
        Keys must match the feature names used during training, e.g.
        {"Age": 30, "Years of Experience": 5.0, "Gender": "Female",
         "Education Level": "Master's", "Job Title": "Data Scientist",
         "Location": "Remote"}
    metadata : dict
        The metadata dict produced by `load_metadata()`.

    Returns
    -------
    pd.DataFrame
        A single-row DataFrame ready to be passed to `model.predict()`.
    """
    expected_columns = metadata["numeric_features"] + metadata["categorical_features"]

    missing = [col for col in expected_columns if col not in raw_input]
    if missing:
        raise ValueError(f"Missing required input fields: {missing}")

    row = {col: raw_input[col] for col in expected_columns}
    return pd.DataFrame([row])


def validate_numeric_ranges(raw_input: dict, metadata: dict) -> list:
    """
    Soft-validate numeric inputs against the ranges observed in training data.
    Returns a list of human-readable warning strings (empty list if all good).
    This does not block prediction -- it only warns the user that the model
    is extrapolating outside its training distribution.
    """
    warnings = []
    for feature, (lo, hi) in metadata.get("numeric_ranges", {}).items():
        if feature in raw_input:
            value = raw_input[feature]
            if value < lo or value > hi:
                warnings.append(
                    f"'{feature}' = {value} is outside the training data range "
                    f"({lo:.1f} - {hi:.1f}). Prediction may be less reliable."
                )
    return warnings
