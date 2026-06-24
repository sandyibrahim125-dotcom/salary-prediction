"""
Inference logic for the Salary Prediction app.

Loads the trained scikit-learn pipeline (preprocessing + model bundled
together, as exported by the notebook) and exposes a single `inference()`
function that the Streamlit app calls.
"""

import os

import joblib
import numpy as np

from preprocess import build_input_dataframe, load_metadata, validate

_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH = os.path.join(_MODEL_DIR, "salary_model.pkl")

_model = None
_metadata = None


def _get_model():
    """Lazily load and cache the trained pipeline."""
    global _model
    if _model is None:
        _model = joblib.load(_MODEL_PATH)
    return _model


def _get_metadata():
    """Lazily load and cache feature metadata."""
    global _metadata
    if _metadata is None:
        _metadata = load_metadata()
    return _metadata


def _estimate_confidence_interval(model, X) -> tuple:
    """
    Estimate a rough prediction interval around the point prediction.

    For RandomForestRegressor, individual trees are independent estimators
    of the same target, so we use the spread of their predictions
    (10th-90th percentile) as an interval. GradientBoostingRegressor's trees
    are sequential/additive (each tree corrects the residual of the
    previous ones), so a single tree's prediction is not a standalone
    salary estimate -- for that model (and any other model type) we fall
    back to a symmetric margin based on held-out test RMSE.

    This is a simple heuristic, not a formal statistical confidence interval.
    """
    metadata = _get_metadata()
    rmse = metadata.get("test_rmse", None)

    underlying_model = model.named_steps.get("model")
    point_pred = float(model.predict(X)[0])

    # RandomForestRegressor: flat list of independently-trained trees in
    # `estimators_`, each one a full, independent predictor of the target.
    is_random_forest = (
        hasattr(underlying_model, "estimators_")
        and type(underlying_model).__name__ == "RandomForestRegressor"
    )

    if is_random_forest:
        preprocessor = model.named_steps["preprocessor"]
        X_transformed = preprocessor.transform(X)
        tree_preds = np.array([tree.predict(X_transformed) for tree in underlying_model.estimators_])
        lower = float(np.percentile(tree_preds, 10, axis=0)[0])
        upper = float(np.percentile(tree_preds, 90, axis=0)[0])
        return lower, upper

    # Fallback (e.g. GradientBoostingRegressor, LinearRegression, etc.):
    # symmetric margin based on held-out test RMSE.
    if rmse is not None:
        return point_pred - rmse, point_pred + rmse

    return None, None


def inference(data: dict) -> dict:
    """
    Run inference for a single employee record.

    Parameters
    ----------
    data : dict
        Raw feature values keyed by the same column names used in training,
        e.g.:
        {
            "Age": 30,
            "Years of Experience": 5.0,
            "Gender": "Female",
            "Education Level": "Master's",
            "Job Title": "Data Scientist",
            "Location": "Remote",
        }

    Returns
    -------
    dict
        {
            "predicted_salary": float,
            "confidence_low": float,
            "confidence_high": float,
            "warnings": list[str],
        }
    """
    model = _get_model()
    metadata = _get_metadata()

    warnings = validate_numeric_ranges(data, metadata)
    X = build_input_dataframe(data, metadata)

    predicted_salary = float(model.predict(X)[0])
    confidence_low, confidence_high = _estimate_confidence_interval(model, X)

    return {
        "predicted_salary": predicted_salary,
        "confidence_low": confidence_low,
        "confidence_high": confidence_high,
        "warnings": warnings,
    }


def get_feature_metadata() -> dict:
    """Expose metadata to the Streamlit UI layer (for building form widgets)."""
    return _get_metadata()
