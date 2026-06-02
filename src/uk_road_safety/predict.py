from pathlib import Path

import joblib
import pandas as pd


def load_model_artifacts(model_dir):
    """Load saved model, scaler, and label encoders from disk.

    Returns a dict with keys: severe_model, balanced_model, scaler, label_encoders.
    Any missing model file is set to None.
    """
    model_dir = Path(model_dir)

    artifacts = {
        "severe_model": None,
        "balanced_model": None,
        "scaler": None,
        "label_encoders": None,
    }

    severe_path = model_dir / "accident_severity_model_severe_optimized.pkl"
    balanced_path = model_dir / "accident_severity_model_balanced.pkl"
    scaler_path = model_dir / "accident_severity_model_scaler.pkl"
    encoders_path = model_dir / "accident_severity_model_label_encoders.pkl"

    if severe_path.exists():
        artifacts["severe_model"] = joblib.load(severe_path)
    if balanced_path.exists():
        artifacts["balanced_model"] = joblib.load(balanced_path)
    if scaler_path.exists():
        artifacts["scaler"] = joblib.load(scaler_path)
    if encoders_path.exists():
        artifacts["label_encoders"] = joblib.load(encoders_path)

    return artifacts


def save_model_artifacts(
    model_dir, severe_model=None, balanced_model=None, scaler=None, label_encoders=None
):
    """Save model artifacts to disk."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    if severe_model is not None:
        joblib.dump(severe_model, model_dir / "accident_severity_model_severe_optimized.pkl")
    if balanced_model is not None:
        joblib.dump(balanced_model, model_dir / "accident_severity_model_balanced.pkl")
    if scaler is not None:
        joblib.dump(scaler, model_dir / "accident_severity_model_scaler.pkl")
    if label_encoders is not None:
        joblib.dump(label_encoders, model_dir / "accident_severity_model_label_encoders.pkl")


def predict_severity(model, scaler, features_df, threshold=None):
    """Predict accident severity from a feature DataFrame.

    Parameters
    ----------
    model : sklearn estimator
        Trained classifier (severe-optimized or balanced).
    scaler : StandardScaler
        Fitted scaler. Applied only if the model expects scaled input
        (e.g., LogisticRegression). Pass None to skip scaling.
    features_df : pd.DataFrame
        Feature values, same columns as the training set.
    threshold : float, optional
        Custom probability threshold for the severe class (1).
        If provided, any sample whose P(severe) >= threshold is
        predicted as class 1; otherwise the default argmax is used.

    Returns
    -------
    predictions : np.ndarray
        Predicted class labels (1, 2, or 3).
    probabilities : np.ndarray or None
        Class probabilities if the model supports predict_proba, else None.
    """
    X = features_df.values if isinstance(features_df, pd.DataFrame) else features_df

    if scaler is not None:
        X = scaler.transform(X)

    if threshold is not None and hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        class_list = list(model.classes_)
        severe_idx = class_list.index(1) if 1 in class_list else 0

        predictions = model.predict(X)
        severe_mask = probabilities[:, severe_idx] >= threshold
        predictions[severe_mask] = 1
    else:
        predictions = model.predict(X)
        probabilities = model.predict_proba(X) if hasattr(model, "predict_proba") else None

    return predictions, probabilities
