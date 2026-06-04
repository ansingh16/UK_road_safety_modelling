from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_dft_data(data_dir, years=None, merge_vehicles=True):
    """Load DfT collision CSVs, optionally merged with vehicle CSVs.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing the DfT CSV files.
    years : list[int], optional
        Which years to load (default [2023]).
    merge_vehicles : bool
        If True, merge collision and vehicle tables on accident_index.
        Set to False to load collision data only (matches the original
        36-feature training set).

    Returns a DataFrame indexed by accident_index.
    """
    if years is None:
        years = [2023]

    data_dir = Path(data_dir)

    collision_files = [
        data_dir / f"dft-road-casualty-statistics-collision-{year}.csv" for year in years
    ]

    df_collisions = pd.concat([pd.read_csv(f, low_memory=False) for f in collision_files])

    if not merge_vehicles:
        return df_collisions

    df_collisions.set_index("accident_index", inplace=True)

    vehicle_files = [
        data_dir / f"dft-road-casualty-statistics-vehicle-{year}.csv" for year in years
    ]

    df_vehicles = pd.concat([pd.read_csv(f, low_memory=False) for f in vehicle_files])
    df_vehicles.set_index("accident_index", inplace=True)

    df = df_collisions.merge(df_vehicles, left_index=True, right_index=True, how="left")
    return df


def preprocess_features(
    df,
    target_col="accident_severity",
    test_size=0.2,
    random_state=42,
):
    """Encode categoricals, impute missing values, scale, and split.

    Returns (X_train, X_test, y_train, y_test, scaler, label_encoders, feature_names).
    """
    # enhanced_severity_collision is derived from the target — drop to avoid leakage
    leak_cols = [c for c in ["enhanced_severity_collision"] if c in df.columns]
    X = df.drop(columns=[target_col] + leak_cols)
    y = df[target_col]

    numerical_cols = X.select_dtypes(include=[np.number]).columns
    X[numerical_cols] = X[numerical_cols].fillna(X[numerical_cols].median())

    categorical_cols = X.select_dtypes(include=["object"]).columns
    label_encoders = {}
    for col in categorical_cols:
        X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else "Unknown")
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "scaler": scaler,
        "label_encoders": label_encoders,
        "feature_names": feature_names,
    }
