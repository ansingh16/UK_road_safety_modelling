from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

LEAK_COLS = ["enhanced_severity_collision"]
NOISE_COLS = ["accident_year", "accident_reference", "local_authority_district"]
ID_COLS = ["accident_index"]
RAW_GEO_COLS = ["latitude", "longitude", "location_easting_osgr", "location_northing_osgr"]
RAW_TEMPORAL_COLS = ["time", "date"]


def load_dft_data(data_dir, years=None, merge_vehicles=True):
    """Load DfT collision CSVs, optionally merged with vehicle-level aggregates.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing the DfT CSV files.
    years : list[int], optional
        Which years to load (default [2023]).
    merge_vehicles : bool
        If True, load the vehicle table, aggregate to collision level,
        and merge. Set to False for collision-only features.

    Returns a DataFrame with one row per collision.
    """
    if years is None:
        years = [2023]

    data_dir = Path(data_dir)

    collision_files = [
        data_dir / f"dft-road-casualty-statistics-collision-{year}.csv" for year in years
    ]
    df = pd.concat([pd.read_csv(f, low_memory=False) for f in collision_files])

    if not merge_vehicles:
        return df

    vehicle_files = [
        data_dir / f"dft-road-casualty-statistics-vehicle-{year}.csv" for year in years
    ]
    vdf = pd.concat([pd.read_csv(f, low_memory=False) for f in vehicle_files])

    agg = build_vehicle_aggregates(vdf)
    df = df.merge(agg, on="accident_index", how="left")
    return df


def build_vehicle_aggregates(vdf):
    """Aggregate vehicle-level data to one row per collision.

    Creates binary flags (motorcycle, HGV, pedestrian, bicycle), driver
    demographics (age min/max, pct male), vehicle characteristics (engine cc,
    vehicle age), and incident mechanics (skidding, side impact).
    """
    return vdf.groupby("accident_index").agg(
        has_motorcycle=("vehicle_type", lambda x: int(any(x.isin([2, 3, 4, 5])))),
        has_hgv=("vehicle_type", lambda x: int(any(x.isin([19, 20, 21])))),
        has_pedestrian=("vehicle_type", lambda x: int(any(x == 90))),
        has_bicycle=("vehicle_type", lambda x: int(any(x == 1))),
        driver_age_min=("age_of_driver", lambda x: x[x > 0].min() if (x > 0).any() else -1),
        driver_age_max=("age_of_driver", lambda x: x[x > 0].max() if (x > 0).any() else -1),
        any_skidding=(
            "skidding_and_overturning", lambda x: int(any(x.isin([1, 2, 3, 4, 5]))),
        ),
        any_side_impact=("first_point_of_impact", lambda x: int(any(x.isin([3, 4])))),
        engine_cc_max=(
            "engine_capacity_cc", lambda x: x[x > 0].max() if (x > 0).any() else 0,
        ),
        age_of_vehicle_max=(
            "age_of_vehicle", lambda x: x[x > 0].max() if (x > 0).any() else 0,
        ),
        pct_male_drivers=("sex_of_driver", lambda x: (x == 1).mean()),
    ).reset_index()


def engineer_features(df):
    """Create derived features from raw collision columns.

    Adds: hour, is_night, is_weekend, month, lat_grid, lon_grid.
    Drops raw temporal and geographic columns that would cause overfitting.
    """
    out = df.copy()

    if "time" in out.columns:
        out["hour"] = pd.to_datetime(out["time"], format="%H:%M", errors="coerce").dt.hour
        out["is_night"] = out["hour"].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
    if "day_of_week" in out.columns:
        out["is_weekend"] = out["day_of_week"].isin([1, 7]).astype(int)
    if "date" in out.columns:
        parsed = pd.to_datetime(out["date"], format="%d/%m/%Y", errors="coerce")
        if parsed.isna().all():
            parsed = pd.to_datetime(out["date"], errors="coerce")
        out["month"] = parsed.dt.month
    if "latitude" in out.columns and "longitude" in out.columns:
        out["lat_grid"] = (out["latitude"] * 10).round()
        out["lon_grid"] = (out["longitude"] * 10).round()

    drop = [c for c in RAW_TEMPORAL_COLS + RAW_GEO_COLS if c in out.columns]
    out = out.drop(columns=drop)
    return out


def preprocess_features(
    df,
    target_col="accident_severity",
    test_size=0.2,
    random_state=42,
):
    """Engineer features, encode categoricals, impute, scale, and split."""
    feat = engineer_features(df)

    drop = [c for c in LEAK_COLS + NOISE_COLS + ID_COLS if c in feat.columns]
    X = feat.drop(columns=[target_col] + drop)
    y = feat[target_col]

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
