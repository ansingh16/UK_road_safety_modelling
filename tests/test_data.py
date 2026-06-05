import numpy as np

from uk_road_safety.data import (
    build_vehicle_aggregates,
    engineer_features,
    load_dft_data,
    preprocess_features,
)


def test_load_collision_only_keeps_accident_index(synthetic_collisions):
    df = load_dft_data(synthetic_collisions, years=[2023], merge_vehicles=False)
    assert "accident_index" in df.columns
    assert "accident_severity" in df.columns
    assert len(df) == 200


def test_load_with_vehicle_merge(synthetic_collisions):
    df = load_dft_data(synthetic_collisions, years=[2023], merge_vehicles=True)
    assert "has_motorcycle" in df.columns
    assert "driver_age_min" in df.columns
    assert "engine_cc_max" in df.columns
    assert len(df) == 200


def test_build_vehicle_aggregates(synthetic_collisions):
    import pandas as pd

    vdf = pd.read_csv(
        synthetic_collisions / "dft-road-casualty-statistics-vehicle-2023.csv"
    )
    agg = build_vehicle_aggregates(vdf)
    assert "has_motorcycle" in agg.columns
    assert "has_hgv" in agg.columns
    assert "pct_male_drivers" in agg.columns
    assert agg["has_motorcycle"].dtype in [np.int64, np.int32, int]
    assert agg["pct_male_drivers"].between(0, 1).all()


def test_engineer_features_creates_time_columns(sample_frame):
    out = engineer_features(sample_frame)
    assert "hour" in out.columns
    assert "is_night" in out.columns
    assert "is_weekend" in out.columns
    assert "month" in out.columns
    # raw columns should be dropped
    assert "time" not in out.columns
    assert "date" not in out.columns
    assert "latitude" not in out.columns


def test_preprocess_splits_and_scales(sample_frame):
    prep = preprocess_features(sample_frame, target_col="accident_severity", test_size=0.25)

    n_total = len(sample_frame)
    n_test = len(prep["X_test"])
    n_train = len(prep["X_train"])

    assert n_train + n_test == n_total
    assert abs(n_test - 0.25 * n_total) <= 2

    assert prep["X_train_scaled"].shape[0] == n_train
    assert prep["X_test_scaled"].shape[0] == n_test
    assert prep["X_train_scaled"].shape[1] == len(prep["feature_names"])


def test_preprocess_encodes_categoricals(sample_frame):
    prep = preprocess_features(sample_frame, target_col="accident_severity")

    assert "road_type" in prep["label_encoders"]
    assert np.issubdtype(prep["X_train"]["road_type"].dtype, np.integer)
    assert "month" in prep["feature_names"]


def test_preprocess_imputes_missing_values(sample_frame):
    frame = sample_frame.copy()
    frame.loc[frame.index[:10], "speed_limit"] = np.nan
    frame.loc[frame.index[:5], "road_type"] = np.nan

    prep = preprocess_features(frame, target_col="accident_severity")

    assert not prep["X_train"].isnull().any().any()
    assert not prep["X_test"].isnull().any().any()


def test_target_excluded_from_features(sample_frame):
    prep = preprocess_features(sample_frame, target_col="accident_severity")
    assert "accident_severity" not in prep["feature_names"]


def test_leaky_columns_excluded(sample_frame):
    frame = sample_frame.copy()
    frame["enhanced_severity_collision"] = np.random.default_rng(0).integers(1, 8, len(frame))
    prep = preprocess_features(frame, target_col="accident_severity")
    assert "enhanced_severity_collision" not in prep["feature_names"]


def test_noise_columns_excluded(sample_frame):
    frame = sample_frame.copy()
    frame["accident_year"] = 2023
    frame["local_authority_district"] = -1
    prep = preprocess_features(frame, target_col="accident_severity")
    assert "accident_year" not in prep["feature_names"]
    assert "local_authority_district" not in prep["feature_names"]
    assert "accident_index" not in prep["feature_names"]


def test_preprocess_merged_includes_vehicle_features(merged_frame):
    prep = preprocess_features(merged_frame, target_col="accident_severity")
    assert "has_motorcycle" in prep["feature_names"]
    assert "driver_age_min" in prep["feature_names"]
    assert "hour" in prep["feature_names"]
    assert "lat_grid" in prep["feature_names"]
