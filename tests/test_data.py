import numpy as np

from uk_road_safety.data import load_dft_data, preprocess_features


def test_load_collision_only_keeps_accident_index(synthetic_collisions):
    df = load_dft_data(synthetic_collisions, years=[2023], merge_vehicles=False)
    assert "accident_index" in df.columns
    assert "accident_severity" in df.columns
    assert len(df) == 200


def test_preprocess_splits_and_scales(sample_frame):
    prep = preprocess_features(sample_frame, target_col="accident_severity", test_size=0.25)

    n_total = len(sample_frame)
    n_test = len(prep["X_test"])
    n_train = len(prep["X_train"])

    assert n_train + n_test == n_total
    # 25% test split, allow rounding slack
    assert abs(n_test - 0.25 * n_total) <= 2

    # scaled arrays line up with the split frames
    assert prep["X_train_scaled"].shape[0] == n_train
    assert prep["X_test_scaled"].shape[0] == n_test
    assert prep["X_train_scaled"].shape[1] == len(prep["feature_names"])


def test_preprocess_encodes_categoricals(sample_frame):
    prep = preprocess_features(sample_frame, target_col="accident_severity")

    # categorical columns must be present in the encoder map and now numeric
    assert "road_type" in prep["label_encoders"]
    assert np.issubdtype(prep["X_train"]["road_type"].dtype, np.integer)
    # date is consumed into month by engineer_features, so no longer categorical
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
