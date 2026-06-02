import pandas as pd

from uk_road_safety.data import preprocess_features
from uk_road_safety.models import _get_model_configs, train_models


def test_model_configs_contain_both_strategies():
    severe, balanced = _get_model_configs()

    assert "RF_SevereOptim" in severe
    assert "LR_SevereOptim" in severe
    assert "RF_Balanced" in balanced
    assert "BalancedRF" in balanced

    # every severe model carries the heavy severe-class weighting
    rf = severe["RF_SevereOptim"]
    assert rf.class_weight == {1: 50, 2: 5, 3: 1}


def test_train_models_runs_on_small_sample(sample_frame):
    prep = preprocess_features(sample_frame, target_col="accident_severity")

    sampling = {"Original": (prep["X_train"], prep["y_train"])}

    results = train_models(
        prep["X_train"],
        prep["y_train"],
        prep["X_test"],
        prep["y_test"],
        prep["X_train_scaled"],
        prep["X_test_scaled"],
        sampling,
    )

    assert isinstance(results, pd.DataFrame)
    assert len(results) > 0

    expected_cols = {
        "Sampling",
        "Model",
        "Recall_Severe",
        "Recall_Macro",
        "Accuracy",
        "Model_Object",
        "Predictions",
    }
    assert expected_cols.issubset(results.columns)


def test_train_models_metrics_in_valid_range(sample_frame):
    prep = preprocess_features(sample_frame, target_col="accident_severity")
    sampling = {"Original": (prep["X_train"], prep["y_train"])}

    results = train_models(
        prep["X_train"],
        prep["y_train"],
        prep["X_test"],
        prep["y_test"],
        prep["X_train_scaled"],
        prep["X_test_scaled"],
        sampling,
    )

    for col in ["Recall_Severe", "Recall_Macro", "Accuracy"]:
        assert results[col].between(0.0, 1.0).all()
