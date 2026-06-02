import numpy as np
import pandas as pd

from uk_road_safety.evaluate import (
    calculate_business_impact,
    find_best_models,
    optimize_threshold,
)


def test_business_impact_on_known_matrix():
    # rows = true [1,2,3], cols = pred [1,2,3]
    cm = np.array(
        [
            [8, 1, 1],  # 8 severe detected, 2 missed
            [3, 20, 2],  # 3 false alarms (serious predicted severe)
            [1, 4, 50],  # 1 false alarm (slight predicted severe)
        ]
    )
    impact = calculate_business_impact(cm)

    assert impact["detected"] == 8
    assert impact["missed"] == 2
    assert impact["false_alarms"] == 4
    # precision = 8 / (8 + 4)
    assert impact["precision"] == 8 / 12


def test_business_impact_handles_no_predictions():
    cm = np.zeros((3, 3), dtype=int)
    impact = calculate_business_impact(cm)
    assert impact["precision"] == 0


def test_find_best_models_picks_top_rows():
    df = pd.DataFrame(
        {
            "Model": ["A", "B", "C"],
            "Recall_Severe": [0.5, 0.9, 0.7],
            "Recall_Macro": [0.8, 0.4, 0.6],
        }
    )
    best_severe, best_balanced = find_best_models(df, y_test=None)

    assert best_severe["Model"] == "B"  # highest Recall_Severe
    assert best_balanced["Model"] == "A"  # highest Recall_Macro


class _DummyModel:
    """Minimal model exposing predict_proba for threshold tests."""

    classes_ = np.array([1, 2, 3])

    def __init__(self, severe_scores):
        self._severe = np.asarray(severe_scores)

    def predict_proba(self, X):
        severe = self._severe
        rest = (1 - severe) / 2
        return np.column_stack([severe, rest, rest])


def test_optimize_threshold_returns_options():
    # 10 samples, 4 severe; severe ones have clearly higher scores
    y_test = pd.Series([1, 1, 1, 1, 2, 2, 3, 3, 3, 3])
    severe_scores = [0.9, 0.85, 0.8, 0.7, 0.2, 0.15, 0.1, 0.05, 0.05, 0.02]
    model = _DummyModel(severe_scores)

    options = optimize_threshold(model, X_test=np.zeros((10, 3)), y_test=y_test, target_recall=0.75)

    assert options is not None
    assert "Best F1" in options
    for opt in options.values():
        assert 0.0 <= opt["threshold"] <= 1.0
        assert 0.0 <= opt["recall"] <= 1.0


def test_optimize_threshold_without_proba_returns_none():
    class NoProba:
        pass

    assert optimize_threshold(NoProba(), np.zeros((4, 3)), pd.Series([1, 2, 3, 1])) is None
