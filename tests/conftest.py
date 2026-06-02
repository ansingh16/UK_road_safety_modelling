import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture
def synthetic_collisions(tmp_path):
    """Write a small synthetic DfT-style collision CSV and return its directory.

    Mirrors the columns the loader keys on (accident_index, accident_severity)
    plus a mix of numeric and categorical fields so preprocessing has work to do.
    """
    rng = np.random.default_rng(42)
    n = 200

    df = pd.DataFrame(
        {
            "accident_index": [f"2023{i:06d}" for i in range(n)],
            "accident_severity": rng.choice([1, 2, 3], size=n, p=[0.05, 0.2, 0.75]),
            "number_of_vehicles": rng.integers(1, 4, size=n),
            "number_of_casualties": rng.integers(1, 3, size=n),
            "speed_limit": rng.choice([20, 30, 40, 60, 70], size=n),
            "light_conditions": rng.integers(1, 7, size=n),
            "weather_conditions": rng.integers(1, 9, size=n),
            "road_type": rng.choice(
                ["Single carriageway", "Dual carriageway", "Roundabout"], size=n
            ),
            "date": rng.choice(["2023-01-01", "2023-06-15", "2023-12-31"], size=n),
        }
    )

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    df.to_csv(data_dir / "dft-road-casualty-statistics-collision-2023.csv", index=False)
    return data_dir


@pytest.fixture
def sample_frame(synthetic_collisions):
    """Load the synthetic collision frame (collision-only mode)."""
    from uk_road_safety.data import load_dft_data

    return load_dft_data(synthetic_collisions, years=[2023], merge_vehicles=False)
