import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture
def synthetic_collisions(tmp_path):
    """Write small synthetic DfT collision + vehicle CSVs and return the directory."""
    rng = np.random.default_rng(42)
    n = 200

    collision_df = pd.DataFrame(
        {
            "accident_index": [f"2023{i:06d}" for i in range(n)],
            "accident_severity": rng.choice([1, 2, 3], size=n, p=[0.05, 0.2, 0.75]),
            "accident_year": 2023,
            "accident_reference": range(n),
            "number_of_vehicles": rng.integers(1, 4, size=n),
            "number_of_casualties": rng.integers(1, 3, size=n),
            "speed_limit": rng.choice([20, 30, 40, 60, 70], size=n),
            "light_conditions": rng.integers(1, 7, size=n),
            "weather_conditions": rng.integers(1, 9, size=n),
            "road_type": rng.choice(
                ["Single carriageway", "Dual carriageway", "Roundabout"], size=n
            ),
            "date": rng.choice(["01/01/2023", "15/06/2023", "31/12/2023"], size=n),
            "time": rng.choice(["08:30", "14:15", "22:45"], size=n),
            "day_of_week": rng.integers(1, 8, size=n),
            "urban_or_rural_area": rng.choice([1, 2], size=n),
            "latitude": rng.uniform(50.0, 55.0, size=n),
            "longitude": rng.uniform(-3.0, 1.0, size=n),
            "location_easting_osgr": rng.integers(100000, 600000, size=n),
            "location_northing_osgr": rng.integers(100000, 600000, size=n),
            "local_authority_district": -1,
            "enhanced_severity_collision": rng.integers(1, 8, size=n),
        }
    )

    # Vehicle table: ~2 vehicles per collision on average
    vehicle_rows = []
    for idx in collision_df["accident_index"]:
        n_veh = rng.integers(1, 4)
        for v in range(n_veh):
            vehicle_rows.append({
                "accident_index": idx,
                "vehicle_reference": v + 1,
                "vehicle_type": rng.choice([1, 2, 3, 5, 9, 19, 20, 90]),
                "age_of_driver": int(rng.integers(17, 80)),
                "sex_of_driver": rng.choice([1, 2, 3]),
                "first_point_of_impact": rng.choice([0, 1, 2, 3, 4, 9]),
                "skidding_and_overturning": rng.choice([0, 0, 0, 1, 2, 5, 9]),
                "engine_capacity_cc": int(rng.choice([0, 998, 1199, 1598, 1998, 2998])),
                "age_of_vehicle": int(rng.integers(0, 25)),
                "vehicle_manoeuvre": rng.choice([2, 4, 9, 18]),
            })
    vehicle_df = pd.DataFrame(vehicle_rows)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    collision_df.to_csv(data_dir / "dft-road-casualty-statistics-collision-2023.csv", index=False)
    vehicle_df.to_csv(data_dir / "dft-road-casualty-statistics-vehicle-2023.csv", index=False)
    return data_dir


@pytest.fixture
def sample_frame(synthetic_collisions):
    """Load the synthetic collision frame (collision-only mode)."""
    from uk_road_safety.data import load_dft_data

    return load_dft_data(synthetic_collisions, years=[2023], merge_vehicles=False)


@pytest.fixture
def merged_frame(synthetic_collisions):
    """Load the synthetic collision + vehicle frame."""
    from uk_road_safety.data import load_dft_data

    return load_dft_data(synthetic_collisions, years=[2023], merge_vehicles=True)
