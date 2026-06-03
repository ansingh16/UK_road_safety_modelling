"""Streamlit dashboard for the UK road accident severity classifier.

Lets a user set the interpretable conditions of a collision (speed limit,
lighting, weather, road surface, etc.) and see what each of the two trained
models predicts:

* Severe-optimized (LogisticRegression) — tuned for high recall on severe cases.
* Balanced (RandomForest) — tuned for overall accuracy across all classes.

The DfT model expects 36 features. Only a handful are meaningful for a human
to set, so the rest are held at their median value from the 2023 training set
(baseline below) — this also means the app runs without the raw CSVs, which
are gitignored.

Run with:
    streamlit run app.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from uk_road_safety.predict import load_model_artifacts, predict_severity  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parent / "models"

# Median value of every model feature on the 2023 training split. Features the
# UI does not expose stay at these values. Order is irrelevant here — the row is
# reindexed to the scaler's feature order before prediction.
FEATURE_BASELINE = {
    "accident_index": 52106.5,
    "accident_year": 2023.0,
    "accident_reference": 52106.5,
    "location_easting_osgr": 462096.5,
    "location_northing_osgr": 216156.5,
    "longitude": -1.0851565,
    "latitude": 51.832629,
    "police_force": 22.0,
    "number_of_vehicles": 2.0,
    "number_of_casualties": 1.0,
    "date": 179.0,
    "day_of_week": 4.0,
    "time": 900.0,
    "local_authority_district": -1.0,
    "local_authority_ons_district": 203.0,
    "local_authority_highway": 112.0,
    "first_road_class": 4.0,
    "first_road_number": 30.0,
    "road_type": 6.0,
    "speed_limit": 30.0,
    "junction_detail": 2.0,
    "junction_control": 2.0,
    "second_road_class": 3.0,
    "second_road_number": 0.0,
    "pedestrian_crossing_human_control": 0.0,
    "pedestrian_crossing_physical_facilities": 0.0,
    "light_conditions": 1.0,
    "weather_conditions": 1.0,
    "road_surface_conditions": 1.0,
    "special_conditions_at_site": 0.0,
    "carriageway_hazards": 0.0,
    "urban_or_rural_area": 1.0,
    "did_police_officer_attend_scene_of_accident": 1.0,
    "trunk_road_flag": 2.0,
    "lsoa_of_accident_location": 12321.0,
    "enhanced_severity_collision": 3.0,
}

# DfT code lookups for the interpretable features the UI exposes.
SPEED_LIMITS = [20, 30, 40, 50, 60, 70]
DAY_OF_WEEK = {1: "Sunday", 2: "Monday", 3: "Tuesday", 4: "Wednesday", 5: "Thursday",
               6: "Friday", 7: "Saturday"}
LIGHT_CONDITIONS = {
    1: "Daylight",
    4: "Darkness — lights lit",
    5: "Darkness — lights unlit",
    6: "Darkness — no lighting",
    7: "Darkness — lighting unknown",
}
WEATHER_CONDITIONS = {
    1: "Fine, no high winds",
    2: "Raining, no high winds",
    3: "Snowing, no high winds",
    4: "Fine + high winds",
    5: "Raining + high winds",
    6: "Snowing + high winds",
    7: "Fog or mist",
    8: "Other",
    9: "Unknown",
}
ROAD_SURFACE = {1: "Dry", 2: "Wet or damp", 3: "Snow", 4: "Frost or ice", 5: "Flood"}
ROAD_TYPE = {
    1: "Roundabout",
    2: "One way street",
    3: "Dual carriageway",
    6: "Single carriageway",
    7: "Slip road",
    9: "Unknown",
}
JUNCTION_DETAIL = {
    0: "Not at junction",
    1: "Roundabout",
    2: "Mini-roundabout",
    3: "T or staggered junction",
    5: "Slip road",
    6: "Crossroads",
    7: "More than 4 arms",
    8: "Private drive or entrance",
    9: "Other junction",
}
URBAN_RURAL = {1: "Urban", 2: "Rural"}

SEVERITY_LABELS = {1: "Severe / Fatal", 2: "Serious", 3: "Slight"}


@st.cache_resource
def get_artifacts():
    """Load and cache the trained models, scaler and label encoders."""
    return load_model_artifacts(MODEL_DIR)


def build_feature_row(overrides, feature_order):
    """Return a one-row DataFrame in the model's feature order."""
    row = dict(FEATURE_BASELINE)
    row.update(overrides)
    return pd.DataFrame([row])[list(feature_order)]


def render_prediction(name, model, scaler, features_df):
    pred, proba = predict_severity(model, scaler, features_df)
    label = SEVERITY_LABELS.get(int(pred[0]), str(pred[0]))
    st.subheader(name)
    st.metric("Predicted severity", label)
    if proba is not None:
        prob_df = pd.DataFrame(
            {"Probability": proba[0]},
            index=[SEVERITY_LABELS[c] for c in model.classes_],
        )
        st.bar_chart(prob_df)


def main():
    st.set_page_config(page_title="UK Road Accident Severity", layout="wide")
    st.title("UK Road Accident Severity Classifier")
    st.caption(
        "Set the conditions of a collision and compare predictions from the "
        "severe-optimized and balanced models. Trained on DfT 2023 road safety data."
    )

    artifacts = get_artifacts()
    severe_model = artifacts["severe_model"]
    balanced_model = artifacts["balanced_model"]
    scaler = artifacts["scaler"]

    if severe_model is None or balanced_model is None or scaler is None:
        st.error(
            f"Model artifacts not found in {MODEL_DIR}. "
            "Train the models or place the .pkl files there first."
        )
        st.stop()

    feature_order = list(scaler.feature_names_in_)

    with st.sidebar:
        st.header("Collision conditions")
        speed = st.select_slider("Speed limit (mph)", SPEED_LIMITS, value=30)
        n_vehicles = st.slider("Number of vehicles", 1, 10, 2)
        n_casualties = st.slider("Number of casualties", 1, 10, 1)
        hour = st.slider("Hour of day", 0, 23, 9)
        dow = st.selectbox("Day of week", list(DAY_OF_WEEK), format_func=DAY_OF_WEEK.get, index=3)
        light = st.selectbox(
            "Light conditions", list(LIGHT_CONDITIONS), format_func=LIGHT_CONDITIONS.get
        )
        weather = st.selectbox(
            "Weather", list(WEATHER_CONDITIONS), format_func=WEATHER_CONDITIONS.get
        )
        surface = st.selectbox(
            "Road surface", list(ROAD_SURFACE), format_func=ROAD_SURFACE.get
        )
        road = st.selectbox(
            "Road type", list(ROAD_TYPE), format_func=ROAD_TYPE.get, index=3
        )
        junction = st.selectbox(
            "Junction detail", list(JUNCTION_DETAIL), format_func=JUNCTION_DETAIL.get, index=2
        )
        area = st.radio("Area", list(URBAN_RURAL), format_func=URBAN_RURAL.get, horizontal=True)

    overrides = {
        "speed_limit": float(speed),
        "number_of_vehicles": float(n_vehicles),
        "number_of_casualties": float(n_casualties),
        "time": float(hour * 100),  # DfT encodes time as HHMM
        "day_of_week": float(dow),
        "light_conditions": float(light),
        "weather_conditions": float(weather),
        "road_surface_conditions": float(surface),
        "road_type": float(road),
        "junction_detail": float(junction),
        "urban_or_rural_area": float(area),
    }
    features_df = build_feature_row(overrides, feature_order)

    col1, col2 = st.columns(2)
    with col1:
        # Severe-optimized model is LogisticRegression — needs scaled input.
        render_prediction("Severe-optimized model", severe_model, scaler, features_df)
    with col2:
        # Balanced model is a tree ensemble — uses raw (unscaled) features.
        render_prediction("Balanced model", balanced_model, None, features_df)

    st.divider()
    st.subheader("Top feature importances (balanced model)")
    if hasattr(balanced_model, "feature_importances_"):
        imp = (
            pd.Series(balanced_model.feature_importances_, index=feature_order)
            .sort_values(ascending=False)
            .head(15)
        )
        st.bar_chart(imp)
    else:
        st.info("The balanced model does not expose feature importances.")


if __name__ == "__main__":
    main()
