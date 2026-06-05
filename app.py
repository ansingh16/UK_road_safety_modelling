"""Streamlit dashboard for the UK road accident severity classifier.

Lets a user set collision conditions and vehicle characteristics, then
compares predictions from two LightGBM models:

* Severe-optimized — tuned for high recall on severe/fatal cases.
* Balanced — tuned for overall accuracy across all three classes.

The models use 42 features (collision context + vehicle aggregates +
engineered time/geo features). The sidebar exposes the interpretable
ones; the rest stay at their 2023 training-set medians.

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

FEATURE_BASELINE = {
    "police_force": 22.0,
    "number_of_vehicles": 2.0,
    "number_of_casualties": 1.0,
    "day_of_week": 4.0,
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
    "has_motorcycle": 0.0,
    "has_hgv": 0.0,
    "has_pedestrian": 0.0,
    "has_bicycle": 0.0,
    "driver_age_min": 31.0,
    "driver_age_max": 44.0,
    "any_skidding": 0.0,
    "any_side_impact": 0.0,
    "engine_cc_max": 1598.0,
    "age_of_vehicle_max": 9.0,
    "pct_male_drivers": 0.5,
    "hour": 15.0,
    "is_night": 0.0,
    "is_weekend": 0.0,
    "month": 7.0,
    "lat_grid": 518.0,
    "lon_grid": -11.0,
}

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
    return load_model_artifacts(MODEL_DIR)


def build_feature_row(overrides, feature_order):
    row = dict(FEATURE_BASELINE)
    row.update(overrides)
    return pd.DataFrame([row])[list(feature_order)]


def render_prediction(name, model, scaler, features_df, description="", key_metric=""):
    pred, proba = predict_severity(model, scaler, features_df)
    label = SEVERITY_LABELS.get(int(pred[0]), str(pred[0]))
    st.subheader(name)
    if description:
        st.caption(description)
    st.metric("Predicted severity", label)
    if key_metric:
        st.caption(key_metric)
    if proba is not None:
        prob_df = pd.DataFrame(
            {"Probability": proba[0]},
            index=[SEVERITY_LABELS[c] for c in model.classes_],
        )
        st.caption("Class probabilities:")
        st.bar_chart(prob_df)
    return label


def main():
    st.set_page_config(page_title="UK Road Accident Severity", layout="wide")
    st.title("UK Road Accident Severity Classifier")
    st.caption(
        "Set the conditions of a collision and compare predictions from the "
        "severe-optimized and balanced models. Trained on DfT 2023 road safety data."
    )

    with st.expander("About this tool"):
        st.markdown(
            """
**Two models, two strategies.** Severe/fatal collisions are only ~1.4% of the
data, so no single model handles them well. This tool runs two LightGBM
classifiers side-by-side:

- **Severe-optimized** — heavy class weighting ({1:50, 2:3, 3:1}) to push
  recall on severe cases higher, at the cost of more false alarms.
- **Balanced** — balanced class weights for the best overall accuracy and
  macro recall across all three severity levels.

**Severity classes** follow the DfT definitions: *Severe / Fatal* = at least
one fatality or life-threatening injury; *Serious* = hospital admission or
significant injury; *Slight* = minor or no hospital treatment.

**How the inputs work.** The models use 42 features from DfT collision and
vehicle records. The sidebar exposes the ones that are meaningful to set by
hand; the remaining features (location codes, police force, etc.) are held
at their 2023 training-set median values.
"""
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
        st.header("Scene conditions")
        speed = st.select_slider(
            "Speed limit (mph)", SPEED_LIMITS, value=30,
            help="Posted speed limit on the road where the collision occurred.",
        )
        n_vehicles = st.slider(
            "Number of vehicles", 1, 10, 2,
            help="Total vehicles involved in the collision.",
        )
        n_casualties = st.slider(
            "Number of casualties", 1, 10, 1,
            help="Total persons injured across all severity levels.",
        )
        hour = st.slider(
            "Hour of day", 0, 23, 9,
            help="Approximate hour the collision occurred (24h format).",
        )
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
        area = st.radio(
            "Area", list(URBAN_RURAL), format_func=URBAN_RURAL.get, horizontal=True,
            help="DfT urban/rural classification of the collision location.",
        )

        st.header("Vehicle & driver")
        motorcycle = st.checkbox("Motorcycle involved")
        hgv = st.checkbox("HGV / lorry involved")
        bicycle = st.checkbox("Bicycle involved")
        driver_age = st.slider(
            "Youngest driver age", 16, 90, 35,
            help="Age of the youngest driver involved.",
        )

    overrides = {
        "speed_limit": float(speed),
        "number_of_vehicles": float(n_vehicles),
        "number_of_casualties": float(n_casualties),
        "hour": float(hour),
        "is_night": float(hour in [22, 23, 0, 1, 2, 3, 4, 5]),
        "is_weekend": float(dow in [1, 7]),
        "day_of_week": float(dow),
        "light_conditions": float(light),
        "weather_conditions": float(weather),
        "road_surface_conditions": float(surface),
        "road_type": float(road),
        "junction_detail": float(junction),
        "urban_or_rural_area": float(area),
        "has_motorcycle": float(motorcycle),
        "has_hgv": float(hgv),
        "has_bicycle": float(bicycle),
        "driver_age_min": float(driver_age),
    }
    features_df = build_feature_row(overrides, feature_order)

    col1, col2 = st.columns(2)
    with col1:
        sev_label = render_prediction(
            "Severe-optimized model", severe_model, scaler, features_df,
            description="LightGBM — heavy class weighting to flag severe cases.",
            key_metric="52.7% macro recall · 66.8% overall accuracy",
        )
    with col2:
        bal_label = render_prediction(
            "Balanced model", balanced_model, None, features_df,
            description="LightGBM — balanced class weights for overall accuracy.",
            key_metric="52.0% macro recall · 64.7% overall accuracy",
        )

    if sev_label != bal_label:
        st.info(
            f"The models disagree: severe-optimized predicts **{sev_label}** while "
            f"balanced predicts **{bal_label}**. This is expected — the severe model "
            f"is deliberately aggressive about flagging high-severity cases."
        )

    st.divider()
    st.subheader("Top feature importances (balanced model)")
    st.caption(
        "How much each feature contributes to the LightGBM's predictions "
        "(split count). Higher bars = the model relies more on that feature."
    )
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
