# UK Road Accident Severity Classification — Dual-Strategy Approach (2023)

[![CI](https://github.com/ansingh16/UK_road_safety_modelling/actions/workflows/ci.yml/badge.svg)](https://github.com/ansingh16/UK_road_safety_modelling/actions/workflows/ci.yml)

Predicting the severity of UK road collisions from **Department for Transport (DfT)
2023 road safety data** (104K collisions + 190K vehicle records), using two
LightGBM classifiers tuned for different real-world objectives:

* **Severe-optimized** — heavy class weighting ({1:50, 2:3, 3:1}) to push
  recall on severe/fatal cases higher, for triage settings where a missed
  severe case is far costlier than a false alarm.
* **Balanced** — balanced class weights for overall accuracy across all three
  severity levels, for general traffic-management and resource planning.

Severe collisions are only ~1.4% of the data. The models combine collision
context (road, weather, time, location) with vehicle-level aggregates
(motorcycle/HGV involvement, driver age, engine capacity) — 42 features total.

## Results

Measured on the held-out 20% test split — **20,852 collisions** (304 severe,
4,688 serious, 15,860 slight). Reproduce with `python scripts/generate_results.py`
(writes [`results/metrics.md`](results/metrics.md) and the plots below).

| Metric | Severe-Optimized (LightGBM) | Balanced (LightGBM) |
|--------|---------------------------|-------------------------|
| Severe recall | **0.342** | 0.316 |
| Severe precision | 0.084 | **0.082** |
| Macro recall | **0.527** | 0.520 |
| Overall accuracy | 0.668 | **0.647** |

Predicting collision severity from pre-crash data is genuinely hard — the
strongest individual features (driver age, engine capacity, hour of day) have
mutual information under 0.02 with the target. See
[`notebooks/03_Feature_Analysis.ipynb`](notebooks/03_Feature_Analysis.ipynb)
for the full analysis: MI ranking, vehicle data exploration, and threshold
optimization curves.

### Visuals

| | |
|---|---|
| ![Confusion matrices](results/confusion_matrices.png) | ![Per-class recall](results/per_class_recall.png) |
| ![Precision–recall (severe)](results/precision_recall.png) | ![Feature importance](results/feature_importance.png) |

## Interactive dashboard

A Streamlit app (`app.py`) lets you set the conditions of a collision — speed
limit, lighting, weather, road surface, junction type, etc. — and compare what
each model predicts, alongside the balanced model's top feature importances.

**Try it live:** [uk-road-safety-modelling.streamlit.app](https://uk-road-safety-modelling.streamlit.app/)

To run locally:

```bash
pip install -e ".[app]"
streamlit run app.py
```

The non-interactive features are held at their 2023 training-set medians, so the
app runs without the raw DfT CSVs.

## Data

The 2023 road safety data is publicly available from the DfT:
https://www.data.gov.uk/dataset/cb7ae6f0-4be6-4935-9277-47e5ce24a11f/road-safety-data

Place the collision and vehicle CSVs in `data/`. The shipped models are
trained on both tables merged (42 features after vehicle aggregation and
feature engineering).

* **Collisions 2023** — accident details, location, conditions, timing
* **Vehicles 2023** — vehicle characteristics, manoeuvres, damage
* **Casualties 2023** — injury severity, demographics, roles

## Tech stack

* **Python** — data pipeline and modelling
* **LightGBM** — gradient-boosted classifiers (both models)
* **scikit-learn** — scaling, metrics, train/test split
* **imbalanced-learn** — SMOTE, ADASYN, SMOTE+Tomek (experimental, not used in shipped models)
* **pandas / NumPy** — preprocessing and feature engineering
* **Matplotlib / Seaborn** — evaluation plots
* **Streamlit** — interactive dashboard
* **joblib** — model serialization

## Project structure

```
UK_road_safety_modelling/
├── src/uk_road_safety/      # Installable package
│   ├── data.py              # DfT CSV loading, vehicle aggregation, feature engineering
│   ├── models.py            # LightGBM training + sampling strategies
│   ├── evaluate.py          # Metrics, threshold optimization, plots
│   └── predict.py           # Load artifacts + predict severity
├── configs/                 # emergency.yaml / balanced.yaml strategies
├── scripts/
│   └── generate_results.py  # Regenerate metrics table + evaluation plots
├── tests/                   # pytest suite (data / models / evaluate)
├── results/                 # Generated metrics.md and plots
├── notebooks/               # Original exploration notebooks
├── models/                  # Trained model pickles
├── data/                    # DfT CSV files (gitignored)
├── app.py                   # Streamlit dashboard
└── pyproject.toml
```

## How to run

```bash
git clone https://github.com/ansingh16/UK_road_safety_modelling.git
cd UK_road_safety_modelling
pip install -e ".[lightgbm,app,dev]"
```

* **Regenerate results:** `python scripts/generate_results.py --data-dir data/ --model-dir models/`
* **Run the dashboard:** `streamlit run app.py`
* **Run the tests:** `pytest`

The exploration notebooks document the analysis journey:

1. `notebooks/Data_Wrangling.ipynb` — loads, merges, and preprocesses the DfT CSVs
2. `notebooks/Data_Modelling.ipynb` — trains both models, evaluates, saves artifacts
3. `notebooks/03_Feature_Analysis.ipynb` — MI analysis, vehicle data exploration, LightGBM comparison
