import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


def get_sampling_techniques(X_train, y_train):
    """Apply SMOTE, ADASYN, SMOTE+Tomek, and random undersampling.

    Returns a dict mapping technique name to (X_resampled, y_resampled).
    """
    techniques = {"Original": (X_train, y_train)}

    smote = SMOTE(random_state=42)
    X_sm, y_sm = smote.fit_resample(X_train, y_train)
    techniques["SMOTE"] = (X_sm, y_sm)

    try:
        adasyn = ADASYN(random_state=42)
        X_ad, y_ad = adasyn.fit_resample(X_train, y_train)
        techniques["ADASYN"] = (X_ad, y_ad)
    except Exception:
        pass

    smote_tomek = SMOTETomek(random_state=42)
    X_st, y_st = smote_tomek.fit_resample(X_train, y_train)
    techniques["SMOTE+Tomek"] = (X_st, y_st)

    class_counts = pd.Series(y_train).value_counts()
    min_class_count = class_counts.min()
    rus = RandomUnderSampler(
        random_state=42,
        sampling_strategy={
            3: 10000,
            2: 5000,
            1: min_class_count,
        },
    )
    X_ru, y_ru = rus.fit_resample(X_train, y_train)
    techniques["RandomUnderSampling"] = (X_ru, y_ru)

    return techniques


def _get_model_configs():
    """Return (severe_optimized_models, balanced_models) dicts."""
    severe = {
        "RF_SevereOptim": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight={1: 50, 2: 5, 3: 1},
            random_state=42,
        ),
        "LR_SevereOptim": LogisticRegression(
            class_weight={1: 50, 2: 5, 3: 1},
            random_state=42,
            max_iter=1000,
        ),
        "GB_SevereOptim": GradientBoostingClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42,
        ),
    }

    balanced = {
        "RF_Balanced": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight="balanced",
            random_state=42,
        ),
        "LR_Balanced": LogisticRegression(
            class_weight="balanced",
            random_state=42,
            max_iter=1000,
        ),
        "BalancedRF": BalancedRandomForestClassifier(
            n_estimators=100,
            random_state=42,
        ),
    }

    if LIGHTGBM_AVAILABLE:
        severe["LightGBM_SevereOptim"] = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=3,
            class_weight={1: 50, 2: 5, 3: 1},
            n_estimators=100,
            random_state=42,
            verbosity=-1,
        )
        balanced["LightGBM_Balanced"] = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=3,
            class_weight="balanced",
            n_estimators=100,
            random_state=42,
            verbosity=-1,
        )

    return severe, balanced


def train_models(
    X_train,
    y_train,
    X_test,
    y_test,
    X_train_scaled,
    X_test_scaled,
    sampling_techniques,
):
    """Train all model/sampling combinations and return a results DataFrame.

    Each row contains metrics plus the trained model object and predictions.
    """
    severe_models, balanced_models = _get_model_configs()
    all_models = {**severe_models, **balanced_models}
    results = []

    for sampling_name, (X_samp, y_samp) in sampling_techniques.items():
        if sampling_name == "Original":
            X_samp_scaled = X_train_scaled
        else:
            from sklearn.preprocessing import StandardScaler

            scaler_tmp = StandardScaler()
            X_samp_scaled = scaler_tmp.fit_transform(X_samp)

        for model_name, model in all_models.items():
            if model_name == "BalancedRF" and sampling_name != "Original":
                continue

            try:
                if "LR_" in model_name:
                    model.fit(X_samp_scaled, y_samp)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_samp, y_samp)
                    y_pred = model.predict(X_test)

                recall_severe = recall_score(
                    y_test, y_pred, labels=[1], average="macro", zero_division=0
                )
                precision_severe = precision_score(
                    y_test, y_pred, labels=[1], average="macro", zero_division=0
                )
                f1_severe = f1_score(y_test, y_pred, labels=[1], average="macro", zero_division=0)

                recall_macro = recall_score(y_test, y_pred, average="macro")
                f1_macro = f1_score(y_test, y_pred, average="macro")
                accuracy = accuracy_score(y_test, y_pred)

                recall_per_class = recall_score(y_test, y_pred, average=None)

                cm = confusion_matrix(y_test, y_pred, labels=[1, 2, 3])
                severe_tp = cm[0, 0]
                severe_fp = cm[1:, 0].sum()
                severe_precision_actual = (
                    severe_tp / (severe_tp + severe_fp) if (severe_tp + severe_fp) > 0 else 0
                )

                model_type = "Severe-Optimized" if "SevereOptim" in model_name else "Balanced"

                results.append(
                    {
                        "Sampling": sampling_name,
                        "Model": model_name,
                        "Model_Type": model_type,
                        "Recall_Severe": recall_severe,
                        "Precision_Severe": precision_severe,
                        "F1_Severe": f1_severe,
                        "Severe_Precision_Actual": severe_precision_actual,
                        "Recall_Macro": recall_macro,
                        "F1_Macro": f1_macro,
                        "Accuracy": accuracy,
                        "Recall_Class1": recall_per_class[0] if len(recall_per_class) > 0 else 0,
                        "Recall_Class2": recall_per_class[1] if len(recall_per_class) > 1 else 0,
                        "Recall_Class3": recall_per_class[2] if len(recall_per_class) > 2 else 0,
                        "Model_Object": model,
                        "Predictions": y_pred,
                    }
                )
            except Exception as e:
                print(f"Error training {model_name} with {sampling_name}: {e}")

    return pd.DataFrame(results)
