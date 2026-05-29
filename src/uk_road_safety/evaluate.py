import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    recall_score,
    precision_score,
    accuracy_score,
)


def find_best_models(results_df, y_test):
    """Pick the best severe-optimized and best balanced model from results.

    Returns (best_severe_row, best_balanced_row) as Series.
    """
    severe_sorted = results_df.sort_values("Recall_Severe", ascending=False)
    balanced_sorted = results_df.sort_values("Recall_Macro", ascending=False)
    return severe_sorted.iloc[0], balanced_sorted.iloc[0]


def calculate_business_impact(cm):
    """Business-oriented metrics from a 3x3 confusion matrix (labels [1,2,3])."""
    severe_detected = cm[0, 0]
    severe_missed = cm[0, 1] + cm[0, 2]
    false_alarms = cm[1, 0] + cm[2, 0]
    precision = (
        severe_detected / (severe_detected + false_alarms)
        if (severe_detected + false_alarms) > 0
        else 0
    )
    return {
        "detected": int(severe_detected),
        "missed": int(severe_missed),
        "false_alarms": int(false_alarms),
        "precision": precision,
    }


def plot_model_comparison(y_test, severe_pred, balanced_pred, severe_info, balanced_info, save_path=None):
    """Side-by-side confusion matrices and per-class recall bars."""
    target_names = ["Severe (1)", "Serious (2)", "Slight (3)"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    cm_severe = confusion_matrix(y_test, severe_pred, labels=[1, 2, 3])
    cm_balanced = confusion_matrix(y_test, balanced_pred, labels=[1, 2, 3])

    sns.heatmap(
        cm_severe, annot=True, fmt="d", cmap="Reds",
        xticklabels=target_names, yticklabels=target_names, ax=axes[0, 0],
    )
    axes[0, 0].set_title(f"Severe-Optimized\n{severe_info['Model']} + {severe_info['Sampling']}")
    axes[0, 0].set_ylabel("True Label")
    axes[0, 0].set_xlabel("Predicted Label")

    sns.heatmap(
        cm_balanced, annot=True, fmt="d", cmap="Blues",
        xticklabels=target_names, yticklabels=target_names, ax=axes[0, 1],
    )
    axes[0, 1].set_title(f"Balanced\n{balanced_info['Model']} + {balanced_info['Sampling']}")
    axes[0, 1].set_ylabel("True Label")
    axes[0, 1].set_xlabel("Predicted Label")

    severe_recalls = [severe_info["Recall_Class1"], severe_info["Recall_Class2"], severe_info["Recall_Class3"]]
    balanced_recalls = [balanced_info["Recall_Class1"], balanced_info["Recall_Class2"], balanced_info["Recall_Class3"]]

    x = np.arange(3)
    width = 0.35
    axes[1, 0].bar(x - width / 2, severe_recalls, width, label="Severe-Optimized", color="red", alpha=0.7)
    axes[1, 0].bar(x + width / 2, balanced_recalls, width, label="Balanced", color="blue", alpha=0.7)
    axes[1, 0].set_xlabel("Accident Severity Class")
    axes[1, 0].set_ylabel("Recall")
    axes[1, 0].set_title("Per-Class Recall Comparison")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(target_names)
    axes[1, 0].legend()
    axes[1, 0].set_ylim(0, 1)

    severe_impact = calculate_business_impact(cm_severe)
    balanced_impact = calculate_business_impact(cm_balanced)

    impact_labels = ["Severe Detected", "False Alarms", "Severe Missed", "Precision %"]
    severe_vals = [severe_impact["detected"], severe_impact["false_alarms"],
                   severe_impact["missed"], severe_impact["precision"] * 100]
    balanced_vals = [balanced_impact["detected"], balanced_impact["false_alarms"],
                     balanced_impact["missed"], balanced_impact["precision"] * 100]

    x_imp = np.arange(len(impact_labels))
    axes[1, 1].bar(x_imp - width / 2, severe_vals, width, label="Severe-Optimized", color="red", alpha=0.7)
    axes[1, 1].bar(x_imp + width / 2, balanced_vals, width, label="Balanced", color="blue", alpha=0.7)
    axes[1, 1].set_xlabel("Impact Metrics")
    axes[1, 1].set_ylabel("Count / Percentage")
    axes[1, 1].set_title("Business Impact Comparison")
    axes[1, 1].set_xticks(x_imp)
    axes[1, 1].set_xticklabels(impact_labels, rotation=45)
    axes[1, 1].legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def optimize_threshold(model, X_test, y_test, target_recall=0.95):
    """Find the probability threshold for the severe class (1) that meets the target recall.

    Returns a dict of threshold options or None if optimization fails.
    """
    if not hasattr(model, "predict_proba"):
        return None

    y_proba = model.predict_proba(X_test)
    class_1_idx = list(model.classes_).index(1) if 1 in model.classes_ else 0
    y_proba_severe = y_proba[:, class_1_idx]

    y_true_binary = (y_test == 1).astype(int)
    if y_true_binary.sum() == 0:
        return None

    precision, recall, thresholds = precision_recall_curve(y_true_binary, y_proba_severe)

    options = {}

    valid = recall >= target_recall
    if valid.any() and len(thresholds) > 0:
        positions = np.where(valid)[0]
        if len(positions) > 0 and positions[0] < len(thresholds):
            idx = positions[0]
            options[f"High Recall ({target_recall:.0%})"] = {
                "threshold": float(thresholds[idx]),
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
            }

    if len(thresholds) > 0:
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_f1_idx = np.argmax(f1)
        if best_f1_idx < len(thresholds):
            options["Best F1"] = {
                "threshold": float(thresholds[best_f1_idx]),
                "precision": float(precision[best_f1_idx]),
                "recall": float(recall[best_f1_idx]),
            }

    return options if options else None


def plot_precision_recall_curve(model, X_test, y_test, title="Precision-Recall Curve", save_path=None):
    """Plot the precision-recall curve for severe case detection."""
    if not hasattr(model, "predict_proba"):
        return

    y_proba = model.predict_proba(X_test)
    class_1_idx = list(model.classes_).index(1) if 1 in model.classes_ else 0
    y_proba_severe = y_proba[:, class_1_idx]

    y_true_binary = (y_test == 1).astype(int)
    precision, recall, thresholds = precision_recall_curve(y_true_binary, y_proba_severe)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, marker=".", alpha=0.7)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
