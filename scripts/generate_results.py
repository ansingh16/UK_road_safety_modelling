"""Generate evaluation plots from saved models and DfT data.

Produces confusion matrices, feature importance (top 15),
and precision-recall curve, saved to results/.

Usage:
    python scripts/generate_results.py --data-dir data/ --model-dir models/
"""

import argparse
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, recall_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from uk_road_safety.data import load_dft_data, preprocess_features


def plot_confusion_matrices(y_test, severe_pred, balanced_pred, save_path):
    target_names = ["Severe (1)", "Serious (2)", "Slight (3)"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cm_sev = confusion_matrix(y_test, severe_pred, labels=[1, 2, 3])
    cm_bal = confusion_matrix(y_test, balanced_pred, labels=[1, 2, 3])

    sns.heatmap(
        cm_sev, annot=True, fmt="d", cmap="Reds",
        xticklabels=target_names, yticklabels=target_names, ax=axes[0],
    )
    axes[0].set_title("Severe-Optimized Model")
    axes[0].set_ylabel("True Label")
    axes[0].set_xlabel("Predicted Label")

    sns.heatmap(
        cm_bal, annot=True, fmt="d", cmap="Blues",
        xticklabels=target_names, yticklabels=target_names, ax=axes[1],
    )
    axes[1].set_title("Balanced Model")
    axes[1].set_ylabel("True Label")
    axes[1].set_xlabel("Predicted Label")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_feature_importance(model, feature_names, top_n, save_path):
    if not hasattr(model, "feature_importances_"):
        print(f"Skipping feature importance — {type(model).__name__} has no feature_importances_")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(indices)), importances[indices], color="steelblue")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Top {top_n} Features — {type(model).__name__}")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_precision_recall(model, X_test, y_test, scaler, save_path):
    if not hasattr(model, "predict_proba"):
        print(f"Skipping PR curve — {type(model).__name__} has no predict_proba")
        return

    needs_scaling = type(model).__name__ == "LogisticRegression"
    X = scaler.transform(X_test) if needs_scaling else X_test

    proba = model.predict_proba(X)
    class_list = list(model.classes_)
    severe_idx = class_list.index(1) if 1 in class_list else 0
    y_binary = (y_test == 1).astype(int)

    precision, recall, _ = precision_recall_curve(y_binary, proba[:, severe_idx])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color="darkred", linewidth=2)
    ax.fill_between(recall, precision, alpha=0.15, color="red")
    ax.set_xlabel("Recall (Severe)")
    ax.set_ylabel("Precision (Severe)")
    ax.set_title("Precision-Recall Curve — Severe Accident Detection")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_per_class_recall(y_test, severe_pred, balanced_pred, save_path):
    target_names = ["Severe (1)", "Serious (2)", "Slight (3)"]
    severe_recalls = recall_score(y_test, severe_pred, labels=[1, 2, 3], average=None)
    balanced_recalls = recall_score(y_test, balanced_pred, labels=[1, 2, 3], average=None)

    x = np.arange(3)
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, severe_recalls, width, label="Severe-Optimized", color="indianred")
    ax.bar(x + width / 2, balanced_recalls, width, label="Balanced", color="steelblue")
    ax.set_ylabel("Recall")
    ax.set_title("Per-Class Recall Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(target_names)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation plots")
    parser.add_argument("--data-dir", default="data/", help="Path to DfT CSV directory")
    parser.add_argument("--model-dir", default="models/", help="Path to saved model PKLs")
    parser.add_argument("--output-dir", default="results/", help="Where to save plots")
    parser.add_argument("--top-n", type=int, default=15, help="Number of top features to plot")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(exist_ok=True)
    model_dir = Path(args.model_dir)

    print("Loading models...")
    severe_model = joblib.load(model_dir / "accident_severity_model_severe_optimized.pkl")
    balanced_model = joblib.load(model_dir / "accident_severity_model_balanced.pkl")
    scaler = joblib.load(model_dir / "accident_severity_model_scaler.pkl")

    print("Loading and preprocessing data...")
    df = load_dft_data(args.data_dir, years=[2023], merge_vehicles=False)
    prep = preprocess_features(df, target_col="accident_severity")
    X_test = prep["X_test"]
    y_test = prep["y_test"]
    X_test_scaled = prep["X_test_scaled"]
    feature_names = prep["feature_names"]

    needs_scaling_severe = type(severe_model).__name__ == "LogisticRegression"
    X_sev = X_test_scaled if needs_scaling_severe else X_test
    severe_pred = severe_model.predict(X_sev)

    needs_scaling_bal = type(balanced_model).__name__ == "LogisticRegression"
    X_bal = X_test_scaled if needs_scaling_bal else X_test
    balanced_pred = balanced_model.predict(X_bal)

    plot_confusion_matrices(y_test, severe_pred, balanced_pred, output / "confusion_matrices.png")
    plot_feature_importance(balanced_model, feature_names, args.top_n, output / "feature_importance.png")
    plot_precision_recall(severe_model, X_test, y_test, scaler, output / "precision_recall.png")
    plot_per_class_recall(y_test, severe_pred, balanced_pred, output / "per_class_recall.png")

    print("\nAll plots saved to", output)


if __name__ == "__main__":
    main()
