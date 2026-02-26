import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # Added for professional plots
import torch
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(y_true, y_pred, y_prob):
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()

    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()

    if torch.is_tensor(y_prob):
        y_prob = y_prob.cpu().numpy()

    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        metrics["auprc"] = average_precision_score(y_true, y_prob)
    else:
        metrics["roc_auc"] = 0.0
        metrics["auprc"] = 0.0

    return metrics


def plot_confusion_matrix(y_true, y_pred, save_path="results/confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        xticklabels=["Normal", "Fraud"],
        yticklabels=["Normal", "Fraud"],
    )
    plt.title("Confusion Matrix: Fraud Detection Performance")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)  # Higher DPI for report quality
    plt.close()


def plot_roc_curve(y_true, y_probs, save_path="results/roc_curve.png"):
    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)
    else:
        return

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_precision_recall_curve(y_true, y_probs, save_path="results/pr_curve.png"):
    if len(np.unique(y_true)) > 1:
        precision, recall, _ = precision_recall_curve(y_true, y_probs)
        auprc = average_precision_score(y_true, y_probs)
    else:
        return

    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, color="green", lw=2, label=f"AUPRC = {auprc:.4f}")
    plt.xlabel("Recall (Ability to catch Fraud)")
    plt.ylabel("Precision (Accuracy of Fraud flags)")
    plt.title("Precision–Recall Curve (Key Fraud Metric)")
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
