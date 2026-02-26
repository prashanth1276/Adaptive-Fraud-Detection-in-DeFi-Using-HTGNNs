"""
Evaluation metric computation and visualization utilities for fraud detection.

This module provides functions for computing standard binary classification
metrics and generating publication-quality diagnostic plots. It is consumed
by the main training pipeline (train/train.py), ablation studies (ablation/),
and baseline comparisons (baselines/) to ensure consistent metric reporting
across all experimental conditions. All plot functions render results at 300 DPI
suitable for inclusion in research papers and technical reports. This module
is used at both training time (epoch validation) and evaluation time (final
test set reporting).
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
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
    """Computes a comprehensive set of binary classification metrics.

    Evaluates precision, recall, F1-score, ROC-AUC, and AUPRC for a set of
    binary predictions. Handles both PyTorch tensor and NumPy array inputs
    transparently. AUPRC is the primary metric for imbalanced fraud detection,
    as it accounts for the skewed class distribution.

    Args:
        y_true (torch.Tensor | np.ndarray): Ground-truth binary labels of
            shape (N,), with values in {0, 1}.
        y_pred (torch.Tensor | np.ndarray): Hard binary predictions of shape
            (N,), obtained by thresholding y_prob.
        y_prob (torch.Tensor | np.ndarray): Continuous fraud probability scores
            of shape (N,), output of sigmoid(logits).

    Returns:
        dict: Metric dictionary with keys:
            - 'precision' (float): Positive predictive value.
            - 'recall' (float): True positive rate (sensitivity).
            - 'f1' (float): Harmonic mean of precision and recall.
            - 'roc_auc' (float): Area under the ROC curve. Set to 0.0 if
              only one class is present in y_true.
            - 'auprc' (float): Area under the precision-recall curve. Set
              to 0.0 if only one class is present in y_true.
    """
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

    # ROC-AUC and AUPRC require at least two distinct classes to be defined
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        metrics["auprc"] = average_precision_score(y_true, y_prob)
    else:
        metrics["roc_auc"] = 0.0
        metrics["auprc"] = 0.0

    return metrics


def plot_confusion_matrix(y_true, y_pred, save_path="results/confusion_matrix.png"):
    """Renders and saves a color-coded confusion matrix as a seaborn heatmap.

    Args:
        y_true (np.ndarray): Ground-truth binary labels of shape (N,).
        y_pred (np.ndarray): Hard binary predictions of shape (N,).
        save_path (str, optional): Filesystem path for the output PNG image.
            Defaults to 'results/confusion_matrix.png'.
    """
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
    # Render at 300 DPI for publication-quality output
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_roc_curve(y_true, y_probs, save_path="results/roc_curve.png"):
    """Renders and saves the Receiver Operating Characteristic (ROC) curve.

    Skips rendering if only one class is represented in y_true, as the
    curve is undefined in that case.

    Args:
        y_true (np.ndarray): Ground-truth binary labels of shape (N,).
        y_probs (np.ndarray): Continuous fraud probability scores of shape (N,).
        save_path (str, optional): Filesystem path for the output PNG image.
            Defaults to 'results/roc_curve.png'.
    """
    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)
    else:
        return

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC AUC = {roc_auc:.4f}")
    # Diagonal random-classifier reference line
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
    """Renders and saves the Precision-Recall curve with AUPRC annotation.

    The Precision-Recall curve is the primary diagnostic plot for imbalanced
    fraud detection tasks, as it is insensitive to the large number of true
    negatives that inflate ROC-AUC. Skips rendering if only one class is present.

    Args:
        y_true (np.ndarray): Ground-truth binary labels of shape (N,).
        y_probs (np.ndarray): Continuous fraud probability scores of shape (N,).
        save_path (str, optional): Filesystem path for the output PNG image.
            Defaults to 'results/pr_curve.png'.
    """
    if len(np.unique(y_true)) > 1:
        precision, recall, _ = precision_recall_curve(y_true, y_probs)
        auprc = average_precision_score(y_true, y_probs)
    else:
        return

    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, color="green", lw=2, label=f"AUPRC = {auprc:.4f}")
    plt.xlabel("Recall (Ability to catch Fraud)")
    plt.ylabel("Precision (Accuracy of Fraud flags)")
    plt.title("Precision-Recall Curve (Key Fraud Metric)")
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
