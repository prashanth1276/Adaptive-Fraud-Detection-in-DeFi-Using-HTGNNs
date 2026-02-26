"""
Primary training pipeline for the Heterogeneous Temporal Graph Neural Network.

This module orchestrates the complete offline training procedure for the HTGNN
fraud detection model: temporal train/val/test splitting, StandardScaler feature
normalization, FocalLoss optimization, adaptive learning rate scheduling,
threshold selection via precision-recall curve maximization, early stopping, and
final evaluation with metric visualization. It is the entry point for producing
the best_model.pth checkpoint consumed by the inference dashboard (app.py) and
the adaptive engine (model/adaptive_engine.py). All randomness is seeded for
full experimental reproducibility.
"""

import datetime
import json
import os
import random
import shutil

import dgl
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from model.htg_nn import HTGNN
from utils.metrics import (
    compute_metrics,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
)

# Ensure the results directory exists prior to any checkpoint or plot I/O
os.makedirs("results", exist_ok=True)


def load_graph():
    """Loads the pre-processed DGL heterogeneous graph from disk.

    Reads the binary graph artifact produced by data_prep/dgl_graph_construction.py,
    which contains node feature tensors, edge type structure, temporal timestamps,
    and fraud labels.

    Returns:
        dgl.DGLHeteroGraph: The first (and only) graph stored in graph.bin,
            with node type 'node' and edge types corresponding to DeFi
            transaction relation categories.
    """
    # Load the pre-processed binary graph saved by dgl_graph_construction.py
    graphs, _ = dgl.load_graphs("DataSet/graph.bin")
    return graphs[0]


class FocalLoss(nn.Module):
    """Focal Loss for binary node classification under severe class imbalance.

    Addresses the extreme label skew in DeFi fraud datasets (typical ratio ~1:169)
    by down-weighting the loss contribution of well-classified benign nodes and
    focusing gradient signal on hard-to-classify fraud instances. The formulation
    follows Lin et al. (2017): FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t).

    Attributes:
        alpha (float): Class-balancing weight. Values near 1.0 assign higher loss
            weight to the minority fraud class. Recommended: 0.95.
        gamma (float): Focusing exponent. Higher values increase emphasis on
            misclassified samples. Recommended: 1.5.
    """

    def __init__(self, alpha=0.95, gamma=1.5):
        """Initializes FocalLoss with class balance and focusing parameters.

        Args:
            alpha (float, optional): Per-class weighting factor. Applied as
                alpha for positives (fraud) and (1-alpha) for negatives (benign).
                Defaults to 0.95.
            gamma (float, optional): Modulating exponent controlling the rate
                at which easy examples are down-weighted. Defaults to 1.5.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Higher alpha assigns more loss weight to the fraud class
        self.gamma = gamma  # Higher gamma increases focus on hard-to-classify nodes

    def forward(self, inputs, targets):
        """Computes the mean Focal Loss over a batch of node predictions.

        Args:
            inputs (torch.Tensor): Raw logit tensor of shape (N,) or (N, 1),
                prior to sigmoid activation.
            targets (torch.Tensor): Binary ground-truth labels of shape (N,)
                or (N, 1), with values in {0, 1}.

        Returns:
            torch.Tensor: Scalar mean Focal Loss value. Differentiable with
                respect to inputs for gradient-based optimization.
        """
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        # Compute element-wise binary cross-entropy without reduction
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none"
        )

        # pt = exp(-BCE) represents model confidence; ranges in (0, 1)
        pt = torch.exp(-BCE_loss)

        # Apply per-sample alpha weighting: alpha for positives, (1-alpha) for negatives
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Focal modulation: (1 - pt)^gamma downweights easy examples
        F_loss = alpha_t * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


def ts_to_date(ts):
    """Converts a Unix timestamp to a human-readable UTC date string.

    Args:
        ts (float | int): Unix timestamp in seconds.

    Returns:
        str: Date string formatted as 'YYYY-MM-DD' in UTC timezone.
    """
    return datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).strftime(
        "%Y-%m-%d"
    )


def train():
    """Executes the complete HTGNN training, validation, and evaluation pipeline.

    Implements the following sequential procedure:
      1. Seeds all random number generators for reproducibility.
      2. Loads the DGL heterogeneous graph and extracts node labels and timestamps.
      3. Performs a temporal train/val/test split:
           - Test set: the temporally latest 15% of nodes (by timestamp quantile).
           - Val set: stratified 15% of remaining nodes.
           - Train set: the remaining 85% of non-test nodes.
      4. Fits a StandardScaler on training node features and applies it to all
         splits (preventing data leakage into validation and test sets).
      5. Instantiates HTGNN with the configured hyperparameters.
      6. Trains for up to 100 epochs using FocalLoss + Adam + ReduceLROnPlateau.
      7. At each epoch, selects the optimal decision threshold via F1 maximization
         on the precision-recall curve over the validation set.
      8. Checkpoints the model with the best validation AUPRC.
      9. Applies early stopping with patience=15.
      10. Evaluates the best checkpoint on the held-out test set, generating
          confusion matrix, ROC, and PR curve artifacts.
      11. Serializes training configuration to results/experiment_config.json.
    """
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    dgl.seed(seed)

    # Enforce deterministic CuDNN behavior for reproducibility across runs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cpu")

    g = load_graph().to(device)

    if not hasattr(g, "graph_data"):
        g.graph_data = {}

    labels = g.nodes["node"].data["label"]
    num_classes = 1

    timestamps = g.nodes["node"].data["timestamp"].float()
    g.graph_data["max_timestamp"] = timestamps.max()

    # Temporal test split: retain the latest 15% of nodes by timestamp quantile
    test_threshold = torch.quantile(timestamps, 0.85)
    test_mask = timestamps > test_threshold

    # Partition remaining nodes for train and validation
    remaining_mask = ~test_mask

    remaining_indices = torch.where(remaining_mask)[0].cpu().numpy()
    remaining_labels = labels[remaining_mask].cpu().numpy()

    # Stratified train/val split preserves fraud class distribution in each partition
    train_idx, val_idx = train_test_split(
        remaining_indices, test_size=0.15, stratify=remaining_labels, random_state=42
    )

    train_mask = torch.zeros_like(labels, dtype=torch.bool)
    val_mask = torch.zeros_like(labels, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True

    print("Train nodes:", train_mask.sum().item())
    print("Val nodes:", val_mask.sum().item())
    print("Test nodes:", test_mask.sum().item())

    print("Train fraud count:", labels[train_mask].sum().item())
    print("Val fraud count:", labels[val_mask].sum().item())
    print("Test fraud count:", labels[test_mask].sum().item())

    feat_raw = g.nodes["node"].data["feat_raw"].cpu().numpy()

    # Fit scaler only on training nodes to prevent leakage into val/test distributions
    scaler = StandardScaler()
    feat_raw[train_mask.cpu().numpy()] = scaler.fit_transform(
        feat_raw[train_mask.cpu().numpy()]
    )

    # Transform val & test using the training-fit scaler parameters
    feat_raw[val_mask.cpu().numpy()] = scaler.transform(
        feat_raw[val_mask.cpu().numpy()]
    )

    feat_raw[test_mask.cpu().numpy()] = scaler.transform(
        feat_raw[test_mask.cpu().numpy()]
    )

    # Assign the normalized feature tensor back to the graph node store
    g.nodes["node"].data["feat"] = torch.tensor(feat_raw, dtype=torch.float32).to(
        device
    )

    # Serialize the fitted scaler for use by the inference dashboard and baselines
    joblib.dump(scaler, "DataSet/scaler.pkl")

    model = HTGNN(
        input_dim=g.nodes["node"].data["feat"].shape[1],
        hidden_dim=128,
        output_dim=num_classes,
        time_dim=32,
        edge_types=g.etypes,
    ).to(device)

    # Focal Loss with alpha=0.95 (fraud-focused) and gamma=1.5 down-weights easy negatives to handle the severe class imbalance
    criterion = FocalLoss(alpha=0.95, gamma=1.5)

    # Conservative learning rate for stable convergence in deep heterogeneous graphs
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)

    # ReduceLROnPlateau halves LR after 5 epochs of no AUPRC improvement
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    best_score = 0.0
    final_best_threshold = 0.5

    patience_counter = 0

    for epoch in range(1, 101):
        model.train()
        optimizer.zero_grad()

        # Set temporal boundary to the maximum timestamp in the training split
        train_max_time = timestamps[train_mask].max()

        if train_mask.sum() > 0:
            g.graph_data["current_time"] = train_max_time
        else:
            g.graph_data["current_time"] = timestamps.max()

        logits = model(g)

        # Compute FocalLoss over training node predictions only
        loss = criterion(
            logits[train_mask].view(-1), labels[train_mask].float().view(-1)
        )

        loss.backward()
        # Gradient clipping at L2 norm = 1.0 maintains training stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            # Use the validation split's maximum timestamp as the temporal boundary
            if val_mask.sum() > 0:
                val_max_time = timestamps[val_mask].max()
                g.graph_data["current_time"] = val_max_time
            else:
                g.graph_data["current_time"] = timestamps.max()

            val_logits = model(g)
            val_probs = torch.sigmoid(val_logits[val_mask]).view(-1)

            print("Mean val prob:", val_probs.mean().item())
            print("Max val prob:", val_probs.max().item())

            if val_mask.sum() == 0:
                print("No validation samples in this split. Skipping validation.")
                continue

            y_val = labels[val_mask].cpu().numpy()
            y_prob = val_probs.cpu().numpy()

            best_threshold = 0.5  # Default classification threshold

            if len(np.unique(y_val)) > 1:
                precision, recall, thresholds = precision_recall_curve(y_val, y_prob)
                # Compute F1 for each threshold; select the one maximizing F1
                f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
                optimal_idx = np.argmax(f1_scores)
                if optimal_idx < len(thresholds):
                    best_threshold = thresholds[optimal_idx]

            val_preds = (val_probs > best_threshold).long()
            val_metrics = compute_metrics(labels[val_mask], val_preds, val_probs)

            # AUPRC is the primary model selection criterion for imbalanced detection
            current_score = val_metrics["auprc"]
            scheduler.step(current_score)

            if current_score > best_score:
                best_score = current_score
                final_best_threshold = best_threshold
                checkpoint = {
                    "model_state": model.state_dict(),
                    "threshold": final_best_threshold,
                    "test_cutoff": test_threshold,
                }
                torch.save(checkpoint, "best_model.pth")
                shutil.copy("best_model.pth", "results/best_model.pth")
                print(
                    f"Epoch {epoch:02d}: New Best Val AUPRC: {best_score:.4f} - Model Saved"
                )
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= 15:
                print("Early stopping triggered.")
                break

        print(
            f"Epoch {epoch:02d} | "
            f"Loss: {loss.item():.4f} | "
            f"Val AUPRC: {val_metrics['auprc']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f}"
        )

    # Restore the best checkpoint for final test evaluation
    if os.path.exists("results/best_model.pth"):
        checkpoint = torch.load("results/best_model.pth")
        # Extract model weights from the structured checkpoint dictionary
        model.load_state_dict(checkpoint["model_state"])
        # Restore the optimal threshold determined during validation
        final_best_threshold = checkpoint["threshold"]
        print(f"Loaded best model with threshold: {final_best_threshold:.4f}")

    model.eval()
    with torch.no_grad():
        # Set temporal boundary to the test split's latest timestamp
        if test_mask.sum() > 0:
            g.graph_data["current_time"] = timestamps[test_mask].max()
        else:
            g.graph_data["current_time"] = timestamps.max()

        logits = model(g).view(-1)
        probs = torch.sigmoid(logits[test_mask])

        # Apply the optimal threshold determined during validation
        preds = (probs > final_best_threshold).long()

        y_true = labels[test_mask].cpu().numpy()
        y_pred = preds.cpu().numpy()
        y_probs = probs.cpu().numpy()

        # Compute test set AUPRC as the primary reporting metric
        test_auprc = average_precision_score(y_true, y_probs)
        print(f"Test AUPRC: {test_auprc:.4f}")

        plot_confusion_matrix(y_true, y_pred, save_path="results/confusion_matrix.png")
        plot_roc_curve(y_true, y_probs, save_path="results/roc_curve.png")
        plot_precision_recall_curve(y_true, y_probs, save_path="results/pr_curve.png")
        print(
            f"Final Evaluation complete using Optimal Threshold: {final_best_threshold:.4f}"
        )

    print("Val cutoff:", ts_to_date(test_threshold.item()))

    # Serialize the full experimental configuration for reproducibility tracking
    experiment_config = {
        "seed": seed,
        "hidden_dim": 128,
        "time_dim": 32,
        "learning_rate": 0.0005,
        "weight_decay": 1e-4,
        "alpha": 0.95,
        "gamma": 1.5,
        "test_cutoff": float(test_threshold.item()),
        "train_size": int(train_mask.sum().item()),
        "val_size": int(val_mask.sum().item()),
        "test_size": int(test_mask.sum().item()),
    }

    with open("results/experiment_config.json", "w") as f:
        json.dump(experiment_config, f, indent=4)


if __name__ == "__main__":
    train()
