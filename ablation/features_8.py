import datetime
import os
import random

import dgl
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


def load_graph():
    graphs, _ = dgl.load_graphs("DataSet/graph.bin")
    return graphs[0]


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.95, gamma=1.5):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none"
        )
        pt = torch.exp(-BCE_loss)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        F_loss = alpha_t * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


def ts_to_date(ts):
    return datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).strftime(
        "%Y-%m-%d"
    )


def train():
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    dgl.seed(seed)
    device = torch.device("cpu")

    g = load_graph().to(device)
    if not hasattr(g, "graph_data") or not g.graph_data:
        g.graph_data = {}
    labels = g.nodes["node"].data["label"]
    num_classes = 1
    timestamps = g.nodes["node"].data["timestamp"].float()

    # Temporal split
    test_threshold = torch.quantile(timestamps, 0.85)
    test_mask = timestamps > test_threshold
    remaining_mask = ~test_mask
    remaining_indices = torch.where(remaining_mask)[0].cpu().numpy()
    remaining_labels = labels[remaining_mask].cpu().numpy()
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

    # Use only the first 8 features (original set)
    feat_raw_all = g.nodes["node"].data["feat_raw"].cpu().numpy()
    feat_raw = feat_raw_all[:, :8]  # shape (N, 8)

    scaler = StandardScaler()
    feat_raw[train_mask.cpu().numpy()] = scaler.fit_transform(
        feat_raw[train_mask.cpu().numpy()]
    )
    feat_raw[val_mask.cpu().numpy()] = scaler.transform(
        feat_raw[val_mask.cpu().numpy()]
    )
    feat_raw[test_mask.cpu().numpy()] = scaler.transform(
        feat_raw[test_mask.cpu().numpy()]
    )

    g.nodes["node"].data["feat"] = torch.tensor(feat_raw, dtype=torch.float32).to(
        device
    )

    model = HTGNN(
        input_dim=8,
        hidden_dim=128,
        output_dim=num_classes,
        time_dim=32,
        edge_types=g.etypes,
    ).to(device)

    criterion = FocalLoss(alpha=0.95, gamma=1.5)
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    best_score = 0.0
    final_best_threshold = 0.5
    patience_counter = 0

    for epoch in range(1, 101):
        model.train()
        optimizer.zero_grad()

        train_max_time = timestamps[train_mask].max()
        if train_mask.sum() > 0:
            g.graph_data["current_time"] = train_max_time
        else:
            g.graph_data["current_time"] = timestamps.max()

        logits = model(g)
        loss = criterion(
            logits[train_mask].view(-1), labels[train_mask].float().view(-1)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            if val_mask.sum() > 0:
                val_max_time = timestamps[val_mask].max()
                g.graph_data["current_time"] = val_max_time
            else:
                g.graph_data["current_time"] = timestamps.max()

            val_logits = model(g)
            val_probs = torch.sigmoid(val_logits[val_mask]).view(-1)

            y_val = labels[val_mask].cpu().numpy()
            y_prob = val_probs.cpu().numpy()

            best_threshold = 0.5
            if len(np.unique(y_val)) > 1:
                precision, recall, thresholds = precision_recall_curve(y_val, y_prob)
                f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
                optimal_idx = np.argmax(f1_scores)
                if optimal_idx < len(thresholds):
                    best_threshold = thresholds[optimal_idx]

            val_preds = (val_probs > best_threshold).long()
            val_metrics = compute_metrics(labels[val_mask], val_preds, val_probs)

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
                torch.save(checkpoint, "results/ablation_8features_best.pth")
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
            f"Epoch {epoch:02d} | Loss: {loss.item():.4f} | Val AUPRC: {val_metrics['auprc']:.4f} | Val F1: {val_metrics['f1']:.4f}"
        )

    if os.path.exists("results/ablation_8features_best.pth"):
        checkpoint = torch.load("results/ablation_8features_best.pth")
        model.load_state_dict(checkpoint["model_state"])
        final_best_threshold = checkpoint["threshold"]

    model.eval()
    with torch.no_grad():
        if test_mask.sum() > 0:
            g.graph_data["current_time"] = timestamps[test_mask].max()
        else:
            g.graph_data["current_time"] = timestamps.max()

        logits = model(g).view(-1)
        probs = torch.sigmoid(logits[test_mask])
        preds = (probs > final_best_threshold).long()
        y_true = labels[test_mask].cpu().numpy()
        y_pred = preds.cpu().numpy()
        y_probs = probs.cpu().numpy()

        plot_confusion_matrix(
            y_true, y_pred, save_path="results/ablation_8features_cm.png"
        )
        plot_roc_curve(y_true, y_probs, save_path="results/ablation_8features_roc.png")
        plot_precision_recall_curve(
            y_true, y_probs, save_path="results/ablation_8features_pr.png"
        )
        print(f"Final Test AUPRC: {average_precision_score(y_true, y_probs):.4f}")


if __name__ == "__main__":
    train()
