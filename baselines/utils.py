import dgl
import numpy as np
import torch
from sklearn.model_selection import train_test_split


def load_data():
    """Load graph and return raw features, labels, and masks."""
    graphs, _ = dgl.load_graphs("DataSet/graph.bin")
    g = graphs[0]
    g = g.to(torch.device("cpu"))

    labels = g.nodes["node"].data["label"].cpu().numpy()
    timestamps = g.nodes["node"].data["timestamp"].float()
    feat_raw = g.nodes["node"].data["feat_raw"].cpu().numpy()  # shape (N, D)

    # Temporal split (same as train.py)
    test_threshold = torch.quantile(timestamps, 0.85)
    test_mask = (timestamps > test_threshold).cpu().numpy()
    remaining_mask = ~test_mask

    remaining_indices = np.where(remaining_mask)[0]
    remaining_labels = labels[remaining_mask]

    train_idx, val_idx = train_test_split(
        remaining_indices, test_size=0.15, stratify=remaining_labels, random_state=42
    )

    train_mask = np.zeros_like(labels, dtype=bool)
    val_mask = np.zeros_like(labels, dtype=bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True

    return feat_raw, labels, train_mask, val_mask, test_mask
