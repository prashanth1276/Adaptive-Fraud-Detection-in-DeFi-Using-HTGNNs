"""
Shared data loading utility for baseline model evaluation.

This module provides the `load_data` function, which loads the pre-processed DGL
heterogeneous transaction graph and produces the raw node feature matrix, binary
fraud labels, and temporally-consistent train/val/test boolean masks. The temporal
splitting strategy (85th percentile timestamp threshold for test, followed by
stratified 85/15 train/val split of remaining nodes) is identical to the procedure
in train/train.py, ensuring that all baseline models are evaluated on exactly the
same data partitions as the primary HTGNN. This module is consumed by all modules
in the baselines package.
"""

import dgl
import numpy as np
import torch
from sklearn.model_selection import train_test_split


def load_data():
    """Loads the DGL graph and returns raw features, labels, and temporal split masks.

    Applies the identical temporal splitting procedure as train/train.py to ensure
    a controlled comparison: the test set consists of the temporally latest 15%
    of nodes (by node-level last-activity timestamp), and the remaining nodes are
    stratified into 85% train and 15% validation partitions.

    Returns:
        tuple:
            - feat_raw (np.ndarray): Raw (pre-StandardScaler) node feature matrix
              of shape (N, D), where D is the number of node features.
            - labels (np.ndarray): Binary fraud label array of shape (N,), with
              values in {0, 1}.
            - train_mask (np.ndarray): Boolean array of shape (N,) marking training nodes.
            - val_mask (np.ndarray): Boolean array of shape (N,) marking validation nodes.
            - test_mask (np.ndarray): Boolean array of shape (N,) marking test nodes.
    """
    graphs, _ = dgl.load_graphs("DataSet/graph.bin")
    g = graphs[0]
    g = g.to(torch.device("cpu"))

    labels = g.nodes["node"].data["label"].cpu().numpy()
    timestamps = g.nodes["node"].data["timestamp"].float()
    feat_raw = g.nodes["node"].data["feat_raw"].cpu().numpy()  # shape (N, D)

    # Temporal split: test nodes are those with timestamp above the 85th quantile
    test_threshold = torch.quantile(timestamps, 0.85)
    test_mask = (timestamps > test_threshold).cpu().numpy()
    remaining_mask = ~test_mask

    remaining_indices = np.where(remaining_mask)[0]
    remaining_labels = labels[remaining_mask]

    # Stratified split preserves the fraud-to-benign ratio across train and val partitions
    train_idx, val_idx = train_test_split(
        remaining_indices, test_size=0.15, stratify=remaining_labels, random_state=42
    )

    train_mask = np.zeros_like(labels, dtype=bool)
    val_mask = np.zeros_like(labels, dtype=bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True

    return feat_raw, labels, train_mask, val_mask, test_mask
