"""
Baseline: Graph Convolutional Network (GCN) on a homogeneous projection of the DeFi graph.

This module implements a two-layer GCN baseline (Kipf & Welling, 2017) operating
on a homogeneous projection of the heterogeneous DGL transaction graph. Like
the GAT baseline, heterogeneous edge semantics are collapsed via `dgl.to_homogeneous`,
discarding relational type distinctions. The GCN serves as a foundational
structure-aware baseline without attention mechanisms, enabling comparison against
both the attention-based (GAT, HTGNN) and relation-type-aware (RGCN, HTGNN) models.
This module is exclusively used for comparative baseline evaluation.
"""

import dgl
import dgl.nn as dglnn
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import average_precision_score

from .utils import load_data


class GCN(nn.Module):
    """Two-layer Graph Convolutional Network for homogeneous fraud classification.

    Implements the standard spectral GCN with symmetric normalized adjacency
    (Kipf & Welling, 2017). Uses mean-field neighbourhood aggregation, which
    provides a non-attentive baseline for comparison against the multi-head
    attention mechanism in HTGNN. Operates on a type-collapsed homogeneous
    graph projection.

    Attributes:
        conv1 (dglnn.GraphConv): First GCN layer mapping in_feats to hidden_feats.
        conv2 (dglnn.GraphConv): Second GCN layer mapping hidden_feats to out_feats.
    """

    def __init__(self, in_feats, hidden_feats, out_feats):
        """Constructs the two-layer GCN.

        Args:
            in_feats (int): Input node feature dimensionality.
            hidden_feats (int): Hidden representation size after the first
                graph convolution layer.
            out_feats (int): Output logit dimensionality. Set to 1 for binary
                fraud classification.
        """
        super().__init__()
        self.conv1 = dglnn.GraphConv(in_feats, hidden_feats)
        self.conv2 = dglnn.GraphConv(hidden_feats, out_feats)

    def forward(self, g, x):
        """Performs a two-layer GCN forward pass on a homogeneous graph.

        Args:
            g (dgl.DGLGraph): Homogeneous input graph with self-loops.
            x (torch.Tensor): Node feature tensor of shape (N, in_feats).

        Returns:
            torch.Tensor: Raw logit tensor of shape (N, out_feats). Apply
                torch.sigmoid() to convert to fraud probabilities.
        """
        # First graph convolution with ReLU nonlinearity
        x = F.relu(self.conv1(g, x))
        # Second graph convolution produces raw output logits
        x = self.conv2(g, x)
        return x


def main():
    """Trains and evaluates the GCN baseline on the homogeneous DeFi graph.

    Loads the DGL heterograph, projects it to a homogeneous graph, adds self-loops,
    applies the pre-fitted StandardScaler, and trains a two-layer GCN with
    BCE loss weighted by the training partition's class imbalance ratio. The best
    model checkpoint by validation AUPRC is retained for final test evaluation.
    """
    # Load the heterogeneous graph and project to homogeneous for GCN compatibility
    graphs, _ = dgl.load_graphs("DataSet/graph.bin")
    g_hetero = graphs[0]
    g = dgl.to_homogeneous(g_hetero, ndata=["feat_raw"])
    # Self-loops ensure all nodes receive their own features during aggregation
    g = dgl.add_self_loop(g)
    feat_raw = g.ndata["feat_raw"].numpy()

    # Apply the training-fit scaler for distributional consistency with HTGNN
    scaler = joblib.load("DataSet/scaler.pkl")
    feat_scaled = scaler.transform(feat_raw)
    g.ndata["feat"] = torch.tensor(feat_scaled, dtype=torch.float32)

    # Load temporally-consistent train/val/test masks (order preserved)
    _, labels, train_mask, val_mask, test_mask = load_data()
    labels = torch.tensor(labels, dtype=torch.long)
    train_mask = torch.tensor(train_mask, dtype=torch.bool)
    val_mask = torch.tensor(val_mask, dtype=torch.bool)
    test_mask = torch.tensor(test_mask, dtype=torch.bool)

    device = torch.device("cpu")
    g = g.to(device)
    labels = labels.to(device)

    in_feats = g.ndata["feat"].shape[1]
    hidden_feats = 128
    out_feats = 1

    model = GCN(in_feats, hidden_feats, out_feats).to(device)

    # Compute pos_weight from training partition class counts to handle imbalance
    pos_weight = (
        (labels[train_mask] == 0).sum().float()
        / (labels[train_mask] == 1).sum().float()
    ).item()
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    best_val_auprc = 0.0
    best_model_state = None

    for epoch in range(200):
        model.train()
        logits = model(g, g.ndata["feat"]).squeeze()
        loss = criterion(logits[train_mask], labels[train_mask].float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(g, g.ndata["feat"]).squeeze()
            val_probs = torch.sigmoid(val_logits[val_mask]).cpu().numpy()
            val_labels = labels[val_mask].cpu().numpy()
            # Track the model checkpoint with the highest validation AUPRC
            val_auprc = average_precision_score(val_labels, val_probs)
            if val_auprc > best_val_auprc:
                best_val_auprc = val_auprc
                best_model_state = model.state_dict()
        if epoch % 20 == 0:
            print(
                f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | Val AUPRC: {val_auprc:.4f}"
            )

    # Restore the best validation checkpoint for final test evaluation
    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        test_logits = model(g, g.ndata["feat"]).squeeze()
        test_probs = torch.sigmoid(test_logits[test_mask]).cpu().numpy()
        test_labels = labels[test_mask].cpu().numpy()
        test_auprc = average_precision_score(test_labels, test_probs)
        print(f"\nBest validation AUPRC: {best_val_auprc:.4f}")
        print(f"Test AUPRC: {test_auprc:.4f}")


if __name__ == "__main__":
    main()
