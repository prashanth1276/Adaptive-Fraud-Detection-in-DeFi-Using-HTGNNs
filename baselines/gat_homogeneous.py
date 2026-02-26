"""
Baseline: Graph Attention Network (GAT) on a homogeneous projection of the DeFi graph.

This module implements a two-layer, 8-head GAT baseline operating on a homogeneous
projection of the heterogeneous DGL transaction graph. Heterogeneous edge semantics
are collapsed into a single edge type via `dgl.to_homogeneous`, discarding relational
type information. This baseline quantifies the marginal value of heterogeneous
message passing in the HTGNN by comparing a type-aware model against a type-agnostic
attention network under identical feature preprocessing and training conditions.
It is exclusively used for baseline comparison and does not participate in the
production inference pipeline.
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


class GAT(nn.Module):
    """Two-layer Graph Attention Network for homogeneous fraud classification.

    Implements the standard GAT architecture (Velickovic et al., 2018) with
    multi-head attention in the first layer and single-head output in the
    second layer. Operates on a homogeneous graph projection where all edge
    types are merged into a single relation, providing a structurally-aware
    but type-unaware baseline for comparison against HTGNN.

    Attributes:
        conv1 (dglnn.GATConv): First GAT layer with num_heads attention heads.
            Input: in_feats, output: hidden_feats (via head concatenation).
        conv2 (dglnn.GATConv): Second GAT layer with a single attention head.
            Input: hidden_feats, output: out_feats (raw logit).
    """

    def __init__(self, in_feats, hidden_feats, out_feats, num_heads=8):
        """Constructs the two-layer GAT with configurable attention head count.

        Args:
            in_feats (int): Input node feature dimensionality.
            hidden_feats (int): Hidden representation size after the first
                GAT layer (total across all heads). Must be divisible by num_heads.
            out_feats (int): Output dimensionality. Set to 1 for binary fraud
                classification with BCEWithLogitsLoss.
            num_heads (int, optional): Number of parallel attention heads in
                the first layer. Defaults to 8.
        """
        super().__init__()
        # First GAT layer: hidden_feats // num_heads per-head, flattened to hidden_feats
        self.conv1 = dglnn.GATConv(in_feats, hidden_feats // num_heads, num_heads)
        # Second GAT layer uses a single attention head for the output projection
        self.conv2 = dglnn.GATConv(hidden_feats, out_feats, 1)

    def forward(self, g, x):
        """Performs a two-layer GAT forward pass on a homogeneous graph.

        Args:
            g (dgl.DGLGraph): Homogeneous input graph with self-loops added.
            x (torch.Tensor): Node feature tensor of shape (N, in_feats).

        Returns:
            torch.Tensor: Raw logit tensor of shape (N,). Apply torch.sigmoid()
                to convert to fraud probabilities.
        """
        # Flatten multi-head first layer output and apply ELU nonlinearity
        x = self.conv1(g, x).flatten(1)
        x = F.elu(x)
        # Single-head second layer; squeeze removes the redundant head dimension
        x = self.conv2(g, x).squeeze()
        return x


def main():
    """Trains and evaluates the GAT baseline on the homogeneous DeFi graph.

    Loads the DGL heterograph, projects it to a homogeneous graph via
    `dgl.to_homogeneous`, adds self-loops (required for GAT to handle isolated
    nodes), applies the pre-fitted StandardScaler, and trains a two-layer GAT
    with BCE loss weighted by the training set's class imbalance ratio. The
    best-performing checkpoint (by validation AUPRC) is retained for final
    test evaluation.
    """
    # Load the heterogeneous graph and project to homogeneous for GAT compatibility
    graphs, _ = dgl.load_graphs("DataSet/graph.bin")
    g_hetero = graphs[0]
    g = dgl.to_homogeneous(g_hetero, ndata=["feat_raw"])
    # Self-loops are required by GATConv to prevent zero in-degree errors
    g = dgl.add_self_loop(g)
    feat_raw = g.ndata["feat_raw"].numpy()

    # Apply the training-fit scaler to ensure distributional alignment with HTGNN
    scaler = joblib.load("DataSet/scaler.pkl")
    feat_scaled = scaler.transform(feat_raw)
    g.ndata["feat"] = torch.tensor(feat_scaled, dtype=torch.float32)

    # Load temporally-consistent train/val/test masks from the shared utility
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

    model = GAT(in_feats, hidden_feats, out_feats).to(device)

    # Compute pos_weight from training partition class counts to address imbalance
    pos_weight = (
        (labels[train_mask] == 0).sum().float()
        / (labels[train_mask] == 1).sum().float()
    ).item()
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

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
            # Track the model state achieving the highest validation AUPRC
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
