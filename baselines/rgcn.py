"""
Baseline: Relational Graph Convolutional Network (RGCN) on the DeFi transaction graph.

This module implements the RGCN baseline (Schlichtkrull et al., 2018), which extends
the standard GCN to heterogeneous graphs by maintaining separate weight matrices per
relation type and decomposing them via basis functions to reduce parameter count.
Unlike the full HTGNN, RGCN operates without temporal encoding or edge masking,
providing a heterogeneous-but-static structural baseline. The homogeneous graph
with explicit edge type tensors is constructed manually to satisfy DGL's
RelGraphConv API. This module is exclusively used for comparative baseline evaluation.
"""

import dgl
import dgl.nn as dglnn
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import average_precision_score

from baselines.utils import load_data


class RGCN(nn.Module):
    """Two-layer Relational Graph Convolutional Network for typed-edge fraud detection.

    Implements the RGCN architecture using basis decomposition to share parameters
    across relation types, mitigating over-parameterization in graphs with multiple
    edge types. Provides a heterogeneous-aware but temporally-static baseline for
    comparison against the full HTGNN.

    Attributes:
        conv1 (dglnn.RelGraphConv): First relational GCN layer with ReLU activation.
            Maps in_dim to hidden_dim using basis-decomposed relation-specific weights.
        conv2 (dglnn.RelGraphConv): Second relational GCN layer without activation.
            Maps hidden_dim to out_dim.
        num_rels (int): Number of distinct edge relation types in the graph.
    """

    def __init__(self, in_dim, hidden_dim, out_dim, num_rels):
        """Constructs the two-layer RGCN with basis decomposition.

        Args:
            in_dim (int): Input node feature dimensionality.
            hidden_dim (int): Hidden representation size after the first
                relational convolution layer.
            out_dim (int): Output logit dimensionality. Set to 1 for binary
                fraud classification.
            num_rels (int): Number of distinct canonical edge types. Used as
                both the relation count and the basis count.
        """
        super().__init__()
        # Basis decomposition parameterization reduces parameters from O(R*D^2) to O(B*D^2)
        self.conv1 = dglnn.RelGraphConv(
            in_dim, hidden_dim, num_rels, "basis", num_rels, activation=F.relu
        )
        self.conv2 = dglnn.RelGraphConv(
            hidden_dim, out_dim, num_rels, "basis", num_rels
        )
        self.num_rels = num_rels

    def forward(self, g, x, etypes):
        """Performs a two-layer RGCN forward pass with relation-specific convolutions.

        Args:
            g (dgl.DGLGraph): Homogeneous graph with edge type indices stored
                as a per-edge integer attribute.
            x (torch.Tensor): Node feature tensor of shape (N, in_dim).
            etypes (torch.Tensor): Edge type index tensor of shape (E,), where
                each value is an integer identifying the edge's relation type.

        Returns:
            torch.Tensor: Raw logit tensor of shape (N, out_dim). Apply
                torch.sigmoid() to convert to fraud probabilities.
        """
        # First RGCN layer applies ReLU and relation-specific weight decomposition
        h = self.conv1(g, x, etypes)
        # Second RGCN layer produces the output logit representation
        h = self.conv2(g, h, etypes)
        return h


def main():
    """Trains and evaluates the RGCN baseline on the DeFi transaction graph.

    Constructs a homogeneous DGL graph with explicit edge type tensors from the
    heterogeneous source graph, applies the pre-fitted StandardScaler, and trains
    a two-layer RGCN with BCE loss weighted by class imbalance ratio. The best
    checkpoint by validation AUPRC is retained for final test evaluation.
    """
    graphs, _ = dgl.load_graphs("DataSet/graph.bin")
    g = graphs[0]
    g = g.to(torch.device("cpu"))

    # Build a homogeneous graph with edge types preserved as an integer tensor.
    # RelGraphConv requires a flat graph with a per-edge edge-type index tensor.
    # Iterate over canonical edge types to extract and concatenate all edges.
    ntype = g.ntypes[0]
    feat_raw = g.nodes[ntype].data["feat_raw"].numpy()
    scaler = joblib.load("DataSet/scaler.pkl")
    feat_scaled = scaler.transform(feat_raw)
    num_nodes = g.num_nodes(ntype)

    # Construct edge index and type lists from the heterogeneous graph
    src_list = []
    dst_list = []
    etype_list = []
    etypes = g.canonical_etypes
    # Map canonical edge type tuples to integer indices for RelGraphConv
    etype_to_id = {et: i for i, et in enumerate(etypes)}
    for et in etypes:
        u, v = g.edges(etype=et)
        src_list.append(u)
        dst_list.append(v)
        # Assign integer relation ID to every edge of this type
        etype_list.append(torch.full((u.shape[0],), etype_to_id[et], dtype=torch.long))
    src = torch.cat(src_list)
    dst = torch.cat(dst_list)
    edge_type = torch.cat(etype_list)
    # Assemble the flattened homogeneous graph with all relation edges included
    g_homo = dgl.graph((src, dst), num_nodes=num_nodes)
    g_homo.ndata["feat"] = torch.tensor(feat_scaled, dtype=torch.float32)
    g_homo.edata["etype"] = edge_type

    # Load temporally-consistent train/val/test masks
    _, labels, train_mask, val_mask, test_mask = load_data()
    labels = torch.tensor(labels, dtype=torch.long)
    train_mask = torch.tensor(train_mask, dtype=torch.bool)
    val_mask = torch.tensor(val_mask, dtype=torch.bool)
    test_mask = torch.tensor(test_mask, dtype=torch.bool)

    in_dim = g_homo.ndata["feat"].shape[1]
    hidden_dim = 128
    out_dim = 1
    num_rels = len(etypes)

    model = RGCN(in_dim, hidden_dim, out_dim, num_rels)
    device = torch.device("cpu")
    model.to(device)

    # Compute pos_weight from training partition class distribution to address imbalance
    pos_weight = (
        (labels[train_mask] == 0).sum().float()
        / (labels[train_mask] == 1).sum().float()
    ).item()
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

    best_val_auprc = 0.0
    best_model_state = None

    for epoch in range(200):
        model.train()
        logits = model(g_homo, g_homo.ndata["feat"], g_homo.edata["etype"]).squeeze()
        loss = criterion(logits[train_mask], labels[train_mask].float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(
                g_homo, g_homo.ndata["feat"], g_homo.edata["etype"]
            ).squeeze()
            val_probs = torch.sigmoid(val_logits[val_mask]).cpu().numpy()
            val_labels = labels[val_mask].cpu().numpy()
            val_auprc = average_precision_score(val_labels, val_probs)
            # Track the model checkpoint with the highest validation AUPRC
            if val_auprc > best_val_auprc:
                best_val_auprc = val_auprc
                best_model_state = model.state_dict()
        if epoch % 20 == 0:
            print(
                f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | Val AUPRC: {val_auprc:.4f}"
            )

    # Restore the best checkpoint for final test set evaluation
    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        test_logits = model(
            g_homo, g_homo.ndata["feat"], g_homo.edata["etype"]
        ).squeeze()
        test_probs = torch.sigmoid(test_logits[test_mask]).cpu().numpy()
        test_labels = labels[test_mask].cpu().numpy()
        test_auprc = average_precision_score(test_labels, test_probs)
        print(f"\nBest validation AUPRC: {best_val_auprc:.4f}")
        print(f"Test AUPRC: {test_auprc:.4f}")


if __name__ == "__main__":
    main()
