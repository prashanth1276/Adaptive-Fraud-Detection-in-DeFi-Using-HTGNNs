"""
Ablation study: HTGNN architecture variant without temporal encoding.

This module defines the HTGNN_NoTemporal model, a static graph counterpart
to the full HTGNN (model/htg_nn.py). It removes the sinusoidal time encoding
layer, per-edge-type temporal projections, and temporal edge masking, retaining
only the heterogeneous GAT message-passing layers and the MLP classifier. This
variant serves as the structural baselines in the ablation study to quantify
the isolated performance contribution of the temporal encoding component.
It is consumed exclusively by ablation/no_temporal.py and does not participate
in the production training or inference pipelines.
"""

import dgl.nn as dglnn
import torch
import torch.nn as nn


class HTGNN_NoTemporal(nn.Module):
    """Static heterogeneous GAT model without temporal edge encoding.

    Implements the same two-layer heterogeneous graph attention architecture
    as HTGNN but omits: (1) sinusoidal time encoding, (2) per-edge-type
    temporal feature projection, and (3) temporal edge masking. Node feature
    tensors are passed directly into the first convolution layer without
    temporal augmentation. This isolates the contribution of the temporal
    modeling component in the full HTGNN when compared under identical
    training conditions.

    Attributes:
        conv1 (dglnn.HeteroGraphConv): First heterogeneous GAT layer mapping
            raw node features (input_dim) to hidden_dim via 8 attention heads.
        conv2 (dglnn.HeteroGraphConv): Second heterogeneous GAT layer refining
            hidden_dim representations via 8 attention heads.
        bn (nn.LayerNorm): Layer normalization applied after the residual
            addition of conv1 and conv2 outputs.
        classifier (nn.Sequential): Two-layer MLP mapping hidden_dim to output_dim
            fraud logits.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, edge_types):
        """Constructs the static HTGNN variant without temporal components.

        Args:
            input_dim (int): Dimensionality of the input node feature tensor.
                Unlike the full HTGNN, this is not expanded by time_dim.
            hidden_dim (int): Hidden representation size. Must be divisible by
                the number of attention heads (8). Recommended: 128.
            output_dim (int): Number of output logits per node. Set to 1 for
                binary fraud classification.
            edge_types (list[str]): List of canonical edge type strings from
                the DGL heterograph's `etypes` attribute.
        """
        super().__init__()
        self.conv1 = dglnn.HeteroGraphConv(
            {
                etype: dglnn.GATConv(
                    input_dim, hidden_dim // 8, num_heads=8, allow_zero_in_degree=True
                )
                for etype in edge_types
            },
            aggregate="sum",
        )
        self.conv2 = dglnn.HeteroGraphConv(
            {
                etype: dglnn.GATConv(
                    hidden_dim, hidden_dim // 8, num_heads=8, allow_zero_in_degree=True
                )
                for etype in edge_types
            },
            aggregate="sum",
        )
        self.bn = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, g):
        """Performs a forward pass without any temporal edge conditioning.

        Executes heterogeneous message passing over all edges in the graph
        without applying temporal masking or time embedding augmentation.
        Node features are fed directly into the first attention layer.

        Args:
            g (dgl.DGLHeteroGraph): Input heterogeneous graph with
                g.nodes['node'].data['feat'] of shape (N, input_dim).

        Returns:
            torch.Tensor: Raw logit tensor of shape (N, output_dim). Apply
                torch.sigmoid() to convert to fraud probabilities.
        """
        x = g.nodes["node"].data["feat"]
        # First heterogeneous GAT layer aggregates neighbourhood features
        h1 = self.conv1(g, {"node": x})["node"]
        # Flatten multi-head attention output from (N, num_heads, head_dim) to (N, hidden_dim)
        h1 = h1.view(h1.shape[0], -1)
        h1 = torch.relu(h1)
        # Second GAT layer further refines the aggregated representations
        h2 = self.conv2(g, {"node": h1})["node"]
        h2 = h2.view(h2.shape[0], -1)
        # Residual connection and LayerNorm stabilize gradient flow
        h = self.bn(h2 + h1)
        return self.classifier(h)
