"""
Core Heterogeneous Temporal Graph Neural Network (HTGNN) architecture.

This module defines the primary GNN model used for inductive fraud representation
learning over evolving DeFi transaction graphs. The architecture combines sinusoidal
temporal edge encodings with multi-head heterogeneous graph attention (HeteroGAT)
to produce per-node risk logits. Temporal edge masking is applied at each forward
pass to enforce causal consistency, ensuring that only edges timestamped at or
before the query time are included in the message-passing computation. This module
is shared between training (train/train.py), ablation studies (ablation/), and
live inference (app.py).
"""

import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn

from model.time_encoding import TimeEncoding


class HTGNN(nn.Module):
    """Heterogeneous Temporal Graph Neural Network for DeFi fraud detection.

    Implements a two-layer heterogeneous graph attention network augmented with
    relative temporal encodings. Each forward pass:
      1. Encodes relative edge timestamps (delta_t) via sinusoidal time embeddings.
      2. Applies per-edge-type linear projections to align temporal embeddings.
      3. Aggregates temporal embeddings via message passing to produce per-node
         temporal context vectors.
      4. Concatenates temporal context with raw node feature tensors.
      5. Applies two successive HeteroGraphConv layers (GAT-based) with residual
         connection and LayerNorm.
      6. Passes the normalized representation through a two-layer MLP classifier.

    Temporal edge masking ensures causal integrity: only edges with timestamps
    at or before `current_time` participate in message aggregation, enabling
    temporally consistent inductive inference on unseen future nodes.

    Attributes:
        time_encoder (TimeEncoding): Sinusoidal encoder mapping scalar delta_t
            values to vectors of shape (E, time_dim).
        edge_proj (nn.ModuleDict): Per-edge-type linear projections of shape
            (time_dim, time_dim) to allow edge-type-specific temporal weighting.
        conv1 (dglnn.HeteroGraphConv): First heterogeneous GAT layer; input
            dimension is (input_dim + time_dim), output hidden_dim via 8 heads.
        conv2 (dglnn.HeteroGraphConv): Second heterogeneous GAT layer; input
            hidden_dim, output hidden_dim via 8 heads.
        bn (nn.LayerNorm): Layer normalisation applied after residual addition
            of conv1 and conv2 outputs.
        classifier (nn.Sequential): Two-layer MLP outputting raw fraud logits
            of shape (N, output_dim).
    """

    def __init__(self, input_dim, hidden_dim, output_dim, time_dim, edge_types):
        """Constructs the HTGNN with per-edge-type attention and temporal encoding.

        Args:
            input_dim (int): Dimensionality of the raw node feature tensor
                (feat_raw after StandardScaler normalization).
            hidden_dim (int): Hidden representation size. Must be divisible by
                the number of attention heads (8). Recommended: 128.
            output_dim (int): Number of output logits per node. Set to 1 for
                binary fraud classification with BCEWithLogitsLoss.
            time_dim (int): Dimensionality of temporal embeddings. Must be even.
                Recommended: 32.
            edge_types (list[str]): List of canonical edge type strings drawn
                from the DGL heterograph's `etypes` attribute (e.g.,
                ['eth_transfer', 'token_transfer', 'contract_call']).
        """
        super().__init__()

        # Step 1: Temporal encoding layer mapping delta_t (days) to time_dim vectors
        self.time_encoder = TimeEncoding(time_dim)

        # Edge-type-specific linear projections align temporal embeddings per relation
        self.edge_proj = nn.ModuleDict(
            {etype: nn.Linear(time_dim, time_dim) for etype in edge_types}
        )

        # Step 2: First heterogeneous GAT layer with 8-head attention.
        # Input dimension is expanded by time_dim to incorporate temporal context.
        self.conv1 = dglnn.HeteroGraphConv(
            {
                etype: dglnn.GATConv(
                    input_dim + time_dim,
                    hidden_dim // 8,
                    num_heads=8,
                    allow_zero_in_degree=True,
                )
                for etype in edge_types
            },
            aggregate="sum",
        )

        # Step 3: Second heterogeneous GAT layer refines representations
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

        # Step 4: Two-layer MLP classifier mapping hidden_dim embeddings to risk logits
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, g):
        """Performs a single forward pass producing per-node fraud logits.

        Applies temporal edge masking, sinusoidal time encoding, heterogeneous
        message aggregation, and GNN-based representation learning to generate
        raw (pre-sigmoid) fraud scores for all nodes in the input graph.

        Args:
            g (dgl.DGLHeteroGraph): Input heterogeneous graph containing:
                - g.nodes['node'].data['feat'] (torch.Tensor): Node feature
                  tensor of shape (N, input_dim), pre-scaled.
                - g.edges[etype].data['timestamp'] (torch.Tensor): Edge-level
                  Unix timestamps of shape (E_etype,).
                - g.graph_data['current_time'] (float | torch.Tensor): The
                  temporal query boundary; edges after this time are masked.
                - g.graph_data['max_timestamp'] (float): Fallback reference
                  time when current_time is not set.

        Returns:
            torch.Tensor: Raw logit tensor of shape (N, output_dim). Apply
                torch.sigmoid() to convert to fraud probabilities.
        """
        # Use local_var() to prevent in-place graph data mutation across calls
        g = g.local_var()
        x = g.nodes["node"].data["feat"]

        # Resolve the temporal query boundary from graph metadata
        if hasattr(g, "graph_data") and "current_time" in g.graph_data:
            current_time = g.graph_data["current_time"]
        else:
            if hasattr(g, "graph_data") and "max_timestamp" in g.graph_data:
                current_time = g.graph_data["max_timestamp"]
            else:
                current_time = float("inf")
                print("Warning: No current_time, using all edges.")

        # Temporal edge masking: retain only causally valid edges (t <= current_time)
        edge_mask_dict = {}
        for cetype in g.canonical_etypes:
            if "timestamp" in g.edges[cetype].data:
                edge_times = g.edges[cetype].data["timestamp"]
                curr = torch.as_tensor(
                    current_time, device=edge_times.device, dtype=edge_times.dtype
                )
                mask = edge_times <= curr
                edge_mask_dict[cetype] = mask
        if edge_mask_dict:
            # Construct a temporally-masked subgraph without relabeling node indices
            g = dgl.edge_subgraph(g, edge_mask_dict, relabel_nodes=False)

        # Temporal feature aggregation: compute per-node aggregated temporal embeddings
        temporal_feats = []
        for etype in g.canonical_etypes:
            if g.num_edges(etype) == 0 or "timestamp" not in g.edges[etype].data:
                continue
            t = g.edges[etype].data["timestamp"]
            # Convert absolute timestamps to relative elapsed time in days; clamp
            # to non-negative to handle any fractional ordering inconsistencies
            delta_t = (current_time - t).float().clamp(min=0.0)
            delta_t = delta_t / 86400.0

            # Encode relative time as sinusoidal embeddings of shape (E, time_dim)
            t_emb = self.time_encoder(delta_t)
            # Apply edge-type-specific linear projection for relational conditioning
            t_emb = self.edge_proj[etype[1]](t_emb)
            g.edges[etype].data["t_emb"] = t_emb
            # Aggregate edge temporal embeddings to destination nodes via mean pooling
            g.update_all(
                dgl.function.copy_e("t_emb", "m"),
                dgl.function.mean("m", f"t_agg_{etype[1]}"),
                etype=etype,
            )
            agg_feat = g.nodes["node"].data.get(f"t_agg_{etype[1]}")
            if agg_feat is not None:
                temporal_feats.append(agg_feat)

        # Combine per-relation temporal aggregations via mean, or use zero vector
        if len(temporal_feats) > 0:
            temporal_feature = torch.stack(temporal_feats).mean(dim=0)
        else:
            temporal_feature = torch.zeros((x.shape[0], self.time_encoder.time_dim)).to(
                x.device
            )

        # Concatenate structural node features with aggregated temporal context
        x_combined = torch.cat([x, temporal_feature], dim=1)

        # First GAT layer: multi-head attention over heterogeneous neighbourhoods
        h1 = self.conv1(g, {"node": x_combined})["node"]
        # Flatten multi-head attention output from (N, num_heads, head_dim) to (N, hidden_dim)
        h1 = h1.view(h1.shape[0], -1)
        h1 = torch.relu(h1)

        # Second GAT layer further refines neighbourhood-aggregated representations
        h2 = self.conv2(g, {"node": h1})["node"]
        h2 = h2.view(h2.shape[0], -1)

        # Residual connection stabilizes gradient flow; LayerNorm prevents covariate shift
        h = self.bn(h2 + h1)

        return self.classifier(h)
