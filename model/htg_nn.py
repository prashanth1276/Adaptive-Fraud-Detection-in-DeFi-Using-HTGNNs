import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn

from model.time_encoding import TimeEncoding


class HTGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, time_dim, edge_types):
        super().__init__()

        # Step 1: Temporal Encoding Layer (Relative time Δt)
        self.time_encoder = TimeEncoding(time_dim)

        # Edge-type–specific projections to align time embeddings
        self.edge_proj = nn.ModuleDict(
            {etype: nn.Linear(time_dim, time_dim) for etype in edge_types}
        )

        # Step 2: Heterogeneous Message Passing Layer 1 (GAT-based)
        # Using Multi-head Attention as per research standards for fraud detection
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

        # Step 3: Heterogeneous Message Passing Layer 2
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

        # Step 4: Classifier (Final Risk Scoring)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, g):
        g = g.local_var()
        x = g.nodes["node"].data["feat"]

        # --- Get current_time from graph data ---
        if hasattr(g, "graph_data") and "current_time" in g.graph_data:
            current_time = g.graph_data["current_time"]
        else:
            if hasattr(g, "graph_data") and "max_timestamp" in g.graph_data:
                current_time = g.graph_data["max_timestamp"]
            else:
                current_time = float("inf")
                print("Warning: No current_time, using all edges.")

        # --- Temporal edge masking ---
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
            g = dgl.edge_subgraph(g, edge_mask_dict, relabel_nodes=False)

        # --- Temporal feature aggregation (unchanged except clamp) ---
        temporal_feats = []
        for etype in g.canonical_etypes:
            if g.num_edges(etype) == 0 or "timestamp" not in g.edges[etype].data:
                continue
            t = g.edges[etype].data["timestamp"]
            delta_t = (current_time - t).float().clamp(min=0.0)  # ensure non-negative
            delta_t = delta_t / 86400.0

            t_emb = self.time_encoder(delta_t)
            t_emb = self.edge_proj[etype[1]](t_emb)
            g.edges[etype].data["t_emb"] = t_emb
            g.update_all(
                dgl.function.copy_e("t_emb", "m"),
                dgl.function.mean("m", f"t_agg_{etype[1]}"),
                etype=etype,
            )
            agg_feat = g.nodes["node"].data.get(f"t_agg_{etype[1]}")
            if agg_feat is not None:
                temporal_feats.append(agg_feat)

        # Combine spatial features (x) with temporal context
        if len(temporal_feats) > 0:
            temporal_feature = torch.stack(temporal_feats).mean(dim=0)
        else:
            temporal_feature = torch.zeros((x.shape[0], self.time_encoder.time_dim)).to(
                x.device
            )

        x_combined = torch.cat([x, temporal_feature], dim=1)

        # --- GNN Layers ---
        h1 = self.conv1(g, {"node": x_combined})["node"]
        h1 = h1.view(h1.shape[0], -1)  # Flatten multi-head output
        h1 = torch.relu(h1)

        h2 = self.conv2(g, {"node": h1})["node"]
        h2 = h2.view(h2.shape[0], -1)

        # Residual connection + LayerNorm for stability
        h = self.bn(h2 + h1)

        return self.classifier(h)
