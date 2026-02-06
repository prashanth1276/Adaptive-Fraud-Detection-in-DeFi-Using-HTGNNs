import torch
import torch.nn as nn
import dgl
import dgl.nn as dglnn

from model.time_encoding import TimeEncoding


class HTGNN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        time_dim,
        edge_types
    ):
        super().__init__()

        self.time_encoder = TimeEncoding(time_dim)

        # Edge-type–specific projections
        self.edge_proj = nn.ModuleDict({
            etype: nn.Linear(time_dim, time_dim)
            for etype in edge_types
        })

        # RGCN-style hetero conv
        self.convs = dglnn.HeteroGraphConv(
            {
                etype: dglnn.GATConv(
                    input_dim + time_dim,
                    hidden_dim,
                    num_heads=1,
                    allow_zero_in_degree=True
                )
                for etype in edge_types
            },
            aggregate="sum"
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, g):
        x = g.nodes['node'].data['feat']  # (N, F)

        temporal_feats = []

        current_time = max(
                g.edges[etype].data['timestamp'].max()
                for etype in g.etypes
            )

        for etype in g.etypes:
            t = g.edges[etype].data['timestamp']
            delta_t = current_time - t

            t_emb = self.time_encoder(delta_t)
            t_emb = self.edge_proj[etype](t_emb)

            g.edges[etype].data['t_emb'] = t_emb

            g.update_all(
                dgl.function.copy_e('t_emb', 'm'),
                dgl.function.mean('m', f't_agg_{etype}'),
                etype=etype
            )

            temporal_feats.append(g.nodes['node'].data[f't_agg_{etype}'])

        temporal_feature = torch.stack(temporal_feats).mean(dim=0)

        x = torch.cat([x, temporal_feature], dim=1)

        h = self.convs(g, {'node': x})
        h = h['node'].squeeze(1)

        logits = self.classifier(h)
        return logits
