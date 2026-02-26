import dgl.nn as dglnn
import torch
import torch.nn as nn


class HTGNN_NoTemporal(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, edge_types):
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
        x = g.nodes["node"].data["feat"]
        h1 = self.conv1(g, {"node": x})["node"]
        h1 = h1.view(h1.shape[0], -1)
        h1 = torch.relu(h1)
        h2 = self.conv2(g, {"node": h1})["node"]
        h2 = h2.view(h2.shape[0], -1)
        h = self.bn(h2 + h1)
        return self.classifier(h)
