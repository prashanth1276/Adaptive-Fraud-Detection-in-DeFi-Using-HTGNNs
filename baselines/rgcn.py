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
    def __init__(self, in_dim, hidden_dim, out_dim, num_rels):
        super().__init__()
        self.conv1 = dglnn.RelGraphConv(
            in_dim, hidden_dim, num_rels, "basis", num_rels, activation=F.relu
        )
        self.conv2 = dglnn.RelGraphConv(
            hidden_dim, out_dim, num_rels, "basis", num_rels
        )
        self.num_rels = num_rels

    def forward(self, g, x, etypes):
        h = self.conv1(g, x, etypes)
        h = self.conv2(g, h, etypes)
        return h


def main():
    graphs, _ = dgl.load_graphs("DataSet/graph.bin")
    g = graphs[0]
    g = g.to(torch.device("cpu"))

    # Convert to homogeneous with edge types preserved
    # RelGraphConv works on homogeneous graphs with edge types tensor.
    # We'll use dgl.to_homogeneous with edata mapping.
    # But easier: we can iterate over edge types and build a single graph with edge_type attribute.
    # Let's create a homogeneous graph with all nodes and edges, and store edge types as an integer tensor.
    # First, get all node features and labels
    ntype = g.ntypes[0]
    feat_raw = g.nodes[ntype].data["feat_raw"].numpy()
    scaler = joblib.load("DataSet/scaler.pkl")
    feat_scaled = scaler.transform(feat_raw)
    num_nodes = g.num_nodes(ntype)

    # Build homogeneous graph
    src_list = []
    dst_list = []
    etype_list = []
    etypes = g.canonical_etypes
    etype_to_id = {et: i for i, et in enumerate(etypes)}
    for et in etypes:
        u, v = g.edges(etype=et)
        src_list.append(u)
        dst_list.append(v)
        etype_list.append(torch.full((u.shape[0],), etype_to_id[et], dtype=torch.long))
    src = torch.cat(src_list)
    dst = torch.cat(dst_list)
    edge_type = torch.cat(etype_list)
    g_homo = dgl.graph((src, dst), num_nodes=num_nodes)
    g_homo.ndata["feat"] = torch.tensor(feat_scaled, dtype=torch.float32)
    g_homo.edata["etype"] = edge_type

    # Load masks and labels
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
            if val_auprc > best_val_auprc:
                best_val_auprc = val_auprc
                best_model_state = model.state_dict()
        if epoch % 20 == 0:
            print(
                f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | Val AUPRC: {val_auprc:.4f}"
            )

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
