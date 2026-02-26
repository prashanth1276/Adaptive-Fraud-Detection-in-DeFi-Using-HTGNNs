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
    def __init__(self, in_feats, hidden_feats, out_feats):
        super().__init__()
        self.conv1 = dglnn.GraphConv(in_feats, hidden_feats)
        self.conv2 = dglnn.GraphConv(hidden_feats, out_feats)

    def forward(self, g, x):
        x = F.relu(self.conv1(g, x))
        x = self.conv2(g, x)
        return x


def main():
    # Load heterograph and convert to homogeneous
    graphs, _ = dgl.load_graphs("DataSet/graph.bin")
    g_hetero = graphs[0]
    g = dgl.to_homogeneous(g_hetero, ndata=["feat_raw"])
    g = dgl.add_self_loop(g)
    feat_raw = g.ndata["feat_raw"].numpy()

    # Scale features using saved scaler
    scaler = joblib.load("DataSet/scaler.pkl")
    feat_scaled = scaler.transform(feat_raw)
    g.ndata["feat"] = torch.tensor(feat_scaled, dtype=torch.float32)

    # Load masks (order preserved)
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
        test_logits = model(g, g.ndata["feat"]).squeeze()
        test_probs = torch.sigmoid(test_logits[test_mask]).cpu().numpy()
        test_labels = labels[test_mask].cpu().numpy()
        test_auprc = average_precision_score(test_labels, test_probs)
        print(f"\nBest validation AUPRC: {best_val_auprc:.4f}")
        print(f"Test AUPRC: {test_auprc:.4f}")


if __name__ == "__main__":
    main()
