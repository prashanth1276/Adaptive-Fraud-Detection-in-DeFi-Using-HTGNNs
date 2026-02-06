import torch
import torch.nn as nn
import torch.optim as optim
import dgl
import pandas as pd
import numpy as np

from model.htg_nn import HTGNN
from utils.metrics import compute_metrics


def load_graph():
    import dgl_graph_construction  # your existing script
    return dgl_graph_construction.g


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    g = load_graph().to(device)

    labels = g.nodes['node'].data['label']
    num_classes = 1

    timestamps = g.nodes['node'].data['timestamp']  # you must add this
    threshold = torch.quantile(timestamps.float(), 0.8)

    train_mask = timestamps <= threshold
    test_mask = timestamps > threshold

    model = HTGNN(
        input_dim=g.nodes['node'].data['feat'].shape[1],
        hidden_dim=128,
        output_dim=num_classes,
        time_dim=32,
        edge_types=g.etypes
    ).to(device)

    pos_weight = torch.tensor(
        [(labels == 0).sum() / (labels == 1).sum()],
        device=device
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)

    for epoch in range(1, 31):
        model.train()
        optimizer.zero_grad()

        logits = model(g)
        loss = criterion(
                    logits[train_mask].squeeze(),
                    labels[train_mask].float()
                )
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(logits[test_mask].squeeze())
            preds = (probs > 0.5).long()

            metrics = compute_metrics(
                labels[test_mask],
                preds,
                probs
            )

        print(
            f"Epoch {epoch:02d} | "
            f"Loss: {loss.item():.4f} | "
            f"Precision: {metrics['precision']:.4f} | "
            f"Recall: {metrics['recall']:.4f} | "
            f"F1: {metrics['f1']:.4f} | "
            f"AUC: {metrics['roc_auc']:.4f}"
        )


if __name__ == "__main__":
    train()
