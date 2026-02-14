import torch
import torch.nn as nn
import torch.optim as optim
import dgl
import pandas as pd
import numpy as np
import os

from model.htg_nn import HTGNN
from utils.metrics import (
    compute_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve
) 


def load_graph():
    import dgl_graph_construction  # your existing script
    return dgl_graph_construction.g


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    g = load_graph().to(device)

    labels = g.nodes['node'].data['label']
    num_classes = 1

    timestamps = g.nodes['node'].data['timestamp']  # you must add this
    val_threshold = torch.quantile(timestamps.float(), 0.6)   # 60% for training
    test_threshold = torch.quantile(timestamps.float(), 0.8)  # 20% for validation, 20% for test

    train_mask = timestamps <= val_threshold
    val_mask = (timestamps > val_threshold) & (timestamps <= test_threshold)
    test_mask = timestamps > test_threshold

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
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)

    # 2. Add Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_f1 = 0.0
    final_best_threshold = 0.5

    for epoch in range(1, 101):
        model.train()
        optimizer.zero_grad()

        logits = model(g)

        loss = criterion(logits[train_mask].view(-1), labels[train_mask].float().view(-1))

        loss.backward()
        optimizer.step()

        # REPLACE the evaluation inside the epoch loop with this:
        model.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(logits[val_mask].squeeze())
            
            # Research addition: Find the best threshold for F1-Score
            from sklearn.metrics import precision_recall_curve
            p, r, t = precision_recall_curve(labels[val_mask].cpu(), val_probs.cpu())
            f1 = 2 * (p * r) / (p + r + 1e-8)
            best_threshold = t[np.argmax(f1)] 

            # Now evaluate on Test set using the best threshold found in Validation
            test_probs = torch.sigmoid(logits[test_mask].squeeze())
            test_preds = (test_probs > best_threshold).long()
            metrics = compute_metrics(labels[test_mask], test_preds, test_probs)
            
            # Save Best Model
            current_f1 = metrics['f1']
            scheduler.step(current_f1)
        
            if current_f1 > best_f1:
                best_f1 = current_f1
                final_best_threshold = best_threshold
                torch.save(model.state_dict(), 'best_model.pth')
                print(f"🌟 Epoch {epoch:02d}: New Best F1: {best_f1:.4f} - Model Saved")

        print(
            f"Epoch {epoch:02d} | "
            f"Loss: {loss.item():.4f} | "
            f"Precision: {metrics['precision']:.4f} | "
            f"Recall: {metrics['recall']:.4f} | "
            f"F1: {metrics['f1']:.4f} | "
            f"AUC: {metrics['roc_auc']:.4f}"
        )

    if os.path.exists('best_model.pth'):
        model.load_state_dict(torch.load('best_model.pth'))
    

    model.eval()
    with torch.no_grad():
        logits = model(g).squeeze()
        probs = torch.sigmoid(logits[test_mask])
        
        # Use the best_threshold calculated during the final epoch
        preds = (probs > final_best_threshold).long()

        y_true = labels[test_mask].cpu().numpy()
        y_pred = preds.cpu().numpy()
        y_probs = probs.cpu().numpy()

        plot_confusion_matrix(y_true, y_pred)
        plot_roc_curve(y_true, y_probs)
        plot_precision_recall_curve(y_true, y_probs)
        print(f"✅ Final Evaluation complete using Optimal Threshold: {final_best_threshold:.4f}")

if __name__ == "__main__":
    train()
