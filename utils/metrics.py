import torch
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


def compute_metrics(y_true, y_pred, y_prob):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    y_prob = y_prob.cpu().numpy()

    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob)
    }
