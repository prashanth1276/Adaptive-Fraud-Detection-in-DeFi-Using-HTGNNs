"""
Data preparation: Class imbalance diagnostic report for the node fraud label distribution.

This script quantifies the degree of class imbalance in the node-level fraud label
distribution, providing an empirical basis for the FocalLoss hyperparameter selection
and the positive class weighting strategy used in baseline models. It reads from
either the audited node file (DataSet/nodes_audit.csv) produced by
dgl_graph_construction.py, or falls back to the labeled node CSV if the audited
file is not yet available. This is a standalone diagnostic utility and does not
participate in training or inference.
"""

import pandas as pd

# Attempt to load the audited node file; fall back to the labeled CSV if unavailable
try:
    df = pd.read_csv("DataSet/nodes_audit.csv")
    label_col = "flag"
except FileNotFoundError:
    # Fallback to the original labeled nodes if audit file is not yet generated
    df = pd.read_csv("DataSet/graph_nodes_labeled.csv")
    label_col = "flag"

# Compute per-class node counts from the fraud label column
counts = df[label_col].value_counts()
normal = counts.get(0, 0)
fraud = counts.get(1, 0)
total = len(df)

# Compute fraud prevalence and imbalance ratio for reporting
fraud_percent = (fraud / total) * 100
# Guard against zero-fraud edge case; imbalance_ratio = class_0_count / class_1_count
imbalance_ratio = normal / fraud if fraud > 0 else 0

print("-" * 30)
print("DATASET IMBALANCE REPORT")
print("-" * 30)
print(f"Normal Nodes (0): {normal:,}")
print(f"Fraud Nodes  (1): {fraud:,}")
print(f"Total Nodes:      {total:,}")
print(f"Fraud Percentage: {fraud_percent:.4f}%")
print(f"Imbalance Ratio:  1 : {int(imbalance_ratio)}")
print("-" * 30)

# Flag extreme imbalance that would necessitate FocalLoss or heavy pos_weight tuning
if fraud_percent < 0.1:
    print(
        "EXTREME IMBALANCE DETECTED: High imbalance ratio identified in this dataset."
    )
