"""
Data preparation: Fraud label propagation from transactions to graph nodes.

This script performs the upstream label assignment step in the DeFi graph
construction pipeline. It joins per-transaction fraud flags (sourced from the
raw Ethereum fraud dataset) with a node-level address list, propagating the
maximum observed fraud flag for each Ethereum address across all its transaction
roles (sender and receiver). An address is marked as fraudulent (flag=1) if it
appeared in any fraudulent transaction in either the `from_address` or `to_address`
role. This label assignment is a prerequisite for the heterogeneous graph
construction step in dgl_graph_construction.py. This script is a one-time
data preparation step and does not participate in training or inference.
"""

import pandas as pd

# Load the raw Ethereum transaction dataset containing per-transaction fraud flags
df = pd.read_csv("DataSet/Ethereum_Fraud_Dataset.csv")

# Load the previously generated node address list (output of nodes_edges_generation.py)
node_df = pd.read_csv("DataSet/graph_nodes.csv")

# Extract fraud flags from the sender column and normalize the join key name
from_flags = df[["from_address", "flag"]].rename(columns={"from_address": "node"})
# Extract fraud flags from the receiver column and normalize the join key name
to_flags = df[["to_address", "flag"]].rename(columns={"to_address": "node"})

# Concatenate sender and receiver fraud flag records for union-based label aggregation
all_flags = pd.concat([from_flags, to_flags])
# Assign fraud label (1) if any transaction involving the address has flag=1
node_flags = all_flags.groupby("node")["flag"].max().reset_index()

# Left-join fraud labels onto the node list; addresses with no fraud record default to 0
node_df = node_df.merge(node_flags, on="node", how="left")
node_df["flag"] = node_df["flag"].fillna(0).astype(int)  # default to 0 for unlabeled

# Persist the fraud-labeled node table for consumption by dgl_graph_construction.py
node_df.to_csv("DataSet/graph_nodes_labeled.csv", index=False)

print("Fraud labels (flag) added to nodes and saved as 'graph_nodes_labeled.csv'")
