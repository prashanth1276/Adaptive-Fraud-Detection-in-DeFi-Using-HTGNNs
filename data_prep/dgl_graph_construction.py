"""
Data preparation: Heterogeneous DGL graph construction from node and edge CSVs.

This module is the central graph construction step of the DeFi fraud detection
data pipeline. It reads the labeled node feature CSV and the typed edge CSV
produced by nodes_edges_generation.py and add_fraudlabels_nodes.py, constructs a
DGL heterogeneous graph with node type 'node' and edge types corresponding to
DeFi transaction relation categories (eth_transfer, token_transfer, contract_call),
attaches preprocessed node feature tensors (log1p-transformed), binary fraud labels,
node-level temporal activity timestamps, and normalized edge feature tensors, then
serializes the graph to DataSet/graph.bin for consumption by the training pipeline
and inference dashboard. This script is executed once as part of offline data
preparation and must be rerun whenever the underlying CSVs are updated.
"""

import pickle

import dgl
import numpy as np
import pandas as pd
import torch

# Load CSVs
nodes_df = pd.read_csv("DataSet/graph_nodes_labeled.csv")
edges_df = pd.read_csv("DataSet/graph_edges.csv", low_memory=False)

# Build node universe from the labeled node CSV to define the graph's vertex set.
# Duplicate comment block below is a residual of a prior refactor; both sections
# refer to the same logic (node universe defined from nodes file, not edge endpoints).
all_nodes = nodes_df["node"].unique()
# Map Ethereum address strings to contiguous integer node indices
node_id_map = {node: idx for idx, node in enumerate(all_nodes)}

# Discard edges whose source or destination address is absent from the node universe
edges_df = edges_df[
    edges_df["src"].isin(node_id_map) & edges_df["dst"].isin(node_id_map)
].copy()

# Convert string addresses to integer node indices for DGL tensor construction
edges_df["src_id"] = edges_df["src"].map(node_id_map)
edges_df["dst_id"] = edges_df["dst"].map(node_id_map)

# Align nodes_df ordering with the node_id_map to ensure feature-label correspondence
nodes_df = nodes_df.copy()
nodes_df = nodes_df[nodes_df["node"].isin(all_nodes)]
nodes_df["node_id"] = nodes_df["node"].map(node_id_map)
nodes_df = nodes_df.sort_values("node_id").reset_index(drop=True)
edges_df = edges_df.sort_values(["edge_type", "src_id", "dst_id"]).reset_index(
    drop=True
)

# Construct per-relation edge index tuples for the DGL heterograph factory
edge_dict = {}

for etype in edges_df["edge_type"].unique():
    et_df = edges_df[edges_df["edge_type"] == etype]
    # Canonical edge type format: (src_node_type, relation_type, dst_node_type)
    edge_dict[("node", etype, "node")] = (
        torch.tensor(et_df["src_id"].values, dtype=torch.int64),
        torch.tensor(et_df["dst_id"].values, dtype=torch.int64),
    )


g = dgl.heterograph(edge_dict, num_nodes_dict={"node": len(all_nodes)})

# Store the global maximum edge timestamp for temporal reference in HTGNN forward passes
g.graph_data = {}
g.graph_data["max_timestamp"] = int(edges_df["timestamp"].max())

# Node feature columns: ordered list must match the column order expected by
# dgl_graph_construction.py feature preprocessing and model/database.py
feature_cols = [
    "tx_count",
    "incoming_value",
    "outgoing_value",
    "avg_gas_used",
    "unique_peers",
    "contract_flag",
    "token_flag",
    "bytecode_size",
    "avg_in_value",
    "avg_out_value",
    "avg_gas_in",
    "avg_tx_freq",
]

node_feat_df = nodes_df[feature_cols].copy()
node_feat_df = node_feat_df.fillna(0)

# Apply log1p transform to heavy-tailed continuous features to reduce skewness.
# contract_flag and token_flag are binary and excluded from log-normalization.
for col in [
    "tx_count",
    "incoming_value",
    "outgoing_value",
    "avg_gas_used",
    "unique_peers",
    "bytecode_size",
    "avg_in_value",
    "avg_out_value",
    "avg_gas_in",
    "avg_tx_freq",
]:
    node_feat_df[col] = np.log1p(node_feat_df[col])

# Store the raw (pre-StandardScaler) log-transformed features; StandardScaler
# is applied at training time in train/train.py to prevent data leakage
g.nodes["node"].data["feat_raw"] = torch.tensor(
    node_feat_df.values, dtype=torch.float32
)

# Attach binary fraud labels (0 = benign, 1 = fraud) derived from on-chain flags
g.nodes["node"].data["label"] = torch.tensor(nodes_df["flag"].values, dtype=torch.long)

# Compute per-node last-activity timestamp as the maximum over all incident edge timestamps
node_last_time_src = edges_df.groupby("src_id")["timestamp"].max()
node_last_time_dst = edges_df.groupby("dst_id")["timestamp"].max()

# Take the later of the last outgoing or incoming timestamp per node
node_last_time = pd.concat([node_last_time_src, node_last_time_dst], axis=1).max(axis=1)

nodes_df["timestamp"] = nodes_df["node_id"].map(node_last_time).fillna(0)

# Attach node-level temporal activity timestamps to the graph node store
g.nodes["node"].data["timestamp"] = torch.tensor(
    nodes_df["timestamp"].values.astype("int64"), dtype=torch.long
)

# Attach normalized edge feature tensors for each relation type
for etype in g.etypes:
    et_df = edges_df[edges_df["edge_type"] == etype].copy()

    # Impute missing edge attributes with zero prior to log-normalization
    et_df = et_df.fillna(
        {
            "value": 0,
            "token_value": 0,
            "gas": 0,
            "gas_price": 0,
            "tx_frequency": 0,
            "timestamp": 0,
        }
    )

    # Cast timestamps to int64 explicitly; Windows may default to int32, causing overflow
    g.edges[etype].data["timestamp"] = torch.tensor(
        et_df["timestamp"].values.astype("int64"), dtype=torch.long
    )

    # Apply log1p normalization to edge weight features to reduce ETH/gas value skewness
    for col in ["value", "token_value", "gas", "gas_price", "tx_frequency"]:
        g.edges[etype].data[col] = torch.tensor(
            np.log1p(et_df[col].values), dtype=torch.float32
        )

    # Stack edge features into a combined tensor: [log_value, log_gas, log_tx_frequency]
    combined_edge_feat = torch.stack(
        [
            g.edges[etype].data["value"],
            g.edges[etype].data["gas"],
            g.edges[etype].data["tx_frequency"],
        ],
        dim=1,
    )

    g.edges[etype].data["feat"] = combined_edge_feat

# Sanity checks confirming graph construction completed successfully
print("DGL heterogeneous graph created successfully")
print(g)
print("Node feature shape:", g.nodes["node"].data["feat_raw"].shape)
print("Node label shape:", g.nodes["node"].data["label"].shape)
print("Edge types:", g.etypes)

# Serialize the address-to-integer node mapping for downstream inference lookup
with open("DataSet/node_id_map.pkl", "wb") as f:
    pickle.dump(node_id_map, f)

# Save the processed node table with labels and timestamps for the dashboard
nodes_df.to_csv("DataSet/nodes_audit.csv", index=False)
print("Created DataSet/nodes_audit.csv for the Dashboard")

# Serialize the DGL heterograph binary artifact for the inference pipeline
dgl.save_graphs("DataSet/graph.bin", [g])
print("Created DataSet/graph.bin for the Dashboard")
