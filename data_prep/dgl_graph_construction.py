import pickle

import dgl
import numpy as np
import pandas as pd
import torch

# -----------------------------
# 1. Load CSVs
# -----------------------------
nodes_df = pd.read_csv("DataSet/graph_nodes_labeled.csv")
edges_df = pd.read_csv("DataSet/graph_edges.csv", low_memory=False)

# -----------------------------
# 2. Build node universe FROM EDGES
# -----------------------------
# -----------------------------
# 2. Build node universe FROM NODES FILE
# -----------------------------
all_nodes = nodes_df["node"].unique()
node_id_map = {node: idx for idx, node in enumerate(all_nodes)}

edges_df = edges_df[
    edges_df["src"].isin(node_id_map) & edges_df["dst"].isin(node_id_map)
].copy()

edges_df["src_id"] = edges_df["src"].map(node_id_map)
edges_df["dst_id"] = edges_df["dst"].map(node_id_map)

# -----------------------------
# 3. Align nodes_df to graph nodes
# -----------------------------
nodes_df = nodes_df.copy()
nodes_df = nodes_df[nodes_df["node"].isin(all_nodes)]
nodes_df["node_id"] = nodes_df["node"].map(node_id_map)
nodes_df = nodes_df.sort_values("node_id").reset_index(drop=True)
edges_df = edges_df.sort_values(["edge_type", "src_id", "dst_id"]).reset_index(
    drop=True
)

# -----------------------------
# 4. Build heterograph (single node type, multiple edge types)
# -----------------------------
edge_dict = {}

for etype in edges_df["edge_type"].unique():
    et_df = edges_df[edges_df["edge_type"] == etype]

    edge_dict[("node", etype, "node")] = (
        torch.tensor(et_df["src_id"].values, dtype=torch.int64),
        torch.tensor(et_df["dst_id"].values, dtype=torch.int64),
    )


g = dgl.heterograph(edge_dict, num_nodes_dict={"node": len(all_nodes)})

# Store global max timestamp for temporal reference
g.graph_data = {}
g.graph_data["max_timestamp"] = int(edges_df["timestamp"].max())

# -----------------------------
# 5. Node features
# -----------------------------
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

# 1. Log-scale heavy-tailed features
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

g.nodes["node"].data["feat_raw"] = torch.tensor(
    node_feat_df.values, dtype=torch.float32
)

# 6. Node labels
g.nodes["node"].data["label"] = torch.tensor(nodes_df["flag"].values, dtype=torch.long)

# 6.5 Node timestamp (last activity time)

# Get last transaction timestamp per destination node
node_last_time_src = edges_df.groupby("src_id")["timestamp"].max()
node_last_time_dst = edges_df.groupby("dst_id")["timestamp"].max()

node_last_time = pd.concat([node_last_time_src, node_last_time_dst], axis=1).max(axis=1)

nodes_df["timestamp"] = nodes_df["node_id"].map(node_last_time).fillna(0)

# Attach to graph
g.nodes["node"].data["timestamp"] = torch.tensor(
    nodes_df["timestamp"].values.astype("int64"), dtype=torch.long
)

# 7. Edge features (NORMALIZED)
for etype in g.etypes:
    et_df = edges_df[edges_df["edge_type"] == etype].copy()

    # Fill NaNs with 0
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

    # Ensure timestamp is a long integer (Unix seconds)
    # The .values.astype('int64') ensures Windows doesn't cap it at int32
    g.edges[etype].data["timestamp"] = torch.tensor(
        et_df["timestamp"].values.astype("int64"), dtype=torch.long
    )

    # Log-normalize edge weights
    for col in ["value", "token_value", "gas", "gas_price", "tx_frequency"]:
        g.edges[etype].data[col] = torch.tensor(
            np.log1p(et_df[col].values), dtype=torch.float32
        )

    combined_edge_feat = torch.stack(
        [
            g.edges[etype].data["value"],
            g.edges[etype].data["gas"],
            g.edges[etype].data["tx_frequency"],
        ],
        dim=1,
    )

    g.edges[etype].data["feat"] = combined_edge_feat

# -----------------------------
# 8. Sanity checks
# -----------------------------
print("✅ DGL heterogeneous graph created successfully")
print(g)
print("Node feature shape:", g.nodes["node"].data["feat_raw"].shape)
print("Node label shape:", g.nodes["node"].data["label"].shape)
print("Edge types:", g.etypes)

with open("DataSet/node_id_map.pkl", "wb") as f:
    pickle.dump(node_id_map, f)

# Save the processed nodes with their labels and timestamps for the Dashboard
# This creates the 'Answer Key' we need
nodes_df.to_csv("DataSet/nodes_audit.csv", index=False)
print("✅ Created DataSet/nodes_audit.csv for the Dashboard")

# Save the DGL graph to a binary file so Streamlit doesn't have to rebuild it
dgl.save_graphs("DataSet/graph.bin", [g])
print("✅ Created DataSet/graph.bin for the Dashboard")
