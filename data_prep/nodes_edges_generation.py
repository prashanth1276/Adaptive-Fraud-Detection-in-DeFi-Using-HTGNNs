"""
Data preparation: Node and edge feature extraction from the Ethereum transaction dataset.

This module transforms the cleaned transactional CSV (produced by data_preprocessing.py)
into a structured graph representation suitable for DGL heterogeneous graph construction.
It extracts unique graph nodes (wallets, contracts, token addresses), computes per-node
behavioral feature aggregates (transaction count, ETH flow volumes, gas statistics,
peer diversity, contract/token flags, bytecode size), constructs three typed edge sets
(eth_transfer, token_transfer, contract_call) with associated temporal and value
attributes, and calculates additional edge-derived node-level aggregates for the
extended 12-feature representation. All outputs are serialized to CSV for downstream
consumption by dgl_graph_construction.py. This script is a one-time offline step.
"""

import pandas as pd

# Load the preprocessed Ethereum transaction dataset
df = pd.read_csv("DataSet/Ethereum_Fraud_Dataset.csv")

# Convert block_timestamp to datetime for temporal feature extraction
df["block_timestamp"] = pd.to_datetime(df["block_timestamp"])

# Step 1: Construct the unique node universe from all address columns.
# Nodes represent Ethereum entities: wallets, smart contracts, and token contracts.
wallet_nodes = set(df["from_address"]).union(set(df["to_address"]))
# Identify contract nodes by addresses that received value from bytecode-positive transactions
contract_nodes = set(df[df["bytecode_size"] > 0]["to_address"])
# Identify token contract nodes from the token_address column in token transfer records
token_nodes = set(df["token_address"])

all_nodes = wallet_nodes.union(contract_nodes).union(token_nodes)

node_df = pd.DataFrame({"node": list(all_nodes)})

# Assign semantic node type labels for contract and token flag derivation
node_df["node_type"] = node_df["node"].apply(
    lambda x: (
        "contract" if x in contract_nodes else "token" if x in token_nodes else "wallet"
    )
)

# Step 2: Compute per-node behavioral feature aggregates.

# Bidirectional transaction count: total appearances as sender or receiver
tx_count = (
    df["from_address"].value_counts().add(df["to_address"].value_counts(), fill_value=0)
)

# Cumulative ETH received (to_address) and sent (from_address) per node
incoming_value = df.groupby("to_address")["value"].sum()
outgoing_value = df.groupby("from_address")["value"].sum()

# Average gas consumed per transaction for each sending address
avg_gas = df.groupby("from_address")["gas"].mean()

# Compute the count of unique counterparties (both incoming and outgoing) per node
incoming_peers = df.groupby("to_address")["from_address"].unique().apply(set)
outgoing_peers = df.groupby("from_address")["to_address"].unique().apply(set)


def count_unique_peers(addr):
    """Computes the number of unique counterparty addresses for a given node.

    Combines both incoming (sender) and outgoing (receiver) counterparty sets
    to produce a symmetric measure of peer diversity, which is a behavioral
    indicator of hub or fan-out activity patterns associated with fraud.

    Args:
        addr (str): Ethereum address of the node to evaluate.

    Returns:
        int: Count of unique counterparty addresses across all transaction roles.
    """
    peers = set()
    if addr in incoming_peers.index:
        peers.update(incoming_peers[addr])
    if addr in outgoing_peers.index:
        peers.update(outgoing_peers[addr])
    return len(peers)


# Map computed features onto node_df using address as the join key
node_df.set_index("node", inplace=True)
node_df["tx_count"] = node_df.index.map(tx_count).fillna(0)
node_df["incoming_value"] = node_df.index.map(incoming_value).fillna(0)
node_df["outgoing_value"] = node_df.index.map(outgoing_value).fillna(0)
node_df["avg_gas_used"] = node_df.index.map(avg_gas).fillna(0)
node_df["unique_peers"] = (
    node_df.index.to_series().apply(count_unique_peers).fillna(0).astype(int)
)

# Derive binary contract and token type flags from the semantic node type label
node_df["contract_flag"] = (node_df["node_type"] == "contract").astype(int)
node_df["token_flag"] = (node_df["node_type"] == "token").astype(int)

# Aggregate bytecode size per destination address; take max where multiple records exist
bytecode_map = df[df["bytecode_size"] > 0].groupby("to_address")["bytecode_size"].max()
node_df["bytecode_size"] = node_df.index.map(bytecode_map).fillna(0)

node_df.reset_index(inplace=True)

# Step 3: Construct typed edge sets for the three DeFi transaction relation categories.

# ETH transfer edges: direct Ether value transfers between addresses
eth_edges = df[
    [
        "from_address",
        "to_address",
        "value",
        "gas",
        "gas_price",
        "block_timestamp",
        "block_number",
        "transaction_index",
    ]
].copy()
eth_edges.rename(columns={"from_address": "src", "to_address": "dst"}, inplace=True)
eth_edges["edge_type"] = "eth_transfer"

# Token transfer edges: ERC20/ERC721 token movements with non-zero token_value
token_edges = df[df["token_value"] > 0][
    [
        "from_address",
        "to_address",
        "token_address",
        "token_value",
        "gas",
        "gas_price",
        "block_timestamp",
        "block_number",
        "transaction_index",
    ]
].copy()
token_edges.rename(columns={"from_address": "src", "to_address": "dst"}, inplace=True)
token_edges["edge_type"] = "token_transfer"


# Contract interaction edges: calls to addresses with non-zero bytecode (smart contracts)
contract_edges = df[df["bytecode_size"] > 0][
    [
        "from_address",
        "to_address",
        "gas",
        "gas_price",
        "block_timestamp",
        "block_number",
        "transaction_index",
    ]
].copy()
contract_edges.rename(
    columns={"from_address": "src", "to_address": "dst"}, inplace=True
)
contract_edges["edge_type"] = "contract_call"
# Pad missing value/token columns with zero for schema consistency in the combined edge table
contract_edges["value"] = 0
contract_edges["token_value"] = 0
contract_edges["token_address"] = None


# Combine all typed edge sets into a unified edge table
all_edges = pd.concat([eth_edges, token_edges, contract_edges], ignore_index=True)

# Convert block_timestamp to Unix seconds for temporal edge masking in the HTGNN
all_edges["timestamp"] = all_edges["block_timestamp"].astype("int64") // 10**9

# Compute bidirectional transaction frequency between each address pair as an edge feature
edge_freq = all_edges.groupby(["src", "dst"]).size().reset_index(name="tx_frequency")
all_edges = all_edges.merge(edge_freq, on=["src", "dst"], how="left")

# Step 3.5: Derive additional node-level aggregates from edge-level statistics.
# These form the extended 12-feature representation (vs. the 8-feature ablation baseline).

# Average ETH value received per incoming edge
in_value = all_edges.groupby("dst")["value"].mean().rename("avg_in_value")
# Average ETH value sent per outgoing edge
out_value = all_edges.groupby("src")["value"].mean().rename("avg_out_value")
# Average gas cost for incoming transactions
in_gas = all_edges.groupby("dst")["gas"].mean().rename("avg_gas_in")
# Average edge-pair transaction frequency (symmetric: mean of src-perspective and dst-perspective)
freq_src = all_edges.groupby("src")["tx_frequency"].mean().rename("freq_src")
freq_dst = all_edges.groupby("dst")["tx_frequency"].mean().rename("freq_dst")
freq_all = pd.concat([freq_src, freq_dst], axis=1).mean(axis=1).rename("avg_tx_freq")

# Temporarily index node_df by address for join operations
node_df = node_df.set_index("node")
node_df = node_df.join(in_value, how="left")
node_df = node_df.join(out_value, how="left")
node_df = node_df.join(in_gas, how="left")
node_df = node_df.join(freq_all, how="left")
node_df.reset_index(inplace=True)

# Impute zero for nodes with no incident edges (isolated nodes have no edge-derived features)
fill_cols = ["avg_in_value", "avg_out_value", "avg_gas_in", "avg_tx_freq"]
node_df[fill_cols] = node_df[fill_cols].fillna(0)

# Step 4: Serialize graph construction artifacts.

# Write the final node feature table for consumption by dgl_graph_construction.py
node_df.to_csv("DataSet/graph_nodes.csv", index=False)


# Serialize the typed edge table with all feature columns for the graph construction step
edge_feature_cols = [
    "src",
    "dst",
    "edge_type",
    "timestamp",
    "value",
    "token_value",
    "token_address",
    "gas",
    "gas_price",
    "block_number",
    "transaction_index",
    "tx_frequency",
]
all_edges[edge_feature_cols].to_csv("DataSet/graph_edges.csv", index=False)

print("Nodes and edges (with features) extracted and saved!")
