"""
Data preparation: Graph integrity and quality audit for the DeFi transaction graph.

This module provides a comprehensive pre-construction audit of the node and edge
CSV files that feed into the DGL heterogeneous graph. It validates referential
integrity (no orphan edges referencing addresses absent from the node list),
quantifies the fraud label imbalance, checks temporal alignment between edge
timestamps and the transaction CSV, identifies potential supernodes (high-degree
addresses that could dominate GAT attention), and reports missing values in key
node feature columns. This diagnostic step should be executed before
dgl_graph_construction.py to detect data quality issues that would degrade
model training or produce silent errors in the graph construction pipeline.
"""

import pandas as pd


def audit_dataset():
    """Performs a multi-dimensional data quality audit on the graph input files.

    Loads graph_nodes_labeled.csv, graph_edges.csv, and the cleaned transaction
    CSV, then runs a sequence of integrity and diagnostic checks. Results are
    printed as a structured report. No data is modified or written to disk.

    Checks performed:
        - Orphan edge detection: edges whose src or dst address does not appear
          in the node universe (would cause KeyError in dgl_graph_construction.py).
        - Fraud label imbalance: count and percentage of fraudulent nodes.
        - Temporal alignment: comparison of edge timestamp ranges against the
          block_timestamp range in the transaction CSV.
        - Supernode detection: maximum out-degree address (high connectivity can
          dominate GAT attention and skew learned representations).
        - Missing value audit: null counts in key node feature columns.
    """
    print("--- Starting Dataset Audit ---")

    # Load node and edge files; use dtype=str for address columns to prevent
    # mixed-type warnings from pandas when parsing heterogeneous hex strings
    nodes = pd.read_csv("DataSet/graph_nodes_labeled.csv", dtype={"node": str})
    edges = pd.read_csv(
        "DataSet/graph_edges.csv",
        dtype={"src": str, "dst": str, "token_address": str},
        low_memory=False,
    )
    main = pd.read_csv(
        "DataSet/Ethereum_Fraud_Dataset.csv",
        dtype={"from_address": str, "to_address": str},
    )

    print(f"Nodes loaded: {len(nodes)}")
    print(f"Edges loaded: {len(edges)}")
    print(f"Main Transactions loaded: {len(main)}\n")

    # Referential integrity check: edges referencing addresses outside the node universe
    # will cause missing node_id mappings in dgl_graph_construction.py
    node_universe = set(nodes["node"])
    edge_srcs = set(edges["src"])
    edge_dsts = set(edges["dst"])

    orphan_src = edge_srcs - node_universe
    orphan_dst = edge_dsts - node_universe

    print("--- Integrity Check ---")
    print(f"Orphan Sources (in edges but not in node list): {len(orphan_src)}")
    print(f"Orphan Destinations (in edges but not in node list): {len(orphan_dst)}")

    # Fraud label imbalance: feeds the FocalLoss hyperparameter selection rationale
    if "flag" in nodes.columns:
        fraud_count = nodes["flag"].sum()
        print(
            f"Fraudulent Nodes: {fraud_count} ({(fraud_count / len(nodes)) * 100:.2f}%)"
        )

    # Temporal alignment verification: edge timestamps should be consistent with
    # block_timestamp ranges in the transaction CSV (converted to Unix seconds)
    print("\n--- Temporal Check ---")
    print(
        f"Edge Timestamp Range: {edges['timestamp'].min()} to {edges['timestamp'].max()}"
    )
    # Convert block_timestamp to Unix seconds for direct comparison with edge timestamps
    main_ts = pd.to_datetime(main["block_timestamp"]).view("int64") // 10**9
    print(f"Main CSV Timestamp Range: {main_ts.min()} to {main_ts.max()}")

    # Supernode detection: addresses with disproportionately high edge counts can
    # dominate GAT attention weights and skew the learned fraud representations
    src_counts = edges["src"].value_counts()
    print("\n--- Connectivity Check ---")
    print(f"Top Source Connectivity: {src_counts.max()} edges from one address")

    # Missing value audit in key node feature columns used by the HTGNN
    print("\n--- Missing Values in Features ---")
    print(nodes[["tx_count", "incoming_value", "outgoing_value"]].isnull().sum())


if __name__ == "__main__":
    audit_dataset()
