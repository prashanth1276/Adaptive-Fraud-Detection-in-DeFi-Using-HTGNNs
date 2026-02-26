import pandas as pd


def audit_dataset():
    print("--- 🔍 Starting Dataset Audit ---")

    # 1. Load Datasets
    # Using dtype=str for addresses to prevent mixed-type warnings
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

    # 2. Check for "Orphan" Edges (Crucial for GNN)
    node_universe = set(nodes["node"])
    edge_srcs = set(edges["src"])
    edge_dsts = set(edges["dst"])

    orphan_src = edge_srcs - node_universe
    orphan_dst = edge_dsts - node_universe

    print("--- 🛠 Integrity Check ---")
    print(f"Orphan Sources (in edges but not in node list): {len(orphan_src)}")
    print(f"Orphan Destinations (in edges but not in node list): {len(orphan_dst)}")

    # 3. Check Fraud Imbalance
    if "flag" in nodes.columns:
        fraud_count = nodes["flag"].sum()
        print(
            f"Fraudulent Nodes: {fraud_count} ({(fraud_count / len(nodes)) * 100:.2f}%)"
        )

    # 4. Check Temporal Alignment
    print("\n--- ⏳ Temporal Check ---")
    print(
        f"Edge Timestamp Range: {edges['timestamp'].min()} to {edges['timestamp'].max()}"
    )
    # Ensure it doesn't divide or scale if it's already Unix seconds
    main_ts = pd.to_datetime(main["block_timestamp"]).view("int64") // 10**9
    print(f"Main CSV Timestamp Range: {main_ts.min()} to {main_ts.max()}")

    # 5. Check for "Supernodes"
    # Too much connectivity in one node can crash DGL or skew the GAT attention
    src_counts = edges["src"].value_counts()
    print("\n--- 🕸 Connectivity Check ---")
    print(f"Top Source Connectivity: {src_counts.max()} edges from one address")

    # 6. Null Analysis in Features
    print("\n--- 🕳 Missing Values in Features ---")
    print(nodes[["tx_count", "incoming_value", "outgoing_value"]].isnull().sum())


if __name__ == "__main__":
    audit_dataset()
