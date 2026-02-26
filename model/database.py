"""
Neo4j Knowledge Graph interface for the DeFi fraud detection system.

This module provides the `Neo4jKnowledgeGraph` class, which bridges the live
Neo4j property graph database and the DGL-based HTGNN inference pipeline. It
is responsible for: (1) querying the evolutionary transaction graph stored in
Neo4j, (2) constructing DGL HeteroGraph objects suitable for inductive HTGNN
inference on arbitrary subgraphs, and (3) propagating analyst-confirmed fraud
labels back into the knowledge graph. This module is exclusively used at
inference time by the Streamlit dashboard (app.py) and the adaptive engine
(model/adaptive_engine.py). It does not participate in offline training.
"""

import os

import dgl
import joblib
import numpy as np
import torch
from neo4j import GraphDatabase


class Neo4jKnowledgeGraph:
    """Interface between the Neo4j property graph and the DGL HTGNN inference pipeline.

    Manages a persistent Neo4j driver connection and exposes methods to retrieve
    subgraph data, construct DGL HeteroGraph instances with correctly preprocessed
    node features, and write analyst-confirmed fraud labels back to the database.
    Feature preprocessing (log1p transformation and StandardScaler normalization)
    is applied identically to the procedure used in dgl_graph_construction.py,
    ensuring distributional consistency between offline-trained and online-inferred
    feature representations.

    Attributes:
        driver (neo4j.GraphDatabase.driver): Active Neo4j Bolt protocol driver.
        scaler (sklearn.preprocessing.StandardScaler | None): Fitted scaler loaded
            from disk. If absent, features are passed unscaled with a warning.
    """

    def __init__(self, uri, user, password, scaler_path="DataSet/scaler.pkl"):
        """Initializes the Neo4j driver connection and loads the feature scaler.

        Args:
            uri (str): Neo4j Bolt URI, e.g., 'bolt://localhost:7687'.
            user (str): Neo4j authentication username.
            password (str): Neo4j authentication password.
            scaler_path (str, optional): Filesystem path to the serialized
                StandardScaler artifact produced by train/train.py.
                Defaults to 'DataSet/scaler.pkl'.
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        # Load the scaler used during training to ensure feature distribution alignment
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print(f"Scaler loaded from {scaler_path}")
        else:
            self.scaler = None
            print(
                f"Warning: Scaler not found at {scaler_path}. Features will not be scaled."
            )

    def close(self):
        """Closes the Neo4j driver connection and releases associated resources."""
        self.driver.close()

    def _build_dgl_graph(self, neo4j_nodes, neo4j_rels, target_address, current_time):
        """Converts Neo4j structural data into a DGL HeteroGraph for HTGNN inference.

        Constructs an inference-ready DGL HeteroGraph from a list of Neo4j node
        records and relationship records. Applies the same feature preprocessing
        pipeline as dgl_graph_construction.py (log1p on continuous features,
        StandardScaler normalization) to guarantee distributional alignment with
        the training-time feature space.

        Args:
            neo4j_nodes (list[neo4j.graph.Node]): Neo4j node records containing
                address-level behavioral features (tx_count, incoming, outgoing,
                avg_gas_used, unique_peers, bytecode_size, etc.).
            neo4j_rels (list[neo4j.graph.Relationship]): Neo4j relationship records
                encoding transaction edges with type (ETH_TRANSFER, TOKEN_TRANSFER,
                CONTRACT_CALL) and timestamp attributes.
            target_address (str): Ethereum address of the node under analysis.
                Used to retrieve its local graph index for inference.
            current_time (float): Unix timestamp defining the temporal boundary
                for the HTGNN forward pass (set in graph_data['current_time']).

        Returns:
            tuple:
                - g (dgl.DGLHeteroGraph): Constructed heterogeneous graph with
                  preprocessed node features, labels, and edge timestamps.
                - node_id_map (dict[str, int]): Mapping from Ethereum address
                  strings to local graph node indices.
        """
        node_id_map = {}
        node_raw_feats = []  # Raw feature values prior to log1p transformation
        node_labels = []

        # Step 1: Extract node features in the canonical order defined by dgl_graph_construction.py
        for i, node in enumerate(neo4j_nodes):
            addr = node["address"]
            node_id_map[addr] = i

            # Retrieve behavioral features with safe defaults for missing values
            tx_count = node.get("tx_count", 0.0)
            incoming = node.get("incoming", 0.0)  # property name is 'incoming'
            outgoing = node.get("outgoing", 0.0)  # property name is 'outgoing'
            avg_gas_used = node.get("avg_gas_used", 0.0)
            unique_peers = node.get("unique_peers", 0)
            bytecode_size = node.get("bytecode_size", 0.0)
            is_token_prop = node.get("is_token", 0)
            avg_in_value = node.get("avg_in_value", 0.0)
            avg_out_value = node.get("avg_out_value", 0.0)
            avg_gas_in = node.get("avg_gas_in", 0.0)
            avg_tx_freq = node.get("avg_tx_freq", 0.0)

            # Derive binary flags from Neo4j node labels (a frozenset of type strings)
            labels_set = set(node.labels) if hasattr(node, "labels") else set()
            contract_flag = 1 if "Contract" in labels_set else 0
            token_flag = 1 if ("Token" in labels_set or is_token_prop == 1) else 0

            # Assemble feature vector in the same column order as dgl_graph_construction.py
            raw_feats = [
                tx_count,
                incoming,
                outgoing,
                avg_gas_used,
                unique_peers,
                contract_flag,
                token_flag,
                bytecode_size,
                avg_in_value,
                avg_out_value,
                avg_gas_in,
                avg_tx_freq,
            ]
            node_raw_feats.append(raw_feats)
            node_labels.append(node.get("label", 0))

        # Step 2: Apply log1p to continuous features (indices 0,1,2,3,4,7,8,9,10,11)
        # to replicate the heavy-tailed normalization from dgl_graph_construction.py
        node_feats_log = []
        for feats in node_raw_feats:
            log_feats = feats.copy()
            for idx in [0, 1, 2, 3, 4, 7, 8, 9, 10, 11]:
                log_feats[idx] = np.log1p(log_feats[idx])
            node_feats_log.append(log_feats)

        # Step 3: Apply the training-time StandardScaler for distributional alignment
        if self.scaler is not None:
            node_feats_scaled = self.scaler.transform(node_feats_log)
        else:
            node_feats_scaled = node_feats_log

        # Step 4: Group relationships by edge type for heterogeneous graph construction
        edge_data = {}
        edge_timestamps = {}

        for rel in neo4j_rels:
            etype = rel.type.lower()
            src_addr = rel.start_node["address"]
            dst_addr = rel.end_node["address"]

            src_idx = node_id_map.get(src_addr)
            dst_idx = node_id_map.get(dst_addr)
            if src_idx is None or dst_idx is None:
                continue  # Skip edges whose endpoints are not in the node universe

            canonical_etype = ("node", etype, "node")
            if canonical_etype not in edge_data:
                edge_data[canonical_etype] = ([], [])
                edge_timestamps[etype] = []

            edge_data[canonical_etype][0].append(src_idx)
            edge_data[canonical_etype][1].append(dst_idx)
            edge_timestamps[etype].append(rel.get("timestamp", 0))

        # Step 5: Construct the DGL HeteroGraph from edge index lists
        g = dgl.heterograph(
            {k: (torch.tensor(v[0]), torch.tensor(v[1])) for k, v in edge_data.items()}
        )

        # Attach scaled feature tensor and ground-truth labels to the node store
        g.nodes["node"].data["feat"] = torch.tensor(
            node_feats_scaled, dtype=torch.float32
        )
        g.nodes["node"].data["label"] = torch.tensor(node_labels, dtype=torch.long)

        # Attach per-relation edge timestamps and track the global maximum
        max_ts = 0
        for etype_str, ts_list in edge_timestamps.items():
            ts_tensor = torch.tensor(ts_list, dtype=torch.float32)
            canonical_etype = ("node", etype_str, "node")
            g.edges[canonical_etype].data["timestamp"] = ts_tensor
            if ts_tensor.numel() > 0:
                max_ts = max(max_ts, ts_tensor.max().item())

        # Store graph-level metadata required by the HTGNN forward pass
        g.graph_data = {
            "max_timestamp": max_ts,
            "current_time": current_time,
            "target_node_idx": node_id_map[target_address],
        }

        return g, node_id_map

    def update_fraud_label(self, wallet_address, label):
        """Propagates an analyst-confirmed fraud label into the Neo4j knowledge graph.

        Executes a parameterized Cypher MATCH-SET query to update the `label`
        property of the node corresponding to the given Ethereum address. This
        operation is the persistence step in the adaptive feedback loop (Step 12
        of the system pipeline).

        Args:
            wallet_address (str): Lowercase Ethereum address of the target node.
            label (int): Corrected ground-truth label (1 for fraud, 0 for benign).

        Returns:
            neo4j.Record | None: The Neo4j record containing the updated address,
                or None if the node was not found.
        """
        with self.driver.session() as session:
            query = """
            MATCH (n {address: $addr})
            SET n.label = $label
            RETURN n.address AS updated
            """
            return session.run(query, addr=wallet_address, label=int(label)).single()

    def get_node_context(self, address):
        """Retrieves metadata for a specific Ethereum address from Neo4j.

        Executes a COUNT subquery (Neo4j 5.x syntax) to obtain the node's degree
        alongside its fraud label, returning a structured context dictionary for
        display in the Streamlit dashboard sidebar.

        Args:
            address (str): Lowercase Ethereum address to look up.

        Returns:
            dict | None: A dictionary with keys 'address', 'label' (str: 'Fraud'
                or 'Safe'), and 'score' (int: node degree). Returns None if the
                address is not present in the knowledge graph.
        """
        query_safe = """
        MATCH (n {address: $address})
        RETURN n.address AS address, 
               coalesce(n.label, 0) AS label,
               COUNT { (n)--() } AS degree
        LIMIT 1
        """
        with self.driver.session() as session:
            result = session.run(query_safe, address=address)
            record = result.single()
            if record:
                return {
                    "address": record["address"],
                    "label": "Fraud" if record["label"] == 1 else "Safe",
                    "score": record["degree"],
                }
            return None
