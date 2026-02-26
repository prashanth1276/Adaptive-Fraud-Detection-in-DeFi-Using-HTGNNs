import os

import dgl
import joblib
import numpy as np
import torch
from neo4j import GraphDatabase


class Neo4jKnowledgeGraph:
    def __init__(self, uri, user, password, scaler_path="DataSet/scaler.pkl"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        # Load the scaler used during training
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print(f"Scaler loaded from {scaler_path}")
        else:
            self.scaler = None
            print(
                f"Warning: Scaler not found at {scaler_path}. Features will not be scaled."
            )

    def close(self):
        self.driver.close()

    def _build_dgl_graph(self, neo4j_nodes, neo4j_rels, target_address, current_time):
        """
        Converts Neo4j structural data into a DGL HeteroGraph with proper feature preprocessing.
        """
        node_id_map = {}
        node_raw_feats = []  # will hold raw feature values before log1p
        node_labels = []

        # 1. Process Nodes: extract features in the same order as dgl_graph_construction.py
        for i, node in enumerate(neo4j_nodes):
            addr = node["address"]
            node_id_map[addr] = i

            # Basic properties (with defaults)
            tx_count = node.get("tx_count", 0.0)
            incoming = node.get("incoming", 0.0)  # property name is 'incoming'
            outgoing = node.get("outgoing", 0.0)  # property name is 'outgoing'
            avg_gas_used = node.get("avg_gas_used", 0.0)  # new
            unique_peers = node.get("unique_peers", 0)  # new (should be integer)
            bytecode_size = node.get("bytecode_size", 0.0)
            is_token_prop = node.get("is_token", 0)
            avg_in_value = node.get("avg_in_value", 0.0)
            avg_out_value = node.get("avg_out_value", 0.0)
            avg_gas_in = node.get("avg_gas_in", 0.0)
            avg_tx_freq = node.get("avg_tx_freq", 0.0)

            # Determine flags based on labels (node.labels is a frozenset)
            labels_set = set(node.labels) if hasattr(node, "labels") else set()
            contract_flag = 1 if "Contract" in labels_set else 0
            token_flag = 1 if ("Token" in labels_set or is_token_prop == 1) else 0

            # Build raw feature vector (order must match training)
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

        # 2. Apply log1p to continuous features (indices 0,1,2,3,4,7)
        node_feats_log = []
        for feats in node_raw_feats:
            log_feats = feats.copy()
            for idx in [0, 1, 2, 3, 4, 7, 8, 9, 10, 11]:
                log_feats[idx] = np.log1p(log_feats[idx])
            node_feats_log.append(log_feats)

        # 3. Scale features if scaler is available
        if self.scaler is not None:
            node_feats_scaled = self.scaler.transform(node_feats_log)
        else:
            node_feats_scaled = node_feats_log

        # 4. Process Relationships (Group by Type for HTGNN)
        edge_data = {}
        edge_timestamps = {}

        for rel in neo4j_rels:
            etype = rel.type.lower()
            src_addr = rel.start_node["address"]
            dst_addr = rel.end_node["address"]

            src_idx = node_id_map.get(src_addr)
            dst_idx = node_id_map.get(dst_addr)
            if src_idx is None or dst_idx is None:
                continue  # Should not happen, but safeguard

            canonical_etype = ("node", etype, "node")
            if canonical_etype not in edge_data:
                edge_data[canonical_etype] = ([], [])
                edge_timestamps[etype] = []

            edge_data[canonical_etype][0].append(src_idx)
            edge_data[canonical_etype][1].append(dst_idx)
            edge_timestamps[etype].append(rel.get("timestamp", 0))

        # 5. Create DGL HeteroGraph
        g = dgl.heterograph(
            {k: (torch.tensor(v[0]), torch.tensor(v[1])) for k, v in edge_data.items()}
        )

        # Add features and labels
        g.nodes["node"].data["feat"] = torch.tensor(
            node_feats_scaled, dtype=torch.float32
        )
        g.nodes["node"].data["label"] = torch.tensor(node_labels, dtype=torch.long)

        # Add edge timestamps
        max_ts = 0
        for etype_str, ts_list in edge_timestamps.items():
            ts_tensor = torch.tensor(ts_list, dtype=torch.float32)
            canonical_etype = ("node", etype_str, "node")
            g.edges[canonical_etype].data["timestamp"] = ts_tensor
            if ts_tensor.numel() > 0:
                max_ts = max(max_ts, ts_tensor.max().item())

        # Global metadata for the HTGNN Forward Pass
        g.graph_data = {
            "max_timestamp": max_ts,
            "current_time": current_time,
            "target_node_idx": node_id_map[target_address],
        }

        return g, node_id_map

    def update_fraud_label(self, wallet_address, label):
        """
        Step 12: Updates the Knowledge Graph after manual confirmation.
        """
        with self.driver.session() as session:
            query = """
            MATCH (n {address: $addr})
            SET n.label = $label
            RETURN n.address AS updated
            """
            return session.run(query, addr=wallet_address, label=int(label)).single()

    def get_node_context(self, address):
        """
        Fetches metadata for a specific address.
        Updated for Neo4j 5.x syntax using COUNT subqueries.
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
