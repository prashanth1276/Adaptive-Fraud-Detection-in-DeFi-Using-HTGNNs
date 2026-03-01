"""
Streamlit inference dashboard for the Adaptive DeFi Fraud Detection system.

This module implements the production-facing user interface for the HTGNN-based
DeFi fraud detection pipeline. It provides: (1) address-level fraud risk scoring
via temporally-masked 2-hop subgraph inference, (2) Neo4j knowledge graph context
retrieval for enriched address metadata, (3) 1-hop neighbourhood visualization,
(4) subgraph structural detail inspection, and (5) an adaptive engine that allows
analysts to fine-tune the HTGNN on confirmed fraud labels without full retraining.
The dashboard exposes the complete inference-time pipeline: data loading,
feature scaling, temporal edge resolution, HTGNN forward pass, and online model
adaptation via model/adaptive_engine.py. All heavy resources (model, graph,
scaler, node map) are cached using Streamlit's resource caching mechanism to
avoid redundant disk I/O across user sessions.
"""

import os
import pickle
import shutil
import time

import dgl
import joblib
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import streamlit as st
import torch
from dotenv import load_dotenv

from model.adaptive_engine import adapt_model_to_new_fraud
from model.database import Neo4jKnowledgeGraph
from model.htg_nn import HTGNN

# Configure page layout and load environment variables for Neo4j credentials
st.set_page_config(page_title="DeFi Fraud Guard", layout="wide")
load_dotenv()

# Initialize persistent session state keys for cross-interaction state management
if "analyzed_node" not in st.session_state:
    st.session_state.analyzed_node = None
if "last_prob" not in st.session_state:
    st.session_state.last_prob = 0.0
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False


@st.cache_resource
def get_db_connection():
    """Establishes and caches a connection to the Neo4j knowledge graph.

    Reads Neo4j connection parameters from environment variables with sensible
    defaults and validates the connection with a lightweight RETURN 1 query.
    The connection is cached as a Streamlit resource to persist across reruns.

    Returns:
        Neo4jKnowledgeGraph | None: An initialized database connection object,
            or None if the connection attempt fails (e.g., Neo4j not running).
    """
    try:
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "")
        conn = Neo4jKnowledgeGraph(uri, user, password)
        # Validate connectivity with a trivial Cypher query before returning
        with conn.driver.session() as session:
            session.run("RETURN 1").single()
        return conn
    except Exception as e:
        st.error(
            f"Neo4j connection failed: {e}\nPlease check your credentials and ensure Neo4j is running."
        )
        return None


db = get_db_connection()


@st.cache_resource
def load_resources():
    """Loads and caches all inference resources: graph, node map, scaler, and model.

    Performs the complete resource loading procedure required for HTGNN inference:
      1. Loads the pre-serialized DGL heterogeneous graph (graph.bin).
      2. Loads the address-to-node-index mapping (node_id_map.pkl).
      3. Loads the training-fit StandardScaler (scaler.pkl).
      4. Applies scaler to raw node features and assigns the scaled tensor.
      5. Ensures edge timestamp tensors are float for temporal masking.
      6. Resolves the global max_timestamp metadata if absent.
      7. Loads the HTGNN checkpoint (best_model.pth) and sets to eval mode.

    Returns:
        tuple: (g, node_id_map, idx_to_addr, model, threshold, device, scaler)
            - g (dgl.DGLHeteroGraph): Inference-ready heterogeneous graph.
            - node_id_map (dict[str, int]): Address to node index mapping.
            - idx_to_addr (dict[int, str]): Inverse mapping for display.
            - model (HTGNN): Loaded model in eval mode.
            - threshold (float): Optimal classification threshold from checkpoint.
            - device (torch.device): Target computation device.
            - scaler (StandardScaler): Fitted feature normalizer.
    """
    with st.spinner("Loading model and graph... (may take a few seconds)"):
        time.sleep(1)  # Allow the spinner to render before heavy I/O begins

        # Load the pre-serialized DGL heterogeneous graph
        if not os.path.exists("DataSet/graph.bin"):
            st.error(
                "Missing DataSet/graph.bin. Please run dgl_graph_construction.py first."
            )
            st.stop()
        graphs, _ = dgl.load_graphs("DataSet/graph.bin")
        g = graphs[0]

        # Load the address-to-integer node index mapping
        if not os.path.exists("DataSet/node_id_map.pkl"):
            st.error(
                "Missing DataSet/node_id_map.pkl. Please run dgl_graph_construction.py first."
            )
            st.stop()
        with open("DataSet/node_id_map.pkl", "rb") as f:
            node_id_map = pickle.load(f)

        # Derive inverse mapping for address display in the subgraph detail panel
        idx_to_addr = {v: k for k, v in node_id_map.items()}

        # Load the training-fit StandardScaler for feature normalization
        if not os.path.exists("DataSet/scaler.pkl"):
            st.error("Missing DataSet/scaler.pkl. Please run train.py first.")
            st.stop()
        scaler = joblib.load("DataSet/scaler.pkl")

        # Apply the scaler to raw node features to produce the inference-ready feature tensor
        if "feat_raw" not in g.nodes["node"].data:
            st.error(
                "Graph does not contain 'feat_raw'. Check dgl_graph_construction.py output."
            )
            st.stop()
        feat_raw = g.nodes["node"].data["feat_raw"].numpy()
        feat_scaled = scaler.transform(feat_raw)
        g.nodes["node"].data["feat"] = torch.tensor(feat_scaled, dtype=torch.float32)

        # Ensure edge timestamps are float32 for temporal masking arithmetic in HTGNN
        for etype in g.etypes:
            if "timestamp" in g.edges[etype].data:
                g.edges[etype].data["timestamp"] = (
                    g.edges[etype].data["timestamp"].float()
                )

        # Populate graph_data if absent; required by the HTGNN forward pass
        if not hasattr(g, "graph_data") or not g.graph_data:
            g.graph_data = {}
        if "max_timestamp" not in g.graph_data:
            # Compute the global max timestamp from all edge types
            max_ts = 0
            for et in g.etypes:
                if g.num_edges(et) > 0 and "timestamp" in g.edges[et].data:
                    ts_max = g.edges[et].data["timestamp"].max().item()
                    max_ts = max(max_ts, int(ts_max))
            g.graph_data["max_timestamp"] = max_ts

        # Load the HTGNN model checkpoint produced by train/train.py
        if not os.path.exists("results/best_model.pth"):
            st.error("Missing results/best_model.pth. Please run train.py first.")
            st.stop()

        device = torch.device("cpu")
        checkpoint = torch.load("results/best_model.pth", map_location=device)

        input_dim = g.nodes["node"].data["feat"].shape[1]
        model = HTGNN(
            input_dim=input_dim,
            hidden_dim=128,
            output_dim=1,
            time_dim=32,
            edge_types=g.etypes,
        ).to(device)

        # Support both raw state_dict checkpoints and structured checkpoint dicts
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
            threshold = checkpoint.get("threshold", 0.5306)
        else:
            model.load_state_dict(checkpoint)
            threshold = 0.5306

        model.eval()
        st.session_state.model_loaded = True
        return g, node_id_map, idx_to_addr, model, threshold, device, scaler


g, node_id_map, idx_to_addr, model, threshold, device, scaler = load_resources()

# UI header section
st.title("Adaptive DeFi Fraud Detection")
st.markdown("Architecture: **HTGNN** | Data: **Ethereum (DGL + Neo4j)**")
st.caption(f"Model threshold: **{threshold:.4f}** | Validation AUPRC: **0.1950**")

# Sidebar: system description and model performance metrics
with st.sidebar:
    st.header("About")
    st.markdown("""
    - Enter an Ethereum address that exists in the graph.
    - The model returns a **risk score** (0-1).
    - If you disagree with the prediction, use the **Adaptive Engine** to fine-tune the model.
    - The system uses **temporal edge masking** - only transactions that occurred **before** the node's last activity are considered.
    """)
    if db:
        st.success("Neo4j connected")
    else:
        st.warning("Neo4j not connected - some features may be limited.")

    st.divider()
    st.subheader("Model Performance")
    st.metric("Validation AUPRC", "0.1950", delta="+0.0667 from 8-feature model")
    st.metric("Optimal Threshold", f"{threshold:.4f}")
    st.caption("AUPRC is the primary metric for imbalanced fraud detection.")

# Main panel: address input and risk analysis trigger
st.subheader("Analyze an Ethereum Address")

search_addr = st.text_input(
    "Enter address:", 
    placeholder="0x...", 
    key="addr_input",
    autocomplete="new-password"
).strip()

col_analyze, col_clear = st.columns([3, 1])

with col_analyze:
    analyze_clicked = st.button("Analyze Risk Profile", use_container_width=True)

with col_clear:
    if st.button("Clear", use_container_width=True):
        st.session_state.analyzed_node = None
        st.rerun()

if analyze_clicked:
    if not search_addr:
        st.warning("Please enter an address.")
    elif search_addr not in node_id_map:
        st.error(
            "Address not found in the graph. Please check the address or run data preprocessing first."
        )
    else:
        st.session_state.analyzed_node = search_addr

        # Fetch enriched address context from the Neo4j knowledge graph if connected
        if db:
            try:
                context = db.get_node_context(search_addr)
                if context:
                    st.sidebar.markdown("**Neo4j Context**")
                    st.sidebar.json(context)
            except Exception as e:
                st.sidebar.warning(f"Could not retrieve context from Neo4j: {e}")

# Results panel: inference output, neighbourhood visualization, and adaptive engine
if st.session_state.analyzed_node:
    addr = st.session_state.analyzed_node
    node_id = node_id_map[addr]

    # Validate node index is within the loaded graph's node range
    if node_id >= g.num_nodes("node"):
        st.error("Node index out of range - please restart the app.")
        st.stop()

    # Resolve the temporal query boundary from the node's last-activity timestamp
    if "timestamp" in g.nodes["node"].data:
        node_time = g.nodes["node"].data["timestamp"][node_id].item()
        current_time = node_time
    else:
        current_time = g.graph_data.get("max_timestamp", 0)
        st.caption("Node timestamp not available - using global max.")

    # Extract the 2-hop subgraph for inductive inference and assign temporal metadata
    with st.spinner("Extracting subgraph and running inference..."):
        sub_g, _ = dgl.khop_in_subgraph(g, node_id, k=2)
        sub_g.graph_data = {
            "max_timestamp": g.graph_data.get("max_timestamp", 0),
            "current_time": current_time,
        }

        # Run HTGNN inference on the 2-hop subgraph; extract fraud probability for target node
        with torch.no_grad():
            logits = model(sub_g)
            # Locate the target node's local index within the subgraph
            local_mask = sub_g.ndata[dgl.NID] == node_id
            if not local_mask.any():
                st.error("Target node lost during subgraph extraction.")
                st.stop()
            local_id = local_mask.nonzero(as_tuple=True)[0].item()
            # Apply sigmoid to logit to obtain calibrated fraud probability
            prob = torch.sigmoid(logits[local_id]).item()
            st.session_state.last_prob = prob

    # Display risk metrics in a three-column layout
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        st.metric("Risk Score", f"{prob * 100:.2f}%")
        if prob > threshold:
            st.error("HIGH RISK")
        elif prob > threshold * 0.5:
            st.warning("MEDIUM RISK")
        else:
            st.success("LOW RISK")

    with col2:
        st.write("**Metadata**")
        st.caption(f"Address: {addr[:15]}...")
        st.caption(f"Internal ID: {node_id}")
        st.caption(f"Last activity: {time.ctime(current_time)}")

    with col3:
        # Render a 1-hop neighbourhood graph for visual structural inspection
        viz_g, _ = dgl.khop_in_subgraph(g, node_id, k=1)
        nx_viz = nx.MultiDiGraph()
        for i in range(viz_g.num_nodes()):
            nx_viz.add_node(i)

        for etype in viz_g.canonical_etypes:
            u, v = viz_g.edges(etype=etype)
            for s, d in zip(u.tolist(), v.tolist()):
                nx_viz.add_edge(s, d, type=etype[1])

        fig, ax = plt.subplots(figsize=(5, 3))
        try:
            # Kamada-Kawai layout produces better visual separation for small graphs
            pos = nx.kamada_kawai_layout(nx_viz)
        except Exception:
            # Fallback to spring layout if Kamada-Kawai fails (e.g., disconnected graph)
            pos = nx.spring_layout(nx_viz, seed=42)

        # Locate the target node's local id in the visualization subgraph
        target_local_id = (
            (viz_g.ndata[dgl.NID] == node_id).nonzero(as_tuple=True)[0].item()
        )
        # Color the target node red if high-risk, green if low-risk; neighbors in sky blue
        node_colors = [
            "red"
            if i == target_local_id and prob > threshold
            else "green"
            if i == target_local_id
            else "skyblue"
            for i in range(viz_g.num_nodes())
        ]
        nx.draw(
            nx_viz,
            pos,
            ax=ax,
            node_size=100,
            node_color=node_colors,
            with_labels=False,
            edge_color="gray",
            alpha=0.7,
        )
        st.pyplot(fig)
        plt.close(fig)

    # Expandable subgraph detail panel displaying node and edge tables
    with st.expander("Show subgraph details"):
        subgraph_node_ids = viz_g.ndata[dgl.NID].tolist()
        node_details = []
        for local_id, global_id in enumerate(subgraph_node_ids):
            addr_full = idx_to_addr[global_id]
            node_type = "target" if global_id == node_id else "neighbor"
            node_details.append(
                {
                    "Local ID": local_id,
                    "Address": addr_full,
                    "Short Address": addr_full[:10] + "...",
                    "Type": node_type,
                }
            )
        node_df = pd.DataFrame(node_details)
        st.markdown("**Nodes in this subgraph:**")
        st.dataframe(
            node_df[["Local ID", "Short Address", "Type"]], use_container_width=True
        )

        edge_list = []
        for etype in viz_g.canonical_etypes:
            u, v = viz_g.edges(etype=etype)
            src_local = u.tolist()
            dst_local = v.tolist()
            for s, d in zip(src_local, dst_local):
                src_global = subgraph_node_ids[s]
                dst_global = subgraph_node_ids[d]
                src_addr = idx_to_addr[src_global][:10] + "..."
                dst_addr = idx_to_addr[dst_global][:10] + "..."
                edge_list.append(
                    {"Source": src_addr, "Destination": dst_addr, "Type": etype[1]}
                )
        if edge_list:
            edge_df = pd.DataFrame(edge_list)
            st.markdown("**Edges in this subgraph:**")
            st.dataframe(edge_df, use_container_width=True)
        else:
            st.info("No edges in this subgraph (isolated node).")

    # Adaptive engine: analyst-driven online fine-tuning
    st.divider()
    st.subheader("Adaptive Engine")
    st.markdown(
        "If the prediction seems wrong, correct it here. The model will fine-tune on this example (without forgetting old patterns)."
    )

    target_label = st.radio(
        "Correct label:",
        options=[0, 1],
        format_func=lambda x: "Fraud" if x == 1 else "Safe",
        horizontal=True,
        key="adaptive_radio",
    )

    if st.button("Fine-Tune Model"):
        if not db:
            st.error("Neo4j not connected - cannot update knowledge graph.")
        else:
            try:
                old_prob = prob  # Capture pre-adaptation fraud probability
                with st.spinner("Adapting HTGNN weights (15 epochs)..."):
                    # Step 1: Propagate analyst-confirmed label to the Neo4j knowledge graph
                    db.update_fraud_label(addr, target_label)

                    # Step 2: Execute the micro-training adaptation loop on the subgraph
                    status = adapt_model_to_new_fraud(
                        model,
                        sub_g,
                        local_id,
                        label=target_label,
                        current_time=current_time,
                    )

                    # Step 3: Update the in-memory graph's label tensor to reflect the change
                    if "label" in g.nodes["node"].data:
                        g.nodes["node"].data["label"][node_id] = torch.tensor(
                            target_label, dtype=torch.long
                        )
                    else:
                        g.nodes["node"].data["label"] = torch.full(
                            (g.num_nodes("node"),), target_label, dtype=torch.long
                        )

                    # Step 4: Persist the updated graph to disk for session continuity
                    dgl.save_graphs("DataSet/graph.bin", [g])

                    # Step 5: Save the updated model checkpoint with the original threshold
                    checkpoint = {
                        "model_state": model.state_dict(),
                        "threshold": threshold,
                    }
                    torch.save(checkpoint, "best_model.pth")
                    shutil.copy("best_model.pth", "results/best_model.pth")

                    # Step 6: Re-score the target node with the adapted model
                    fresh_sub_g, _ = dgl.khop_in_subgraph(g, node_id, k=2)
                    fresh_sub_g.graph_data = {
                        "max_timestamp": g.graph_data.get("max_timestamp", 0),
                        "current_time": current_time,
                    }
                    fresh_local_mask = fresh_sub_g.ndata[dgl.NID] == node_id
                    fresh_local_id = fresh_local_mask.nonzero(as_tuple=True)[0].item()
                    with torch.no_grad():
                        new_logits = model(fresh_sub_g)
                        new_prob = torch.sigmoid(new_logits[fresh_local_id]).item()

                    improvement = new_prob - old_prob
                    st.success(
                        f"{status}\n\n"
                        f"Risk score changed from **{old_prob:.4f}** to **{new_prob:.4f}** "
                        f"({improvement:+.4f})."
                    )
                    st.balloons()
                    st.info(
                        "The model has been updated. Re-analyze the address to see the new prediction."
                    )
            except Exception as e:
                st.error(f"Fine-tuning failed: {e}")
