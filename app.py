import streamlit as st
import torch
import dgl
import pickle
import pandas as pd
import networkx as nx
import joblib
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
from model.htg_nn import HTGNN
from model.adaptive_engine import adapt_model_to_new_fraud
from model.database import Neo4jKnowledgeGraph

# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------
st.set_page_config(page_title="🛡️ DeFi Fraud Guard", layout="wide")

# Load environment variables (for Neo4j credentials)
load_dotenv()

# -------------------------------------------------------------------
# SESSION STATE
# -------------------------------------------------------------------
if 'analyzed_node' not in st.session_state:
    st.session_state.analyzed_node = None
if 'last_prob' not in st.session_state:
    st.session_state.last_prob = 0.0

# -------------------------------------------------------------------
# DATABASE CONNECTION (Neo4j)
# -------------------------------------------------------------------
@st.cache_resource
def get_db_connection():
    try:
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "")
        conn = Neo4jKnowledgeGraph(uri, user, password)
        # Test connection
        with conn.driver.session() as session:
            session.run("RETURN 1").single()
        return conn
    except Exception as e:
        st.error(f"❌ Neo4j connection failed: {e}\nPlease check your credentials and ensure Neo4j is running.")
        return None

db = get_db_connection()

# -------------------------------------------------------------------
# LOAD TRAINED MODEL, GRAPH, AND MAPPINGS
# -------------------------------------------------------------------
@st.cache_resource
def load_resources():
    try:
        # 1. DGL graph
        if not os.path.exists("DataSet/graph.bin"):
            st.error("Missing DataSet/graph.bin. Please run dgl_graph_construction.py first.")
            st.stop()
        graphs, _ = dgl.load_graphs("DataSet/graph.bin")
        g = graphs[0]

        # 2. Node ID mapping
        if not os.path.exists("DataSet/node_id_map.pkl"):
            st.error("Missing DataSet/node_id_map.pkl. Please run dgl_graph_construction.py first.")
            st.stop()
        with open("DataSet/node_id_map.pkl", "rb") as f:
            node_id_map = pickle.load(f)

        idx_to_addr = {v: k for k, v in node_id_map.items()}

        # 3. Scaler (required to transform features)
        if not os.path.exists("DataSet/scaler.pkl"):
            st.error("Missing DataSet/scaler.pkl. Please run train.py first.")
            st.stop()
        scaler = joblib.load("DataSet/scaler.pkl")

        # 4. Apply scaling to node features
        if 'feat_raw' not in g.nodes['node'].data:
            st.error("Graph does not contain 'feat_raw'. Check dgl_graph_construction.py output.")
            st.stop()
        feat_raw = g.nodes['node'].data['feat_raw'].numpy()   # move to CPU numpy
        feat_scaled = scaler.transform(feat_raw)
        g.nodes['node'].data['feat'] = torch.tensor(feat_scaled, dtype=torch.float32)

        # 5. Ensure edge timestamps are float
        for etype in g.etypes:
            if 'timestamp' in g.edges[etype].data:
                g.edges[etype].data['timestamp'] = g.edges[etype].data['timestamp'].float()

        # 6. Populate graph_data if missing
        if not hasattr(g, 'graph_data') or not g.graph_data:
            g.graph_data = {}
        if 'max_timestamp' not in g.graph_data:
            # Compute from edges
            max_ts = 0
            for et in g.etypes:
                if g.num_edges(et) > 0 and 'timestamp' in g.edges[et].data:
                    ts_max = g.edges[et].data['timestamp'].max().item()
                    max_ts = max(max_ts, int(ts_max))
            g.graph_data['max_timestamp'] = max_ts

        # 7. Trained model
        if not os.path.exists("best_model.pth"):
            st.error("Missing best_model.pth. Please run train.py first.")
            st.stop()

        device = torch.device("cpu")
        checkpoint = torch.load("best_model.pth", map_location=device)

        input_dim = g.nodes['node'].data['feat'].shape[1]
        model = HTGNN(
            input_dim=input_dim,
            hidden_dim=128,
            output_dim=1,
            time_dim=32,
            edge_types=g.etypes
        ).to(device)

        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
            threshold = checkpoint.get("threshold", 0.4838)
        else:
            model.load_state_dict(checkpoint)
            threshold = 0.4838

        model.eval()

        return g, node_id_map, idx_to_addr, model, threshold, device, scaler
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        st.stop()

g, node_id_map, idx_to_addr, model, threshold, device, scaler = load_resources()

# -------------------------------------------------------------------
# UI HEADER
# -------------------------------------------------------------------
st.title("🛡️ Adaptive DeFi Fraud Detection")
st.markdown("Architecture: **HTGNN** | Data: **Ethereum (DGL + Neo4j)**")

# -------------------------------------------------------------------
# SIDEBAR – INFORMATION
# -------------------------------------------------------------------
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    - Enter an Ethereum address that exists in the graph.
    - The model returns a **risk score** (0–1).
    - If you disagree with the prediction, use the **Adaptive Engine** to fine‑tune the model.
    - The system uses **temporal edge masking** – only transactions that occurred **before** the node's last activity are considered.
    """)
    if db:
        st.success("✅ Neo4j connected")
    else:
        st.warning("⚠️ Neo4j not connected – some features may be limited.")

# -------------------------------------------------------------------
# MAIN – ADDRESS ANALYSIS
# -------------------------------------------------------------------
st.subheader("🔍 Analyze an Ethereum Address")

search_addr = st.text_input("Enter wallet / contract address:", placeholder="0x...").strip()

if st.button("Analyze Risk Profile"):
    if not search_addr:
        st.warning("Please enter an address.")
    elif search_addr not in node_id_map:
        st.error("Address not found in the graph. Please check the address or run data preprocessing first.")
    else:
        st.session_state.analyzed_node = search_addr

        # Optionally fetch additional context from Neo4j (if available)
        if db:
            try:
                context = db.get_node_context(search_addr)
                if context:
                    st.sidebar.markdown("**Neo4j Context**")
                    st.sidebar.json(context)
            except Exception as e:
                st.sidebar.warning(f"Could not retrieve context from Neo4j: {e}")

# -------------------------------------------------------------------
# RESULTS & ADAPTIVE ENGINE
# -------------------------------------------------------------------
if st.session_state.analyzed_node:
    addr = st.session_state.analyzed_node
    node_id = node_id_map[addr]

    # Safety check – node index must be within range
    if node_id >= g.num_nodes('node'):
        st.error("Node index out of range – please restart the app.")
        st.stop()

    # --- Determine current_time from node's last activity ---
    if 'timestamp' in g.nodes['node'].data:
        node_time = g.nodes['node'].data['timestamp'][node_id].item()
        current_time = node_time
    else:
        # Fallback to global max (less accurate)
        current_time = g.graph_data.get('max_timestamp', 0)
        st.caption("⚠️ Node timestamp not available – using global max.")

    # --- Extract 2‑hop subgraph (for model input) ---
    sub_g, _ = dgl.khop_in_subgraph(g, node_id, k=2)
    sub_g.graph_data = {
        'max_timestamp': g.graph_data.get('max_timestamp', 0),
        'current_time': current_time
    }

    # --- Inference ---
    with torch.no_grad():
        logits = model(sub_g)
        local_mask = (sub_g.ndata[dgl.NID] == node_id)
        if not local_mask.any():
            st.error("Target node lost during subgraph extraction.")
            st.stop()
        local_id = local_mask.nonzero(as_tuple=True)[0].item()
        prob = torch.sigmoid(logits[local_id]).item()
        st.session_state.last_prob = prob

    # --- Display metrics ---
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        st.metric("Risk Score", f"{prob*100:.2f}%")
        if prob > threshold:
            st.error("🔴 HIGH RISK")
        elif prob > threshold * 0.5:
            st.warning("🟡 MEDIUM RISK")
        else:
            st.success("🟢 LOW RISK")

    with col2:
        st.write("**Metadata**")
        st.caption(f"Address: {addr[:15]}...")
        st.caption(f"Internal ID: {node_id}")
        st.caption(f"Current time: {current_time}")

    with col3:
        # --- Visualise 1‑hop neighbourhood ---
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
            pos = nx.kamada_kawai_layout(nx_viz)
        except:
            pos = nx.spring_layout(nx_viz, seed=42)

        target_local_id = (viz_g.ndata[dgl.NID] == node_id).nonzero(as_tuple=True)[0].item()
        node_colors = [
            "red" if i == target_local_id and prob > threshold
            else "green" if i == target_local_id
            else "skyblue"
            for i in range(viz_g.num_nodes())
        ]
        nx.draw(nx_viz, pos, ax=ax, node_size=100, node_color=node_colors,
                with_labels=False, edge_color="gray", alpha=0.7)
        st.pyplot(fig)
        plt.close(fig)
        # --- Subgraph details (nodes and edges) ---
    with st.expander("🔍 Show subgraph details"):
        # Nodes in 1‑hop subgraph
        subgraph_node_ids = viz_g.ndata[dgl.NID].tolist()
        node_details = []
        for local_id, global_id in enumerate(subgraph_node_ids):
            addr_full = idx_to_addr[global_id]
            node_type = "target" if global_id == node_id else "neighbor"
            node_details.append({
                "Local ID": local_id,
                "Address": addr_full,
                "Short Address": addr_full[:10] + "...",
                "Type": node_type
            })
        node_df = pd.DataFrame(node_details)
        st.markdown("**Nodes in this subgraph:**")
        st.dataframe(node_df[["Local ID", "Short Address", "Type"]], use_container_width=True)

        # Edges in 1‑hop subgraph
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
                edge_list.append({
                    "Source": src_addr,
                    "Destination": dst_addr,
                    "Type": etype[1]
                })
        if edge_list:
            edge_df = pd.DataFrame(edge_list)
            st.markdown("**Edges in this subgraph:**")
            st.dataframe(edge_df, use_container_width=True)
        else:
            st.info("No edges in this subgraph (isolated node).")

    # -------------------------------------------------------------------
    # ADAPTIVE ENGINE – FINE‑TUNE ON USER FEEDBACK
    # -------------------------------------------------------------------
    st.divider()
    st.subheader("🛠️ Adaptive Engine")
    st.markdown("If the prediction seems wrong, correct it here. The model will fine‑tune on this example (without forgetting old patterns).")

    target_label = st.radio(
        "Correct label:",
        options=[0, 1],
        format_func=lambda x: "🚨 Fraud" if x == 1 else "✅ Safe",
        horizontal=True
    )

    if st.button("Fine‑Tune Model"):
        if not db:
            st.error("Neo4j not connected – cannot update knowledge graph.")
        else:
            try:
                with st.spinner("Adapting HTGNN weights..."):
                    # 1. Update Neo4j label
                    db.update_fraud_label(addr, target_label)

                    # 2. Fine‑tune the model on this subgraph (with correct current_time)
                    status = adapt_model_to_new_fraud(
                        model, sub_g, local_id,
                        label=target_label,
                        current_time=current_time
                    )

                    # 3. Update the node's label in the main DGL graph and save
                    if 'label' in g.nodes['node'].data:
                        g.nodes['node'].data['label'][node_id] = torch.tensor(target_label, dtype=torch.long)
                    else:
                        g.nodes['node'].data['label'] = torch.full((g.num_nodes('node'),), target_label, dtype=torch.long)

                    dgl.save_graphs("DataSet/graph.bin", [g])

                    # 4. Save updated model checkpoint
                    checkpoint = {
                        'model_state': model.state_dict(),
                        'threshold': threshold
                    }
                    torch.save(checkpoint, 'best_model.pth')

                    st.success(status)
                    st.balloons()
            except Exception as e:
                st.error(f"Fine‑tuning failed: {e}")