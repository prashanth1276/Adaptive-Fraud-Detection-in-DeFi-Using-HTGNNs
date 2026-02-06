import pandas as pd
import dgl
import torch
import numpy as np

# -----------------------------
# 1. Load CSVs
# -----------------------------
nodes_df = pd.read_csv("DataSet/graph_nodes_labeled.csv")
edges_df = pd.read_csv("DataSet/graph_edges.csv", low_memory=False)

# -----------------------------
# 2. Build node universe FROM EDGES
# -----------------------------
all_nodes = pd.Index(
    pd.concat([edges_df['src'], edges_df['dst']]).unique()
)

node_id_map = {node: idx for idx, node in enumerate(all_nodes)}

edges_df['src_id'] = edges_df['src'].map(node_id_map)
edges_df['dst_id'] = edges_df['dst'].map(node_id_map)

# -----------------------------
# 3. Align nodes_df to graph nodes
# -----------------------------
nodes_df = nodes_df[nodes_df['node'].isin(all_nodes)].copy()
nodes_df['node_id'] = nodes_df['node'].map(node_id_map)
nodes_df = nodes_df.sort_values('node_id').reset_index(drop=True)

# -----------------------------
# 4. Build heterograph (single node type, multiple edge types)
# -----------------------------
edge_dict = {}

for etype in edges_df['edge_type'].unique():
    et_df = edges_df[edges_df['edge_type'] == etype]
    edge_dict[('node', etype, 'node')] = (
        torch.tensor(et_df['src_id'].values, dtype=torch.int64),
        torch.tensor(et_df['dst_id'].values, dtype=torch.int64)
    )

g = dgl.heterograph(edge_dict, num_nodes_dict={'node': len(all_nodes)})

# -----------------------------
# 5. Node features
# -----------------------------
feature_cols = [
    'tx_count',
    'incoming_value',
    'outgoing_value',
    'avg_gas_used',
    'unique_peers',
    'contract_flag',
    'token_flag',
    'bytecode_size'
]

node_feat_df = nodes_df[feature_cols].copy()

# Log-scale heavy-tailed features
for col in ['tx_count', 'incoming_value', 'outgoing_value',
            'avg_gas_used', 'unique_peers', 'bytecode_size']:
    node_feat_df[col] = np.log1p(node_feat_df[col])

node_features = torch.tensor(
    node_feat_df.values,
    dtype=torch.float32
)

g.nodes['node'].data['feat'] = node_features

# -----------------------------
# 6. Node labels
# -----------------------------
g.nodes['node'].data['label'] = torch.tensor(
    nodes_df['flag'].values,
    dtype=torch.long
)

# -----------------------------
# 6.5 Node timestamp (last activity time)
# -----------------------------

# Get last transaction timestamp per destination node
node_last_time = edges_df.groupby('dst')['timestamp'].max()

# Map timestamps to nodes
nodes_df['timestamp'] = nodes_df['node'].map(node_last_time).fillna(0)

# Attach to graph
g.nodes['node'].data['timestamp'] = torch.tensor(
    nodes_df['timestamp'].values,
    dtype=torch.long
)

# -----------------------------
# 7. Edge features (NORMALIZED)
# -----------------------------
for etype in g.etypes:
    et_df = edges_df[edges_df['edge_type'] == etype].copy()

    et_df = et_df.fillna({
        'value': 0,
        'token_value': 0,
        'gas': 0,
        'gas_price': 0,
        'tx_frequency': 0,
        'timestamp': 0
    })

    # 🔥 CRITICAL: log-normalization
    et_df['value'] = np.log1p(et_df['value'])
    et_df['token_value'] = np.log1p(et_df['token_value'])
    et_df['gas'] = np.log1p(et_df['gas'])
    et_df['gas_price'] = np.log1p(et_df['gas_price'])
    et_df['tx_frequency'] = np.log1p(et_df['tx_frequency'])

    g.edges[etype].data['timestamp'] = torch.tensor(
        et_df['timestamp'].values, dtype=torch.long
    )
    g.edges[etype].data['value'] = torch.tensor(
        et_df['value'].values, dtype=torch.float32
    )
    g.edges[etype].data['token_value'] = torch.tensor(
        et_df['token_value'].values, dtype=torch.float32
    )
    g.edges[etype].data['gas'] = torch.tensor(
        et_df['gas'].values, dtype=torch.float32
    )
    g.edges[etype].data['gas_price'] = torch.tensor(
        et_df['gas_price'].values, dtype=torch.float32
    )
    g.edges[etype].data['tx_frequency'] = torch.tensor(
        et_df['tx_frequency'].values, dtype=torch.float32
    )

# -----------------------------
# 8. Sanity checks
# -----------------------------
print("✅ DGL heterogeneous graph created successfully")
print(g)
print("Node feature shape:", g.nodes['node'].data['feat'].shape)
print("Node label shape:", g.nodes['node'].data['label'].shape)
print("Edge types:", g.etypes)
