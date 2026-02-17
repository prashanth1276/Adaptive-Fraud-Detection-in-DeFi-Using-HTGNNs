import pandas as pd
from datetime import datetime

# Load dataset
df = pd.read_csv('DataSet/Ethereum_Fraud_Dataset.csv')

# Ensure timestamps are in datetime format
df['block_timestamp'] = pd.to_datetime(df['block_timestamp'])

# -------------------------------
# Step 1: Create unique node list
# -------------------------------
wallet_nodes = set(df['from_address']).union(set(df['to_address']))
contract_nodes = set(df[df['bytecode_size'] > 0]['to_address'])
token_nodes = set(df['token_address'])

all_nodes = wallet_nodes.union(contract_nodes).union(token_nodes)

node_df = pd.DataFrame({'node': list(all_nodes)})

# Node type labeling
node_df['node_type'] = node_df['node'].apply(
    lambda x: 'contract' if x in contract_nodes
    else 'token' if x in token_nodes
    else 'wallet'
)

# -------------------------------
# Step 2: Node Features
# -------------------------------

# Transaction count (from or to)
tx_count = df['from_address'].value_counts().add(df['to_address'].value_counts(), fill_value=0)

# ETH received and sent
incoming_value = df.groupby('to_address')['value'].sum()
outgoing_value = df.groupby('from_address')['value'].sum()

# Average gas used per node (as sender)
avg_gas = df.groupby('from_address')['gas'].mean()

# --- Compute unique_peers (distinct counterparties, both incoming and outgoing) ---
# Create sets of peers for incoming and outgoing transactions
incoming_peers = df.groupby('to_address')['from_address'].unique().apply(set)
outgoing_peers = df.groupby('from_address')['to_address'].unique().apply(set)

def count_unique_peers(addr):
    peers = set()
    if addr in incoming_peers.index:
        peers.update(incoming_peers[addr])
    if addr in outgoing_peers.index:
        peers.update(outgoing_peers[addr])
    return len(peers)

# Map features to nodes
node_df.set_index('node', inplace=True)
node_df['tx_count'] = node_df.index.map(tx_count).fillna(0)
node_df['incoming_value'] = node_df.index.map(incoming_value).fillna(0)
node_df['outgoing_value'] = node_df.index.map(outgoing_value).fillna(0)
node_df['avg_gas_used'] = node_df.index.map(avg_gas).fillna(0)
node_df['unique_peers'] = node_df.index.to_series().apply(count_unique_peers).fillna(0).astype(int)   # new calculation

node_df['contract_flag'] = (node_df['node_type'] == 'contract').astype(int)
node_df['token_flag'] = (node_df['node_type'] == 'token').astype(int)

# Aggregate bytecode size per to_address (some addresses appear multiple times)
bytecode_map = df[df['bytecode_size'] > 0].groupby('to_address')['bytecode_size'].max()
node_df['bytecode_size'] = node_df.index.map(bytecode_map).fillna(0)

node_df.reset_index(inplace=True)

# -------------------------------
# Step 3: Edge Construction
# -------------------------------

# ETH transfer edges
eth_edges = df[['from_address', 'to_address', 'value', 'gas', 'gas_price', 'block_timestamp', 'block_number', 'transaction_index']].copy()
eth_edges.rename(columns={'from_address': 'src', 'to_address': 'dst'}, inplace=True)
eth_edges['edge_type'] = 'eth_transfer'

# Token transfer edges
token_edges = df[df['token_value'] > 0][['from_address', 'to_address', 'token_address', 'token_value', 'gas', 'gas_price', 'block_timestamp', 'block_number', 'transaction_index']].copy()
token_edges.rename(columns={'from_address': 'src', 'to_address': 'dst'}, inplace=True)
token_edges['edge_type'] = 'token_transfer'


# Contract interaction edges
contract_edges = df[df['bytecode_size'] > 0][['from_address', 'to_address', 'gas', 'gas_price', 'block_timestamp', 'block_number', 'transaction_index']].copy()
contract_edges.rename(columns={'from_address': 'src', 'to_address': 'dst'}, inplace=True)
contract_edges['edge_type'] = 'contract_call'
contract_edges['value'] = 0
contract_edges['token_value'] = 0
contract_edges['token_address'] = None


# Combine all edge types
all_edges = pd.concat([eth_edges, token_edges, contract_edges], ignore_index=True)

# Timestamp conversion (to Unix time)
all_edges['timestamp'] = all_edges['block_timestamp'].astype('int64') // 10**9

# Optional: Edge-level frequency feature (number of transactions between src-dst)
edge_freq = all_edges.groupby(['src', 'dst']).size().reset_index(name='tx_frequency')
all_edges = all_edges.merge(edge_freq, on=['src', 'dst'], how='left')

# -------------------------------
# Step 4: Final Output
# -------------------------------

# Final node features
node_df.to_csv("graph_nodes.csv", index=False)


# Final edge features
edge_feature_cols = [
    'src', 'dst', 'edge_type', 'timestamp', 'value', 'token_value',
    'token_address', 'gas', 'gas_price', 'block_number',
    'transaction_index', 'tx_frequency'
]
all_edges[edge_feature_cols].to_csv("graph_edges.csv", index=False)

print("✅ Nodes and edges (with features) extracted and saved!")
