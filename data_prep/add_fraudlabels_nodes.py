import pandas as pd

# Load original dataset
df = pd.read_csv("DataSet/Ethereum_Fraud_Dataset.csv")

# Load node data
node_df = pd.read_csv("DataSet/graph_nodes.csv")

# Get fraud flags from `from_address` and `to_address`
from_flags = df[['from_address', 'flag']].rename(columns={'from_address': 'node'})
to_flags = df[['to_address', 'flag']].rename(columns={'to_address': 'node'})

# Combine flags from both sides
all_flags = pd.concat([from_flags, to_flags])
# Keep only the highest flag for each node (1 means fraud, 0 means not)
node_flags = all_flags.groupby('node')['flag'].max().reset_index()

# Merge fraud labels into node_df
node_df = node_df.merge(node_flags, on='node', how='left')
node_df['flag'] = node_df['flag'].fillna(0).astype(int)  # default to 0 for unlabeled

# Save labeled node data
node_df.to_csv("graph_nodes_labeled.csv", index=False)

print("✅ Fraud labels (flag) added to nodes and saved as 'graph_nodes_labeled.csv'")
