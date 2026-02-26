"""
Data preparation: Exploratory schema inspection utility for node and edge CSVs.

This script provides a minimal exploratory view of the node CSV file produced
by nodes_edges_generation.py, printing the first few rows and schema metadata
for rapid inspection during the data construction phase. Commented-out variants
allow quick switching between node, edge, and transaction CSV inspection targets.
This is a developer utility script used during data pipeline development and
validation; it does not participate in model training or inference.
"""

import pandas as pd

# Load the node feature CSV for schema and sample inspection
df = pd.read_csv("graph_nodes.csv")

# Deprecated alternative implementation retained for historical reference.
# df = pd.read_csv("DataSet/graph_edges.csv", low_memory=False)

# Deprecated alternative implementation retained for historical reference.
# df = pd.read_csv("DataSet/Ethereum_Fraud_Dataset.csv")

# Print the first few rows to verify column presence and data types
print(df.head())
# Print schema, non-null value counts, and inferred dtypes for all columns
print(df.info())

# Deprecated alternative implementation retained for historical reference.
# flag_0_count = df[df['flag'] == 0].shape[0]
# flag_1_count = df[df['flag'] == 1].shape[0]

# Deprecated alternative implementation retained for historical reference.
# print(f"Total rows with Flag = 0: {flag_0_count}")
# print(f"Total rows with Flag = 1: {flag_1_count}")
