"""
Data preparation: Raw Ethereum transaction dataset cleaning and normalization.

This script performs the initial preprocessing stage of the DeFi fraud detection
data pipeline. It ingests the raw labeled Ethereum transaction CSV, standardizes
timestamp formats, imputes missing values in numeric and categorical fields,
normalizes ERC token flags to a ternary integer encoding, lowercases Ethereum
addresses to ensure consistent join semantics across downstream processing steps,
and removes fully-duplicated rows. The output constitutes the cleaned transaction
dataset consumed by nodes_edges_generation.py for graph node and edge construction.
This is a one-time data preparation step executed prior to graph construction.
"""

import pandas as pd

# Load the raw labeled Ethereum transaction dataset with fraud flags pre-assigned
df = pd.read_csv(r"DataSet/Dataset_with_updated_flags_labeled.csv")


# Preserve the original dataset for auditing or rollback purposes
df_cleaned = df.copy()

# Parse block_timestamp to a timezone-naive datetime object using mixed-format parsing
# to handle records with varying sub-second precision or trailing timezone tokens
df_cleaned["block_timestamp"] = pd.to_datetime(
    df_cleaned["block_timestamp"], format="mixed"
)

# Deprecated alternative implementation retained for historical reference.
# df_cleaned['block_timestamp'] = pd.to_datetime(df_cleaned['block_timestamp'].str.replace(" UTC", "", regex=False), format="%Y-%m-%d %H:%M:%S.%f")


# Impute missing values in numeric columns with -1 as a sentinel for missing data
numeric_cols = ["function_count", "bytecode_size", "token_value"]
df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(-1)

# Impute missing categorical address fields with descriptive sentinel strings
df_cleaned["to_address"] = df_cleaned["to_address"].fillna("unknown_to")
df_cleaned["token_address"] = df_cleaned["token_address"].fillna("unknown_token")


def normalize_erc_column(col):
    """Normalizes ERC standard flags from heterogeneous string representations to integers.

    Converts free-form boolean strings (True/False/NaN/None/empty) in ERC token
    columns to a ternary integer encoding: 1 (True), 0 (False), -1 (Unknown).
    This encoding is required because raw CSV exports from BigQuery may contain
    mixed-type ERC flag values depending on the query execution environment.

    Args:
        col (pd.Series): Raw ERC flag column containing string or boolean values.

    Returns:
        pd.Series: Integer-encoded column with values in {-1, 0, 1}.
    """
    # Normalize to title case strings to handle True/true/TRUE variations
    col = col.astype(str).str.strip().str.title()
    col = col.replace({"Nan": "Unknown", "None": "Unknown", "": "Unknown"})
    return col.map({"True": 1, "False": 0, "Unknown": -1})


df_cleaned["is_erc20"] = normalize_erc_column(df["is_erc20"])
df_cleaned["is_erc721"] = normalize_erc_column(df["is_erc721"])


# Lowercase all Ethereum addresses to ensure case-insensitive join semantics
# across nodes_edges_generation.py, add_fraudlabels_nodes.py, and Neo4j queries
df_cleaned["from_address"] = df_cleaned["from_address"].str.lower()
df_cleaned["to_address"] = df_cleaned["to_address"].str.lower()
df_cleaned["token_address"] = df_cleaned["token_address"].str.lower()

# Remove fully-duplicated rows arising from multi-log transactions in the raw data
df_cleaned = df_cleaned.drop_duplicates()

# Validate that ERC flag columns contain no residual missing values post-normalization
assert df_cleaned["is_erc20"].isna().sum() == 0, "Missing values in is_erc20"
assert df_cleaned["is_erc721"].isna().sum() == 0, "Missing values in is_erc721"


# Write the cleaned dataset to disk for consumption by downstream pipeline stages
df_cleaned.to_csv("DataSet/Ethereum_Fraud_Dataset.csv", index=False)

print("Cleaned dataset saved as 'Ethereum_Fraud_Dataset.csv'")

# Report per-class transaction counts from the original (pre-cleaning) dataset
flag_0_count = df[df["flag"] == 0].shape[0]
flag_1_count = df[df["flag"] == 1].shape[0]

print(f"Total rows with Flag = 0: {flag_0_count}")
print(f"Total rows with Flag = 1: {flag_1_count}")
