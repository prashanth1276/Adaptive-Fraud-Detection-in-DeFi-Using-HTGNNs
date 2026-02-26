import pandas as pd

# Load the original dataset
df = pd.read_csv(r"DataSet/Dataset_with_updated_flags_labeled.csv")


# Make a copy to preserve the original dataset
df_cleaned = df.copy()

# Convert block_timestamp to datetime (standardize time-based features)
# Option 1 (recommended for varying formats):
df_cleaned["block_timestamp"] = pd.to_datetime(
    df_cleaned["block_timestamp"], format="mixed"
)

# Option 2 (if all timestamps have " UTC" and microseconds):
# df_cleaned['block_timestamp'] = pd.to_datetime(df_cleaned['block_timestamp'].str.replace(" UTC", "", regex=False), format="%Y-%m-%d %H:%M:%S.%f")


# Fill missing numeric values with -1 (placeholder for missing data)
numeric_cols = ["function_count", "bytecode_size", "token_value"]
df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(-1)

# Fill missing categorical values
df_cleaned["to_address"] = df_cleaned["to_address"].fillna("unknown_to")
df_cleaned["token_address"] = df_cleaned["token_address"].fillna("unknown_token")


# Normalize string values and fill missing
def normalize_erc_column(col):
    # Convert everything to string, strip, title case, and handle known issues
    col = col.astype(str).str.strip().str.title()
    col = col.replace({"Nan": "Unknown", "None": "Unknown", "": "Unknown"})
    return col.map({"True": 1, "False": 0, "Unknown": -1})


df_cleaned["is_erc20"] = normalize_erc_column(df["is_erc20"])
df_cleaned["is_erc721"] = normalize_erc_column(df["is_erc721"])


# Standardize address format (important for graph nodes)
df_cleaned["from_address"] = df_cleaned["from_address"].str.lower()
df_cleaned["to_address"] = df_cleaned["to_address"].str.lower()
df_cleaned["token_address"] = df_cleaned["token_address"].str.lower()

# Drop any fully duplicated rows (due to multiple logs per tx)
df_cleaned = df_cleaned.drop_duplicates()

# Safety checks to ensure no missing values in critical fields
assert df_cleaned["is_erc20"].isna().sum() == 0, "Missing values in is_erc20"
assert df_cleaned["is_erc721"].isna().sum() == 0, "Missing values in is_erc721"


# Save cleaned dataset to a new CSV file
df_cleaned.to_csv("DataSet/Ethereum_Fraud_Dataset.csv", index=False)

print("✅ Cleaned dataset saved as 'Ethereum_Fraud_Dataset.csv'")

# Count the number of occurrences for each Flag value
flag_0_count = df[df["flag"] == 0].shape[0]
flag_1_count = df[df["flag"] == 1].shape[0]

# Print the results
print(f"Total rows with Flag = 0: {flag_0_count}")
print(f"Total rows with Flag = 1: {flag_1_count}")
