import pandas as pd

df = pd.read_csv("Dataset/graph_nodes_labeled.csv")

#df = pd.read_csv("Dataset/Ethereum_Fraud_Dataset.csv")
print(df.head())
print(df.info())
# Count the number of occurrences for each Flag value
#flag_0_count = df[df['flag'] == 0].shape[0]
#flag_1_count = df[df['flag'] == 1].shape[0]

# Print the results
#print(f"Total rows with Flag = 0: {flag_0_count}")
#print(f"Total rows with Flag = 1: {flag_1_count}")