import pandas as pd

# Load the file we created/used
try:
    df = pd.read_csv('DataSet/nodes_audit.csv')
    label_col = 'flag'
except FileNotFoundError:
    # Fallback to the original labeled nodes if audit isn't there yet
    df = pd.read_csv('DataSet/graph_nodes_labeled.csv')
    label_col = 'flag'

# Count the values
counts = df[label_col].value_counts()
normal = counts.get(0, 0)
fraud = counts.get(1, 0)
total = len(df)

# Calculate percentages
fraud_percent = (fraud / total) * 100
imbalance_ratio = normal / fraud if fraud > 0 else 0

print("-" * 30)
print("📊 DATASET IMBALANCE REPORT")
print("-" * 30)
print(f"✅ Normal Nodes (0): {normal:,}")
print(f"🚨 Fraud Nodes  (1): {fraud:,}")
print(f"📈 Total Nodes:      {total:,}")
print(f"🔥 Fraud Percentage: {fraud_percent:.4f}%")
print(f"⚖️  Imbalance Ratio:  1 : {int(imbalance_ratio)}")
print("-" * 30)

if fraud_percent < 0.1:
    print("⚠️ EXTREME IMBALANCE DETECTED: This is why the model is cautious!")