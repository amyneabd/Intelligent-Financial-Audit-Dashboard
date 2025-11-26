import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest

# ==========================
# 1. LOAD THE CSV
# ==========================

dataset_path = r"C:\Users\amine\OneDrive\Bureau\Pfe Preparation\Intelligent Financial Audit Dashboard with AI-driven Anomaly Detection\Dataset\clean_dataset.csv"

df = pd.read_csv(
    dataset_path,
    delimiter=';',
    engine='python'
)

print("Dataset loaded.")
print(df.head())
print("\nCOLUMNS:")
print(df.columns.tolist())

# ==========================
# 2. CLEAN COLUMNS
# ==========================

df = df.rename(columns={
    "Product Name": "Product_Name",
    "Payment Method": "Payment_Method",
    "Transacion Status": "Transaction_Status"   # dataset typo
})

# ==========================
# 3. HANDLE NUMERIC FIELDS
# ==========================

df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

# ==========================
# 4. ENCODE CATEGORICAL FIELDS
# ==========================

cat_cols = ["Product_Name", "Payment_Method", "Transaction_Status"]

encoder = LabelEncoder()
for col in cat_cols:
    df[col] = encoder.fit_transform(df[col].astype(str))

print("\nAFTER ENCODING:")
print(df[cat_cols].head())

# ==========================
# 5. FEATURE MATRIX
# ==========================

features = ["Quantity", "Price", "Product_Name", "Payment_Method", "Transaction_Status"]
X = df[features]

print("\nFEATURE MATRIX SAMPLE:")
print(X.head())

# ==========================
# 6. TRAIN ISOLATION FOREST
# ==========================

model = IsolationForest(
    contamination=0.05,
    random_state=42
)

model.fit(X)

# ==========================
# 7. PREDICTIONS
# ==========================

df["Anomaly"] = model.predict(X)        # 1 = normal, -1 = anomaly
df["Anomaly_Score"] = model.score_samples(X)

anomalies = df[df["Anomaly"] == -1]

print("\nDETECTED ANOMALIES:")
print(anomalies.head())

# ==========================
# 8. EXPORT RESULTS (same folder)
# ==========================

export_base_path = r"C:\Users\amine\OneDrive\Bureau\Pfe Preparation\Intelligent Financial Audit Dashboard with AI-driven Anomaly Detection\Dataset"

df.to_csv(fr"{export_base_path}\anomaly_output.csv", index=False)
anomalies.to_csv(fr"{export_base_path}\anomalies_only.csv", index=False)

print("\nFiles exported to:")
print(fr"{export_base_path}\anomaly_output.csv")
print(fr"{export_base_path}\anomalies_only.csv")
