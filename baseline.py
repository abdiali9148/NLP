import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from helper import evaluate, set_seed

set_seed(42)

print("Loading datasets...")
df_train = pd.read_parquet("dataset/yelp_train.parquet")
df_dev = pd.read_parquet("dataset/yelp_dev.parquet")

def build_features(df):
    """Extracts metadata features directly from raw columns."""
    df_out = df.copy()
    df_out["word_count"] = df_out["text"].str.split().str.len()
    df_out["review_year"] = df_out["date"].dt.year
    
    # Because train and dev are split by business_id (no overlap), 
    # counting within the df accurately reflects the business's frequency in that split.
    df_out["business_review_count"] = df_out.groupby("business_id")["review_id"].transform("count")
    return df_out

print("Building metadata features...")
df_train = build_features(df_train)
df_dev = build_features(df_dev)

feature_cols = ["word_count", "stars", "review_year", "business_review_count"]

X_train = df_train[feature_cols]
# Target is log(1 + useful)
y_train = np.log1p(df_train["useful"])

X_dev = df_dev[feature_cols]
y_dev = np.log1p(df_dev["useful"])

# Gradient Boosted Decision Tree
print(f"Training GBDT Metadata Baseline on {len(X_train)} train rows...")
model = HistGradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)

print(f"Evaluating on {len(X_dev)} dev rows...")
dev_preds = model.predict(X_dev)
metrics = evaluate(y_dev, dev_preds)

print("\nMetadata Baseline Results (log scale) on Yelp Dev:")
for name, val in metrics.items():
    print(f"{name}: {val:.4f}")