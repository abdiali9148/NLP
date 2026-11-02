import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit
from helper import evaluate, set_seed

set_seed(42)

print("Loading dataset...")
df = pd.read_parquet("dataset/yelp_train.parquet")

print("Building metadata features...")
df["word_count"] = df["text"].str.split().str.len()
df["review_year"] = df["date"].dt.year
df["business_review_count"] = df.groupby("business_id")["review_id"].transform("count")

feature_cols = ["word_count", "stars", "review_year", "business_review_count"]
X = df[feature_cols]

# Line 227: Target is log(1 + useful)
y = np.log1p(df["useful"])

# Line 218: Split by business_id so no restaurant is in both train and val
splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
train_idx, val_idx = next(splitter.split(X, y, groups=df["business_id"]))

X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

# Line 305: Gradient Boosted Decision Tree
print("Training GBDT Metadata Baseline...")
model = HistGradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)

print("Evaluating on validation split...")
val_preds = model.predict(X_val)
metrics = evaluate(y_val, val_preds)

print("\nMetadata Baseline Results (log scale):")
for name, val in metrics.items():
    print(f"{name}: {val:.4f}")