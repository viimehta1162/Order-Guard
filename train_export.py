# Run this in Google Colab (after uploading Amazon_Sale_Report.csv)
import json, joblib, sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)

df = pd.read_csv("Amazon_Sale_Report.csv", low_memory=False)
df["Cancelled"] = (df["Status"] == "Cancelled").astype(int)

# IMPORTANT: 7,800 orders have no Amount and 97% of them are cancelled.
# Filling them with the median would let the model "cheat" (Amount == median
# becomes a hidden sign of cancellation), so we train only on orders that
# have a real Amount.
df = df.dropna(subset=["Amount"]).copy()

df["ship-state"] = df["ship-state"].fillna("Unknown")
df["B2B"] = df["B2B"].astype(int)

cat_cols = ["Fulfilment", "Sales Channel", "ship-service-level",
            "Category", "Size", "ship-state"]
num_cols = ["B2B", "Amount"]

X = df[cat_cols + num_cols]
y = df["Cancelled"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

model = Pipeline([
    ("prep", ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
        remainder="passthrough")),
    ("clf", RandomForestClassifier(
        n_estimators=100, min_samples_leaf=5, class_weight="balanced",
        n_jobs=-1, random_state=42))
])
model.fit(X_train, y_train)

pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]
metrics = {
    "Accuracy":  round(accuracy_score(y_test, pred), 3),
    "Precision": round(precision_score(y_test, pred), 3),
    "Recall":    round(recall_score(y_test, pred), 3),
    "F1 Score":  round(f1_score(y_test, pred), 3),
    "ROC-AUC":   round(roc_auc_score(y_test, prob), 3),
}
print(metrics)

# Files needed by the app
joblib.dump(model, "cancel_model.pkl", compress=3)

options = {c: sorted(df[c].dropna().astype(str).unique().tolist()) for c in cat_cols}
options["amount_min"] = float(df["Amount"].min())
options["amount_max"] = float(df["Amount"].max())
options["amount_median"] = float(df["Amount"].median())
options["cancel_rate"] = round(float(y.mean()), 3)
json.dump(options, open("options.json", "w"))
json.dump(metrics, open("metrics.json", "w"))

print("\nPUT THIS LINE IN requirements.txt ->  scikit-learn==" + sklearn.__version__)

from google.colab import files
for f in ["cancel_model.pkl", "options.json", "metrics.json"]:
    files.download(f)
