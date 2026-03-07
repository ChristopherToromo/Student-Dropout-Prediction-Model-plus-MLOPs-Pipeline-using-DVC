import pandas as pd
import pickle

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# ============================
# Load Training Data
# ============================

train_data = pd.read_csv("data/features/train_features.csv")

X_train = train_data.drop(columns=['dropout'])
y_train = train_data['dropout']

# ============================
# Optimized Logistic Model
# ============================

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        C=0.1,
        l1_ratio=0,
        class_weight="balanced",
        solver="saga",
        max_iter=5000,
        random_state=42
    ))
])

print("Training optimized Logistic Regression...")

pipeline.fit(X_train, y_train)

# ============================
# Save Model
# ============================

with open("model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("Model saved as model.pkl")