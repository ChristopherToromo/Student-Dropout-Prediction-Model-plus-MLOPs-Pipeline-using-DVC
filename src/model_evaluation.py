import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score, classification_report, f1_score

import pickle
import json

# Load the XGBoost model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
    
# Load the test data
test_data = pd.read_csv("data/features/test_features.csv")

X_test = test_data.drop(columns=['dropout'])
y_test = test_data['dropout']    

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("---------------------------------------------------")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

metrics_dict = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1
}

with open("metrics.json", "w") as f:
    json.dump(metrics_dict, f, indent=4)