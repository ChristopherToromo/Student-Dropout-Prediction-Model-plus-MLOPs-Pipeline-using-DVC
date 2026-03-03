import pandas as pd
import numpy as np
import xgboost as xgb

import pickle

# Define and train the XGBoost model
train_data = pd.read_csv("data/features/train_features.csv")

X_train = train_data.drop(columns=['dropout'])
y_train = train_data['dropout']

# Define and train the XGBoost model
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)

# Save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)