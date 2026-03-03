import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


import os

train_data = pd.read_csv(os.path.join("data","processed","train_processed.csv"))
test_data = pd.read_csv(os.path.join("data","processed","test_processed.csv"))

X_train = train_data.drop(columns=['student_id', 'dropout'])
y_train = train_data['dropout']

X_test = test_data.drop(columns=['student_id', 'dropout'])
y_test = test_data['dropout']

#  Encoding all categorical variables using LabelEncoder
def encoder(df):
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object','string']).columns:
        df[col] = le.fit_transform(df[col])
    return df

X_train = encoder(X_train)
X_test = encoder(X_test)

train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

# store the data inside data/features
data_path = os.path.join("data","features")

os.makedirs(data_path)

train_df.to_csv(os.path.join(data_path,"train_features.csv"))
test_df.to_csv(os.path.join(data_path,"test_features.csv"))




