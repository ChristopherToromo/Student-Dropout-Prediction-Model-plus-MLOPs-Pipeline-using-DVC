import numpy as np
import pandas as pd

import os

# Fetching data from raw folder
data_path = os.path.join("data", "raw")
train_data = pd.read_csv(os.path.join(data_path, "train.csv"))
test_data = pd.read_csv(os.path.join(data_path, "test.csv"))

def preprocessing(data: pd.DataFrame) -> pd.DataFrame:
    try:
        df = data.copy()

        # Missing values
        df['Parental_Education'] = df['Parental_Education'].fillna("Unknown")
        df['Family_Income'] = df['Family_Income'].fillna(df['Family_Income'].median())
        df['Stress_Index'] = df['Stress_Index'].fillna(df['Stress_Index'].mean())
        df['Study_Hours_per_Day'] = df['Study_Hours_per_Day'].fillna(
            df['Study_Hours_per_Day'].mean()
        )

        # Outlier removal (IQR)
        Q1 = df['Family_Income'].quantile(0.25)
        Q3 = df['Family_Income'].quantile(0.75)
        IQR = Q3 - Q1

        df = df[
            (df['Family_Income'] >= Q1 - 1.5 * IQR) &
            (df['Family_Income'] <= Q3 + 1.5 * IQR)
        ]

        # Lowercase columns
        df.columns = [col.lower() for col in df.columns]

        return df

    except KeyError as e:
        print(f"Column not found: {e}")
        raise
    except Exception as e:
        print(f"An error occurred: {e}")
    
    
train_processed_data = preprocessing(train_data)
test_processed_data = preprocessing(test_data)    


# store the data inside data/processed
data_path = os.path.join("data","processed")

os.makedirs(data_path)

train_processed_data.to_csv(os.path.join(data_path,"train_processed.csv"))
test_processed_data.to_csv(os.path.join(data_path,"test_processed.csv"))