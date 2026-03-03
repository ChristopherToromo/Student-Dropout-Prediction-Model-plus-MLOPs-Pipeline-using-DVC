import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import logging
import os
import kagglehub


# ---------------------------
# Logging configuration
# ---------------------------

logger = logging.getLogger("data_ingestion")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

file_handler = logging.FileHandler("errors.log")
file_handler.setLevel("ERROR")

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# ---------------------------
# Load the data
# ---------------------------

def load_data(file_path: str) -> pd.DataFrame:
    try:
        data = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully from {file_path}")
        return data
    except pd.errors.ParserError as e:
        logger.error(f"Failed to parse the data from {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"An error occurred while loading the data: {e}")
        raise



# ---------------------------
# Save data
# ---------------------------

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str):
    try:
        data_path = os.path.join(data_path, "raw")
        os.makedirs(data_path, exist_ok=True)

        train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)

        logger.info("Train and test data saved successfully.")
    except Exception as e:
        logger.error(f"An error occurred while saving the data: {e}")
        raise


# ---------------------------
# Main pipeline
# ---------------------------

def main():
    try:
        # Download dataset via kagglehub
        dataset_path = kagglehub.dataset_download(
            "meharshanali/student-dropout-prediction-dataset"
        )

        logger.info(f"Dataset downloaded to: {dataset_path}")

        # Detect CSV file automatically
        files = os.listdir(dataset_path)
        csv_files = [f for f in files if f.endswith(".csv")]

        if not csv_files:
            raise FileNotFoundError("No CSV file found in downloaded dataset.")

        csv_path = os.path.join(dataset_path, csv_files[0])

        # Load
        df = load_data(csv_path)

        # Train-test split
        train_data, test_data = train_test_split(
            df, test_size=0.2, random_state=42
        )

        # Save
        save_data(train_data, test_data, "data")

        logger.info("Data ingestion pipeline completed successfully.")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()