# 🎓 Student Dropout Prediction — MLOps Pipeline with DVC

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://student-dropout-prediction-model-plus-mlops-pipeline-using-dvc.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![DVC](https://img.shields.io/badge/DVC-Pipeline-purple)
![License](https://img.shields.io/badge/License-CC0--1.0-green)

A complete end-to-end machine learning project that predicts student dropout risk using demographic, academic, and socioeconomic features — with a fully tracked MLOps pipeline built using DVC and deployed on Streamlit Cloud.

🔗 **Live App:** https://student-dropout-prediction-model-plus-mlops-pipeline-using-dvc.streamlit.app/

---

## Project Overview

Student dropout is a critical issue in higher education, with significant social and financial consequences. This project builds a **Logistic Regression classifier** to identify at-risk students early, enabling timely intervention.

The project demonstrates a production-style MLOps workflow including:
- Automated data ingestion from Kaggle
- Data preprocessing and feature engineering
- Reproducible pipeline tracking with DVC
- Model deployment via Streamlit

---

## Project Structure

```
├── app.py                        # Streamlit web application
├── dvc.yaml                      # DVC pipeline definition
├── dvc.lock                      # DVC pipeline lock file
├── model.pkl                     # Trained Logistic Regression model
├── requirements.txt              # Python dependencies
├── errors.log                    # Pipeline error logs
├── data/
│   ├── raw/                      # Raw train/test CSVs from Kaggle
│   ├── processed/                # Cleaned and preprocessed data
│   └── features/                 # Final feature-engineered datasets
├── src/
│   ├── data_ingestion.py         # Downloads data from Kaggle via kagglehub
│   ├── data_preprocessing.py     # Cleans data, handles missing values & outliers
│   ├── feature_engineering.py    # Feature engineering + OHE + corr removal
│   └── model_building.py         # Trains and saves the Logistic Regression model
└── Notebook/
    └── dropout_prediction.ipynb  # Exploratory data analysis notebook
```

---

## 🔬 Features Used

After preprocessing and feature engineering, the model trains on **26 features**:

| Category | Features |
|---|---|
| **Demographic** | Age, Gender, Family Income, Internet Access, Travel Time |
| **Academic** | Study Hours/Day, Attendance Rate, Assignment Delay, Semester, Department |
| **Socioeconomic** | Part-Time Job, Scholarship, Parental Education |
| **Performance** | GPA, Stress Index |
| **Engineered** | GPA Trend, Study Efficiency, Low Income + High Travel Risk |

---

## MLOps Pipeline (DVC)

The pipeline is defined in `dvc.yaml` and consists of 4 stages:

```
data_ingestion → data_preprocessing → feature_engineering → model_building
```

| Stage | Script | Output |
|---|---|---|
| `data_ingestion` | `src/data_ingestion.py` | `data/raw/train.csv`, `data/raw/test.csv` |
| `data_preprocessing` | `src/data_preprocessing.py` | `data/processed/train_processed.csv` |
| `feature_engineering` | `src/feature_engineering.py` | `data/features/train_features.csv` |
| `model_building` | `src/model_building.py` | `model.pkl` |

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/ChristopherToromo/Student-Dropout-Prediction-Model-plus-MLOPs-Pipeline-using-DVC.git
cd Student-Dropout-Prediction-Model-plus-MLOPs-Pipeline-using-DVC
```

### 2. Create and activate environment

```bash
conda create -n myenv python=3.12 -y
conda activate myenv
pip install -r requirements.txt
```

### 3. Set up Kaggle credentials

Make sure your Kaggle API key is configured. Download it from [kaggle.com/settings](https://www.kaggle.com/settings) and place it at:

```
~/.kaggle/kaggle.json       # Linux/Mac
C:\Users\<user>\.kaggle\kaggle.json   # Windows
```

### 4. Run the full DVC pipeline

```bash
dvc init        # only needed on first run
dvc repro       # runs all pipeline stages in order
```

To run individual stages:

```bash
python src/data_ingestion.py
python src/data_preprocessing.py
python src/feature_engineering.py
python src/model_building.py
```

### 5. Launch the Streamlit app

```bash
streamlit run app.py
```

---

## Streamlit App Features

The deployed app has three tabs:

- ** Model Performance** — Upload `train_features.csv` to view confusion matrix, ROC-AUC, probability distribution, and classification report
- **🔮 Predict a Student** — Fill in a student's details and get an instant dropout risk prediction with a probability gauge
- **🗂️ Dataset Explorer** — Browse the dataset, view target distribution, and explore feature statistics

---

## Model Details

| Property | Value |
|---|---|
| Algorithm | Logistic Regression |
| Regularization | L2 (C=0.1) |
| Class Weighting | Balanced |
| Solver | SAGA |
| Preprocessing | StandardScaler |
| Train/Test Split | 80/20 |

---

## Requirements

```
streamlit
pandas
numpy
seaborn
matplotlib
xgboost
scikit-learn
kagglehub
dvc
PyYAML
```

---

## DVC Commands Reference

```bash
dvc repro           # Reproduce the full pipeline
dvc repro --force   # Force re-run all stages
dvc dag             # Visualize pipeline DAG
dvc metrics show    # Show tracked metrics
dvc status          # Check pipeline status
```

---

## Dataset

- **Source:** [Student Dropout Prediction Dataset](https://www.kaggle.com/datasets/meharshanali/student-dropout-prediction-dataset) on Kaggle
- **Downloaded via:** `kagglehub`
- **Target column:** `dropout` (0 = Not Dropout, 1 = Dropout)

---

## 📝 License

This project is licensed under the [CC0-1.0 License](LICENSE).
