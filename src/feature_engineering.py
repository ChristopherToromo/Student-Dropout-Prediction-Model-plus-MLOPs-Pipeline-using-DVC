import numpy as np
import pandas as pd
import os

# ============================
# 1. Load Data
# ============================

train_data = pd.read_csv(os.path.join("data","processed","train_processed.csv"))
test_data = pd.read_csv(os.path.join("data","processed","test_processed.csv"))

X_train = train_data.drop(columns=['student_id','dropout'])
y_train = train_data['dropout']

X_test = test_data.drop(columns=['student_id','dropout'])
y_test = test_data['dropout']

# ============================
# 2. One-Hot Encode categorical columns
# ============================

cat_cols = X_train.select_dtypes(include=['object','string']).columns.tolist()
print("Categorical columns:", cat_cols)

combined = pd.concat([X_train, X_test], axis=0)

combined = pd.get_dummies(
    combined,
    columns=cat_cols,
    drop_first=True
)

# split back
X_train = combined.iloc[:len(X_train)].copy()
X_test = combined.iloc[len(X_train):].copy()

print("One-hot encoding complete.")

# ============================
# 3. Feature Engineering
# ============================

def safe_add_feature(df_train, df_test, feature_name, func):
    try:
        df_train[feature_name] = func(df_train)
        df_test[feature_name] = func(df_test)
    except KeyError:
        print(f"Skipping {feature_name}: missing column")

# GPA trend
safe_add_feature(
    X_train, X_test,
    "gpa_trend",
    lambda df: df['semester_gpa'] - df['gpa']
)

# Study efficiency
safe_add_feature(
    X_train, X_test,
    "study_efficiency",
    lambda df: df['study_hours_per_day']/(df['gpa'] + 1e-5)
)

# Job-study load
safe_add_feature(
    X_train, X_test,
    "job_study_load",
    lambda df: df['part_time_job'] * df['study_hours_per_day']
)

# Low income + high travel risk
income_median = X_train['family_income'].median() if 'family_income' in X_train.columns else 0
travel_median = X_train['travel_time_minutes'].median() if 'travel_time_minutes' in X_train.columns else 0

safe_add_feature(
    X_train, X_test,
    "low_income_high_travel",
    lambda df: (
        (df['family_income'] < income_median) &
        (df['travel_time_minutes'] > travel_median)
    ).astype(int)
)

# Attendance per semester
safe_add_feature(
    X_train, X_test,
    "attendance_per_semester",
    lambda df: df['attendance_rate']/(df['semester'] + 1e-5)
)

# Parental support
safe_add_feature(
    X_train, X_test,
    "parental_support",
    lambda df: df['parental_education'] * np.log1p(df['family_income'])
)

# Risk index
safe_add_feature(
    X_train, X_test,
    "risk_index",
    lambda df: (
        df['stress_index'] +
        df['assignment_delay_days'] +
        (100 - df['attendance_rate'])
    )/3
)

print("Feature engineering complete.")

# ============================
# 4. Multicollinearity Removal
# ============================

print("Checking correlation > 0.8")

corr_matrix = X_train.corr().abs()

upper = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

to_drop = [col for col in upper.columns if any(upper[col] > 0.8)]

print("Removing correlated features:", to_drop)

X_train = X_train.drop(columns=to_drop)
X_test = X_test.drop(columns=to_drop)

# ============================
# 5. Final Dataset
# ============================

print("\nFinal Features:", X_train.columns.tolist())
print("Total Features:", X_train.shape[1])

train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

# ============================
# 6. Save Feature Dataset
# ============================

data_path = os.path.join("data","features")

os.makedirs(data_path, exist_ok=True)

train_df.to_csv(os.path.join(data_path,"train_features.csv"), index=False)
test_df.to_csv(os.path.join(data_path,"test_features.csv"), index=False)

print("\nFeature datasets saved to data/features/")