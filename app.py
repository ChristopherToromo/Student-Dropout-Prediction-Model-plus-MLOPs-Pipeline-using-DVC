import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Student Dropout Predictor", page_icon="🎓", layout="wide")

st.markdown("""
<style>
    .predict-box-dropout {
        background: #fff0f0; border-left: 5px solid #e74c3c; border-radius: 10px;
        padding: 1.2rem 1.6rem; font-size: 1.3rem; font-weight: 600; color: #c0392b;
    }
    .predict-box-ok {
        background: #f0fff4; border-left: 5px solid #27ae60; border-radius: 10px;
        padding: 1.2rem 1.6rem; font-size: 1.3rem; font-weight: 600; color: #1e8449;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Exact feature columns from train_features.csv
# (after OHE + feature engineering + corr removal)
# ─────────────────────────────────────────────
MODEL_FEATURES = [
    'age', 'family_income', 'study_hours_per_day', 'attendance_rate',
    'assignment_delay_days', 'travel_time_minutes', 'stress_index', 'gpa',
    'gender_Male', 'internet_access_Yes', 'part_time_job_Yes', 'scholarship_Yes',
    'semester_Year 2', 'semester_Year 3', 'semester_Year 4',
    'department_Business', 'department_CS', 'department_Engineering', 'department_Science',
    'parental_education_High School', 'parental_education_Master',
    'parental_education_PhD', 'parental_education_Unknown',
    'gpa_trend', 'study_efficiency', 'low_income_high_travel',
]

# ─────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists("model.pkl"):
        return None
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    path = "data/features/train_features.csv"
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

train_df = load_data()

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown("""
<h1 style="font-size:3.8rem; font-weight:800; color:#ffffff; margin-bottom:0.1rem; line-height:1.1;">
    🎓 Student Dropout Prediction
</h1>
<p style="font-size:1.1rem; color:#aaaaaa; margin-top:0; margin-bottom:2rem;">
    Identify at-risk students early using Logistic Regression
</p>
""", unsafe_allow_html=True)

if model is None:
    st.error("**`model.pkl` not found.** Place it in the same directory as `app.py` and restart.")
    st.info("Generate it by running:\n```bash\npython train.py\n```")
    st.stop()


tab1, tab2, tab3 = st.tabs(["📊 Model Performance", "🔮 Predict a Student", "🗂️ Dataset Explorer"])

# ═══════════════════════════════════════════════
# TAB 1 — Model Performance
# ═══════════════════════════════════════════════
with tab1:
    if train_df is None:
        st.warning("`data/features/train_features.csv` not found.")
    else:
        X = train_df[MODEL_FEATURES]
        y = train_df["dropout"]

        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy",     f"{accuracy_score(y_test, y_pred):.1%}")
        c2.metric("ROC-AUC",      f"{roc_auc_score(y_test, y_prob):.3f}")
        c3.metric("Test Samples", f"{len(y_test):,}")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("### Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=["Not Dropout", "Dropout"],
                        yticklabels=["Not Dropout", "Dropout"])
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            fig.tight_layout(); st.pyplot(fig)

        with col_b:
            st.markdown("### Predicted Probability Distribution")
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            ax2.hist(y_prob[y_test == 0], bins=30, alpha=0.65, color="steelblue", label="Not Dropout")
            ax2.hist(y_prob[y_test == 1], bins=30, alpha=0.65, color="salmon",    label="Dropout")
            ax2.axvline(0.5, color="gray", linestyle="--", linewidth=1, label="threshold = 0.5")
            ax2.set_xlabel("Predicted Probability"); ax2.set_ylabel("Count")
            ax2.legend(); fig2.tight_layout(); st.pyplot(fig2)

        st.markdown("### Classification Report")
        st.code(classification_report(y_test, y_pred, target_names=["Not Dropout", "Dropout"]))

# ═══════════════════════════════════════════════
# TAB 2 — Predict a Student
# ═══════════════════════════════════════════════
with tab2:
    st.markdown("### Enter student information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**👤 Personal**")
        age           = st.slider("Age", 15, 30, 7) # Max age 30
        gender        = st.selectbox("Gender", ["Female", "Male"])
        family_income = st.number_input("Family Income", min_value=0.0, value=35000.0, step=1000.0)
        internet      = st.selectbox("Internet Access", ["No", "Yes"])
        part_time_job = st.selectbox("Part-Time Job", ["No", "Yes"])
        travel_time   = st.slider("Travel Time (minutes)", 0, 180, 30)

    with col2:
        st.markdown("**📚 Academic**")
        semester      = st.selectbox("Semester", ["Year 1", "Year 2", "Year 3", "Year 4"])
        department    = st.selectbox("Department", ["Arts", "Business", "CS", "Engineering", "Science"])
        study_hours   = st.slider("Study Hours per Day", 0.0, 16.0, 4.0, 0.5)
        attendance    = st.slider("Attendance Rate (%)", 0.0, 100.0, 80.0, 0.5)
        assign_delay  = st.slider("Assignment Delay (days)", 0, 30, 2)
        scholarship   = st.selectbox("Scholarship", ["No", "Yes"])

    with col3:
        st.markdown("**📊 Performance & Wellbeing**")
        gpa           = st.slider("GPA",          0.0, 4.0, 2.8, 0.01)
        sem_gpa       = st.slider("Semester GPA", 0.0, 4.0, 2.8, 0.01)
        stress_index  = st.slider("Stress Index", 0.0, 10.0, 5.0, 0.1)
        parental_edu  = st.selectbox("Parental Education",
                                     ["Below High School", "High School", "Master", "PhD", "Unknown"])

    if st.button("🔮 Predict Dropout Risk", use_container_width=True):

        # Compute engineered features
        income_median = train_df["family_income"].median() if train_df is not None else 35000
        travel_median = train_df["travel_time_minutes"].median() if train_df is not None else 30
        gpa_trend         = sem_gpa - gpa
        study_efficiency  = study_hours / (gpa + 1e-5)
        low_inc_high_trav = int(family_income < income_median and travel_time > travel_median)

        # Build input row matching MODEL_FEATURES exactly
        row = {
            'age':                              age,
            'family_income':                    family_income,
            'study_hours_per_day':              study_hours,
            'attendance_rate':                  attendance,
            'assignment_delay_days':            assign_delay,
            'travel_time_minutes':              travel_time,
            'stress_index':                     stress_index,
            'gpa':                              gpa,
            # OHE binary features
            'gender_Male':                      int(gender == "Male"),
            'internet_access_Yes':              int(internet == "Yes"),
            'part_time_job_Yes':                int(part_time_job == "Yes"),
            'scholarship_Yes':                  int(scholarship == "Yes"),
            # OHE semester (Year 1 is the dropped baseline)
            'semester_Year 2':                  int(semester == "Year 2"),
            'semester_Year 3':                  int(semester == "Year 3"),
            'semester_Year 4':                  int(semester == "Year 4"),
            # OHE department (Arts is the dropped baseline)
            'department_Business':              int(department == "Business"),
            'department_CS':                    int(department == "CS"),
            'department_Engineering':           int(department == "Engineering"),
            'department_Science':               int(department == "Science"),
            # OHE parental education (Below High School is the dropped baseline)
            'parental_education_High School':   int(parental_edu == "High School"),
            'parental_education_Master':        int(parental_edu == "Master"),
            'parental_education_PhD':           int(parental_edu == "PhD"),
            'parental_education_Unknown':       int(parental_edu == "Unknown"),
            # Engineered
            'gpa_trend':                        gpa_trend,
            'study_efficiency':                 study_efficiency,
            'low_income_high_travel':           low_inc_high_trav,
        }

        input_df = pd.DataFrame([row])[MODEL_FEATURES]

        try:
            prob  = model.predict_proba(input_df)[0][1]
            label = model.predict(input_df)[0]

            st.markdown("---")
            if label == 1:
                st.markdown(
                    f'<div class="predict-box-dropout">⚠️ High Dropout Risk — {prob:.1%} probability</div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="predict-box-ok">✅ Low Dropout Risk — {prob:.1%} probability</div>',
                    unsafe_allow_html=True)

            fig3, ax3 = plt.subplots(figsize=(6, 1.2))
            color = "salmon" if prob > 0.5 else "mediumseagreen"
            ax3.barh([""], [prob],     color=color,  height=0.5)
            ax3.barh([""], [1 - prob], left=[prob],  color="#444", height=0.5)
            ax3.axvline(0.5, color="white", linestyle="--", linewidth=1)
            ax3.set_xlim(0, 1)
            ax3.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax3.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
            ax3.set_title("Dropout Probability", color="white")
            ax3.tick_params(colors="white")
            fig3.patch.set_alpha(0)
            ax3.set_facecolor("#1e1e1e")
            fig3.tight_layout(); st.pyplot(fig3)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ═══════════════════════════════════════════════
# TAB 3 — Dataset Explorer
# ═══════════════════════════════════════════════
with tab3:
    df_show = train_df
    if df_show is None:
        st.warning("`data/features/train_features.csv` not found.")
        up = st.file_uploader("Upload train_features.csv", type=["csv"])
        if up:
            df_show = pd.read_csv(up)

    if df_show is not None:
        st.markdown(f"### Dataset — {len(df_show):,} rows × {len(df_show.columns)} columns")
        st.dataframe(df_show.head(100), use_container_width=True)

        if "dropout" in df_show.columns:
            st.markdown("### Target Distribution")
            counts = df_show["dropout"].value_counts().rename({0: "Not Dropout", 1: "Dropout"})
            fig4, ax4 = plt.subplots(figsize=(4, 3))
            counts.plot.bar(ax=ax4, color=["steelblue", "salmon"], edgecolor="white")
            ax4.set_xticklabels(counts.index, rotation=0)
            ax4.set_ylabel("Count")
            fig4.tight_layout(); st.pyplot(fig4)

            st.markdown("### GPA vs Dropout")
            fig5, ax5 = plt.subplots(figsize=(6, 3))
            for lbl, color in [(0, "steelblue"), (1, "salmon")]:
                ax5.hist(df_show[df_show["dropout"] == lbl]["gpa"], bins=30,
                         alpha=0.65, color=color, label="Not Dropout" if lbl == 0 else "Dropout")
            ax5.set_xlabel("GPA"); ax5.set_ylabel("Count"); ax5.legend()
            fig5.tight_layout(); st.pyplot(fig5)

            st.markdown("### Attendance Rate vs Dropout")
            fig6, ax6 = plt.subplots(figsize=(6, 3))
            for lbl, color in [(0, "steelblue"), (1, "salmon")]:
                ax6.hist(df_show[df_show["dropout"] == lbl]["attendance_rate"], bins=30,
                         alpha=0.65, color=color, label="Not Dropout" if lbl == 0 else "Dropout")
            ax6.set_xlabel("Attendance Rate (%)"); ax6.set_ylabel("Count"); ax6.legend()
            fig6.tight_layout(); st.pyplot(fig6)

        st.markdown("### Feature Statistics")
        num_cols = df_show.select_dtypes(include="number").columns.tolist()
        st.dataframe(df_show[num_cols].describe().T.round(3), use_container_width=True)