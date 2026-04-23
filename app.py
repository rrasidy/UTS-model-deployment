import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Placement Prediction App", layout="wide")

# LOAD MODELS
@st.cache_resource
def load_models():
    clf_model = None
    reg_model = None
    pre_clf = None
    pre_reg = None

    # Try pipeline artifacts first
    if os.path.exists("artifact/best_classification_model.pkl"):
        clf_model = joblib.load("artifact/best_classification_model.pkl")
    if os.path.exists("artifact/best_regression_model.pkl"):
        reg_model = joblib.load("artifact/best_regression_model.pkl")
    if os.path.exists("artifact/preprocessor_clf.pkl"):
        pre_clf = joblib.load("artifact/preprocessor_clf.pkl")
    if os.path.exists("artifact/preprocessor_reg.pkl"):
        pre_reg = joblib.load("artifact/preprocessor_reg.pkl")

    # fallback simple models
    if clf_model is None and os.path.exists("placement_model.pkl"):
        clf_model = joblib.load("placement_model.pkl")
    if reg_model is None and os.path.exists("salary_model.pkl"):
        reg_model = joblib.load("salary_model.pkl")

    return clf_model, reg_model, pre_clf, pre_reg


clf_model, reg_model, pre_clf, pre_reg = load_models()

# LOAD DATA (for UI)
def load_data():
    if os.path.exists("B.csv"):
        return pd.read_csv("B.csv")
    elif os.path.exists("ingested/B.csv"):
        return pd.read_csv("ingested/B.csv")
    return None


df = load_data()

# SIDEBAR
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Placement Prediction", "Salary Prediction", "About"]
)

st.sidebar.markdown("---")
st.sidebar.info("Simple ML Web App using Streamlit")

# HELPER: BUILD INPUT FORM
def build_input_form():
    input_data = {}

    if df is not None:
        cols = [c for c in df.columns if c.lower() not in ["placement_status", "salary"]]

        for col in cols:
            if df[col].dtype == "object":
                input_data[col] = st.selectbox(col, df[col].dropna().unique())
            else:
                input_data[col] = st.number_input(col, float(df[col].mean()))
    else:
        st.warning("Dataset not found. Using manual input.")
        input_data["internships_completed"] = st.number_input("internships_completed", 0)
        input_data["projects_completed"] = st.number_input("projects_completed", 0)
        input_data["cgpa"] = st.number_input("cgpa", 0.0)

    return pd.DataFrame([input_data])


if page == "Placement Prediction":
    st.title("🎓 Placement Prediction")

    st.write("Fill the form below to predict placement status.")

    X_input = build_input_form()

    # Auto feature engineering
    if "internships_completed" in X_input.columns and "projects_completed" in X_input.columns:
        X_input["experience_score"] = (
            X_input["internships_completed"] + X_input["projects_completed"]
        )

    if st.button("Predict Placement"):
        if clf_model is None:
            st.error("Classification model not found!")
        else:
            X_proc = pre_clf.transform(X_input) if pre_clf else X_input
            pred = clf_model.predict(X_proc)[0]

            st.success(f"Prediction: {pred}")

            if hasattr(clf_model, "predict_proba"):
                prob = clf_model.predict_proba(X_proc)[0]
                st.write("Probability:", prob)


elif page == "Salary Prediction":
    st.title("💰 Salary Prediction")

    st.write("Predict expected salary package.")

    X_input = build_input_form()

    if "internships_completed" in X_input.columns and "projects_completed" in X_input.columns:
        X_input["experience_score"] = (
            X_input["internships_completed"] + X_input["projects_completed"]
        )

    if st.button("Predict Salary"):
        if reg_model is None:
            st.error("Regression model not found!")
        else:
            X_proc = pre_reg.transform(X_input) if pre_reg else X_input
            pred = reg_model.predict(X_proc)[0]

            st.success(f"Estimated Salary: {pred:,.2f}")


else:
    st.title("📌 About")

    st.markdown("""
    This is a simple Machine Learning web application built using **Streamlit**.

    ### Features:
    - Placement prediction (classification)
    - Salary prediction (regression)
    - Uses trained `.pkl` models
    - Clean UI with sidebar navigation

    ### Tech Stack:
    - Python
    - Scikit-learn
    - Streamlit

    ### Author:
    Your Name Here
    """)

    if df is not None:
        st.subheader("Dataset Preview")
        st.dataframe(df.head())