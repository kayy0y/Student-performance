import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="Student Performance Predictor",
                   page_icon="🎓", layout="wide")

st.title("🎓 Student Performance Prediction System")
st.markdown("### Machine Learning-Based Academic Performance Predictor")
st.markdown("---")

# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------
if 'df' not in st.session_state:
    st.session_state.df = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'label_encoders' not in st.session_state:
    st.session_state.label_encoders = {}
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None


# ---------------------------------------------------------
# UTILITY: DOWNLOAD EXCEL
# ---------------------------------------------------------
def download_excel(df, filename="data.xlsx"):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name="Sheet1")
    st.download_button(
        label="📥 Download Excel File",
        data=buffer,
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


# ---------------------------------------------------------
# SAMPLE DATA GENERATOR
# ---------------------------------------------------------
@st.cache_data
def generate_sample_data(n_samples=500):
    np.random.seed(42)
    data = {
        'study_hours': np.random.randint(1, 15, n_samples),
        'attendance': np.random.randint(50, 100, n_samples),
        'previous_grade': np.random.randint(40, 100, n_samples),
        'absences': np.random.randint(0, 30, n_samples),
        'parent_education': np.random.choice(['Primary', 'High School', 'Bachelor', 'Master'], n_samples),
        'extra_activities': np.random.choice(['Yes', 'No'], n_samples),
        'internet_access': np.random.choice(['Yes', 'No'], n_samples),
        'family_support': np.random.choice(['Yes', 'No'], n_samples),
    }

    df = pd.DataFrame(data)

    df['final_grade'] = (
        df['study_hours'] * 2.5 +
        df['attendance'] * 0.3 +
        df['previous_grade'] * 0.4 +
        df['absences'] * -0.5 +
        df['parent_education'].map({'Primary': 2, 'High School': 5, 'Bachelor': 8, 'Master': 10}) +
        (df['extra_activities'] == 'Yes') * 3 +
        (df['internet_access'] == 'Yes') * 2 +
        (df['family_support'] == 'Yes') * 4 +
        np.random.normal(0, 5, n_samples)
    )

    df['final_grade'] = df['final_grade'].clip(0, 100)
    return df


# ---------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------
def preprocess_data(df):
    df_encoded = df.copy()
    label_encoders = {}
    categorical_cols = ['parent_education', 'extra_activities', 'internet_access', 'family_support']

    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    df_encoded['study_attendance_ratio'] = df_encoded['study_hours'] / (df_encoded['attendance'] + 1)
    df_encoded['effective_study_time'] = df_encoded['study_hours'] * (df_encoded['attendance'] / 100)

    return df_encoded, label_encoders


# ---------------------------------------------------------
# TRAINING FUNCTION
# ---------------------------------------------------------
def train_model(df_encoded, model_type='Random Forest'):
    X = df_encoded.drop('final_grade', axis=1)
    y = df_encoded['final_grade']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
    }

    model = models[model_type]
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE': mean_absolute_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred),
    }

    return model, scaler, metrics, X_test, y_test, y_pred, X.columns


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.header("📊 Navigation")
page = st.sidebar.radio("Go to", [
    "Home", "Upload Data", "Data Overview", "Train Model",
    "Make Prediction", "Model Comparison"
])

# ---------------------------------------------------------
# HOME
# ---------------------------------------------------------
if page == "Home":
    st.header("Welcome to Student Performance Predictor! 🧑‍💻")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📚 About This App")
        st.write("""
        Predict student academic performance using machine learning.
        Upload your dataset or generate a sample one.
        """)
    with col2:
        st.subheader("🎯 Features")
        st.write("""
        - Upload Excel/CSV
        - Train multiple ML models  
        - Predict individual student results  
        - Visualize performance  
        - Export results to Excel  
        """)

    if st.button("Generate Sample Dataset"):
        st.session_state.df = generate_sample_data()
        st.success("Sample data generated successfully!")
        st.balloons()


# ---------------------------------------------------------
# UPLOAD DATA
# ---------------------------------------------------------
elif page == "Upload Data":
    st.header("📂 Upload Dataset")

    uploaded = st.file_uploader("Upload Excel or CSV file",
                                type=["xlsx", "xls", "csv"])

    if uploaded:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)

        st.session_state.df = df
        st.success("File uploaded successfully!")

        st.subheader("Preview")
        st.dataframe(df.head())

        download_excel(df, "uploaded_data.xlsx")


# ---------------------------------------------------------
# DATA OVERVIEW
# ---------------------------------------------------------
elif page == "Data Overview":
    st.header("📊 Data Overview")

    if st.session_state.df is None:
        st.warning("No data available.")
    else:
        df = st.session_state.df
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        st.subheader("Download Raw Data")
        download_excel(df, "raw_dataset.xlsx")


# TRAIN MODEL

elif page == "Train Model":
    st.header("🤖 Train Model")

    if st.session_state.df is None:
        st.warning("Upload or generate data first.")
    else:
        df = st.session_state.df

        model_type = st.selectbox(
            "Select ML Model",
            ["Random Forest", "Gradient Boosting", "Decision Tree", "Linear Regression"]
        )

        if st.button("Train"):
            df_encoded, encoders = preprocess_data(df)
            st.session_state.label_encoders = encoders

            model, scaler, metrics, X_test, y_test, y_pred, feature_names = train_model(df_encoded, model_type)

            st.session_state.model = model
            st.session_state.scaler = scaler
            st.session_state.feature_names = feature_names

            st.success("Model trained successfully!")

            st.metric("R²", f"{metrics['R2']:.4f}")
            st.metric("RMSE", f"{metrics['RMSE']:.2f}")
            st.metric("MAE", f"{metrics['MAE']:.2f}")

            pred_df = pd.DataFrame({
                "Actual": y_test,
                "Predicted": y_pred
            })

            st.subheader("Download Predictions")
            download_excel(pred_df, "predictions.xlsx")


# MAKE PREDICTION

elif page == "Make Prediction":
    st.header("🔮 Predict Student Performance")

    if st.session_state.model is None:
        st.warning("Train a model first.")
    else:
        study_hours = st.slider("Study Hours", 1, 15, 7)
        attendance = st.slider("Attendance %", 50, 100, 85)
        previous_grade = st.slider("Previous Grade", 40, 100, 75)
        absences = st.slider("Absences", 0, 30, 5)

        parent_education = st.selectbox("Parent Education", ["Primary", "High School", "Bachelor", "Master"])
        extra_activities = st.radio("Extra Activities", ["Yes", "No"])
        internet_access = st.radio("Internet Access", ["Yes", "No"])
        family_support = st.radio("Family Support", ["Yes", "No"])

        if st.button("Predict"):
            input_data = pd.DataFrame({
                'study_hours': [study_hours],
                'attendance': [attendance],
                'previous_grade': [previous_grade],
                'absences': [absences],
                'parent_education': [parent_education],
                'extra_activities': [extra_activities],
                'internet_access': [internet_access],
                'family_support': [family_support]
            })

            for col in st.session_state.label_encoders:
                input_data[col] = st.session_state.label_encoders[col].transform(input_data[col])

            input_data["study_attendance_ratio"] = study_hours / (attendance + 1)
            input_data["effective_study_time"] = study_hours * (attendance / 100)

            scaled = st.session_state.scaler.transform(input_data)
            prediction = st.session_state.model.predict(scaled)[0]

            st.success(f"Predicted Final Grade: {prediction:.2f}")


# MODEL COMPARISON

elif page == "Model Comparison":
    st.header("📊 Model Comparison")

    if st.session_state.df is None:
        st.warning("Upload or generate data first.")
    else:
        st.info("This section trains all models and compares performance.")
