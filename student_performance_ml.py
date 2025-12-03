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

# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🧑‍💻",
    layout="wide"
)

# Title and description
st.title("🎓 Student Performance Prediction System")
st.markdown("### Machine Learning-Based Academic Performance Predictor")
st.markdown("---")

# Sidebar for navigation
st.sidebar.header("📊 Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Overview", "Train Model", "Make Prediction", "Model Comparison"])

# Initialize session state
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

# Function to generate sample data
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
    
    # Generate target variable
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

# Function to preprocess data
def preprocess_data(df):
    df_encoded = df.copy()
    label_encoders = {}
    categorical_cols = ['parent_education', 'extra_activities', 'internet_access', 'family_support']
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Feature engineering
    df_encoded['study_attendance_ratio'] = df_encoded['study_hours'] / (df_encoded['attendance'] + 1)
    df_encoded['effective_study_time'] = df_encoded['study_hours'] * (df_encoded['attendance'] / 100)
    
    return df_encoded, label_encoders

# Function to train model
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
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    
    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'CV_R2_mean': cv_scores.mean(),
        'CV_R2_std': cv_scores.std()
    }
    
    return model, scaler, metrics, X_test, y_test, y_pred, X.columns

# ==================== HOME PAGE ====================
if page == "Home":
    st.header("Welcome to Student Performance Predictor!🧑‍💻")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📚 About This App")
        st.write("""
        This application uses Machine Learning to predict student academic performance 
        based on various factors including:
        - Study hours per week
        - Attendance percentage
        - Previous grades
        - Number of absences
        - Parental education level
        - Extra-curricular activities
        - Internet access
        - Family support
        """)
    
    with col2:
        st.subheader("🎯 Features")
        st.write("""
        - **Data Overview**: Explore and visualize the dataset
        - **Train Model**: Train ML models and compare performance
        - **Make Prediction**: Predict grades for individual students
        - **Model Comparison**: Compare different ML algorithms
        """)
    
    st.info("👈 Use the sidebar to navigate through different sections!")
    
    # Generate or load data
    if st.button("🔄 Generate Sample Dataset", type="primary"):
        st.session_state.df = generate_sample_data(500)
        st.success("✅ Sample dataset generated successfully!")
        st.balloons()

# ==================== DATA OVERVIEW PAGE ====================
elif page == "Data Overview":
    st.header("📊 Data Overview")
    
    if st.session_state.df is None:
        st.warning("⚠️ No data loaded. Please generate sample data from the Home page.")
    else:
        df = st.session_state.df
        
        # Dataset info
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Students", len(df))
        col2.metric("Average Grade", f"{df['final_grade'].mean():.2f}")
        col3.metric("Highest Grade", f"{df['final_grade'].max():.2f}")
        col4.metric("Lowest Grade", f"{df['final_grade'].min():.2f}")
        
        st.subheader("📋 Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.subheader("📈 Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
        
        # Visualizations
        st.subheader("📊 Data Visualizations")
        
        tab1, tab2, tab3 = st.tabs(["Grade Distribution", "Feature Correlations", "Categorical Analysis"])
        
        with tab1:
            fig = px.histogram(df, x='final_grade', nbins=30, 
                             title='Distribution of Final Grades',
                             labels={'final_grade': 'Final Grade'},
                             color_discrete_sequence=['#1f77b4'])
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            corr_matrix = df[numeric_cols].corr()
            
            fig = px.imshow(corr_matrix, 
                          text_auto=True, 
                          aspect="auto",
                          title='Feature Correlation Heatmap',
                          color_continuous_scale='RdBu_r')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.box(df, x='parent_education', y='final_grade',
                           title='Grades by Parent Education Level',
                           color='parent_education')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.box(df, x='extra_activities', y='final_grade',
                           title='Grades by Extra Activities',
                           color='extra_activities')
                st.plotly_chart(fig, use_container_width=True)

# ==================== TRAIN MODEL PAGE ====================
elif page == "Train Model":
    st.header("🤖 Train Machine Learning Model")
    
    if st.session_state.df is None:
        st.warning("⚠️ No data loaded. Please generate sample data from the Home page.")
    else:
        df = st.session_state.df
        
        st.subheader("⚙️ Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_type = st.selectbox(
                "Select ML Algorithm",
                ["Random Forest", "Gradient Boosting", "Decision Tree", "Linear Regression"]
            )
        
        with col2:
            st.write("")
            st.write("")
            train_button = st.button("🚀 Train Model", type="primary")
        
        if train_button:
            with st.spinner("Training model... Please wait."):
                # Preprocess data
                df_encoded, label_encoders = preprocess_data(df)
                st.session_state.label_encoders = label_encoders
                
                # Train model
                model, scaler, metrics, X_test, y_test, y_pred, feature_names = train_model(df_encoded, model_type)
                
                # Save to session state
                st.session_state.model = model
                st.session_state.scaler = scaler
                st.session_state.feature_names = feature_names
                
            st.success(f"✅ {model_type} model trained successfully!")
            
            # Display metrics
            st.subheader("📊 Model Performance Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("R² Score", f"{metrics['R2']:.4f}")
            col2.metric("RMSE", f"{metrics['RMSE']:.2f}")
            col3.metric("MAE", f"{metrics['MAE']:.2f}")
            col4.metric("CV R² Score", f"{metrics['CV_R2_mean']:.4f}")
            
            # Prediction vs Actual plot
            st.subheader("📈 Predictions vs Actual Values")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers',
                                   name='Predictions',
                                   marker=dict(color='blue', size=8, opacity=0.6)))
            fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], 
                                   y=[y_test.min(), y_test.max()],
                                   mode='lines', name='Perfect Prediction',
                                   line=dict(color='red', dash='dash')))
            fig.update_layout(xaxis_title='Actual Grades', 
                            yaxis_title='Predicted Grades',
                            title='Model Predictions vs Actual Grades')
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance (if available)
            if hasattr(model, 'feature_importances_'):
                st.subheader("🎯 Feature Importance")
                
                feature_importance = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig = px.bar(feature_importance, x='Importance', y='Feature',
                           orientation='h',
                           title='Feature Importance Ranking',
                           color='Importance',
                           color_continuous_scale='viridis')
                st.plotly_chart(fig, use_container_width=True)

# ==================== MAKE PREDICTION PAGE ====================
elif page == "Make Prediction":
    st.header("🎯 Make Individual Prediction")
    
    if st.session_state.model is None:
        st.warning("⚠️ No trained model found. Please train a model first.")
    else:
        st.subheader("👨‍🎓 Enter Student Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            study_hours = st.slider("Study Hours per Week", 1, 15, 7)
            attendance = st.slider("Attendance Percentage", 50, 100, 85)
            previous_grade = st.slider("Previous Grade", 40, 100, 75)
            absences = st.slider("Number of Absences", 0, 30, 5)
        
        with col2:
            parent_education = st.selectbox("Parent Education Level", 
                                          ["Primary", "High School", "Bachelor", "Master"])
            extra_activities = st.radio("Extra-curricular Activities", ["Yes", "No"])
            internet_access = st.radio("Internet Access at Home", ["Yes", "No"])
            family_support = st.radio("Family Support", ["Yes", "No"])
        
        if st.button("🔮 Predict Grade", type="primary"):
            # Prepare input data
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
            
            # Encode categorical variables
            for col in ['parent_education', 'extra_activities', 'internet_access', 'family_support']:
                input_data[col] = st.session_state.label_encoders[col].transform(input_data[col])
            
            # Add engineered features
            input_data['study_attendance_ratio'] = input_data['study_hours'] / (input_data['attendance'] + 1)
            input_data['effective_study_time'] = input_data['study_hours'] * (input_data['attendance'] / 100)
            
            # Scale and predict
            input_scaled = st.session_state.scaler.transform(input_data)
            prediction = st.session_state.model.predict(input_scaled)[0]
            
            # Display result
            st.markdown("---")
            st.subheader("📊 Prediction Result")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                st.metric("Predicted Final Grade", f"{prediction:.2f}", 
                         delta=f"{prediction - previous_grade:.2f} from previous grade")
                
                # Grade category
                if prediction >= 90:
                    grade_cat = "🌟 Excellent (A)"
                    color = "green"
                elif prediction >= 80:
                    grade_cat = "👍 Very Good (B)"
                    color = "blue"
                elif prediction >= 70:
                    grade_cat = "✅ Good (C)"
                    color = "orange"
                elif prediction >= 60:
                    grade_cat = "⚠️ Satisfactory (D)"
                    color = "orange"
                else:
                    grade_cat = "❌ Needs Improvement (F)"
                    color = "red"
                
                st.markdown(f"**Performance Category:** :{color}[{grade_cat}]")
                
                # Progress bar
                st.progress(int(prediction))

# ==================== MODEL COMPARISON PAGE ====================
elif page == "Model Comparison":
    st.header("🔬 Model Comparison")
    
    if st.session_state.df is None:
        st.warning("⚠️ No data loaded. Please generate sample data from the Home page.")
    else:
        if st.button("📊 Compare All Models", type="primary"):
            with st.spinner("Training and comparing all models..."):
                df_encoded, _ = preprocess_data(st.session_state.df)
                
                model_names = ["Linear Regression", "Decision Tree", "Random Forest", "Gradient Boosting"]
                comparison_results = []
                
                for model_name in model_names:
                    _, _, metrics, _, _, _, _ = train_model(df_encoded, model_name)
                    comparison_results.append({
                        'Model': model_name,
                        'R² Score': metrics['R2'],
                        'RMSE': metrics['RMSE'],
                        'MAE': metrics['MAE'],
                        'CV R² Score': metrics['CV_R2_mean']
                    })
                
                results_df = pd.DataFrame(comparison_results)
            
            st.success("✅ Model comparison completed!")
            
            # Display comparison table
            st.subheader("📋 Performance Comparison Table")
            st.dataframe(results_df.style.highlight_max(axis=0, subset=['R² Score', 'CV R² Score'])
                        .highlight_min(axis=0, subset=['RMSE', 'MAE']), 
                        use_container_width=True)
            
            # Visualization
            st.subheader("📊 Visual Comparison")
            
            fig = go.Figure()
            
            metrics_to_plot = ['R² Score', 'RMSE', 'MAE']
            
            for metric in metrics_to_plot:
                fig.add_trace(go.Bar(
                    name=metric,
                    x=results_df['Model'],
                    y=results_df[metric],
                    text=results_df[metric].round(2),
                    textposition='auto'
                ))
            
            fig.update_layout(
                barmode='group',
                title='Model Performance Comparison',
                xaxis_title='Model',
                yaxis_title='Score',
                legend_title='Metrics'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Best model recommendation
            best_model = results_df.loc[results_df['R² Score'].idxmax(), 'Model']
            st.success(f"🏆 **Recommended Model:** {best_model} (Highest R² Score)")
