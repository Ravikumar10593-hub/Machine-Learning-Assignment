"""
ML Assignment 2 - Streamlit Web Application
Heart Disease Classification Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                            recall_score, f1_score, matthews_corrcoef,
                            confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Heart Disease Classifier",
    page_icon="❤️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
        .stMetric {
            background-color: #000000 !important;
            color: #ffffff !important;
            padding: 10px;
            border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("❤️ Heart Disease Classification System")
st.markdown("### ML Assignment 2 - Multi-Model Comparison Dashboard")
st.markdown("---")

# Sidebar
st.sidebar.header("📊 Configuration")
st.sidebar.markdown("Upload your test dataset and select a model to evaluate.")

# Load models function
@st.cache_resource
def load_models():
    models = {}
    model_names = ['Logistic Regression', 'Decision Tree', 'K-Nearest Neighbors',
                   'Naive Bayes', 'Random Forest', 'XGBoost']
    model_files = ['logistic_regression.pkl', 'decision_tree.pkl', 'knn.pkl',
                   'naive_bayes.pkl', 'random_forest.pkl', 'xgboost.pkl']
    
    for name, file in zip(model_names, model_files):
        try:
            with open(f'model/{file}', 'rb') as f:
                models[name] = pickle.load(f)
        except:
            st.warning(f"Model {name} not found. Please train models first.")
    
    # Load scaler
    try:
        with open('model/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
    except:
        scaler = None
    
    return models, scaler

# Load feature names
@st.cache_data
def load_feature_names():
    try:
        with open('model/feature_names.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        return None

# File upload
st.sidebar.markdown("### 📁 Upload Test Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file with test data",
    type=['csv'],
    help="Upload the test_data.csv file generated during training"
)

# Model selection
st.sidebar.markdown("### 🤖 Select Model")
model_choice = st.sidebar.selectbox(
    "Choose a classification model:",
    ['Logistic Regression', 'Decision Tree', 'K-Nearest Neighbors',
     'Naive Bayes', 'Random Forest', 'XGBoost']
)

# Load models
models, scaler = load_models()
feature_names = load_feature_names()

# Main content
if uploaded_file is not None:
    # Read the uploaded file
    data = pd.read_csv(uploaded_file)
    
    # Display dataset info
    st.header("📋 Dataset Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Samples", data.shape[0])
    with col2:
        st.metric("Features", data.shape[1] - 1)  # Excluding target
    with col3:
        if 'target' in data.columns:
            st.metric("Classes", data['target'].nunique())
    
    # Show data preview
    with st.expander("🔍 View Data Preview"):
        st.dataframe(data.head(10))
    
    # Prepare data
    if 'target' in data.columns:
        X_test = data.drop('target', axis=1)
        y_test = data['target']
        
        # Make predictions
        if model_choice in models:
            model = models[model_choice]
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Get probabilities if available
            try:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            except:
                y_pred_proba = None
            
            # Display evaluation metrics
            st.header(f"📊 Model Performance: {model_choice}")
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            mcc = matthews_corrcoef(y_test, y_pred)
            
            # Display metrics in columns
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Accuracy", f"{accuracy:.4f}")
                st.metric("Precision", f"{precision:.4f}")
            with col2:
                st.metric("Recall", f"{recall:.4f}")
                st.metric("F1 Score", f"{f1:.4f}")
            with col3:
                if y_pred_proba is not None:
                    auc = roc_auc_score(y_test, y_pred_proba)
                    st.metric("AUC Score", f"{auc:.4f}")
                st.metric("MCC Score", f"{mcc:.4f}")
            
            st.markdown("---")
            
            # Confusion Matrix
            st.header("🎯 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['No Disease', 'Disease'],
                       yticklabels=['No Disease', 'Disease'],
                       cbar_kws={'label': 'Count'})
            ax.set_xlabel('Predicted Label', fontsize=12)
            ax.set_ylabel('True Label', fontsize=12)
            ax.set_title(f'Confusion Matrix - {model_choice}', fontsize=14, fontweight='bold')
            st.pyplot(fig)
            plt.close()
            
            # Classification Report
            st.header("📈 Classification Report")
            report = classification_report(y_test, y_pred, 
                                          target_names=['No Disease', 'Disease'],
                                          output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='RdYlGn', subset=['f1-score']))
            
            # Predictions table
            with st.expander("🔮 View Predictions"):
                predictions_df = pd.DataFrame({
                    'True Label': y_test.values,
                    'Predicted Label': y_pred,
                    'Correct': y_test.values == y_pred
                })
                if y_pred_proba is not None:
                    predictions_df['Probability'] = y_pred_proba
                
                st.dataframe(predictions_df.head(20))
        else:
            st.error(f"Model '{model_choice}' not found. Please train the model first.")
    else:
        st.error("The uploaded file must contain a 'target' column.")
        
    # Model Comparison Section
    st.markdown("---")
    st.header("🏆 All Models Comparison")
    
    # Pre-computed results (you should replace these with actual values)
    comparison_data = {
    'ML Model': [
        'Logistic Regression',
        'Decision Tree',
        'K-Nearest Neighbors',
        'Naive Bayes',
        'Random Forest',
        'XGBoost'
    ],
    'Accuracy': [0.8333, 0.7000, 0.8833, 0.8833, 0.8667, 0.8667],
    'AUC':      [0.9498, 0.7450, 0.9492, 0.9375, 0.9414, 0.8917],
    'Precision':[0.8462, 0.7500, 0.9200, 0.8889, 0.8846, 0.8846],
    'Recall':   [0.7857, 0.5357, 0.8214, 0.8571, 0.8214, 0.8214],
    'F1':       [0.8148, 0.6250, 0.8679, 0.8727, 0.8519, 0.8519],
    'MCC':      [0.6652, 0.4016, 0.7680, 0.7655, 0.7326, 0.7326]
}
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df.style.background_gradient(cmap='YlGnBu', subset=comparison_df.columns[1:]))
    
else:
    # Instructions when no file is uploaded
    st.info("👈 Please upload your test data CSV file from the sidebar to begin evaluation.")
    
    st.markdown("""
    ### 📖 Instructions:
    
    1. **Upload Test Data**: Use the file uploader in the sidebar to upload your `test_data.csv`
    2. **Select Model**: Choose a classification model from the dropdown
    3. **View Results**: The app will display:
        - Dataset information
        - Evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
        - Confusion matrix visualization
        - Detailed classification report
        - Prediction results
    
    ### 📊 Available Models:
    - Logistic Regression
    - Decision Tree
    - K-Nearest Neighbors
    - Naive Bayes (Gaussian)
    - Random Forest (Ensemble)
    - XGBoost (Ensemble)
    
    ### 🎯 Dataset Requirements:
    - Minimum 12 features
    - Minimum 500 instances
    - Must include 'target' column for evaluation
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>ML Assignment 2 | M.Tech (AIML/DSE) | Machine Learning</p>
        <p>Developed with ❤️ using Streamlit</p>
    </div>
""", unsafe_allow_html=True)