"""
ML Assignment 2 - Model Training and Evaluation
Dataset: Heart Disease Classification
Author: [Your Name]
Date: January 2026
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                            recall_score, f1_score, matthews_corrcoef, 
                            confusion_matrix, classification_report)
import pickle
import os

# Create model directory if it doesn't exist
if not os.path.exists('model'):
    os.makedirs('model')

# Load the dataset
# TODO: Replace with your chosen dataset
# For Heart Disease dataset from UCI
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

# Load data
df = pd.read_csv(url, names=column_names, na_values='?')

# Data preprocessing
df = df.dropna()  # Remove missing values
df['target'] = (df['target'] > 0).astype(int)  # Binary classification: 0 = no disease, 1 = disease

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

print(f"Dataset shape: {X.shape}")
print(f"Number of features: {X.shape[1]}")
print(f"Number of instances: {X.shape[0]}")
print(f"Class distribution:\n{y.value_counts()}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
with open('model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save test data for Streamlit
test_data = pd.concat([pd.DataFrame(X_test_scaled, columns=X.columns), 
                       pd.DataFrame(y_test.values, columns=['target'])], axis=1)
test_data.to_csv('test_data.csv', index=False)

print("\nTest data saved as 'test_data.csv' for Streamlit upload")

# Dictionary to store models and results
models = {}
results = []

# 1. Logistic Regression
print("\n" + "="*50)
print("Training Logistic Regression...")
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train_scaled, y_train)
y_pred = lr.predict(X_test_scaled)
y_pred_proba = lr.predict_proba(X_test_scaled)[:, 1]

lr_metrics = {
    'Model': 'Logistic Regression',
    'Accuracy': accuracy_score(y_test, y_pred),
    'AUC': roc_auc_score(y_test, y_pred_proba),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1': f1_score(y_test, y_pred),
    'MCC': matthews_corrcoef(y_test, y_pred)
}
results.append(lr_metrics)
models['Logistic Regression'] = lr
print(f"Accuracy: {lr_metrics['Accuracy']:.4f}")

# Save model
with open('model/logistic_regression.pkl', 'wb') as f:
    pickle.dump(lr, f)

# 2. Decision Tree
print("\n" + "="*50)
print("Training Decision Tree...")
dt = DecisionTreeClassifier(random_state=42, max_depth=5)
dt.fit(X_train_scaled, y_train)
y_pred = dt.predict(X_test_scaled)
y_pred_proba = dt.predict_proba(X_test_scaled)[:, 1]

dt_metrics = {
    'Model': 'Decision Tree',
    'Accuracy': accuracy_score(y_test, y_pred),
    'AUC': roc_auc_score(y_test, y_pred_proba),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1': f1_score(y_test, y_pred),
    'MCC': matthews_corrcoef(y_test, y_pred)
}
results.append(dt_metrics)
models['Decision Tree'] = dt
print(f"Accuracy: {dt_metrics['Accuracy']:.4f}")

# Save model
with open('model/decision_tree.pkl', 'wb') as f:
    pickle.dump(dt, f)

# 3. K-Nearest Neighbors
print("\n" + "="*50)
print("Training K-Nearest Neighbors...")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)
y_pred_proba = knn.predict_proba(X_test_scaled)[:, 1]

knn_metrics = {
    'Model': 'K-Nearest Neighbors',
    'Accuracy': accuracy_score(y_test, y_pred),
    'AUC': roc_auc_score(y_test, y_pred_proba),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1': f1_score(y_test, y_pred),
    'MCC': matthews_corrcoef(y_test, y_pred)
}
results.append(knn_metrics)
models['K-Nearest Neighbors'] = knn
print(f"Accuracy: {knn_metrics['Accuracy']:.4f}")

# Save model
with open('model/knn.pkl', 'wb') as f:
    pickle.dump(knn, f)

# 4. Naive Bayes (Gaussian)
print("\n" + "="*50)
print("Training Naive Bayes...")
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)
y_pred = nb.predict(X_test_scaled)
y_pred_proba = nb.predict_proba(X_test_scaled)[:, 1]

nb_metrics = {
    'Model': 'Naive Bayes',
    'Accuracy': accuracy_score(y_test, y_pred),
    'AUC': roc_auc_score(y_test, y_pred_proba),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1': f1_score(y_test, y_pred),
    'MCC': matthews_corrcoef(y_test, y_pred)
}
results.append(nb_metrics)
models['Naive Bayes'] = nb
print(f"Accuracy: {nb_metrics['Accuracy']:.4f}")

# Save model
with open('model/naive_bayes.pkl', 'wb') as f:
    pickle.dump(nb, f)

# 5. Random Forest
print("\n" + "="*50)
print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred = rf.predict(X_test_scaled)
y_pred_proba = rf.predict_proba(X_test_scaled)[:, 1]

rf_metrics = {
    'Model': 'Random Forest',
    'Accuracy': accuracy_score(y_test, y_pred),
    'AUC': roc_auc_score(y_test, y_pred_proba),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1': f1_score(y_test, y_pred),
    'MCC': matthews_corrcoef(y_test, y_pred)
}
results.append(rf_metrics)
models['Random Forest'] = rf
print(f"Accuracy: {rf_metrics['Accuracy']:.4f}")

# Save model
with open('model/random_forest.pkl', 'wb') as f:
    pickle.dump(rf, f)

# 6. XGBoost
print("\n" + "="*50)
print("Training XGBoost...")
xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train_scaled, y_train)
y_pred = xgb.predict(X_test_scaled)
y_pred_proba = xgb.predict_proba(X_test_scaled)[:, 1]

xgb_metrics = {
    'Model': 'XGBoost',
    'Accuracy': accuracy_score(y_test, y_pred),
    'AUC': roc_auc_score(y_test, y_pred_proba),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1': f1_score(y_test, y_pred),
    'MCC': matthews_corrcoef(y_test, y_pred)
}
results.append(xgb_metrics)
models['XGBoost'] = xgb
print(f"Accuracy: {xgb_metrics['Accuracy']:.4f}")

# Save model
with open('model/xgboost.pkl', 'wb') as f:
    pickle.dump(xgb, f)

# Create results DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.round(4)

# Display results
print("\n" + "="*80)
print("FINAL RESULTS - ALL MODELS")
print("="*80)
print(results_df.to_string(index=False))
print("="*80)

# Save results
results_df.to_csv('model_results.csv', index=False)
print("\nResults saved to 'model_results.csv'")

# Save feature names
with open('model/feature_names.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)

print("\n✓ All models trained and saved successfully!")
print("✓ Test data prepared for Streamlit")