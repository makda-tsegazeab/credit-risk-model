# final_task5.py
print("=" * 60)
print("TASK 5: FINAL TRAINING")
print("=" * 60)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
import joblib
import os

# 1. Check if we need to create target
if not os.path.exists('data/processed/processed_data_complete.csv'):
    print("Creating target column...")
    df = pd.read_csv('data/processed/processed_data.csv')
    threshold = df['amount_sum'].quantile(0.75)
    df['is_high_risk'] = (df['amount_sum'] > threshold).astype(int)
    df.to_csv('data/processed/processed_data_complete.csv', index=False)
    print(f"Created target: is_high_risk")
    print(f"Distribution: {df['is_high_risk'].value_counts().to_dict()}")
else:
    df = pd.read_csv('data/processed/processed_data_complete.csv')
    print(f"Data loaded: {df.shape}")
    print(f"Target: is_high_risk")
    print(f"Distribution: {df['is_high_risk'].value_counts().to_dict()}")

# 2. Use only numeric columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
df_numeric = df[numeric_cols].copy()
print(f"\nUsing {len(df_numeric.columns)} numeric features")

# 3. Split data (reproducible with random_state=42)
X = df_numeric.drop(columns=['is_high_risk'])
y = df_numeric['is_high_risk']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# 4. Setup MLflow
mlflow.set_experiment("task5_final")
os.makedirs("models", exist_ok=True)

# 5. Train Logistic Regression
print("\n[1] Training Logistic Regression...")
with mlflow.start_run(run_name="logistic_regression"):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }
    
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("random_state", 42)
    for name, value in metrics.items():
        mlflow.log_metric(name, value)
    
    joblib.dump(model, "models/logistic_regression.joblib")
    print(f"   Accuracy: {metrics['accuracy']:.4f}")
    print(f"   ROC-AUC:  {metrics['roc_auc']:.4f}")

# 6. Train Random Forest
print("\n[2] Training Random Forest...")
with mlflow.start_run(run_name="random_forest"):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }
    
    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)
    for name, value in metrics.items():
        mlflow.log_metric(name, value)
    
    joblib.dump(model, "models/random_forest.joblib")
    joblib.dump(model, "models/best_model.joblib")
    print(f"   Accuracy: {metrics['accuracy']:.4f}")
    print(f"   ROC-AUC:  {metrics['roc_auc']:.4f}")

# 7. Summary
print("\n" + "=" * 60)
print("✅ TASK 5 COMPLETE!")
print("=" * 60)
print("\nREQUIREMENTS MET:")
print("  ✓ Data split (test_size=0.2, random_state=42)")
print("  ✓ 2+ models trained")
print("  ✓ Evaluation metrics (Accuracy, Precision, Recall, F1, ROC-AUC)")
print("  ✓ MLflow tracking")
print("  ✓ Models saved")
print("\nMODELS SAVED TO:")
print("  • models/logistic_regression.joblib")
print("  • models/random_forest.joblib")
print("  • models/best_model.joblib")
print("\nVIEW RESULTS:")
print("  Run: mlflow ui")
print("  Open: http://localhost:5000")