# run_task5.py - SIMPLE VERSION
print("=" * 60)
print("TASK 5: TRAINING MODELS")
print("=" * 60)

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
import joblib
import os

# 1. Load Task 4 complete data
df = pd.read_csv('data/processed/processed_data_complete.csv')
print(f"Data: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Target: 'is_high_risk'")
print(f"Target distribution: {df['is_high_risk'].value_counts().to_dict()}")

# 2. Split data
X = df.drop(columns=['is_high_risk'])
y = df['is_high_risk']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

# 3. Setup MLflow
mlflow.set_experiment("credit_risk")
os.makedirs("models", exist_ok=True)

# 4. Train Logistic Regression
print("\n[1/2] Training Logistic Regression...")
with mlflow.start_run(run_name="logistic_regression"):
    # Grid Search
    param_grid = {'C': [0.01, 0.1, 1, 10]}
    grid = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), 
                       param_grid, cv=3, scoring='roc_auc')
    grid.fit(X_train, y_train)
    
    model = grid.best_estimator_
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }
    
    # Log to MLflow
    mlflow.log_params(grid.best_params_)
    for name, value in metrics.items():
        mlflow.log_metric(name, value)
    
    # Save
    joblib.dump(model, "models/logistic_regression.joblib")
    
    print(f"   [OK] Accuracy: {metrics['accuracy']:.4f}")
    print(f"   [OK] ROC-AUC: {metrics['roc_auc']:.4f}")

# 5. Train Random Forest
print("\n[2/2] Training Random Forest...")
with mlflow.start_run(run_name="random_forest"):
    # Grid Search
    param_grid = {'n_estimators': [50, 100], 'max_depth': [5, 10]}
    grid = GridSearchCV(RandomForestClassifier(random_state=42), 
                       param_grid, cv=3, scoring='roc_auc')
    grid.fit(X_train, y_train)
    
    model = grid.best_estimator_
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }
    
    # Log to MLflow
    mlflow.log_params(grid.best_params_)
    for name, value in metrics.items():
        mlflow.log_metric(name, value)
    
    # Save
    joblib.dump(model, "models/random_forest.joblib")
    joblib.dump(model, "models/best_model.joblib")
    
    print(f"   [OK] Accuracy: {metrics['accuracy']:.4f}")
    print(f"   [OK] ROC-AUC: {metrics['roc_auc']:.4f}")

print("\n" + "=" * 60)
print("[DONE] TASK 5 COMPLETE!")
print("=" * 60)
print("\nModels saved to 'models/' folder")
print("Run: mlflow ui")
print("Open: http://localhost:5000")