# src/model_utils.py
"""
Model utilities for training, hyperparameter tuning, and evaluation.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Union
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)
from xgboost import XGBClassifier
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')


class ModelFactory:
    """Factory class to create and configure ML models."""
    
    @staticmethod
    def get_models() -> Dict[str, Any]:
        """
        Get dictionary of models with their default configurations.
        
        Returns:
            Dict[str, Any]: Dictionary of model names and instances
        """
        return {
            'logistic_regression': LogisticRegression(
                random_state=42, 
                max_iter=1000,
                class_weight='balanced'
            ),
            'decision_tree': DecisionTreeClassifier(
                random_state=42,
                max_depth=10,
                min_samples_split=5,
                class_weight='balanced'
            ),
            'random_forest': RandomForestClassifier(
                random_state=42,
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                n_jobs=-1,
                class_weight='balanced'
            ),
            'gradient_boosting': GradientBoostingClassifier(
                random_state=42,
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5
            ),
            'xgboost': XGBClassifier(
                random_state=42,
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                eval_metric='logloss',
                use_label_encoder=False,
                n_jobs=-1
            ),
            'lightgbm': lgb.LGBMClassifier(
                random_state=42,
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                n_jobs=-1,
                class_weight='balanced'
            )
        }
    
    @staticmethod
    def get_hyperparameter_grids() -> Dict[str, Dict[str, List]]:
        """
        Get hyperparameter grids for each model.
        
        Returns:
            Dict[str, Dict[str, List]]: Hyperparameter grids for each model
        """
        return {
            'logistic_regression': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'decision_tree': {
                'max_depth': [3, 5, 7, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4, 8],
                'criterion': ['gini', 'entropy']
            },
            'random_forest': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.001, 0.01, 0.1, 0.2],
                'max_depth': [3, 4, 5, 6, 7],
                'min_samples_split': [2, 5, 10],
                'subsample': [0.8, 0.9, 1.0]
            },
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.001, 0.01, 0.1, 0.2],
                'max_depth': [3, 4, 5, 6, 7],
                'min_child_weight': [1, 3, 5],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            'lightgbm': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.001, 0.01, 0.1, 0.2],
                'max_depth': [3, 4, 5, 6, 7, -1],  # -1 means no limit
                'num_leaves': [31, 50, 100, 150],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
        }


class HyperparameterTuner:
    """Class for hyperparameter tuning using Grid Search and Random Search."""
    
    def __init__(self, scoring: str = 'roc_auc', cv: int = 5, n_jobs: int = -1):
        """
        Initialize the hyperparameter tuner.
        
        Args:
            scoring (str): Scoring metric for tuning
            cv (int): Number of cross-validation folds
            n_jobs (int): Number of parallel jobs
        """
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
    
    def grid_search(self, model, X_train: pd.DataFrame, y_train: pd.Series, 
                   param_grid: Dict[str, List]) -> Tuple[Any, Dict]:
        """
        Perform Grid Search CV for hyperparameter tuning.
        
        Args:
            model: Base model instance
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training labels
            param_grid (Dict[str, List]): Parameter grid
            
        Returns:
            Tuple[Any, Dict]: Best model and best parameters
        """
        print(f"Performing Grid Search for {model.__class__.__name__}...")
        
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring=self.scoring,
            cv=self.cv,
            n_jobs=self.n_jobs,
            verbose=1,
            error_score='raise'
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best {self.scoring} score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
    
    def random_search(self, model, X_train: pd.DataFrame, y_train: pd.Series,
                     param_distributions: Dict[str, List], n_iter: int = 20) -> Tuple[Any, Dict]:
        """
        Perform Random Search CV for hyperparameter tuning.
        
        Args:
            model: Base model instance
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training labels
            param_distributions (Dict[str, List]): Parameter distributions
            n_iter (int): Number of iterations
            
        Returns:
            Tuple[Any, Dict]: Best model and best parameters
        """
        print(f"Performing Random Search for {model.__class__.__name__}...")
        
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring=self.scoring,
            cv=self.cv,
            n_jobs=self.n_jobs,
            random_state=42,
            verbose=1,
            error_score='raise'
        )
        
        random_search.fit(X_train, y_train)
        
        print(f"Best parameters: {random_search.best_params_}")
        print(f"Best {self.scoring} score: {random_search.best_score_:.4f}")
        
        return random_search.best_estimator_, random_search.best_params_, random_search.best_score_


class ModelEvaluator:
    """Class for evaluating model performance."""
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                         y_pred_proba: np.ndarray = None) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            y_pred_proba (np.ndarray): Predicted probabilities
            
        Returns:
            Dict[str, float]: Dictionary of evaluation metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }
        
        # Calculate ROC-AUC if probabilities are available
        if y_pred_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            except:
                metrics['roc_auc'] = 0.5
        else:
            metrics['roc_auc'] = 0.5
        
        return metrics
    
    @staticmethod
    def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate a trained model on test data.
        
        Args:
            model: Trained model
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test labels
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Get probabilities if available
        y_pred_proba = None
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = ModelEvaluator.calculate_metrics(y_test.values, y_pred, y_pred_proba)
        
        return metrics
    
    @staticmethod
    def print_metrics(metrics: Dict[str, float], model_name: str = "Model"):
        """
        Print evaluation metrics in a formatted way.
        
        Args:
            metrics (Dict[str, float]): Evaluation metrics
            model_name (str): Name of the model
        """
        print(f"\n{'='*50}")
        print(f"{model_name} Evaluation Metrics")
        print(f"{'='*50}")
        for metric, value in metrics.items():
            print(f"{metric.replace('_', ' ').title():<15}: {value:.4f}")
        print(f"{'='*50}")


def prepare_data_for_training(data_path: str, target_column: str = 'is_high_risk', 
                             test_size: float = 0.2, random_state: int = 42) -> Tuple:
    """
    Prepare data for model training.
    
    Args:
        data_path (str): Path to processed data
        target_column (str): Name of target column
        test_size (float): Proportion of test data
        random_state (int): Random seed for reproducibility
        
    Returns:
        Tuple: (X_train, X_test, y_train, y_test)
    """
    # Load data
    data = pd.read_csv(data_path)
    
    # Check if target column exists
    if target_column not in data.columns:
        available_columns = data.columns.tolist()
        raise ValueError(f"Target column '{target_column}' not found. Available columns: {available_columns}")
    
    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Check class distribution
    print(f"Class distribution in full dataset:")
    print(y.value_counts(normalize=True))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\nData split:")
    print(f"Training set: {X_train.shape}")
    print(f"Testing set: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test