# src/evaluate.py
"""
Model evaluation and metrics calculation.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os


class ModelEvaluation:
    """Comprehensive model evaluation class."""
    
    def __init__(self, model_name: str = "model"):
        """
        Initialize the ModelEvaluation class.
        
        Args:
            model_name (str): Name of the model for reporting
        """
        self.model_name = model_name
        self.metrics = {}
        self.y_true = None
        self.y_pred = None
        self.y_pred_proba = None
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, 
                y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Evaluate model predictions.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            y_pred_proba (np.ndarray, optional): Predicted probabilities
            
        Returns:
            Dict[str, float]: Dictionary of evaluation metrics
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba
        
        # Basic metrics
        self.metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }
        
        # Probability-based metrics
        if y_pred_proba is not None:
            try:
                self.metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
                self.metrics['average_precision'] = average_precision_score(y_true, y_pred_proba)
            except Exception as e:
                print(f"Warning: Could not calculate probability-based metrics: {e}")
                self.metrics['roc_auc'] = 0.5
                self.metrics['average_precision'] = 0.5
        
        # Additional metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        self.metrics['true_negative'] = tn
        self.metrics['false_positive'] = fp
        self.metrics['false_negative'] = fn
        self.metrics['true_positive'] = tp
        self.metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return self.metrics
    
    def get_classification_report(self) -> Dict:
        """
        Get detailed classification report.
        
        Returns:
            Dict: Classification report
        """
        if self.y_true is None or self.y_pred is None:
            raise ValueError("Must call evaluate() first")
        
        return classification_report(self.y_true, self.y_pred, output_dict=True)
    
    def get_confusion_matrix(self) -> np.ndarray:
        """
        Get confusion matrix.
        
        Returns:
            np.ndarray: Confusion matrix
        """
        if self.y_true is None or self.y_pred is None:
            raise ValueError("Must call evaluate() first")
        
        return confusion_matrix(self.y_true, self.y_pred)
    
    def plot_roc_curve(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot ROC curve.
        
        Args:
            save_path (str, optional): Path to save the plot
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        if self.y_pred_proba is None:
            raise ValueError("ROC curve requires predicted probabilities")
        
        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'ROC Curve - {self.model_name}', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to {save_path}")
        
        return fig
    
    def plot_precision_recall_curve(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot Precision-Recall curve.
        
        Args:
            save_path (str, optional): Path to save the plot
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        if self.y_pred_proba is None:
            raise ValueError("Precision-Recall curve requires predicted probabilities")
        
        precision, recall, _ = precision_recall_curve(self.y_true, self.y_pred_proba)
        avg_precision = average_precision_score(self.y_true, self.y_pred_proba)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, color='green', lw=2, 
                label=f'Precision-Recall curve (AP = {avg_precision:.3f})')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(f'Precision-Recall Curve - {self.model_name}', fontsize=14, fontweight='bold')
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Precision-Recall curve saved to {save_path}")
        
        return fig
    
    def plot_confusion_matrix_heatmap(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot confusion matrix as heatmap.
        
        Args:
            save_path (str, optional): Path to save the plot
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        cm = self.get_confusion_matrix()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Predicted 0', 'Predicted 1'],
                   yticklabels=['Actual 0', 'Actual 1'],
                   ax=ax)
        ax.set_title(f'Confusion Matrix - {self.model_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        return fig
    
    def plot_metrics_comparison(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot comparison of all metrics.
        
        Args:
            save_path (str, optional): Path to save the plot
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        # Filter out non-metric keys
        metric_keys = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'specificity']
        metrics_to_plot = {k: self.metrics[k] for k in metric_keys if k in self.metrics}
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(metrics_to_plot.keys(), metrics_to_plot.values(), 
                     color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
        ax.set_title(f'Model Metrics - {self.model_name}', fontsize=14, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, (metric, value) in zip(bars, metrics_to_plot.items()):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Metrics comparison saved to {save_path}")
        
        return fig
    
    def generate_report(self, output_dir: str = "evaluation_reports") -> Dict:
        """
        Generate comprehensive evaluation report.
        
        Args:
            output_dir (str): Directory to save reports
            
        Returns:
            Dict: Complete evaluation report
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report = {
            'model_name': self.model_name,
            'timestamp': timestamp,
            'metrics': self.metrics,
            'classification_report': self.get_classification_report(),
            'confusion_matrix': self.get_confusion_matrix().tolist()
        }
        
        # Save JSON report
        json_path = os.path.join(output_dir, f"{self.model_name}_report_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=4, default=str)
        
        print(f"JSON report saved to {json_path}")
        
        # Generate plots
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        try:
            self.plot_roc_curve(os.path.join(plots_dir, f"{self.model_name}_roc_{timestamp}.png"))
            self.plot_precision_recall_curve(os.path.join(plots_dir, f"{self.model_name}_pr_{timestamp}.png"))
            self.plot_confusion_matrix_heatmap(os.path.join(plots_dir, f"{self.model_name}_cm_{timestamp}.png"))
            self.plot_metrics_comparison(os.path.join(plots_dir, f"{self.model_name}_metrics_{timestamp}.png"))
        except Exception as e:
            print(f"Warning: Could not generate some plots: {e}")
        
        return report
    
    def print_summary(self):
        """Print a summary of evaluation results."""
        print(f"\n{'='*60}")
        print(f"EVALUATION SUMMARY - {self.model_name}")
        print(f"{'='*60}")
        
        print("\n📊 Performance Metrics:")
        for metric, value in self.metrics.items():
            if metric not in ['true_negative', 'false_positive', 'false_negative', 'true_positive']:
                print(f"  {metric.replace('_', ' ').title():<20}: {value:.4f}")
        
        print("\n🔢 Confusion Matrix:")
        cm = self.get_confusion_matrix()
        print(f"                 Predicted")
        print(f"              0         1")
        print(f"Actual 0   {cm[0,0]:^8}  {cm[0,1]:^8}")
        print(f"       1   {cm[1,0]:^8}  {cm[1,1]:^8}")
        
        print(f"\n{'='*60}")


def evaluate_model_from_path(model_path: str, test_data_path: str, 
                           target_column: str = 'is_high_risk') -> ModelEvaluation:
    """
    Evaluate a model from file path.
    
    Args:
        model_path (str): Path to saved model
        test_data_path (str): Path to test data
        target_column (str): Name of target column
        
    Returns:
        ModelEvaluation: Evaluation results
    """
    import joblib
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    
    # Load test data
    print(f"Loading test data from {test_data_path}...")
    test_data = pd.read_csv(test_data_path)
    
    # Separate features and target
    X_test = test_data.drop(columns=[target_column])
    y_test = test_data[target_column]
    
    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Evaluate
    evaluator = ModelEvaluation(model_name=os.path.basename(model_path))
    evaluator.evaluate(y_test.values, y_pred, y_pred_proba)
    
    return evaluator


if __name__ == "__main__":
    # Example usage
    print("Model Evaluation Module")
    print("Use evaluate_model_from_path() to evaluate a saved model")