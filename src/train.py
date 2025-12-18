# src/train.py
"""
Main training script with MLflow tracking and model registry.
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from datetime import datetime
import joblib

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our custom modules
try:
    from src.model_utils import ModelFactory, HyperparameterTuner, ModelEvaluator, prepare_data_for_training
    from src.evaluate import ModelEvaluation
except ImportError:
    # For direct execution
    from model_utils import ModelFactory, HyperparameterTuner, ModelEvaluator, prepare_data_for_training
    from evaluate import ModelEvaluation


class MLflowTrainer:
    """Main training class with MLflow integration."""
    
    def __init__(self, experiment_name: str = "credit_risk_modeling"):
        """
        Initialize the MLflow trainer.
        
        Args:
            experiment_name (str): MLflow experiment name
        """
        self.experiment_name = experiment_name
        self.tuner = HyperparameterTuner(scoring='roc_auc', cv=3, n_jobs=-1)
        self.models = {}
        self.results = {}
        
        # Set up MLflow
        self._setup_mlflow()
        
        # Create directories
        self.models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        self.mlflow_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "mlruns")
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.mlflow_dir, exist_ok=True)
    
    def _setup_mlflow(self):
        """Set up MLflow tracking."""
        mlflow.set_tracking_uri(f"file:///{os.path.join(os.path.dirname(os.path.dirname(__file__)), 'mlruns')}")
        mlflow.set_experiment(self.experiment_name)
        print(f"MLflow experiment: {self.experiment_name}")
        print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    
    def load_data(self, data_path: str = None, target_column: str = 'is_high_risk'):
        """
        Load and prepare data for training.
        
        Args:
            data_path (str): Path to processed data
            target_column (str): Name of target column
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        if data_path is None:
            # Default path
            data_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "data", "processed", "processed_data.csv"
            )
        
        print(f"Loading data from: {data_path}")
        print(f"Target column: {target_column}")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Use the function from model_utils
        X_train, X_test, y_train, y_test = prepare_data_for_training(
            data_path=data_path,
            target_column=target_column,
            test_size=0.2,
            random_state=42
        )
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        # Log data info
        self.data_info = {
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'n_features': X_train.shape[1],
            'train_class_distribution': dict(y_train.value_counts()),
            'test_class_distribution': dict(y_test.value_counts())
        }
        
        print("\n📊 Data Information:")
        for key, value in self.data_info.items():
            print(f"  {key}: {value}")
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, model_name: str, use_grid_search: bool = True, 
                   use_random_search: bool = True):
        """
        Train a specific model with hyperparameter tuning.
        
        Args:
            model_name (str): Name of the model to train
            use_grid_search (bool): Whether to use Grid Search
            use_random_search (bool): Whether to use Random Search
            
        Returns:
            dict: Training results
        """
        if self.X_train is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print(f"\n{'='*60}")
        print(f"TRAINING MODEL: {model_name.upper()}")
        print(f"{'='*60}")
        
        model_results = {}
        
        # Get model and hyperparameters
        models = ModelFactory.get_models()
        param_grids = ModelFactory.get_hyperparameter_grids()
        
        if model_name not in models:
            raise ValueError(f"Model {model_name} not found. Available: {list(models.keys())}")
        
        base_model = models[model_name]
        param_grid = param_grids[model_name]
        
        # Grid Search
        if use_grid_search:
            print(f"\n🔍 Running Grid Search for {model_name}...")
            try:
                with mlflow.start_run(run_name=f"{model_name}_grid_search", nested=True):
                    best_model, best_params, cv_score = self.tuner.grid_search(
                        base_model, self.X_train, self.y_train, param_grid
                    )
                    
                    # Evaluate
                    train_metrics = ModelEvaluator.evaluate_model(best_model, self.X_train, self.y_train)
                    test_metrics = ModelEvaluator.evaluate_model(best_model, self.X_test, self.y_test)
                    
                    # Log to MLflow
                    mlflow.log_params(best_params)
                    mlflow.log_metric("cv_score", cv_score)
                    for metric, value in train_metrics.items():
                        mlflow.log_metric(f"train_{metric}", value)
                    for metric, value in test_metrics.items():
                        mlflow.log_metric(f"test_{metric}", value)
                    
                    # Save model artifact
                    model_path = os.path.join(self.models_dir, f"{model_name}_grid.joblib")
                    joblib.dump(best_model, model_path)
                    mlflow.log_artifact(model_path)
                    
                    # Store results
                    model_results['grid_search'] = {
                        'model': best_model,
                        'params': best_params,
                        'cv_score': cv_score,
                        'train_metrics': train_metrics,
                        'test_metrics': test_metrics,
                        'model_path': model_path
                    }
                    
                    print(f"✅ Grid Search completed for {model_name}")
                    ModelEvaluator.print_metrics(test_metrics, f"{model_name} (Grid Search)")
                    
            except Exception as e:
                print(f"❌ Grid Search failed for {model_name}: {e}")
        
        # Random Search
        if use_random_search:
            print(f"\n🎲 Running Random Search for {model_name}...")
            try:
                with mlflow.start_run(run_name=f"{model_name}_random_search", nested=True):
                    best_model, best_params, cv_score = self.tuner.random_search(
                        base_model, self.X_train, self.y_train, param_grid, n_iter=10
                    )
                    
                    # Evaluate
                    train_metrics = ModelEvaluator.evaluate_model(best_model, self.X_train, self.y_train)
                    test_metrics = ModelEvaluator.evaluate_model(best_model, self.X_test, self.y_test)
                    
                    # Log to MLflow
                    mlflow.log_params(best_params)
                    mlflow.log_metric("cv_score", cv_score)
                    for metric, value in train_metrics.items():
                        mlflow.log_metric(f"train_{metric}", value)
                    for metric, value in test_metrics.items():
                        mlflow.log_metric(f"test_{metric}", value)
                    
                    # Save model artifact
                    model_path = os.path.join(self.models_dir, f"{model_name}_random.joblib")
                    joblib.dump(best_model, model_path)
                    mlflow.log_artifact(model_path)
                    
                    # Store results
                    model_results['random_search'] = {
                        'model': best_model,
                        'params': best_params,
                        'cv_score': cv_score,
                        'train_metrics': train_metrics,
                        'test_metrics': test_metrics,
                        'model_path': model_path
                    }
                    
                    print(f"✅ Random Search completed for {model_name}")
                    ModelEvaluator.print_metrics(test_metrics, f"{model_name} (Random Search)")
                    
            except Exception as e:
                print(f"❌ Random Search failed for {model_name}: {e}")
        
        self.models[model_name] = model_results
        return model_results
    
    def train_all_models(self, model_names: list = None, 
                        use_grid_search: bool = True, 
                        use_random_search: bool = True):
        """
        Train multiple models.
        
        Args:
            model_names (list): List of model names to train
            use_grid_search (bool): Whether to use Grid Search
            use_random_search (bool): Whether to use Random Search
            
        Returns:
            dict: All training results
        """
        if model_names is None:
            model_names = ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm']
        
        print(f"\n🎯 Training {len(model_names)} models: {model_names}")
        
        all_results = {}
        for model_name in model_names:
            try:
                results = self.train_model(
                    model_name=model_name,
                    use_grid_search=use_grid_search,
                    use_random_search=use_random_search
                )
                all_results[model_name] = results
            except Exception as e:
                print(f"❌ Failed to train {model_name}: {e}")
        
        self.results = all_results
        return all_results
    
    def select_best_model(self, metric: str = 'roc_auc') -> tuple:
        """
        Select the best model based on test performance.
        
        Args:
            metric (str): Metric to use for comparison
            
        Returns:
            tuple: (best_model_name, best_model_info, best_score)
        """
        best_score = -1
        best_model_name = None
        best_model_info = None
        best_search_type = None
        
        for model_name, model_results in self.results.items():
            for search_type, results in model_results.items():
                if 'test_metrics' in results and metric in results['test_metrics']:
                    score = results['test_metrics'][metric]
                    if score > best_score:
                        best_score = score
                        best_model_name = model_name
                        best_model_info = results
                        best_search_type = search_type
        
        if best_model_name is None:
            raise ValueError("No trained models found")
        
        print(f"\n🏆 Best Model: {best_model_name} ({best_search_type})")
        print(f"📈 Best {metric}: {best_score:.4f}")
        
        return best_model_name, best_model_info, best_search_type, best_score
    
    def register_best_model(self, metric: str = 'roc_auc', stage: str = "Staging"):
        """
        Register the best model in MLflow Model Registry.
        
        Args:
            metric (str): Metric to use for comparison
            stage (str): Model stage in registry
            
        Returns:
            str: Model URI
        """
        # Select best model
        best_model_name, best_model_info, search_type, best_score = self.select_best_model(metric)
        
        # Start a new run for the best model
        with mlflow.start_run(run_name=f"best_model_{best_model_name}_{search_type}"):
            # Log data info
            mlflow.log_params(self.data_info)
            mlflow.log_param("best_model_name", best_model_name)
            mlflow.log_param("search_type", search_type)
            mlflow.log_param("metric_used", metric)
            
            # Log metrics
            mlflow.log_metric(f"best_{metric}", best_score)
            for metric_name, metric_value in best_model_info['test_metrics'].items():
                mlflow.log_metric(f"test_{metric_name}", metric_value)
            
            if 'train_metrics' in best_model_info:
                for metric_name, metric_value in best_model_info['train_metrics'].items():
                    mlflow.log_metric(f"train_{metric_name}", metric_value)
            
            # Log model
            best_model = best_model_info['model']
            mlflow.sklearn.log_model(best_model, "model")
            
            # Log parameters
            if 'params' in best_model_info:
                mlflow.log_params(best_model_info['params'])
            
            # Register the model
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
            
            try:
                registered_model = mlflow.register_model(model_uri, "credit_risk_model")
                
                # Transition to specified stage
                client = mlflow.tracking.MlflowClient()
                client.transition_model_version_stage(
                    name="credit_risk_model",
                    version=registered_model.version,
                    stage=stage
                )
                
                print(f"\n✅ Model registered successfully!")
                print(f"   Name: {registered_model.name}")
                print(f"   Version: {registered_model.version}")
                print(f"   Stage: {stage}")
                print(f"   URI: {model_uri}")
                
            except Exception as e:
                print(f"\n⚠️  Could not register model in MLflow registry: {e}")
                print(f"   Model URI: {model_uri}")
            
            # Save best model locally
            best_model_path = os.path.join(self.models_dir, "best_model.joblib")
            joblib.dump(best_model, best_model_path)
            print(f"   Local copy: {best_model_path}")
            
            return model_uri
    
    def generate_comparison_report(self):
        """Generate comparison report of all models."""
        if not self.results:
            print("No results to compare")
            return
        
        print(f"\n{'='*60}")
        print("MODEL COMPARISON REPORT")
        print(f"{'='*60}")
        
        comparison_data = []
        
        for model_name, model_results in self.results.items():
            for search_type, results in model_results.items():
                if 'test_metrics' in results:
                    row = {
                        'Model': model_name,
                        'Search Type': search_type,
                        'ROC-AUC': results['test_metrics'].get('roc_auc', 0),
                        'Accuracy': results['test_metrics'].get('accuracy', 0),
                        'Precision': results['test_metrics'].get('precision', 0),
                        'Recall': results['test_metrics'].get('recall', 0),
                        'F1 Score': results['test_metrics'].get('f1_score', 0)
                    }
                    comparison_data.append(row)
        
        # Create DataFrame and sort by ROC-AUC
        df = pd.DataFrame(comparison_data)
        if not df.empty:
            df = df.sort_values('ROC-AUC', ascending=False)
            print(df.to_string(index=False))
            
            # Save to CSV
            report_path = os.path.join(self.models_dir, "model_comparison.csv")
            df.to_csv(report_path, index=False)
            print(f"\n📄 Comparison report saved to: {report_path}")
    
    def run(self, data_path: str = None, target_column: str = 'is_high_risk'):
        """
        Run the complete training pipeline.
        
        Args:
            data_path (str): Path to processed data
            target_column (str): Name of target column
        """
        print(f"\n{'='*60}")
        print("CREDIT RISK MODEL TRAINING PIPELINE")
        print(f"{'='*60}")
        
        # Step 1: Load data
        print("\n📁 STEP 1: Loading data...")
        self.load_data(data_path=data_path, target_column=target_column)
        
        # Step 2: Train models
        print("\n🤖 STEP 2: Training models...")
        self.train_all_models(
            model_names=['logistic_regression', 'random_forest', 'xgboost', 'lightgbm'],
            use_grid_search=True,
            use_random_search=True
        )
        
        # Step 3: Generate comparison
        print("\n📊 STEP 3: Generating model comparison...")
        self.generate_comparison_report()
        
        # Step 4: Register best model
        print("\n🏆 STEP 4: Registering best model...")
        model_uri = self.register_best_model(metric='roc_auc', stage="Staging")
        
        # Step 5: Summary
        print(f"\n{'='*60}")
        print("TRAINING PIPELINE COMPLETE! 🎉")
        print(f"{'='*60}")
        print(f"\n📋 Next steps:")
        print(f"1. View MLflow UI: mlflow ui --backend-store-uri {self.mlflow_dir}")
        print(f"2. Evaluate model: python -m src.evaluate")
        print(f"3. Test predictions: python -m src.predict")
        print(f"\n📁 Outputs:")
        print(f"   • Models saved to: {self.models_dir}")
        print(f"   • MLflow runs: {self.mlflow_dir}")
        print(f"   • Best model URI: {model_uri}")
        print(f"{'='*60}")


def main():
    """Main function to run the training pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train credit risk models with MLflow tracking')
    parser.add_argument('--data-path', type=str, help='Path to processed data')
    parser.add_argument('--target-column', type=str, default='is_high_risk', 
                       help='Name of target column')
    parser.add_argument('--experiment-name', type=str, default='credit_risk_modeling',
                       help='MLflow experiment name')
    
    args = parser.parse_args()
    
    # Create and run trainer
    trainer = MLflowTrainer(experiment_name=args.experiment_name)
    trainer.run(
        data_path=args.data_path,
        target_column=args.target_column
    )


if __name__ == "__main__":
    main()