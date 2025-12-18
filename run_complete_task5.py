#!/usr/bin/env python3
"""
Complete Task 5 Runner Script
Runs the full Task 5 pipeline including:
1. Data preparation
2. Model training
3. Hyperparameter tuning
4. MLflow tracking
5. Model evaluation
6. Unit tests
"""

import os
import sys
import subprocess
import argparse
import pandas as pd
import numpy as np

def setup_environment():
    """Setup directories and environment"""
    directories = [
        'models',
        'mlruns',
        'data/processed',
        'tests'
    ]
    
    print("Setting up environment...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  ✓ {directory}")
    
    # Create sample data if needed
    sample_data_path = 'data/processed/processed_data.csv'
    if not os.path.exists(sample_data_path):
        print(f"\nCreating sample data at {sample_data_path}...")
        os.makedirs('data/processed', exist_ok=True)
        
        # Create realistic sample data
        np.random.seed(42)
        n_samples = 1000
        n_features = 10
        
        data = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i+1}' for i in range(n_features)]
        )
        
        # Create target with some pattern
        data['is_high_risk'] = (
            (data['feature_1'] > 0.5) | 
            (data['feature_2'] < -0.5) |
            ((data['feature_3'] > 0.3) & (data['feature_4'] < -0.3))
        ).astype(int)
        
        data.to_csv(sample_data_path, index=False)
        print(f"  ✓ Created sample data with {n_samples} samples, {n_features} features")

def run_unit_tests():
    """Run the unit tests"""
    print("\n" + "="*60)
    print("Running Unit Tests for Task 5")
    print("="*60)
    
    # Run the test file
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', 'tests/test_train_task5.py', '-v'],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return result.returncode == 0

def run_training_pipeline():
    """Run the main training pipeline"""
    print("\n" + "="*60)
    print("Running Training Pipeline")
    print("="*60)
    
    try:
        # Import and run the trainer
        from src.train import MLflowTrainer
        
        trainer = MLflowTrainer(experiment_name="task5_credit_risk_modeling")
        
        # Run with sample data
        data_path = 'data/processed/processed_data.csv'
        if os.path.exists(data_path):
            trainer.run(data_path=data_path, target_column='is_high_risk')
        else:
            # Run with default settings
            trainer.run()
            
        return True
    except Exception as e:
        print(f"Training pipeline failed: {e}")
        return False

def verify_outputs():
    """Verify that expected outputs were created"""
    print("\n" + "="*60)
    print("Verifying Outputs")
    print("="*60)
    
    expected_files = [
        'models/best_model.joblib',
        'models/logistic_regression_grid.joblib',
        'models/random_forest_grid.joblib',
        'mlruns/0/meta.yaml'
    ]
    
    all_exist = True
    for file_path in expected_files:
        if os.path.exists(file_path):
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} (missing)")
            all_exist = False
    
    return all_exist

def create_task5_readme():
    """Create a README file documenting Task 5 completion"""
    readme_content = """# Task 5: Model Training with MLflow Tracking - COMPLETED ✅"""