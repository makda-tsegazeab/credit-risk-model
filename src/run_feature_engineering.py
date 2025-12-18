# tests/test_data_processing.py
import pytest
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_processing import DataPreprocessor, create_feature_engineering_pipeline, engineer_features

@pytest.fixture
def sample_data():
    """Create minimal sample data"""
    dates = pd.date_range('2019-01-01', periods=100, freq='D')
    
    data = {
        'TransactionId': [f'TransactionId_{i}' for i in range(100)],
        'CustomerId': ['CustomerId_1'] * 50 + ['CustomerId_2'] * 50,
        'TransactionStartTime': dates.tolist(),
        'Amount': np.random.normal(1000, 500, 100),
        'Value': np.random.normal(1200, 600, 100),
        'ProductCategory': ['airtime'] * 40 + ['financial_services'] * 40 + ['utility_bill'] * 20,
        'ProviderId': ['ProviderId_1'] * 60 + ['ProviderId_2'] * 40,
        'ChannelId': ['ChannelId_1'] * 70 + ['ChannelId_2'] * 30,
        'CountryCode': [256] * 100,
        'PricingStrategy': [2] * 100
    }
    
    return pd.DataFrame(data)

def test_data_preprocessor(sample_data):
    """Test data preprocessing"""
    preprocessor = DataPreprocessor()
    transformed = preprocessor.transform(sample_data)
    
    assert 'transaction_hour' in transformed.columns
    assert 'transaction_day' in transformed.columns
    assert 'transaction_month' in transformed.columns
    assert 'transaction_year' in transformed.columns

def test_feature_engineering_pipeline(sample_data):
    """Test pipeline creation and transformation"""
    pipeline = create_feature_engineering_pipeline()
    X_processed = pipeline.fit_transform(sample_data)
    
    # Check shape
    assert X_processed.shape[0] == sample_data.shape[0]
    assert X_processed.shape[1] > 0
    
    # Check no NaN values
    assert not np.isnan(X_processed).any()

def test_engineer_features(sample_data):
    """Test main feature engineering function"""
    X_processed, feature_names, customer_df = engineer_features(sample_data)
    
    # Check processed features
    assert X_processed.shape[0] == sample_data.shape[0]
    assert len(feature_names) == X_processed.shape[1]
    
    # Check customer features
    assert 'CustomerId' in customer_df.columns
    assert 'recency' in customer_df.columns
    assert 'frequency' in customer_df.columns
    assert 'monetary' in customer_df.columns
    
    # Check we have all customers
    assert len(customer_df) == sample_data['CustomerId'].nunique()