# src/data_processing.py - COMPLETE FIXED VERSION
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

class DataPreprocessor(BaseEstimator, TransformerMixin):
    """Minimal data preprocessing for credit risk"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Convert date
        if 'TransactionStartTime' in X.columns:
            X['TransactionStartTime'] = pd.to_datetime(X['TransactionStartTime'])
            
            # Extract basic temporal features
            X['transaction_hour'] = X['TransactionStartTime'].dt.hour
            X['transaction_day'] = X['TransactionStartTime'].dt.day
            X['transaction_month'] = X['TransactionStartTime'].dt.month
            X['transaction_year'] = X['TransactionStartTime'].dt.year
        
        return X

def create_feature_engineering_pipeline():
    """Create the minimal feature engineering pipeline"""
    
    # Define column types
    numerical_features = ['Amount', 'Value', 'PricingStrategy', 'CountryCode']
    categorical_features = ['ProviderId', 'ProductCategory', 'ChannelId']
    
    # Numerical preprocessing
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical preprocessing
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Create full pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', DataPreprocessor()),
        ('feature_processor', preprocessor)
    ])
    
    return pipeline, numerical_features, categorical_features

def engineer_features(df):
    """Main function to engineer features"""
    
    print("Starting feature engineering...")
    
    # Create customer-level features (for Task 4)
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    snapshot_date = df['TransactionStartTime'].max()
    
    # Calculate RFM features
    customer_features = []
    
    for customer_id, group in df.groupby('CustomerId'):
        # Recency
        recency = (snapshot_date - group['TransactionStartTime'].max()).days
        
        # Frequency
        frequency = len(group)
        
        # Monetary (only positive amounts)
        monetary = group[group['Amount'] > 0]['Amount'].sum()
        
        # Additional features
        avg_amount = group['Amount'].mean()
        std_amount = group['Amount'].std()
        max_amount = group['Amount'].max()
        min_amount = group['Amount'].min()
        
        customer_features.append({
            'CustomerId': customer_id,
            'recency': recency,
            'frequency': frequency,
            'monetary': monetary,
            'avg_amount': avg_amount,
            'std_amount': std_amount if not pd.isna(std_amount) else 0,
            'max_amount': max_amount,
            'min_amount': min_amount,
            'transaction_count': frequency,
            'total_amount': group['Amount'].sum()
        })
    
    # Create customer dataframe
    customer_df = pd.DataFrame(customer_features)
    
    # Save for Task 4
    import os
    os.makedirs('data/processed', exist_ok=True)
    customer_df.to_csv('data/processed/customer_features.csv', index=False)
    
    # Create pipeline for transaction-level features
    pipeline, numerical_features, categorical_features = create_feature_engineering_pipeline()
    
    # Fit and transform
    X_processed = pipeline.fit_transform(df)
    
    # Get feature names
    feature_names = []
    
    # Numerical features
    feature_names.extend(numerical_features)
    
    # Categorical features (from one-hot encoding)
    categorical_encoder = pipeline.named_steps['feature_processor'].transformers_[1][1].named_steps['onehot']
    cat_features = categorical_features
    cat_feature_names = categorical_encoder.get_feature_names_out(cat_features)
    feature_names.extend(cat_feature_names)
    
    print(f"Feature engineering complete. Shape: {X_processed.shape}")
    print(f"Number of features: {len(feature_names)}")
    
    # Save processed features
    np.save('data/processed/features.npy', X_processed)
    pd.Series(feature_names).to_csv('data/processed/feature_names.csv', index=False)
    
    return X_processed, feature_names, customer_df

if __name__ == "__main__":
    # Load data
    df = pd.read_csv('data/raw/data.csv')
    
    # Engineer features
    X_processed, feature_names, customer_df = engineer_features(df)
    
    print(f"\n✅ Task 3 Complete!")
    print(f"📊 Original data: {df.shape}")
    print(f"📈 Processed features: {X_processed.shape}")
    print(f"👥 Customer features: {customer_df.shape}")
    print(f"\n📁 Outputs saved to data/processed/")