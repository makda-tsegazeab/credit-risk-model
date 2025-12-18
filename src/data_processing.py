# src/data_processing.py

"""
Feature Engineering Pipeline for Credit Risk Model
Transforms raw transaction data into features for credit risk prediction
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import logging
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataLoader(BaseEstimator, TransformerMixin):
    """Load and preprocess raw transaction data"""
    
    def __init__(self, filepath=None):
        self.filepath = filepath
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X=None):
        """Load data from file or use provided dataframe"""
        if X is None:
            if self.filepath:
                logger.info(f"Loading data from {self.filepath}")
                df = pd.read_csv(self.filepath)
            else:
                raise ValueError("Either provide data or filepath")
        else:
            df = X.copy()
            
        # Initial preprocessing
        df = self._preprocess_data(df)
        return df
    
    def _preprocess_data(self, df):
        """Basic data preprocessing"""
        logger.info("Preprocessing data...")
        
        # Convert TransactionStartTime to datetime
        if 'TransactionStartTime' in df.columns:
            df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        return df
    
    def _handle_missing_values(self, df):
        """Handle missing values appropriately"""
        # For Amount/Value, fill with median
        if 'Amount' in df.columns:
            df['Amount'] = df['Amount'].fillna(df['Amount'].median())
        if 'Value' in df.columns:
            df['Value'] = df['Value'].fillna(df['Value'].median())
            
        # For categorical, fill with mode
        categorical_cols = ['ProductCategory', 'ChannelId', 'ProviderId']
        for col in categorical_cols:
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode()[0])
                
        return df


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Create engineered features from raw data"""
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Create all engineered features"""
        logger.info("Engineering features...")
        df = X.copy()
        
        # 1. Temporal Features
        df = self._create_temporal_features(df)
        
        # 2. Customer-level RFM Features
        df = self._create_rfm_features(df)
        
        # 3. Behavioral Features
        df = self._create_behavioral_features(df)
        
        # 4. Statistical Features
        df = self._create_statistical_features(df)
        
        # 5. Interaction Features
        df = self._create_interaction_features(df)
        
        logger.info(f"Engineered features complete. Total features: {len(df.columns)}")
        return df
    
    def _create_temporal_features(self, df):
        """Create time-based features"""
        if 'TransactionStartTime' in df.columns:
            # Extract basic time features
            df['TransactionHour'] = df['TransactionStartTime'].dt.hour
            df['TransactionDay'] = df['TransactionStartTime'].dt.day
            df['TransactionMonth'] = df['TransactionStartTime'].dt.month
            df['TransactionYear'] = df['TransactionStartTime'].dt.year
            df['TransactionDayOfWeek'] = df['TransactionStartTime'].dt.dayofweek
            df['TransactionIsWeekend'] = df['TransactionDayOfWeek'].isin([5, 6]).astype(int)
            
            # Time since first transaction (customer engagement duration)
            if 'CustomerId' in df.columns:
                first_transaction = df.groupby('CustomerId')['TransactionStartTime'].min()
                df['DaysSinceFirstTransaction'] = df.apply(
                    lambda row: (row['TransactionStartTime'] - first_transaction[row['CustomerId']]).days,
                    axis=1
                )
                
        return df
    
    def _create_rfm_features(self, df):
        """Create Recency, Frequency, Monetary features per customer"""
        if 'CustomerId' in df.columns:
            # Calculate snapshot date (most recent transaction + 1 day)
            snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)
            
            # Group by customer
            rfm = df.groupby('CustomerId').agg({
                'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,  # Recency
                'TransactionId': 'count',  # Frequency
                'Amount': ['sum', 'mean', 'std']  # Monetary
            })
            
            # Flatten multi-index columns
            rfm.columns = ['Recency', 'Frequency', 'MonetarySum', 'MonetaryMean', 'MonetaryStd']
            
            # Merge back to original dataframe
            df = df.merge(rfm, on='CustomerId', how='left')
            
        return df
    
    def _create_behavioral_features(self, df):
        """Create behavioral pattern features"""
        if 'CustomerId' in df.columns:
            # Number of unique product categories per customer
            category_counts = df.groupby('CustomerId')['ProductCategory'].nunique().reset_index()
            category_counts.columns = ['CustomerId', 'UniqueCategoriesCount']
            df = df.merge(category_counts, on='CustomerId', how='left')
            
            # Preferred channel
            preferred_channel = df.groupby(['CustomerId', 'ChannelId']).size().reset_index(name='count')
            preferred_channel = preferred_channel.loc[preferred_channel.groupby('CustomerId')['count'].idxmax()]
            preferred_channel = preferred_channel[['CustomerId', 'ChannelId']].rename(columns={'ChannelId': 'PreferredChannel'})
            df = df.merge(preferred_channel, on='CustomerId', how='left')
            
            # Transaction time variability
            hour_std = df.groupby('CustomerId')['TransactionHour'].std().reset_index()
            hour_std.columns = ['CustomerId', 'TransactionHourStd']
            df = df.merge(hour_std, on='CustomerId', how='left')
            
        return df
    
    def _create_statistical_features(self, df):
        """Create statistical features"""
        if 'CustomerId' in df.columns:
            # Customer-level aggregates
            customer_stats = df.groupby('CustomerId').agg({
                'Amount': ['mean', 'std', 'min', 'max', 'skew'],
                'Value': ['mean', 'std']
            })
            
            # Flatten columns
            customer_stats.columns = [f'{col[0]}_{col[1]}' for col in customer_stats.columns]
            customer_stats = customer_stats.reset_index()
            
            # Merge back
            df = df.merge(customer_stats, on='CustomerId', how='left')
            
        return df
    
    def _create_interaction_features(self, df):
        """Create interaction features"""
        # Amount relative to customer average
        if 'Amount' in df.columns and 'Amount_mean' in df.columns:
            df['AmountRelativeToMean'] = df['Amount'] / df['Amount_mean']
            
        # Time of day × day of week
        if 'TransactionHour' in df.columns and 'TransactionDayOfWeek' in df.columns:
            df['HourDayInteraction'] = df['TransactionHour'] * df['TransactionDayOfWeek']
            
        return df


class TargetVariableCreator(BaseEstimator, TransformerMixin):
    """Create proxy target variable using RFM clustering"""
    
    def __init__(self, n_clusters=3, high_risk_cluster=0):
        self.n_clusters = n_clusters
        self.high_risk_cluster = high_risk_cluster
        from sklearn.cluster import KMeans
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        
    def fit(self, X, y=None):
        """Fit KMeans on RFM features"""
        # Check if required columns exist
        required_cols = ['Recency', 'Frequency', 'MonetarySum']
        missing_cols = [col for col in required_cols if col not in X.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns for clustering: {missing_cols}")
            
        # Prepare RFM features
        rfm_features = X[['CustomerId'] + required_cols].drop_duplicates()
        
        # Scale features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm_features[required_cols])
        
        # Fit KMeans
        self.kmeans.fit(rfm_scaled)
        self.scaler_ = scaler
        self.customer_labels_ = pd.DataFrame({
            'CustomerId': rfm_features['CustomerId'],
            'RFM_Cluster': self.kmeans.labels_
        })
        
        return self
    
    def transform(self, X):
        """Add target variable to dataframe"""
        logger.info("Creating proxy target variable...")
        df = X.copy()
        
        # Merge cluster labels
        df = df.merge(self.customer_labels_, on='CustomerId', how='left')
        
        # Create binary target: 1 for high risk, 0 otherwise
        # Assuming cluster 0 is high-risk (least engaged)
        df['is_high_risk'] = (df['RFM_Cluster'] == self.high_risk_cluster).astype(int)
        
        logger.info(f"Target variable created. High risk customers: {df['is_high_risk'].sum()}")
        logger.info(f"Risk distribution: {df['is_high_risk'].value_counts().to_dict()}")
        
        return df


class DataPreprocessor(BaseEstimator, TransformerMixin):
    """Final preprocessing for model training"""
    
    def __init__(self, target_col='is_high_risk'):
        self.target_col = target_col
        self.numerical_features = None
        self.categorical_features = None
        
    def fit(self, X, y=None):
        """Identify feature types"""
        # Exclude non-feature columns
        exclude_cols = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 
                       'CustomerId', 'TransactionStartTime', self.target_col, 'RFM_Cluster']
        
        feature_cols = [col for col in X.columns if col not in exclude_cols]
        
        # Identify numerical and categorical features
        self.numerical_features = X[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = X[feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()
        
        logger.info(f"Identified {len(self.numerical_features)} numerical features")
        logger.info(f"Identified {len(self.categorical_features)} categorical features")
        
        # Create preprocessing pipeline
        self.preprocessor_ = self._create_preprocessing_pipeline()
        
        # Fit preprocessor
        X_features = X[self.numerical_features + self.categorical_features]
        self.preprocessor_.fit(X_features)
        
        return self
    
    def _create_preprocessing_pipeline(self):
        """Create sklearn preprocessing pipeline"""
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ],
            remainder='drop'
        )
        
        return preprocessor
    
    def transform(self, X):
        """Apply preprocessing and return features and target"""
        logger.info("Preprocessing data for modeling...")
        
        # Extract target if it exists
        if self.target_col in X.columns:
            y = X[self.target_col].values
        else:
            y = None
            
        # Apply preprocessing to features
        X_features = X[self.numerical_features + self.categorical_features]
        X_processed = self.preprocessor_.transform(X_features)
        
        # Get feature names
        feature_names = self._get_feature_names()
        
        logger.info(f"Preprocessing complete. Features shape: {X_processed.shape}")
        
        return X_processed, y, feature_names
    
    def _get_feature_names(self):
        """Get feature names after preprocessing"""
        # Numerical features
        num_features = self.numerical_features
        
        # Categorical features after one-hot encoding
        cat_features = []
        if self.categorical_features:
            ohe = self.preprocessor_.named_transformers_['cat'].named_steps['onehot']
            cat_features = ohe.get_feature_names_out(self.categorical_features)
        
        return list(num_features) + list(cat_features)


def create_feature_pipeline():
    """Create complete feature engineering pipeline"""
    pipeline = Pipeline([
        ('loader', DataLoader()),
        ('engineer', FeatureEngineer()),
        ('target_creator', TargetVariableCreator()),
        ('preprocessor', DataPreprocessor())
    ])
    
    return pipeline


def main():
    """Main function to test the pipeline"""
    logger.info("Testing feature engineering pipeline...")
    
    # Create sample data
    sample_data = pd.DataFrame({
        'TransactionId': ['T1', 'T2', 'T3', 'T4'],
        'CustomerId': ['C1', 'C1', 'C2', 'C2'],
        'Amount': [100, 200, 50, 150],
        'Value': [100, 200, 50, 150],
        'TransactionStartTime': pd.date_range('2024-01-01', periods=4, freq='H'),
        'ProductCategory': ['A', 'B', 'A', 'C'],
        'ChannelId': ['Web', 'Web', 'Mobile', 'Web']
    })
    
    # Test pipeline
    pipeline = create_feature_pipeline()
    X_processed, y, feature_names = pipeline.fit_transform(sample_data)
    
    logger.info("Pipeline test successful!")
    logger.info(f"Processed features shape: {X_processed.shape}")
    logger.info(f"Target shape: {y.shape if y is not None else 'None'}")
    logger.info(f"Number of features: {len(feature_names)}")
    
    return pipeline


if __name__ == "__main__":
    pipeline = main()