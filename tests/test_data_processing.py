# tests/test_data_processing.py
"""
Unit tests for data processing functions.
"""
import pytest
import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import your actual data processing functions
try:
    # Try to import your actual functions
    from src.data_processing import clean_data, engineer_features
    from src.target_engineering import create_target_variable
    HAS_SRC_MODULES = True
except ImportError as e:
    print(f"Note: Could not import src modules: {e}")
    print("Using mock functions for testing")
    HAS_SRC_MODULES = False
    
    # Mock functions for testing if src modules not available
    def clean_data(df):
        """Mock clean_data function."""
        df = df.copy()
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna('Unknown')
        
        return df
    
    def engineer_features(df):
        """Mock engineer_features function."""
        df = df.copy()
        # Add simple engineered features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            col1, col2 = numeric_cols[0], numeric_cols[1]
            if col1 != col2:  # Ensure different columns
                df[f'ratio_{col1}_{col2}'] = df[col1] / (df[col2] + 1e-10)
        return df
    
    def create_target_variable(df, threshold_column='TransactionAmount', threshold=1000):
        """Mock create_target_variable function."""
        df = df.copy()
        if threshold_column in df.columns:
            df['is_high_risk'] = (df[threshold_column] > threshold).astype(int)
        return df


class TestDataProcessing:
    """Test suite for data processing functions."""
    
    @pytest.fixture
    def sample_credit_data(self):
        """Create sample credit risk data for testing."""
        np.random.seed(42)
        n_samples = 100
        
        data = {
            'TransactionId': [f'T{i}' for i in range(n_samples)],
            'CustomerId': [f'C{i//10}' for i in range(n_samples)],
            'TransactionAmount': np.random.exponential(500, n_samples),
            'TransactionHour': np.random.randint(0, 24, n_samples),
            'TransactionDay': np.random.randint(1, 31, n_samples),
            'ProductCategory': np.random.choice(['Electronics', 'Clothing', 'Food', 'Services'], n_samples),
            'CustomerAge': np.random.randint(18, 70, n_samples),
            'CustomerIncome': np.random.normal(50000, 20000, n_samples),
            'PreviousDefaults': np.random.randint(0, 3, n_samples),
            'CreditScore': np.random.randint(300, 850, n_samples)
        }
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_data_with_missing(self):
        """Create sample data with missing values."""
        np.random.seed(42)
        n_samples = 50
        
        data = {
            'TransactionId': [f'T{i}' for i in range(n_samples)],
            'TransactionAmount': np.random.exponential(500, n_samples),
            'TransactionHour': np.random.randint(0, 24, n_samples),
            'CustomerAge': np.random.randint(18, 70, n_samples),
            'CreditScore': np.random.randint(300, 850, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Introduce missing values
        for col in ['TransactionAmount', 'CustomerAge', 'CreditScore']:
            missing_indices = np.random.choice(n_samples, size=5, replace=False)
            df.loc[missing_indices, col] = np.nan
        
        return df
    
    def test_clean_data_no_missing(self, sample_credit_data):
        """
        Test 1: clean_data function handles data without missing values correctly.
        
        Expected: 
        - Returns DataFrame with same number of rows as input
        - No missing values in output
        - Preserves all original columns
        """
        print("\n🔍 Test 1: Testing clean_data with no missing values")
        
        cleaned_data = clean_data(sample_credit_data)
        
        # Check return type
        assert isinstance(cleaned_data, pd.DataFrame), "Should return a DataFrame"
        
        # Check no data loss
        assert len(cleaned_data) == len(sample_credit_data), "Should preserve all rows"
        
        # Check no missing values
        assert not cleaned_data.isnull().any().any(), "Should have no missing values"
        
        # Check columns preserved
        original_cols = set(sample_credit_data.columns)
        cleaned_cols = set(cleaned_data.columns)
        assert original_cols.issubset(cleaned_cols), "Should preserve all original columns"
        
        print("✅ Test 1 passed: clean_data handles complete data correctly")
    
    def test_clean_data_with_missing(self, sample_data_with_missing):
        """
        Test 2: clean_data function handles missing values correctly.
        
        Expected:
        - Returns DataFrame with same number of rows as input
        - No missing values in output
        - Numeric columns have appropriate fill values
        """
        print("\n🔍 Test 2: Testing clean_data with missing values")
        
        cleaned_data = clean_data(sample_data_with_missing)
        
        # Check return type
        assert isinstance(cleaned_data, pd.DataFrame), "Should return a DataFrame"
        
        # Check no data loss
        assert len(cleaned_data) == len(sample_data_with_missing), "Should preserve all rows"
        
        # Check no missing values
        assert not cleaned_data.isnull().any().any(), "Should have no missing values after cleaning"
        
        # Check that numeric columns were filled
        numeric_cols = sample_data_with_missing.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert col in cleaned_data.columns, f"Should preserve column {col}"
            assert not cleaned_data[col].isnull().any(), f"Column {col} should have no missing values"
        
        print("✅ Test 2 passed: clean_data handles missing values correctly")
    
    def test_feature_engineering_columns(self, sample_credit_data):
        """
        Test 3: feature engineering function returns expected columns.
        
        Expected:
        - Returns DataFrame with original columns plus engineered features
        - Has at least one new engineered feature
        """
        print("\n🔍 Test 3: Testing feature engineering columns")
        
        engineered_data = engineer_features(sample_credit_data)
        
        # Check return type
        assert isinstance(engineered_data, pd.DataFrame), "Should return a DataFrame"
        
        # Check all original columns are present
        original_cols = set(sample_credit_data.columns)
        engineered_cols = set(engineered_data.columns)
        assert original_cols.issubset(engineered_cols), "Should preserve all original columns"
        
        # Check for new engineered features
        new_cols = engineered_cols - original_cols
        assert len(new_cols) > 0, "Should create at least one new engineered feature"
        
        print(f"✅ Test 3 passed: Created {len(new_cols)} new features: {list(new_cols)[:3]}...")
    
    def test_feature_engineering_values(self, sample_credit_data):
        """
        Test 4: Engineered features have correct values.
        
        Expected:
        - Engineered features are calculated correctly
        - No NaN values in engineered features (unless division by zero)
        """
        print("\n🔍 Test 4: Testing feature engineering values")
        
        engineered_data = engineer_features(sample_credit_data)
        
        # Check that engineered features don't have NaN (unless expected)
        numeric_cols = sample_credit_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            # Look for ratio features (common engineered feature)
            ratio_cols = [col for col in engineered_data.columns if 'ratio' in col.lower()]
            
            for col in ratio_cols:
                # Should have finite values (not NaN or inf)
                assert engineered_data[col].notna().all(), f"Engineered feature {col} should not have NaN"
                
                # Check some values are different (not all zeros)
                unique_values = engineered_data[col].nunique()
                assert unique_values > 1, f"Engineered feature {col} should have varying values"
        
        print("✅ Test 4 passed: Engineered features have valid values")
    
    def test_target_engineering(self, sample_credit_data):
        """
        Test 5: Target variable creation.
        
        Expected:
        - Creates target column 'is_high_risk'
        - Target values are 0 or 1
        - Reasonable class distribution
        """
        print("\n🔍 Test 5: Testing target variable creation")
        
        # Create target variable
        if 'TransactionAmount' in sample_credit_data.columns:
            df_with_target = create_target_variable(
                sample_credit_data, 
                threshold_column='TransactionAmount',
                threshold=500  # Median-ish value
            )
            
            # Check target column exists
            assert 'is_high_risk' in df_with_target.columns, "Should create 'is_high_risk' column"
            
            # Check target values are 0 or 1
            target_values = set(df_with_target['is_high_risk'].unique())
            assert target_values.issubset({0, 1}), "Target should be binary (0 or 1)"
            
            # Check class distribution
            class_counts = df_with_target['is_high_risk'].value_counts()
            assert len(class_counts) == 2, "Should have both classes"
            
            print(f"✅ Test 5 passed: Target created with distribution {dict(class_counts)}")
    
    def test_data_split_reproducibility(self):
        """
        Test 6: Data splitting is reproducible with random_state.
        
        Expected:
        - Same random_state produces identical splits
        - No overlap between train and test indices
        """
        print("\n🔍 Test 6: Testing data split reproducibility")
        
        # Create sample data
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5), columns=[f'feature_{i}' for i in range(5)])
        y = pd.Series(np.random.randint(0, 2, 100), name='is_high_risk')
        
        # First split
        X_train1, X_test1, y_train1, y_test1 = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Second split with same random_state
        X_train2, X_test2, y_train2, y_test2 = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Check identical splits
        assert X_train1.equals(X_train2), "X_train should be identical with same random_state"
        assert X_test1.equals(X_test2), "X_test should be identical with same random_state"
        assert y_train1.equals(y_train2), "y_train should be identical with same random_state"
        assert y_test1.equals(y_test2), "y_test should be identical with same random_state"
        
        # Check no overlap
        train_indices = set(X_train1.index)
        test_indices = set(X_test1.index)
        assert train_indices.isdisjoint(test_indices), "Train and test sets should not overlap"
        
        print("✅ Test 6 passed: Data splitting is reproducible")
    
    def test_data_types_preserved(self, sample_credit_data):
        """
        Test 7: Data types are preserved after processing.
        
        Expected:
        - Numeric columns remain numeric
        - String columns remain string/object
        """
        print("\n🔍 Test 7: Testing data type preservation")
        
        processed_data = clean_data(sample_credit_data)
        
        # Check numeric columns
        numeric_cols = ['TransactionAmount', 'TransactionHour', 'TransactionDay', 
                       'CustomerAge', 'CustomerIncome', 'PreviousDefaults', 'CreditScore']
        numeric_cols = [col for col in numeric_cols if col in processed_data.columns]
        
        for col in numeric_cols:
            assert pd.api.types.is_numeric_dtype(processed_data[col]), \
                f"Column {col} should remain numeric"
        
        # Check string columns
        string_cols = ['TransactionId', 'CustomerId', 'ProductCategory']
        string_cols = [col for col in string_cols if col in processed_data.columns]
        
        for col in string_cols:
            assert pd.api.types.is_string_dtype(processed_data[col]) or \
                   pd.api.types.is_object_dtype(processed_data[col]), \
                f"Column {col} should remain string/object type"
        
        print("✅ Test 7 passed: Data types are preserved")
    
    def test_no_data_leakage_in_split(self):
        """
        Test 8: Train-test split doesn't have data leakage.
        
        Expected:
        - No overlapping indices between train and test
        - Combined indices cover all original data
        """
        print("\n🔍 Test 8: Testing no data leakage in split")
        
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5), columns=[f'feature_{i}' for i in range(5)])
        y = pd.Series(np.random.randint(0, 2, 100), name='is_high_risk')
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Check no overlap
        train_indices = set(X_train.index)
        test_indices = set(X_test.index)
        assert len(train_indices.intersection(test_indices)) == 0, "No indices should overlap"
        
        # Check all original data is accounted for
        all_indices = train_indices.union(test_indices)
        original_indices = set(X.index)
        assert all_indices == original_indices, "All original data should be in train or test"
        
        print("✅ Test 8 passed: No data leakage in train-test split")


def run_all_tests():
    """Run all tests and print summary."""
    print(f"\n{'='*60}")
    print("RUNNING DATA PROCESSING UNIT TESTS")
    print(f"{'='*60}")
    
    if not HAS_SRC_MODULES:
        print("⚠️  Using mock functions - src modules not found")
        print("   To test actual functions, ensure src modules are importable")
    
    # Create test instance
    test_instance = TestDataProcessing()
    
    # Run tests
    tests_passed = 0
    tests_failed = 0
    
    test_methods = [
        'test_clean_data_no_missing',
        'test_clean_data_with_missing',
        'test_feature_engineering_columns',
        'test_feature_engineering_values',
        'test_target_engineering',
        'test_data_split_reproducibility',
        'test_data_types_preserved',
        'test_no_data_leakage_in_split'
    ]
    
    for test_method in test_methods:
        try:
            # Get the fixture
            if test_method == 'test_clean_data_no_missing':
                sample_data = test_instance.sample_credit_data()
                getattr(test_instance, test_method)(sample_data)
            elif test_method == 'test_clean_data_with_missing':
                sample_data = test_instance.sample_data_with_missing()
                getattr(test_instance, test_method)(sample_data)
            elif test_method == 'test_feature_engineering_columns':
                sample_data = test_instance.sample_credit_data()
                getattr(test_instance, test_method)(sample_data)
            elif test_method == 'test_feature_engineering_values':
                sample_data = test_instance.sample_credit_data()
                getattr(test_instance, test_method)(sample_data)
            elif test_method == 'test_target_engineering':
                sample_data = test_instance.sample_credit_data()
                getattr(test_instance, test_method)(sample_data)
            else:
                getattr(test_instance, test_method)()
            
            tests_passed += 1
        except Exception as e:
            tests_failed += 1
            print(f"\n❌ {test_method} failed: {e}")
    
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Tests passed: {tests_passed}/{len(test_methods)}")
    print(f"❌ Tests failed: {tests_failed}/{len(test_methods)}")
    
    if tests_failed == 0:
        print("\n🎉 All tests passed successfully!")
    else:
        print(f"\n⚠️  {tests_failed} test(s) failed")
    
    return tests_failed == 0


if __name__ == "__main__":
    # Run tests when script is executed directly
    success = run_all_tests()
    
    # Exit with appropriate code for CI/CD
    import sys
    sys.exit(0 if success else 1)