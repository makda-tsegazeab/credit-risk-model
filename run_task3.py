# # run_task3.py
# import sys
# sys.path.append('src')

# from data_processing import engineer_features
# import pandas as pd

# print("=" * 60)
# print("TASK 3: FEATURE ENGINEERING")
# print("=" * 60)

# # Load data
# print("Loading data...")
# df = pd.read_csv('data/raw/data.csv')
# print(f"Data shape: {df.shape}")

# # Engineer features
# print("\nEngineering features...")
# X_processed, feature_names, customer_df = engineer_features(df)

# print(f"\n✅ TASK 3 COMPLETE!")
# print(f"📊 Original data: {df.shape}")
# print(f"📈 Processed features: {X_processed.shape}")
# print(f"👥 Customer features (for Task 4): {customer_df.shape}")
# print(f"🔢 Number of features: {len(feature_names)}")

# print(f"\n📁 Outputs created:")
# print(f"  - data/processed/features.npy")
# print(f"  - data/processed/feature_names.csv")
# print(f"  - data/processed/customer_features.csv")

# print(f"\n🎉 Ready for Task 4!")

# run_task3.py - UPDATED TO PROCESS FULL DATASET
import sys
sys.path.append('src')

from data_processing import engineer_features
import pandas as pd

print("=" * 60)
print("TASK 3: FEATURE ENGINEERING (FULL DATASET)")
print("=" * 60)

# Load FULL data
print("Loading FULL dataset...")
df = pd.read_csv('data/raw/data.csv')
print(f"Data shape: {df.shape}")
print(f"Unique customers: {df['CustomerId'].nunique()}")

# Engineer features on FULL dataset
print("\nEngineering features on FULL dataset...")
X_processed, feature_names, customer_df = engineer_features(df)

print(f"\n✅ TASK 3 COMPLETE!")
print(f"📊 Original data: {df.shape}")
print(f"📈 Processed features: {X_processed.shape}")
print(f"👥 Customer features: {customer_df.shape}")
print(f"🔢 Number of features: {len(feature_names)}")

print(f"\n📁 Outputs created:")
print(f"  - data/processed/features.npy")
print(f"  - data/processed/feature_names.csv")
print(f"  - data/processed/customer_features.csv")

print(f"\n🎉 Ready for Task 4 with FULL customer data!")