# finish_task4.py - SIMPLE VERSION
import pandas as pd
import numpy as np

print("=" * 60)
print("COMPLETING TASK 4")
print("=" * 60)

# Load your processed data
df = pd.read_csv('data/processed/processed_data.csv')
print(f"Data shape: {df.shape}")

# Create target column (is_high_risk)
threshold = df['amount_sum'].quantile(0.75)  # Top 25% are high risk
df['is_high_risk'] = (df['amount_sum'] > threshold).astype(int)

print(f"\n[OK] Created target column: 'is_high_risk'")
print(f"    Threshold (75th percentile): {threshold:.2f}")
print(f"    Distribution: {df['is_high_risk'].value_counts().to_dict()}")
print(f"    High risk %: {(df['is_high_risk'].mean()*100):.1f}%")

# Save complete data
output_file = 'data/processed/processed_data_complete.csv'
df.to_csv(output_file, index=False)
print(f"\n[SAVED] {output_file}")
print(f"    New shape: {df.shape}")
print("=" * 60)
print("TASK 4 COMPLETE!")
print("=" * 60)