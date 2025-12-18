# src/target_engineering.py - COMPLETE FIXED VERSION
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime, timedelta

def create_proxy_target_variable():
    """Create proxy target variable using RFM clustering"""
    
    print("=" * 60)
    print("TASK 4: PROXY TARGET VARIABLE ENGINEERING")
    print("=" * 60)
    
    # Load customer features from Task 3
    customer_features_path = 'data/processed/customer_features.csv'
    
    if not os.path.exists(customer_features_path):
        print("❌ Error: customer_features.csv not found.")
        print("Please run Task 3 first: python run_task3.py")
        return None
    
    print("📊 Loading customer features...")
    df = pd.read_csv(customer_features_path)
    
    # Check if we have enough data
    if len(df) < 10:
        print(f"❌ Error: Only {len(df)} customers found. Need more data.")
        return None
    
    print(f"✅ Loaded {len(df)} customer records")
    
    # FIX: Recalculate recency with proper snapshot date
    print("\n🔍 Recalculating recency with proper snapshot date...")
    
    try:
        original_data = pd.read_csv('data/raw/data.csv', nrows=10000)
        original_data['TransactionStartTime'] = pd.to_datetime(original_data['TransactionStartTime'])
        
        max_date = original_data['TransactionStartTime'].max()
        snapshot_date = max_date + timedelta(days=30)
        
        print(f"   Max transaction date in data: {max_date.date()}")
        print(f"   Snapshot date for recency: {snapshot_date.date()}")
        
        customer_last_transaction = original_data.groupby('CustomerId')['TransactionStartTime'].max()
        
        recency_values = []
        for customer_id in df['CustomerId']:
            if customer_id in customer_last_transaction.index:
                last_transaction = customer_last_transaction[customer_id]
                recency = (snapshot_date - last_transaction).days
            else:
                recency = 180
            recency_values.append(recency)
        
        df['recency'] = recency_values
        
        print(f"   Recency range: {df['recency'].min()} to {df['recency'].max()} days")
        print(f"   Average recency: {df['recency'].mean():.1f} days")
        
    except Exception as e:
        print(f"⚠️ Could not recalculate recency: {e}")
    
    # Ensure we have required features
    required_features = ['recency', 'frequency', 'monetary']
    for feature in required_features:
        if feature not in df.columns:
            print(f"❌ Missing required feature: {feature}")
            return None
    
    # Handle monetary outliers
    print("\n🗑️ Handling extreme monetary values...")
    monetary_q1 = df['monetary'].quantile(0.25)
    monetary_q3 = df['monetary'].quantile(0.75)
    monetary_iqr = monetary_q3 - monetary_q1
    monetary_upper = monetary_q3 + 1.5 * monetary_iqr
    
    df['monetary_capped'] = np.where(
        df['monetary'] > monetary_upper, 
        monetary_upper, 
        df['monetary']
    )
    
    # Use RFM features for clustering
    rfm_features = ['recency', 'frequency', 'monetary_capped']
    print(f"\n📈 Using RFM features for clustering: {rfm_features}")
    
    # Prepare data
    X = df[rfm_features].copy()
    X['monetary_log'] = np.log1p(X['monetary_capped'])
    clustering_features = ['recency', 'frequency', 'monetary_log']
    X_cluster = X[clustering_features].copy()
    X_cluster = X_cluster.fillna(X_cluster.median())
    
    # Scale and cluster
    print("🔢 Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    print("🎯 Applying K-Means clustering (k=3)...")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Analyze clusters
    print("\n" + "=" * 60)
    print("CLUSTER ANALYSIS")
    print("=" * 60)
    
    cluster_stats = []
    for cluster in range(3):
        cluster_data = df[df['cluster'] == cluster]
        if len(cluster_data) > 0:
            stats = {
                'cluster': cluster,
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(df) * 100,
                'avg_recency': cluster_data['recency'].mean(),
                'avg_frequency': cluster_data['frequency'].mean(),
                'median_monetary': cluster_data['monetary'].median()
            }
            cluster_stats.append(stats)
    
    stats_df = pd.DataFrame(cluster_stats)
    print("\nCluster Characteristics:")
    print(stats_df.round(2).to_string(index=False))
    
    # FIXED RISK ASSESSMENT
    print("\n" + "=" * 60)
    print("RISK ASSESSMENT")
    print("=" * 60)
    
    risk_scores = []
    
    for cluster in range(3):
        cluster_data = df[df['cluster'] == cluster]
        
        if len(cluster_data) == 0:
            risk_scores.append(0)
            continue
        
        # Get statistics
        avg_recency = cluster_data['recency'].mean()
        avg_frequency = cluster_data['frequency'].mean()
        median_monetary = cluster_data['monetary'].median()
        
        # Calculate normalized risk scores (0 to 1)
        # Higher recency = higher risk
        recency_max = df['recency'].max()
        recency_risk = avg_recency / recency_max if recency_max > 0 else 0
        
        # Lower frequency = higher risk
        frequency_max = df['frequency'].max()
        frequency_risk = 1 - (avg_frequency / frequency_max) if frequency_max > 0 else 0
        
        # Lower monetary = higher risk
        monetary_max = df['monetary'].max()
        monetary_risk = 1 - (median_monetary / monetary_max) if monetary_max > 0 else 0
        
        # Combined risk score (weighted average)
        risk_score = (0.5 * recency_risk + 0.3 * frequency_risk + 0.2 * monetary_risk)
        risk_score = max(0, min(1, risk_score))  # Clamp to 0-1
        risk_scores.append(risk_score)
        
        print(f"\nCluster {cluster}:")
        print(f"  Size: {len(cluster_data)} customers ({len(cluster_data)/len(df)*100:.1f}%)")
        print(f"  Avg Recency: {avg_recency:.1f} days")
        print(f"  Avg Frequency: {avg_frequency:.1f} transactions")
        print(f"  Median Monetary: ${median_monetary:.0f}")
        print(f"  Risk Score: {risk_score:.3f}")
    
    # Identify high-risk cluster
    high_risk_cluster = np.argmax(risk_scores)
    high_risk_data = df[df['cluster'] == high_risk_cluster]
    
    print(f"\n⚠️ IDENTIFIED HIGH-RISK CLUSTER: {high_risk_cluster}")
    print(f"   Risk Score: {risk_scores[high_risk_cluster]:.3f}")
    print(f"   Size: {len(high_risk_data)} customers")
    
    # Create initial target variable
    df['is_high_risk'] = (df['cluster'] == high_risk_cluster).astype(int)
    
    # BALANCE THE TARGET VARIABLE
    print("\n" + "=" * 60)
    print("BALANCING TARGET VARIABLE")
    print("=" * 60)
    
    high_risk_percentage = df['is_high_risk'].mean() * 100
    
    if high_risk_percentage > 60:
        print(f"⚠️ High-risk percentage is {high_risk_percentage:.1f}% (>60%)")
        print("   Rebalancing for better model training...")
        
        # For high-risk cluster, only mark top 40% as high-risk
        high_risk_cluster_data = df[df['cluster'] == high_risk_cluster].copy()
        
        # Calculate composite risk score within high-risk cluster
        high_risk_cluster_data['composite_risk'] = (
            high_risk_cluster_data['recency'] / high_risk_cluster_data['recency'].max() +
            (1 - high_risk_cluster_data['frequency'] / high_risk_cluster_data['frequency'].max())
        ) / 2
        
        # Take only top 40% as high-risk
        risk_threshold = high_risk_cluster_data['composite_risk'].quantile(0.6)
        true_high_risk_indices = high_risk_cluster_data[
            high_risk_cluster_data['composite_risk'] >= risk_threshold
        ].index
        
        # Reset target variable
        df['is_high_risk'] = 0
        df.loc[true_high_risk_indices, 'is_high_risk'] = 1
        
        print(f"✅ Rebalanced: High-risk from {high_risk_percentage:.1f}% to {df['is_high_risk'].mean()*100:.1f}%")
    
    # Show final distribution
    print("\n" + "=" * 60)
    print("FINAL TARGET DISTRIBUTION")
    print("=" * 60)
    
    risk_counts = df['is_high_risk'].value_counts().sort_index()
    total_customers = len(df)
    
    for risk_level in [0, 1]:
        count = risk_counts.get(risk_level, 0)
        percentage = count / total_customers * 100
        risk_label = "HIGH-RISK" if risk_level == 1 else "LOW-RISK"
        print(f"  {risk_label}: {count} customers ({percentage:.1f}%)")
    
    # Visualize clusters
    print("\n📊 Creating visualizations...")
    try:
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Recency distribution
        plt.subplot(1, 3, 1)
        colors = ['blue', 'orange', 'green']
        for cluster in range(3):
            cluster_data = df[df['cluster'] == cluster]['recency']
            plt.hist(cluster_data, alpha=0.5, label=f'Cluster {cluster}', 
                    color=colors[cluster], bins=20)
        plt.xlabel('Recency (days)')
        plt.ylabel('Frequency')
        plt.title('Recency by Cluster')
        plt.legend()
        
        # Plot 2: Monetary vs Frequency with risk
        plt.subplot(1, 3, 2)
        scatter = plt.scatter(df['frequency'], df['monetary'], 
                            c=df['is_high_risk'], cmap='RdYlGn',
                            alpha=0.6, s=20, edgecolor='k', linewidth=0.5)
        plt.xlabel('Frequency')
        plt.ylabel('Monetary Value ($)')
        plt.title('High-Risk vs Low-Risk')
        plt.colorbar(scatter, label='High Risk')
        
        # Plot 3: Target distribution
        plt.subplot(1, 3, 3)
        labels = ['Low-Risk', 'High-Risk']
        sizes = [risk_counts.get(0, 0), risk_counts.get(1, 0)]
        colors_pie = ['lightgreen', 'lightcoral']
        plt.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
        plt.title('Target Variable Distribution')
        plt.axis('equal')
        
        plt.tight_layout()
        
        # Save visualization
        os.makedirs('data/processed', exist_ok=True)
        plt.savefig('data/processed/cluster_visualization.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✅ Cluster visualization saved: data/processed/cluster_visualization.png")
        
    except Exception as e:
        print(f"⚠️ Could not create visualization: {e}")
    
    # Save results
    output_path = 'data/processed/target_variable.csv'
    
    # Keep only essential columns
    output_columns = ['CustomerId', 'recency', 'frequency', 'monetary', 
                     'cluster', 'is_high_risk']
    
    existing_columns = [col for col in output_columns if col in df.columns]
    df[existing_columns].to_csv(output_path, index=False)
    
    # Create summary
    summary = {
        'task': 'Task 4 - Proxy Target Variable Engineering',
        'status': 'COMPLETED',
        'total_customers': int(len(df)),
        'high_risk_customers': int(df['is_high_risk'].sum()),
        'high_risk_percentage': float(df['is_high_risk'].mean() * 100),
        'high_risk_cluster': int(high_risk_cluster),
        'risk_scores': [float(score) for score in risk_scores],
        'clustering_features': clustering_features,
        'cluster_sizes': [int(len(df[df['cluster'] == i])) for i in range(3)],
        'cluster_percentages': [float(len(df[df['cluster'] == i])/len(df)*100) for i in range(3)],
        'target_distribution': {
            'low_risk': int(df[df['is_high_risk'] == 0].shape[0]),
            'high_risk': int(df[df['is_high_risk'] == 1].shape[0]),
            'low_risk_pct': float(df[df['is_high_risk'] == 0].shape[0] / len(df) * 100),
            'high_risk_pct': float(df[df['is_high_risk'] == 1].shape[0] / len(df) * 100)
        }
    }
    
    with open('data/processed/target_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Target variable saved: {output_path}")
    print(f"📄 Summary saved: data/processed/target_summary.json")
    
    # Final summary
    print("\n" + "=" * 60)
    print("TASK 4 COMPLETE! ✅")
    print("=" * 60)
    
    print(f"\n📊 FINAL RESULTS:")
    print(f"  • Total customers: {summary['total_customers']}")
    print(f"  • High-risk customers: {summary['high_risk_customers']}")
    print(f"  • High-risk percentage: {summary['high_risk_percentage']:.1f}%")
    print(f"  • Balance: {summary['target_distribution']['low_risk_pct']:.1f}% low-risk, {summary['target_distribution']['high_risk_pct']:.1f}% high-risk")
    
    print(f"\n🎯 TASK 4 DELIVERABLES:")
    print(f"  1. ✅ RFM metrics calculated")
    print(f"  2. ✅ Customers clustered (K=3)")
    print(f"  3. ✅ High-risk cluster identified")
    print(f"  4. ✅ Binary target variable created")
    print(f"  5. ✅ Target integrated with customer data")
    
    print(f"\n📁 OUTPUTS CREATED:")
    print(f"  • data/processed/target_variable.csv")
    print(f"  • data/processed/target_summary.json")
    print(f"  • data/processed/cluster_visualization.png")
    
    print(f"\n🎉 READY FOR TASK 5: Model Training!")
    
    return df

if __name__ == "__main__":
    result = create_proxy_target_variable()

