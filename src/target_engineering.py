def save_modeling_dataset(
    features_path="data/processed/processed_data.csv",
    output_path="data/processed/modeling_data.csv"
):
    """
    Merge engineered proxy target with features and save final dataset.
    """
    df = pd.read_csv(features_path)

    # Simple proxy target logic (same logic as Task 4)
    threshold = df["amount_sum"].quantile(0.75)
    df["is_high_risk"] = (df["amount_sum"] < threshold).astype(int)

    df.to_csv(output_path, index=False)
    print(f"Modeling dataset saved to {output_path}")
