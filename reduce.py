import pandas as pd

def reduce_csv(input_path, output_path, target_column=None, num_samples=7000, random_state=42):
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    print(f"Original dataset size: {len(df)} rows")

    # Case 1: If target column is provided, sample equally from each category
    if target_column and target_column in df.columns:
        unique_categories = df[target_column].unique()
        print(f"Found {len(unique_categories)} unique categories in '{target_column}': {unique_categories}")

        # Calculate samples per category
        samples_per_category = num_samples // len(unique_categories)
        sampled_df = pd.DataFrame()

        for category in unique_categories:
            category_df = df[df[target_column] == category]
            n_samples = min(samples_per_category, len(category_df))  # Avoid oversampling
            sampled_category = category_df.sample(n=n_samples, random_state=random_state)
            sampled_df = pd.concat([sampled_df, sampled_category], ignore_index=True)
        
        print(f"Sampled {samples_per_category} data points per category.")
    
    # Case 2: If no target column is provided, randomly sample the dataset
    else:
        print("No target column provided. Performing random sampling.")
        sampled_df = df.sample(n=min(num_samples, len(df)), random_state=random_state)
    
    sampled_df.to_csv(output_path, index=False)
    print(f"Reduced dataset saved to: {output_path}")
    print(f"Reduced dataset size: {len(sampled_df)} rows")



input_csv = r'C:\Users\sarat\Downloads\cleaned_reviews.csv'  
output_csv = r'C:\Users\sarat\Downloads\reduced_dataset.csv' 
target_col = 'sentiments'

reduce_csv(input_csv, output_csv, target_column=target_col)
