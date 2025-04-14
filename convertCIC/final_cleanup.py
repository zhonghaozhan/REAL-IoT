import pandas as pd

# File paths
input_file = '/media/ssd/test/GNN/kaggle/input/CSE-CIC-IDS2018/CSE-CIC-ID2018-Converted.csv'
output_file = '/media/ssd/test/GNN/kaggle/input/CSE-CIC-IDS2018/CSE-CIC-ID2018-Converted.csv'

try:
    print(f"Reading converted dataset from {input_file}...")
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Drop the "Unnamed: 0" column if it exists
    if 'Unnamed: 0' in df.columns:
        print("Removing 'Unnamed: 0' column...")
        df = df.drop('Unnamed: 0', axis=1)
    
    # Save with spaces after commas to match ISCX format
    print(f"Saving final cleaned dataset to {output_file}...")
    
    # Use modified to_csv to add spaces after commas
    with open(output_file, 'w') as f:
        # Write header with spaces after commas
        header = ', '.join(df.columns)
        f.write(header + '\n')
        
        # Write data rows with spaces after commas
        for _, row in df.iterrows():
            row_str = ', '.join([str(val) for val in row])
            f.write(row_str + '\n')
    
    print("Final cleanup completed successfully!")
    
except Exception as e:
    print(f"Error during cleanup: {str(e)}")
