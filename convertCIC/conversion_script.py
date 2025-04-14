import pandas as pd
import numpy as np

# File paths
input_file = '/media/ssd/test/GNN/kaggle/input/CSE-CIC-IDS2018/CSE-CIC-ID2018.csv'
output_file = '/media/ssd/test/GNN/kaggle/input/CSE-CIC-IDS2018/CSE-CIC-ID2018-Converted.csv'
reference_file = '/media/ssd/test/GNN/kaggle/input/network-intrusion-dataset/Wednesday-workingHours.pcap_ISCX.csv'

print(f"Reading original CSE-CIC-ID2018 dataset from {input_file}...")

# Load datasets
try:
    # Check header rows of both files to understand column structures
    cse_df = pd.read_csv(input_file, nrows=0)
    iscx_df = pd.read_csv(reference_file, nrows=0)
    
    # Get column names from both datasets
    cse_columns = cse_df.columns.tolist()
    iscx_columns = iscx_df.columns.tolist()
    
    print(f"CSE-CIC-ID2018 columns: {len(cse_columns)}")
    print(f"ISCX columns: {len(iscx_columns)}")
    
    # Now load the full CSE dataset (this may take some time if the file is large)
    print("Loading full CSE-CIC-ID2018 dataset...")
    cse_df = pd.read_csv(input_file)
    print(f"Loaded {len(cse_df)} rows.")
    
    # Create column mapping based on column names
    # First, map exact matches and known similar names
    column_mapping = {}
    
    # Common mapping patterns
    mapping_patterns = {
        'Dst Port': 'Destination Port',
        'Tot Fwd Pkts': 'Total Fwd Packets',
        'Tot Bwd Pkts': 'Total Backward Packets',
        'TotLen Fwd Pkts': 'Total Length of Fwd Packets',
        'TotLen Bwd Pkts': 'Total Length of Bwd Packets',
        'Fwd Pkt Len Max': 'Fwd Packet Length Max',
        'Fwd Pkt Len Min': 'Fwd Packet Length Min',
        'Fwd Pkt Len Mean': 'Fwd Packet Length Mean',
        'Fwd Pkt Len Std': 'Fwd Packet Length Std',
        'Bwd Pkt Len Max': 'Bwd Packet Length Max',
        'Bwd Pkt Len Min': 'Bwd Packet Length Min',
        'Bwd Pkt Len Mean': 'Bwd Packet Length Mean',
        'Bwd Pkt Len Std': 'Bwd Packet Length Std',
        'Flow Byts/s': 'Flow Bytes/s',
        'Flow Pkts/s': 'Flow Packets/s',
        'Pkt Len Min': 'Min Packet Length',
        'Pkt Len Max': 'Max Packet Length',
        'Pkt Len Mean': 'Packet Length Mean',
        'Pkt Len Std': 'Packet Length Std',
        'Pkt Len Var': 'Packet Length Variance',
        'FIN Flag Cnt': 'FIN Flag Count',
        'SYN Flag Cnt': 'SYN Flag Count',
        'RST Flag Cnt': 'RST Flag Count',
        'PSH Flag Cnt': 'PSH Flag Count',
        'ACK Flag Cnt': 'ACK Flag Count',
        'URG Flag Cnt': 'URG Flag Count',
        'CWE Flag Count': 'CWE Flag Count',
        'ECE Flag Cnt': 'ECE Flag Count',
        'Pkt Size Avg': 'Average Packet Size',
        'Fwd Seg Size Avg': 'Avg Fwd Segment Size',
        'Bwd Seg Size Avg': 'Avg Bwd Segment Size',
        'Init Fwd Win Byts': 'Init_Win_bytes_forward',
        'Init Bwd Win Byts': 'Init_Win_bytes_backward',
        'Fwd Act Data Pkts': 'act_data_pkt_fwd',
        'Fwd Seg Size Min': 'min_seg_size_forward',
    }
    
    # Map the columns based on patterns
    for cse_col in cse_columns:
        if cse_col in mapping_patterns:
            column_mapping[cse_col] = mapping_patterns[cse_col]
        elif cse_col == '':  # First unnamed column
            column_mapping[cse_col] = 'Record ID'
        else:
            # For columns without explicit mapping, keep the original name
            column_mapping[cse_col] = cse_col
    
    # Rename columns
    print("Renaming columns to match ISCX format...")
    cse_df.rename(columns=column_mapping, inplace=True)
    
    # If there's an unnamed first column, handle it
    if 'Record ID' in cse_df.columns:
        print("Dropping 'Record ID' column...")
        cse_df.drop('Record ID', axis=1, inplace=True)
    
    # Check for label values
    unique_labels = cse_df['Label'].unique()
    print(f"Unique label values in the original dataset: {unique_labels}")
    
    # If labels are numeric, convert them to text-based labels if needed
    if pd.api.types.is_numeric_dtype(cse_df['Label']):
        print("Converting numeric labels to text-based labels...")
        # Simple mapping - adjust based on your specific dataset information
        label_mapping = {
            0: 'BENIGN',
            1: 'ATTACK'  # Generic label, adjust as needed
        }
        
        # For more specific attack types, you'd need domain knowledge
        # This is a simplified mapping
        try:
            cse_df['Label'] = cse_df['Label'].map(label_mapping)
        except:
            print("Could not map labels. Keeping original values.")
    
    # Make sure all data is in consistent format
    # For example, convert any numeric columns that should be strings, etc.
    
    # Add spaces after commas in the column names to match ISCX format
    new_columns = {}
    for col in cse_df.columns:
        new_columns[col] = col
    cse_df.rename(columns=new_columns, inplace=True)
    
    # Save the transformed dataset
    print(f"Saving converted dataset to {output_file}...")
    cse_df.to_csv(output_file, index=False)
    
    print("Conversion completed successfully!")

except Exception as e:
    print(f"Error during conversion: {str(e)}")
