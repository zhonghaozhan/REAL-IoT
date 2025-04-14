import pandas as pd
import numpy as np

# File paths
input_file = '/media/ssd/test/GNN/kaggle/input/CSE-CIC-IDS2018/CSE-CIC-ID2018.csv'
output_file = '/media/ssd/test/GNN/kaggle/input/CSE-CIC-IDS2018/CSE-CIC-ID2018-Converted.csv'
reference_file = '/media/ssd/test/GNN/kaggle/input/network-intrusion-dataset/Wednesday-workingHours.pcap_ISCX.csv'

print(f"Reading original CSE-CIC-ID2018 dataset from {input_file}...")

try:
    # Load datasets
    cse_df = pd.read_csv(input_file)
    print(f"Loaded {len(cse_df)} rows from CSE-CIC-ID2018 dataset.")
    
    # Check header structure of reference file
    iscx_df = pd.read_csv(reference_file, nrows=5)
    print(f"Reference file has {len(iscx_df.columns)} columns.")
    
    # Create mapping for column names
    column_mapping = {
        '': 'Record ID',  # First unnamed column
        'Dst Port': 'Destination Port',
        'Protocol': 'Protocol',
        'Flow Duration': 'Flow Duration',
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
        'Flow IAT Mean': 'Flow IAT Mean',
        'Flow IAT Std': 'Flow IAT Std',
        'Flow IAT Max': 'Flow IAT Max',
        'Flow IAT Min': 'Flow IAT Min',
        'Fwd IAT Tot': 'Fwd IAT Total',
        'Fwd IAT Mean': 'Fwd IAT Mean',
        'Fwd IAT Std': 'Fwd IAT Std',
        'Fwd IAT Max': 'Fwd IAT Max',
        'Fwd IAT Min': 'Fwd IAT Min',
        'Bwd IAT Tot': 'Bwd IAT Total',
        'Bwd IAT Mean': 'Bwd IAT Mean',
        'Bwd IAT Std': 'Bwd IAT Std',
        'Bwd IAT Max': 'Bwd IAT Max',
        'Bwd IAT Min': 'Bwd IAT Min',
        'Fwd PSH Flags': 'Fwd PSH Flags',
        'Bwd PSH Flags': 'Bwd PSH Flags',
        'Fwd URG Flags': 'Fwd URG Flags',
        'Bwd URG Flags': 'Bwd URG Flags',
        'Fwd Header Len': 'Fwd Header Length',
        'Bwd Header Len': 'Bwd Header Length',
        'Fwd Pkts/s': 'Fwd Packets/s',
        'Bwd Pkts/s': 'Bwd Packets/s',
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
        'Fwd Byts/b Avg': 'Fwd Avg Bytes/Bulk',
        'Fwd Pkts/b Avg': 'Fwd Avg Packets/Bulk',
        'Fwd Blk Rate Avg': 'Fwd Avg Bulk Rate',
        'Bwd Byts/b Avg': 'Bwd Avg Bytes/Bulk',
        'Bwd Pkts/b Avg': 'Bwd Avg Packets/Bulk',
        'Bwd Blk Rate Avg': 'Bwd Avg Bulk Rate',
        'Subflow Fwd Pkts': 'Subflow Fwd Packets',
        'Subflow Fwd Byts': 'Subflow Fwd Bytes',
        'Subflow Bwd Pkts': 'Subflow Bwd Packets',
        'Subflow Bwd Byts': 'Subflow Bwd Bytes',
        'Init Fwd Win Byts': 'Init_Win_bytes_forward',
        'Init Bwd Win Byts': 'Init_Win_bytes_backward',
        'Fwd Act Data Pkts': 'act_data_pkt_fwd',
        'Fwd Seg Size Min': 'min_seg_size_forward',
        'Active Mean': 'Active Mean',
        'Active Std': 'Active Std',
        'Active Max': 'Active Max',
        'Active Min': 'Active Min',
        'Idle Mean': 'Idle Mean',
        'Idle Std': 'Idle Std',
        'Idle Max': 'Idle Max',
        'Idle Min': 'Idle Min',
        'Label': 'Label'
    }
    
    # Rename columns
    print("Renaming columns to match ISCX format...")
    cse_df.rename(columns=column_mapping, inplace=True)
    
    # Drop the Record ID column (first unnamed column)
    if 'Record ID' in cse_df.columns:
        print("Removing 'Record ID' column...")
        cse_df.drop('Record ID', axis=1, inplace=True)
    
    # Map numeric labels to more descriptive attack types
    # Based on common understanding of CSE-CIC-IDS2018 dataset
    print("Converting numeric labels to descriptive attack types...")
    unique_labels = cse_df['Label'].unique()
    print(f"Original unique labels: {unique_labels}")
    
    # More specific label mapping based on CSE-CIC-IDS2018 documentation
    # Modify this mapping based on your specific knowledge of the dataset
    label_mapping = {
        1: 'BENIGN',
        2: 'DoS', 
        3: 'PortScan',
        4: 'BruteForce',
        5: 'DDoS',
        6: 'WebAttack',
        7: 'Bot',
        8: 'Infiltration',
        9: 'SQL_Injection',
        10: 'FTP-Patator',
        11: 'SSH-Patator'
        # Add other labels as needed
    }
    
    cse_df['Label'] = cse_df['Label'].map(label_mapping)
    print(f"Mapped labels: {cse_df['Label'].unique()}")
    
    # Save with spaces after commas to match ISCX format
    print(f"Saving converted dataset to {output_file}...")
    
    # Use modified to_csv to add spaces after commas
    with open(output_file, 'w') as f:
        # Write header with spaces after commas
        header = ', '.join(cse_df.columns)
        f.write(header + '\n')
        
        # Write data rows with spaces after commas
        for _, row in cse_df.iterrows():
            row_str = ', '.join([str(val) for val in row])
            f.write(row_str + '\n')
    
    print("Conversion completed successfully!")

except Exception as e:
    print(f"Error during conversion: {str(e)}")
