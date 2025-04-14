#!/usr/bin/env python3
# bot_iot_adapter.py - Adapts and standardizes the BoT-IoT reduced dataset

import os
import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path

def standardize_bot_iot_reduced(input_file, output_dir):
    """
    Standardize BoT-IoT reduced dataset
    
    Args:
        input_file: Path to the BoT-IoT reduced dataset file
        output_dir: Directory to save the standardized dataset
    
    Returns:
        Path to the standardized dataset file
    """
    print(f"Standardizing BoT-IoT reduced dataset from {input_file}")
    
    # Load dataset
    df = pd.read_csv(input_file)
    
    print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
    
    # Create standardized dataframe
    std_df = pd.DataFrame()
    
    # Map fields to common schema from the reduced format
    std_df['src_ip'] = df['saddr'].astype(str)
    
    # Handle hexadecimal port values
    def safe_convert_port(port_val):
        try:
            if isinstance(port_val, str) and port_val.startswith('0x'):
                return int(port_val, 16)
            return int(port_val)
        except (ValueError, TypeError):
            return 0
    
    # Convert ports safely
    std_df['src_port'] = df['sport'].apply(safe_convert_port)
    std_df['dst_ip'] = df['daddr'].astype(str)
    std_df['dst_port'] = df['dport'].apply(safe_convert_port)
    
    # Protocol mapping based on proto_number column
    proto_map = {1: 'tcp', 2: 'udp', 3: 'icmp'}
    std_df['protocol'] = df['proto_number'].astype(int)
    std_df['protocol_name'] = df['proto']
    
    # Duration and other metrics
    std_df['duration_ms'] = df['dur'].fillna(0) * 1000  # Convert to milliseconds
    std_df['bytes_in'] = df['sbytes'].fillna(0).astype(int)
    std_df['bytes_out'] = df['dbytes'].fillna(0).astype(int)
    std_df['packets_in'] = df['spkts'].fillna(0).astype(int)
    std_df['packets_out'] = df['dpkts'].fillna(0).astype(int)
    std_df['tcp_flags'] = df.get('flgs_number', 0).fillna(0).astype(int)
    
    # Handle labels - For BoT-IoT, 'attack' is the binary indicator and 'label' is the attack type
    std_df['binary_label'] = df['attack'].astype(int)
    std_df['attack_type'] = df['label'].fillna('Normal')
    std_df.loc[std_df['attack_type'] == '-', 'attack_type'] = 'Normal'
    std_df.loc[std_df['binary_label'] == 0, 'attack_type'] = 'Normal'
    
    # Create attack type encoding
    attack_types = sorted(std_df['attack_type'].unique())
    attack_mapping = {attack: idx for idx, attack in enumerate(attack_types)}
    std_df['attack_type_encoded'] = std_df['attack_type'].map(attack_mapping).astype(int)
    
    # Add flow_id
    std_df['flow_id'] = range(len(std_df))
    
    # Transfer any additional numeric features from original dataset
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in [
            'sport', 'dport', 'proto_number', 'dur', 'sbytes', 'dbytes', 
            'spkts', 'dpkts', 'flgs_number', 'attack'
        ]:
            # Add this column to the standardized dataframe
            col_name = col.lower()
            std_df[col_name] = df[col].fillna(0)
    
    print(f"Created standardized dataframe with {len(std_df)} rows and {len(std_df.columns)} columns")
    print(f"Attack types found: {attack_types}")
    
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {os.path.abspath(output_dir)}")
    
    # Save attack mapping
    mapping_file = os.path.join(output_dir, 'bot_iot_attack_mapping.json')
    with open(mapping_file, 'w') as f:
        json.dump(attack_mapping, f, indent=2)
    print(f"Saved attack mapping to {mapping_file}")
    
    # Save standardized dataset
    output_file = os.path.join(output_dir, 'bot_iot_standardized.csv')
    std_df.to_csv(output_file, index=False)
    print(f"Saved standardized dataset to {output_file}")
    
    # Verify the file was created
    if os.path.exists(output_file):
        print(f"Verified file exists: {output_file} ({os.path.getsize(output_file)} bytes)")
    else:
        print(f"ERROR: File was not created: {output_file}")
    
    print(f"Standardized BoT-IoT dataset saved to {output_file}")
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Standardize BoT-IoT reduced dataset')
    parser.add_argument('--input_file', required=True, help='Path to bot_reduced.csv file')
    parser.add_argument('--output_dir', default='standard', help='Output directory for standardized dataset')
    
    args = parser.parse_args()
    
    print(f"Input file: {os.path.abspath(args.input_file)}")
    print(f"Output directory: {os.path.abspath(args.output_dir)}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Standardize the dataset
    standardize_bot_iot_reduced(args.input_file, args.output_dir)

if __name__ == '__main__':
    main() 