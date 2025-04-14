#!/usr/bin/env python3
# standardize_datasets.py - Standardizes NetFlow datasets for cross-model compatibility
#
# This script performs Step 2 (STANDARDIZATION) in the 3-step workflow:
# 1. REDUCTION: Use reduction scripts (reduce_*.py) to reduce dataset sizes
# 2. STANDARDIZATION (this script): Standardize formats without additional filtering
# 3. COMBINATION: Use combine_datasets.py to merge standardized datasets

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import argparse
import json

# Input and output directory paths
INPUT_DIR = "/media/ssd/test/GNN/kaggle/input"
OUTPUT_DIR = "/media/ssd/test/standardized-datasets"
NETFLOW_DIR = os.path.join(OUTPUT_DIR, "netflow")
COMBINED_DIR = os.path.join(OUTPUT_DIR, "combined")

# Input dataset paths
BOT_IOT_PATH = os.path.join(INPUT_DIR, "NF-BoT-IoT/NF-BoT-IoT.csv")
BOT_IOT_V2_PATH = os.path.join(INPUT_DIR, "NF-BoT-IoT/NF-BoT-IoT-v2.csv")
CIC_IDS_PATH = os.path.join(INPUT_DIR, "NF-CSE-CIC-IDS2018-v2/NF-CSE-CIC-IDS2018-v2.csv")
UNSW_NB15_PATH = os.path.join(INPUT_DIR, "nf-unsw-nb15-v2/NF-UNSW-NB15-v2.csv")

# NetFlow V1 core features (common to all datasets)
CORE_FEATURES = [
    "IPV4_SRC_ADDR", "L4_SRC_PORT", 
    "IPV4_DST_ADDR", "L4_DST_PORT", 
    "PROTOCOL", "L7_PROTO", 
    "IN_BYTES", "OUT_BYTES", 
    "IN_PKTS", "OUT_PKTS", 
    "TCP_FLAGS", "FLOW_DURATION_MILLISECONDS",
    "Label", "Attack"
]

def ensure_directories():
    """Create output directories if they don't exist"""
    os.makedirs(NETFLOW_DIR, exist_ok=True)
    os.makedirs(COMBINED_DIR, exist_ok=True)
    print(f"Created output directories: {NETFLOW_DIR} and {COMBINED_DIR}")

def standardize_bot_iot():
    """Standardize the BoT-IoT NetFlow dataset"""
    print("Standardizing BoT-IoT NetFlow dataset...")
    
    # Load the dataset
    try:
        df = pd.read_csv(BOT_IOT_PATH)
        print(f"Loaded BoT-IoT dataset with {len(df)} rows")
    except Exception as e:
        print(f"Error loading BoT-IoT dataset: {e}")
        return None
    
    # Check if required columns exist
    missing_cols = [col for col in CORE_FEATURES if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns in BoT-IoT: {missing_cols}")
    
    # Ensure column names are consistent
    df.columns = [col.strip() for col in df.columns]
    
    # Add flow_id column if not present
    if 'flow_id' not in df.columns:
        df['flow_id'] = range(len(df))
    
    # Add dataset_source column
    df['dataset_source'] = 'bot_iot'
    
    # Ensure all columns have correct data types
    # String columns
    str_cols = ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'Attack']
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    # Integer columns
    int_cols = ['L4_SRC_PORT', 'L4_DST_PORT', 'PROTOCOL', 'IN_PKTS', 'OUT_PKTS', 'TCP_FLAGS', 'Label', 'flow_id']
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    # Float columns
    float_cols = ['L7_PROTO', 'IN_BYTES', 'OUT_BYTES', 'FLOW_DURATION_MILLISECONDS']
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(float)
    
    # Save the standardized dataset
    output_file = os.path.join(NETFLOW_DIR, "nf_bot_iot_standardized.csv")
    df.to_csv(output_file, index=False)
    print(f"Saved standardized BoT-IoT dataset to {output_file}")
    
    return df

def is_reduced_dataset(file_path):
    """Check if a dataset file is a reduced version"""
    return 'reduced' in os.path.basename(file_path).lower()

def standardize_bot_iot_v2():
    """
    Standardize the BoT-IoT v2 NetFlow dataset
    
    This function only standardizes the data format and does NOT perform
    additional filtering. Any data reduction should be done separately
    using the reduce_bot_iot_v2.py script before standardization.
    """
    print("Standardizing BoT-IoT v2 NetFlow dataset...")
    
    # Check if reduced dataset exists
    reduced_file = os.path.join(NETFLOW_DIR, "nf_bot_iot_v2_reduced.csv")
    
    if os.path.exists(reduced_file):
        print(f"Reduced BoT-IoT v2 dataset found at {reduced_file}")
        print("Using reduced dataset for standardization...")
        input_path = reduced_file
        # Use a name indicating this is standardized from a reduced dataset
        output_file = os.path.join(NETFLOW_DIR, "nf_bot_iot_v2_reduced_standardized.csv")
        use_chunks = False
    else:
        print("No reduced dataset found, processing full BoT-IoT v2 dataset (this may take a while)...")
        input_path = BOT_IOT_V2_PATH
        output_file = os.path.join(NETFLOW_DIR, "nf_bot_iot_v2_standardized.csv")
        use_chunks = True
    
    # Process either the full dataset in chunks or the reduced dataset
    if use_chunks:
        # For large datasets, process in chunks
        chunk_size = 1000000  # Process 1 million rows at a time
        first_chunk = True
        total_rows = 0
        
        for chunk in pd.read_csv(input_path, chunksize=chunk_size):
            # IMPORTANT: Only standardize format, don't filter rows
            standardized_chunk = standardize_chunk(chunk, 'bot_iot_v2', total_rows)
            
            # Write to file (append mode after first chunk)
            standardized_chunk.to_csv(output_file, mode='w' if first_chunk else 'a', 
                               header=first_chunk, index=False)
            
            first_chunk = False
            total_rows += len(chunk)
            print(f"Processed {total_rows} rows so far...")
        
        print(f"Saved standardized BoT-IoT v2 dataset to {output_file}")
        return None  # Can't return the full dataframe when using chunks
    else:
        # For reduced datasets, process all at once
        try:
            df = pd.read_csv(input_path)
            print(f"Loaded reduced BoT-IoT v2 dataset with {len(df)} rows")
            
            # IMPORTANT: Only standardize format, don't filter rows
            standardized_df = standardize_chunk(df, 'bot_iot_v2', 0)
            
            # Save the standardized dataset
            standardized_df.to_csv(output_file, index=False)
            print(f"Saved standardized BoT-IoT v2 dataset to {output_file}")
            print(f"Original row count: {len(df)}, Standardized row count: {len(standardized_df)}")
            
            if len(df) != len(standardized_df):
                print("WARNING: Row count changed during standardization!")
            
            return standardized_df
            
        except Exception as e:
            print(f"Error standardizing BoT-IoT v2 dataset: {e}")
            return None

def standardize_chunk(chunk, dataset_source, offset=0):
    """
    Helper function to standardize a chunk of data
    This ensures consistent standardization across all datasets
    """
    # Make a copy to avoid modifying the original
    df = chunk.copy()
    
    # Check if required columns exist
    missing_cols = [col for col in CORE_FEATURES if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns in {dataset_source}: {missing_cols}")
    
    # Ensure column names are consistent
    df.columns = [col.strip() for col in df.columns]
    
    # Add flow_id column if not present
    if 'flow_id' not in df.columns:
        df['flow_id'] = list(range(offset, offset + len(df)))
    
    # Add dataset_source column
    df['dataset_source'] = dataset_source
    
    # Ensure all columns have correct data types
    # String columns
    str_cols = ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'Attack']
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    # Integer columns
    int_cols = ['L4_SRC_PORT', 'L4_DST_PORT', 'PROTOCOL', 'IN_PKTS', 'OUT_PKTS', 'TCP_FLAGS', 'Label', 'flow_id']
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    # Float columns
    float_cols = ['L7_PROTO', 'IN_BYTES', 'OUT_BYTES', 'FLOW_DURATION_MILLISECONDS']
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(float)
    
    return df

def standardize_cic_ids():
    """Standardize the CIC-IDS NetFlow dataset"""
    print("Standardizing CIC-IDS NetFlow dataset...")
    
    # Check if reduced dataset exists
    reduced_file = os.path.join(NETFLOW_DIR, "nf_cic_ids2018_reduced.csv")
    
    if os.path.exists(reduced_file):
        print(f"Reduced CIC-IDS dataset found at {reduced_file}")
        print("Using reduced dataset for standardization...")
        input_path = reduced_file
        # Use a name indicating this is standardized from a reduced dataset
        output_file = os.path.join(NETFLOW_DIR, "nf_cic_ids2018_reduced_standardized.csv")
    else:
        print("No reduced dataset found, using full CIC-IDS dataset...")
        input_path = CIC_IDS_PATH
        output_file = os.path.join(NETFLOW_DIR, "nf_cic_ids2018_standardized.csv")
    
    # Load the dataset
    try:
        df = pd.read_csv(input_path)
        print(f"Loaded CIC-IDS dataset with {len(df)} rows")
        
        # Use the common standardization function
        standardized_df = standardize_chunk(df, 'cic_ids2018', 0)
        
        # Save the standardized dataset
        standardized_df.to_csv(output_file, index=False)
        print(f"Saved standardized CIC-IDS dataset to {output_file}")
        print(f"Original row count: {len(df)}, Standardized row count: {len(standardized_df)}")
        
        return standardized_df
        
    except Exception as e:
        print(f"Error standardizing CIC-IDS dataset: {e}")
        return None

def standardize_unsw_nb15():
    """Standardize the UNSW-NB15 NetFlow dataset"""
    print("Standardizing UNSW-NB15 NetFlow dataset...")
    
    # Check if reduced dataset exists
    reduced_file = os.path.join(NETFLOW_DIR, "nf_unsw_nb15_reduced.csv")
    
    if os.path.exists(reduced_file):
        print(f"Reduced UNSW-NB15 dataset found at {reduced_file}")
        print("Using reduced dataset for standardization...")
        input_path = reduced_file
        # Use a name indicating this is standardized from a reduced dataset
        output_file = os.path.join(NETFLOW_DIR, "nf_unsw_nb15_reduced_standardized.csv")
    else:
        print("No reduced dataset found, using full UNSW-NB15 dataset...")
        input_path = UNSW_NB15_PATH
        output_file = os.path.join(NETFLOW_DIR, "nf_unsw_nb15_standardized.csv")
    
    # Load the dataset
    try:
        df = pd.read_csv(input_path)
        print(f"Loaded UNSW-NB15 dataset with {len(df)} rows")
        
        # Use the common standardization function
        standardized_df = standardize_chunk(df, 'unsw_nb15', 0)
        
        # Save the standardized dataset
        standardized_df.to_csv(output_file, index=False)
        print(f"Saved standardized UNSW-NB15 dataset to {output_file}")
        print(f"Original row count: {len(df)}, Standardized row count: {len(standardized_df)}")
        
        return standardized_df
        
    except Exception as e:
        print(f"Error standardizing UNSW-NB15 dataset: {e}")
        return None

def combine_netflow_datasets(datasets):
    """Combine all standardized netflow datasets"""
    print("Combining NetFlow datasets...")
    
    if not datasets or all(df is None for df in datasets):
        print("No datasets to combine")
        return
    
    # Filter out None values
    valid_datasets = [df for df in datasets if df is not None]
    
    # Combine all datasets
    combined_df = pd.concat(valid_datasets, ignore_index=True)
    print(f"Combined dataset has {len(combined_df)} rows")
    
    # Ensure flow_id is unique across the combined dataset
    combined_df['flow_id'] = range(len(combined_df))
    
    # Create a unified attack type mapping
    attack_types = sorted(combined_df['Attack'].unique())
    attack_mapping = {attack: idx for idx, attack in enumerate(attack_types)}
    
    # Add attack_type_encoded column
    combined_df['attack_type_encoded'] = combined_df['Attack'].map(attack_mapping)
    
    # Save unified attack mapping
    with open(os.path.join(COMBINED_DIR, 'combined_netflow_attack_mapping.json'), 'w') as f:
        json.dump(attack_mapping, f, indent=2)
    
    # Save combined dataset
    output_file = os.path.join(COMBINED_DIR, 'combined_netflow.csv')
    combined_df.to_csv(output_file, index=False)
    print(f"Saved combined NetFlow dataset to {output_file}")
    
    # Print statistics
    print(f"Total records: {len(combined_df)}")
    print(f"Records per dataset source:")
    for source in combined_df['dataset_source'].unique():
        count = len(combined_df[combined_df['dataset_source'] == source])
        print(f"  {source}: {count}")
    
    # Print attack distribution
    print(f"Attack type distribution:")
    attack_counts = combined_df['Attack'].value_counts()
    for attack, count in attack_counts.items():
        print(f"  {attack}: {count}")
    
    return combined_df

def prepare_cagn_compatibility(combined_df):
    """
    Special handling for CAGN-GAT compatibility with NetFlow data
    """
    print("Preparing CAGN-GAT compatibility layer for NetFlow data...")
    
    if combined_df is None:
        print("No combined dataset available for CAGN-GAT preparation")
        return
    
    # Create a copy of the combined dataset
    cagn_df = combined_df.copy()
    
    # Combine IP and port for node representation
    cagn_df['src_node'] = cagn_df['IPV4_SRC_ADDR'] + ':' + cagn_df['L4_SRC_PORT'].astype(str)
    cagn_df['dst_node'] = cagn_df['IPV4_DST_ADDR'] + ':' + cagn_df['L4_DST_PORT'].astype(str)
    
    # Create protocol name column from protocol number
    protocol_map = {6: 'TCP', 17: 'UDP', 1: 'ICMP'}
    cagn_df['protocol_name'] = cagn_df['PROTOCOL'].map(protocol_map).fillna('OTHER')
    
    # Select and rename columns for CAGN-GAT compatibility
    cagn_cols = {
        'src_node': 'src',
        'dst_node': 'dst',
        'protocol_name': 'protocol',
        'IN_BYTES': 'bytes_in',
        'OUT_BYTES': 'bytes_out',
        'IN_PKTS': 'packets_in',
        'OUT_PKTS': 'packets_out',
        'FLOW_DURATION_MILLISECONDS': 'duration_ms',
        'TCP_FLAGS': 'tcp_flags',
        'L7_PROTO': 'service',
        'Label': 'binary_label',
        'Attack': 'attack_type',
        'attack_type_encoded': 'attack_type_encoded',
        'flow_id': 'flow_id',
        'dataset_source': 'dataset_source'
    }
    
    # Create the CAGN-compatible dataset
    cagn_compatible = cagn_df[[col for col in cagn_cols.keys() if col in cagn_df.columns]].copy()
    cagn_compatible.columns = [cagn_cols[col] for col in cagn_compatible.columns]
    
    # Save CAGN-compatible dataset
    output_file = os.path.join(COMBINED_DIR, 'cagn_compatible_netflow.csv')
    cagn_compatible.to_csv(output_file, index=False)
    print(f"Saved CAGN-GAT compatible NetFlow dataset to {output_file}")
    
    # Create a node mapping file for graph construction
    nodes = pd.concat([
        pd.DataFrame({'node_id': cagn_df['src_node'].unique()}),
        pd.DataFrame({'node_id': cagn_df['dst_node'].unique()})
    ]).drop_duplicates()
    
    nodes['node_index'] = range(len(nodes))
    node_mapping = dict(zip(nodes['node_id'], nodes['node_index']))
    
    with open(os.path.join(COMBINED_DIR, 'netflow_node_mapping.json'), 'w') as f:
        json.dump(node_mapping, f)
    
    print(f"Created node mapping with {len(node_mapping)} unique nodes")
    
    return cagn_compatible

def main():
    parser = argparse.ArgumentParser(description='Standardize NetFlow datasets for cross-model compatibility')
    parser.add_argument('--skip-bot-iot', action='store_true', help='Skip BoT-IoT dataset standardization')
    parser.add_argument('--skip-bot-iot-v2', action='store_true', help='Skip BoT-IoT v2 dataset standardization')
    parser.add_argument('--skip-cic-ids', action='store_true', help='Skip CIC-IDS2018 dataset standardization')
    parser.add_argument('--skip-unsw', action='store_true', help='Skip UNSW-NB15 dataset standardization')
    parser.add_argument('--skip-combine', action='store_true', help='Skip dataset combination')
    parser.add_argument('--skip-cagn', action='store_true', help='Skip CAGN-GAT compatibility preparation')
    parser.add_argument('--use-v2', action='store_true', help='Use BoT-IoT v2 instead of v1 (overrides skip-bot-iot)')
    
    args = parser.parse_args()
    
    # Create output directories
    ensure_directories()
    
    # Standardize individual datasets
    datasets = []
    
    # Handle Bot-IoT versions
    if args.use_v2:
        # If use-v2 is specified, use v2 instead of v1
        if not args.skip_bot_iot_v2:
            bot_iot_v2_df = standardize_bot_iot_v2()
            if bot_iot_v2_df is not None:
                datasets.append(bot_iot_v2_df)
    else:
        # Otherwise follow the skip flags
        if not args.skip_bot_iot:
            bot_iot_df = standardize_bot_iot()
            datasets.append(bot_iot_df)
            
        if not args.skip_bot_iot_v2:
            bot_iot_v2_df = standardize_bot_iot_v2()
            if bot_iot_v2_df is not None:
                datasets.append(bot_iot_v2_df)
    
    if not args.skip_cic_ids:
        cic_ids_df = standardize_cic_ids()
        datasets.append(cic_ids_df)
    
    if not args.skip_unsw:
        unsw_df = standardize_unsw_nb15()
        datasets.append(unsw_df)
    
    # Combine datasets
    combined_df = None
    if not args.skip_combine and datasets:
        combined_df = combine_netflow_datasets(datasets)
    
    # Prepare CAGN-GAT compatibility
    if not args.skip_cagn and combined_df is not None:
        prepare_cagn_compatibility(combined_df)
    
    print("Dataset standardization completed")

if __name__ == '__main__':
    main() 