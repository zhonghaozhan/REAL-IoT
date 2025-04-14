#!/usr/bin/env python3
"""
Optimized Dataset Standardization Script for Network Intrusion Detection GNN Models

This script standardizes multiple network security datasets using dask for memory efficiency.
"""

import os
import pandas as pd
import numpy as np
import json
import dask.dataframe as dd
from sklearn.preprocessing import LabelEncoder
import networkx as nx
import warnings
import time

warnings.filterwarnings('ignore')

# Base directories
INPUT_DIR = "/media/ssd/test/GNN/kaggle/input"
OUTPUT_DIR = "/media/ssd/test/standardized-datasets"

# Create output directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "standard"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "netflow"), exist_ok=True)

# Define dataset paths
DATASETS = {
    "standard": {
        "bot_iot": os.path.join(INPUT_DIR, "BoT-IoT", "bot_reduced.csv"),
        "cic_ids2018": os.path.join(INPUT_DIR, "CSE-CIC-IDS2018", "CSE-CIC-ID2018-Converted.csv"),
        "unsw_nb15": os.path.join(INPUT_DIR, "unsw-nb15", "UNSW_NB15_training-set.csv")
    },
    "netflow": {
        "nf_bot_iot": os.path.join(INPUT_DIR, "NF-BoT-IoT", "NF-BoT-IoT.csv"),
        "nf_cic_ids2018": os.path.join(INPUT_DIR, "NF-CSE-CIC-IDS2018-v2", "NF-CSE-CIC-IDS2018-v2.csv"),
        "nf_unsw_nb15": os.path.join(INPUT_DIR, "nf-unsw-nb15-v2", "NF-UNSW-NB15-v2.csv")
    }
}

# Attack type standardization
ATTACK_TYPE_MAP = {
    # Bot-IoT attack categories
    "DDoS": "DDoS",
    "DoS": "DoS",
    "Reconnaissance": "Reconnaissance",
    "Theft": "Information_Theft",
    
    # UNSW-NB15 attack categories
    "Fuzzers": "Fuzzers",
    "Analysis": "Analysis",
    "Backdoor": "Backdoor",
    "DoS": "DoS",
    "Exploits": "Exploits",
    "Generic": "Generic",
    "Reconnaissance": "Reconnaissance",
    "Shellcode": "Shellcode",
    "Worms": "Worms",
    
    # CSE-CIC-IDS2018 attack categories
    "Benign": "Normal",
    "Bot": "Bot",
    "Brute Force": "Brute_Force",
    "Brute Force -Web": "Brute_Force",
    "Brute Force -XSS": "Brute_Force",
    "DDOS attack-HOIC": "DDoS",
    "DDoS attacks-LOIC-HTTP": "DDoS",
    "DoS attacks-GoldenEye": "DoS",
    "DoS attacks-Hulk": "DoS",
    "DoS attacks-SlowHTTPTest": "DoS",
    "DoS attacks-Slowloris": "DoS",
    "FTP-BruteForce": "Brute_Force",
    "Infiltration": "Infiltration",
    "SQL Injection": "Injection",
    "SSH-Bruteforce": "Brute_Force"
}

# JSON encoder to handle numpy data types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def load_dataset(path, dataset_type, sample_fraction=None):
    """Load a dataset and return the DataFrame, with option to load a sample"""
    try:
        start_time = time.time()
        print(f"Loading {dataset_type} dataset from {path}")
        
        if dataset_type == "cic_ids2018":
            # CSE-CIC-IDS2018 has leading spaces in column names
            if sample_fraction:
                df = pd.read_csv(path, skipinitialspace=True, nrows=int(sample_fraction * 1000000))
            else:
                df = pd.read_csv(path, skipinitialspace=True)
        else:
            if sample_fraction:
                df = pd.read_csv(path, nrows=int(sample_fraction * 1000000))
            else:
                df = pd.read_csv(path)
                
        end_time = time.time()
        print(f"  Loaded successfully with shape: {df.shape}, took {end_time - start_time:.2f} seconds")
        
        # Print 3 sample rows to understand the data
        print("  Sample data:")
        print(df.head(3))
        
        return df
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def process_chunk(chunk, dataset_type, chunk_id):
    """Process a dataframe chunk based on dataset type"""
    if dataset_type == 'bot_iot':
        return standardize_bot_iot_chunk(chunk)
    elif dataset_type == 'cic_ids2018':
        return standardize_cic_ids2018_chunk(chunk)
    elif dataset_type == 'unsw_nb15':
        return standardize_unsw_nb15_chunk(chunk)
    elif dataset_type == 'nf_bot_iot':
        return standardize_nf_bot_iot_chunk(chunk)
    elif dataset_type == 'nf_cic_ids2018':
        return standardize_nf_cic_ids2018_chunk(chunk)
    elif dataset_type == 'nf_unsw_nb15':
        return standardize_nf_unsw_nb15_chunk(chunk)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

def standardize_bot_iot_chunk(df):
    """Standardize BoT-IoT dataset chunk"""
    # Create a copy with only needed columns
    std_df = pd.DataFrame(index=df.index)
    
    # Map columns to standard schema
    std_df['src_ip'] = df['saddr'].astype(str).str.split(':', n=1).str[0]
    std_df['src_port'] = pd.to_numeric(df['saddr'].astype(str).str.split(':', n=1).str[1], errors='coerce')
    std_df['dst_ip'] = df['daddr'].astype(str).str.split(':', n=1).str[0]
    std_df['dst_port'] = pd.to_numeric(df['daddr'].astype(str).str.split(':', n=1).str[1], errors='coerce')
    
    # Handle protocol
    std_df['protocol'] = df['proto_number'].astype(int)
    std_df['protocol_name'] = df['proto'].astype(str)
    
    # Flow metrics
    std_df['duration_ms'] = df['dur'] * 1000  # Convert to milliseconds
    std_df['bytes_in'] = df['sbytes'].astype(int)
    std_df['bytes_out'] = df['dbytes'].astype(int)
    std_df['packets_in'] = df['spkts'].astype(int)
    std_df['packets_out'] = df['dpkts'].astype(int)
    
    # TCP flags (if available)
    std_df['tcp_flags'] = 0  # Default value
    
    # Labels - the 'attack' field is 0/1, and 'label' contains the attack type
    std_df['binary_label'] = df['attack'].astype(int)
    
    # Use the attack type from the 'label' field
    std_df['attack_type'] = df.apply(
        lambda x: 'Normal' if x['attack'] == 0 else ATTACK_TYPE_MAP.get(x['label'], 'Other'), 
        axis=1
    )
    
    # Add flow ID
    std_df['flow_id'] = df.index
    
    # Add statistical features that might be useful
    for col in ['mean', 'stddev', 'sum', 'min', 'max']:
        if col in df.columns:
            std_df[f'stat_{col}'] = df[col]
    
    return std_df

def standardize_cic_ids2018_chunk(df):
    """Standardize CSE-CIC-IDS2018 dataset chunk"""
    # Create a new dataframe with standardized columns
    std_df = pd.DataFrame(index=df.index)
    
    # Source IP is not directly available - we'll use placeholders
    std_df['src_ip'] = "0.0.0.0"  # Placeholder
    std_df['src_port'] = 0  # Placeholder
    std_df['dst_ip'] = "0.0.0.0"  # Placeholder
    std_df['dst_port'] = df[' Destination Port'].astype(int)
    
    # Protocol 
    std_df['protocol'] = df[' Protocol'].astype(int)
    
    protocol_map = {1: "ICMP", 6: "TCP", 17: "UDP"}
    std_df['protocol_name'] = df[' Protocol'].apply(lambda x: protocol_map.get(x, "Unknown"))
    
    # Flow metrics
    std_df['duration_ms'] = df[' Flow Duration'].astype(float)
    std_df['bytes_in'] = df[' Total Length of Fwd Packets'].astype(int)
    std_df['bytes_out'] = df[' Total Length of Bwd Packets'].astype(int)
    std_df['packets_in'] = df[' Total Fwd Packets'].astype(int)
    std_df['packets_out'] = df[' Total Backward Packets'].astype(int)
    
    # TCP flags
    flags_cols = [' FIN Flag Count', ' SYN Flag Count', ' RST Flag Count', 
                  ' PSH Flag Count', ' ACK Flag Count', ' URG Flag Count']
    
    # Create a combined flag field
    tcp_flags = 0
    for i, flag in enumerate(flags_cols):
        if flag in df.columns:
            tcp_flags |= (df[flag].astype(int) > 0).astype(int) << i
    
    std_df['tcp_flags'] = tcp_flags
    
    # Labels
    std_df['binary_label'] = df[' Label'].apply(lambda x: 0 if x == 'BENIGN' else 1).astype(int)
    std_df['attack_type'] = df[' Label'].apply(lambda x: 'Normal' if x == 'BENIGN' else ATTACK_TYPE_MAP.get(x, 'Other'))
    
    # Add flow ID
    std_df['flow_id'] = df.index
    
    # Add statistical features
    for col_name, std_col in [
        (' Fwd Packet Length Mean', 'stat_fwd_len_mean'),
        (' Fwd Packet Length Std', 'stat_fwd_len_std'),
        (' Bwd Packet Length Mean', 'stat_bwd_len_mean'),
        (' Bwd Packet Length Std', 'stat_bwd_len_std'),
        (' Flow IAT Mean', 'stat_flow_iat_mean'),
        (' Flow IAT Std', 'stat_flow_iat_std')
    ]:
        if col_name in df.columns:
            std_df[std_col] = df[col_name]
    
    return std_df

def standardize_unsw_nb15_chunk(df):
    """Standardize UNSW-NB15 dataset chunk"""
    # Create a new dataframe with standardized columns
    std_df = pd.DataFrame(index=df.index)
    
    # Basic flow identifiers
    std_df['src_ip'] = "0.0.0.0"  # Not available directly in the dataset
    std_df['src_port'] = 0  # Placeholder
    std_df['dst_ip'] = "0.0.0.0"  # Not available directly in the dataset
    std_df['dst_port'] = 0  # Placeholder
    
    # Protocol information
    proto_mapping = {'tcp': 6, 'udp': 17, 'icmp': 1}
    std_df['protocol'] = df['proto'].map(proto_mapping).fillna(0).astype(int)
    std_df['protocol_name'] = df['proto']
    
    # Flow metrics
    std_df['duration_ms'] = df['dur'] * 1000  # Convert to milliseconds
    std_df['bytes_in'] = df['sbytes'].astype(int)
    std_df['bytes_out'] = df['dbytes'].astype(int)
    std_df['packets_in'] = df['spkts'].astype(int)
    std_df['packets_out'] = df['dpkts'].astype(int)
    
    # TCP flags - not directly available in same format
    std_df['tcp_flags'] = 0  # Default placeholder
    
    # Labels
    std_df['binary_label'] = df['label'].astype(int)
    std_df['attack_type'] = df.apply(
        lambda x: 'Normal' if x['label'] == 0 else ATTACK_TYPE_MAP.get(x['attack_cat'], 'Other'), 
        axis=1
    )
    
    # Add flow ID 
    std_df['flow_id'] = df['id']
    
    # Add statistical features
    for col_name, std_col in [
        ('smean', 'stat_src_mean'),
        ('dmean', 'stat_dst_mean'),
        ('sload', 'stat_src_load'),
        ('dload', 'stat_dst_load')
    ]:
        if col_name in df.columns:
            std_df[std_col] = df[col_name]
    
    return std_df

def standardize_nf_bot_iot_chunk(df):
    """Standardize NF-BoT-IoT dataset chunk"""
    std_df = pd.DataFrame(index=df.index)
    
    # Basic flow identifiers
    std_df['src_ip'] = df['IPV4_SRC_ADDR']
    std_df['src_port'] = df['L4_SRC_PORT'].astype(int)
    std_df['dst_ip'] = df['IPV4_DST_ADDR']
    std_df['dst_port'] = df['L4_DST_PORT'].astype(int)
    
    # Protocol information
    std_df['protocol'] = df['PROTOCOL'].astype(int)
    
    protocol_map = {1: "ICMP", 6: "TCP", 17: "UDP"}
    std_df['protocol_name'] = df['PROTOCOL'].apply(lambda x: protocol_map.get(x, "Unknown"))
    
    # Flow metrics
    std_df['duration_ms'] = df['FLOW_DURATION_MILLISECONDS'].astype(float)
    std_df['bytes_in'] = df['IN_BYTES'].astype(int)
    std_df['bytes_out'] = df['OUT_BYTES'].astype(int)
    std_df['packets_in'] = df['IN_PKTS'].astype(int)
    std_df['packets_out'] = df['OUT_PKTS'].astype(int)
    
    # TCP flags
    std_df['tcp_flags'] = df['TCP_FLAGS'].fillna(0).astype(int)
    
    # Labels
    std_df['binary_label'] = df['Label'].astype(int)
    std_df['attack_type'] = df['Attack'].apply(lambda x: 'Normal' if x == 'Normal' else ATTACK_TYPE_MAP.get(x, 'Other'))
    
    # Add flow ID
    std_df['flow_id'] = df.index
    
    # L7 protocol as additional feature
    if 'L7_PROTO' in df.columns:
        std_df['l7_proto'] = df['L7_PROTO']
    
    return std_df

def standardize_nf_cic_ids2018_chunk(df):
    """Standardize NF-CSE-CIC-IDS2018 dataset chunk"""
    std_df = pd.DataFrame(index=df.index)
    
    # Basic flow identifiers
    std_df['src_ip'] = df['IPV4_SRC_ADDR']
    std_df['src_port'] = df['L4_SRC_PORT'].astype(int)
    std_df['dst_ip'] = df['IPV4_DST_ADDR']
    std_df['dst_port'] = df['L4_DST_PORT'].astype(int)
    
    # Protocol information
    std_df['protocol'] = df['PROTOCOL'].astype(int)
    
    protocol_map = {1: "ICMP", 6: "TCP", 17: "UDP"}
    std_df['protocol_name'] = df['PROTOCOL'].apply(lambda x: protocol_map.get(x, "Unknown"))
    
    # Flow metrics
    std_df['duration_ms'] = df['FLOW_DURATION_MILLISECONDS'].astype(float)
    std_df['bytes_in'] = df['IN_BYTES'].astype(int)
    std_df['bytes_out'] = df['OUT_BYTES'].astype(int)
    std_df['packets_in'] = df['IN_PKTS'].astype(int)
    std_df['packets_out'] = df['OUT_PKTS'].astype(int)
    
    # TCP flags
    std_df['tcp_flags'] = df['TCP_FLAGS'].fillna(0).astype(int)
    
    # Labels
    std_df['binary_label'] = df['Label'].astype(int)
    std_df['attack_type'] = df['Attack'].apply(lambda x: 'Normal' if x == 'Normal' else ATTACK_TYPE_MAP.get(x, 'Other'))
    
    # Add flow ID
    std_df['flow_id'] = df.index
    
    # Add additional netflow-specific features
    netflow_cols = [
        'L7_PROTO', 'CLIENT_TCP_FLAGS', 'SERVER_TCP_FLAGS', 
        'MIN_TTL', 'MAX_TTL', 'MIN_IP_PKT_LEN', 'MAX_IP_PKT_LEN'
    ]
    
    for col in netflow_cols:
        if col in df.columns:
            std_df[col.lower()] = df[col]
    
    return std_df

def standardize_nf_unsw_nb15_chunk(df):
    """Standardize NF-UNSW-NB15 dataset chunk"""
    std_df = pd.DataFrame(index=df.index)
    
    # Basic flow identifiers
    std_df['src_ip'] = df['IPV4_SRC_ADDR']
    std_df['src_port'] = df['L4_SRC_PORT'].astype(int)
    std_df['dst_ip'] = df['IPV4_DST_ADDR']
    std_df['dst_port'] = df['L4_DST_PORT'].astype(int)
    
    # Protocol information
    std_df['protocol'] = df['PROTOCOL'].astype(int)
    
    protocol_map = {1: "ICMP", 6: "TCP", 17: "UDP"}
    std_df['protocol_name'] = df['PROTOCOL'].apply(lambda x: protocol_map.get(x, "Unknown"))
    
    # Flow metrics
    std_df['duration_ms'] = df['FLOW_DURATION_MILLISECONDS'].astype(float)
    std_df['bytes_in'] = df['IN_BYTES'].astype(int)
    std_df['bytes_out'] = df['OUT_BYTES'].astype(int)
    std_df['packets_in'] = df['IN_PKTS'].astype(int)
    std_df['packets_out'] = df['OUT_PKTS'].astype(int)
    
    # TCP flags
    std_df['tcp_flags'] = df['TCP_FLAGS'].fillna(0).astype(int)
    
    # Labels
    std_df['binary_label'] = df['Label'].astype(int)
    std_df['attack_type'] = df['Attack'].apply(lambda x: 'Normal' if x == 'Normal' else ATTACK_TYPE_MAP.get(x, 'Other'))
    
    # Add flow ID
    std_df['flow_id'] = df.index
    
    # Add additional netflow-specific features
    netflow_cols = [
        'L7_PROTO', 'CLIENT_TCP_FLAGS', 'SERVER_TCP_FLAGS', 
        'MIN_TTL', 'MAX_TTL', 'MIN_IP_PKT_LEN', 'MAX_IP_PKT_LEN'
    ]
    
    for col in netflow_cols:
        if col in df.columns:
            std_df[col.lower()] = df[col]
    
    return std_df

def encode_attack_types(df):
    """Encode attack types to numerical values"""
    le = LabelEncoder()
    df['attack_type_encoded'] = le.fit_transform(df['attack_type'])
    
    # Save the mapping - convert numpy types to Python native types for JSON serialization
    attack_mapping = {str(k): int(v) for k, v in zip(le.classes_, le.transform(le.classes_))}
    return df, attack_mapping

def create_sample_graph(df, dataset_name, sample_size=10000):
    """Create a sample graph representation for quick analysis"""
    print(f"Creating sample graph representation for {dataset_name} with {sample_size} rows")
    
    # Use a sample for large datasets
    df_sample = df.sample(min(sample_size, len(df)))
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes (IP addresses)
    src_ips = df_sample['src_ip'].unique()
    dst_ips = df_sample['dst_ip'].unique()
    all_ips = list(set(src_ips) | set(dst_ips))
    
    # Add nodes with features
    for ip in all_ips:
        # Aggregate features for this IP
        src_flows = df_sample[df_sample['src_ip'] == ip]
        dst_flows = df_sample[df_sample['dst_ip'] == ip]
        all_flows = pd.concat([src_flows, dst_flows])
        
        if len(all_flows) > 0:
            # Node features - basic statistics
            features = {
                'total_flows': int(len(all_flows)),
                'attack_flows': int(all_flows['binary_label'].sum()),
                'normal_flows': int(len(all_flows) - all_flows['binary_label'].sum()),
                'is_source': bool(len(src_flows) > 0),
                'is_destination': bool(len(dst_flows) > 0)
            }
            
            # Label - majority class
            majority_label = int(1 if all_flows['binary_label'].mean() >= 0.5 else 0)
            majority_attack = str(all_flows['attack_type'].value_counts().index[0])
                
            G.add_node(ip, **features, label=majority_label, attack_type=majority_attack)
    
    # Add edges (connections between IPs)
    for _, flow in df_sample.iterrows():
        src = flow['src_ip']
        dst = flow['dst_ip']
        
        # Skip if source or destination is not in the graph
        if src not in G.nodes or dst not in G.nodes:
            continue
            
        # Convert numpy types to Python native types for networkx compatibility
        binary_label = int(flow['binary_label'])
            
        # Create the edge if it doesn't exist yet
        if not G.has_edge(src, dst):
            G.add_edge(src, dst, weight=1, attack_flows=1 if binary_label == 1 else 0)
    
    print(f"  Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G

def save_standardized_dataset(df, name, dataset_type, attack_mapping=None):
    """Save the standardized dataset to CSV"""
    output_path = os.path.join(OUTPUT_DIR, dataset_type, f"{name}_standardized.csv")
    
    # Save in chunks to avoid memory issues
    chunk_size = 100000
    total_rows = len(df)
    
    print(f"Saving standardized dataset in chunks to {output_path}")
    for i in range(0, total_rows, chunk_size):
        end_idx = min(i + chunk_size, total_rows)
        mode = 'w' if i == 0 else 'a'
        header = i == 0  # only include header in first chunk
        
        # Get chunk and save
        chunk = df.iloc[i:end_idx]
        chunk.to_csv(output_path, mode=mode, header=header, index=False)
        
        print(f"  Saved chunk {i//chunk_size + 1}/{(total_rows-1)//chunk_size + 1} " +
              f"({i}-{end_idx-1} of {total_rows} rows)")
    
    # Save attack mapping
    if attack_mapping:
        mapping_path = os.path.join(OUTPUT_DIR, dataset_type, f"{name}_attack_mapping.json")
        with open(mapping_path, 'w') as f:
            json.dump(attack_mapping, f, indent=2, cls=NumpyEncoder)
        print(f"Saved attack mapping to {mapping_path}")

def save_graph(G, name, dataset_type):
    """Save the graph representation"""
    # Save in GraphML format
    output_path = os.path.join(OUTPUT_DIR, dataset_type, f"{name}_graph.graphml")
    nx.write_graphml(G, output_path)
    print(f"Saved graph to {output_path}")
    
    # Also save basic graph metrics
    metrics = {
        'nodes': int(G.number_of_nodes()),
        'edges': int(G.number_of_edges()),
        'density': float(nx.density(G)),
        'avg_degree': float(sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0),
    }
    
    metrics_path = os.path.join(OUTPUT_DIR, dataset_type, f"{name}_graph_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2, cls=NumpyEncoder)
    print(f"Saved graph metrics to {metrics_path}")

def process_dataset(name, path, dataset_type, standardize_func, sample_size=0.1):
    """Process a dataset: load, standardize, encode, create graph and save"""
    # Load a sample of the dataset for large datasets (only for exploration)
    # For actual processing, we'll use chunks
    df = load_dataset(path, name, sample_fraction=sample_size)
    if df is None:
        return
    
    print(f"Processing dataset {name} in memory-efficient manner...")
    
    # Create a standardized version by processing the dataset in chunks
    chunk_size = 100000
    
    # Create temp directory for chunks
    temp_dir = os.path.join(OUTPUT_DIR, "temp", dataset_type)
    os.makedirs(temp_dir, exist_ok=True)
    
    standardized_chunks = []
    
    # Load and process in chunks
    try:
        # Get total rows
        df_count = len(pd.read_csv(path, nrows=0))
        
        # Process chunks
        for chunk_i, chunk_df in enumerate(pd.read_csv(path, chunksize=chunk_size)):
            print(f"Processing chunk {chunk_i+1}/{(df_count-1)//chunk_size + 1} " +
                  f"({chunk_i*chunk_size}-{min((chunk_i+1)*chunk_size-1, df_count-1)} of ~{df_count} rows)")
            
            # Create standardized chunk
            std_chunk = process_chunk(chunk_df, name, chunk_i)
            
            # Save chunk temporarily
            chunk_path = os.path.join(temp_dir, f"{name}_chunk_{chunk_i}.csv")
            std_chunk.to_csv(chunk_path, index=False)
            
            standardized_chunks.append(chunk_path)
            
            # For testing, only process a limited number of chunks
            if chunk_i >= 9:  # limit to 10 chunks (~1M rows) for test
                break
        
        # Get column names from first chunk
        first_chunk = pd.read_csv(standardized_chunks[0])
        column_names = first_chunk.columns.tolist()
        
        # Now process attack types - this needs to be done on the whole dataset
        # Load all chunks and collect attack types
        attack_types = set()
        for chunk_path in standardized_chunks:
            chunk_df = pd.read_csv(chunk_path)
            attack_types.update(chunk_df['attack_type'].unique())
        
        # Create label encoder
        le = LabelEncoder()
        le.fit(list(attack_types))
        
        # Save the mapping
        attack_mapping = {str(k): int(v) for k, v in zip(le.classes_, le.transform(le.classes_))}
        
        # Now create final file with encoding
        output_path = os.path.join(OUTPUT_DIR, dataset_type, f"{name}_standardized.csv")
        with open(output_path, 'w') as f:
            # Write header
            f.write(','.join(column_names) + ',attack_type_encoded\n')
            
            # Process each chunk and append
            for chunk_i, chunk_path in enumerate(standardized_chunks):
                chunk_df = pd.read_csv(chunk_path)
                
                # Add encoded attack type
                chunk_df['attack_type_encoded'] = chunk_df['attack_type'].apply(
                    lambda x: attack_mapping[str(x)] if str(x) in attack_mapping else -1
                )
                
                # Append to final file
                chunk_df.to_csv(f, mode='a', header=False, index=False)
        
        # Save attack mapping
        mapping_path = os.path.join(OUTPUT_DIR, dataset_type, f"{name}_attack_mapping.json")
        with open(mapping_path, 'w') as f:
            json.dump(attack_mapping, f, indent=2, cls=NumpyEncoder)
        print(f"Saved attack mapping to {mapping_path}")
        
        # Create a sample graph representation for visualization
        sample_df = pd.concat([pd.read_csv(chunk) for chunk in standardized_chunks[:2]], ignore_index=True)
        G = create_sample_graph(sample_df, name)
        save_graph(G, name, dataset_type)
        
        # Cleanup temporary files
        for chunk_path in standardized_chunks:
            if os.path.exists(chunk_path):
                os.remove(chunk_path)
        
        print(f"Finished processing {name} dataset")
        
    except Exception as e:
        print(f"Error processing {name}: {e}")
        import traceback
        traceback.print_exc()
    
    return True

def main():
    """Main function to process all datasets"""
    print("Starting optimized dataset standardization process")
    
    # Process standard datasets
    process_dataset('bot_iot', DATASETS['standard']['bot_iot'], 'standard', standardize_bot_iot_chunk)
    process_dataset('cic_ids2018', DATASETS['standard']['cic_ids2018'], 'standard', standardize_cic_ids2018_chunk)
    process_dataset('unsw_nb15', DATASETS['standard']['unsw_nb15'], 'standard', standardize_unsw_nb15_chunk)
    
    # Process netflow datasets
    process_dataset('nf_bot_iot', DATASETS['netflow']['nf_bot_iot'], 'netflow', standardize_nf_bot_iot_chunk)
    process_dataset('nf_cic_ids2018', DATASETS['netflow']['nf_cic_ids2018'], 'netflow', standardize_nf_cic_ids2018_chunk)
    process_dataset('nf_unsw_nb15', DATASETS['netflow']['nf_unsw_nb15'], 'netflow', standardize_nf_unsw_nb15_chunk)
    
    print("Dataset standardization completed successfully")

if __name__ == "__main__":
    main() 