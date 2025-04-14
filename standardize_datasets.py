#!/usr/bin/env python3
"""
Dataset Standardization Script for Network Intrusion Detection GNN Models

This script standardizes multiple network security datasets (both standard and netflow formats)
to allow cross-model and cross-dataset evaluation of GNN-based intrusion detection systems.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import networkx as nx
import json

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

# Define common schema for standardized datasets
COMMON_SCHEMA = {
    # Common fields across all datasets
    "src_ip": str,
    "src_port": int,
    "dst_ip": str,
    "dst_port": int,
    "protocol": int,  # Numerical protocol identifier
    "protocol_name": str,  # Protocol name (e.g., TCP, UDP)
    "duration_ms": float,  # Flow duration in milliseconds
    "bytes_in": int,  # Bytes from source to destination
    "bytes_out": int,  # Bytes from destination to source
    "packets_in": int,  # Packets from source to destination
    "packets_out": int,  # Packets from destination to source
    "tcp_flags": int,  # TCP flags if applicable
    "binary_label": int,  # Binary label (0=normal, 1=attack)
    "attack_type": str,  # Attack category (e.g., DoS, DDoS, etc.)
    "attack_type_encoded": int  # Encoded attack type
}

# Mapping dictionaries for standardization
PROTOCOL_MAP = {
    1: "ICMP",
    6: "TCP", 
    17: "UDP",
    # Additional protocols as needed
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

def load_dataset(path, dataset_type):
    """Load a dataset and return the DataFrame"""
    print(f"Loading {dataset_type} dataset from {path}")
    try:
        if dataset_type == "cic_ids2018":
            # CSE-CIC-IDS2018 has leading spaces in column names
            df = pd.read_csv(path, skipinitialspace=True)
        else:
            df = pd.read_csv(path)
        print(f"  Loaded successfully with shape: {df.shape}")
        print(f"  Columns: {df.columns.tolist()}")
        
        # Print some sample values for debugging
        if 'label' in df.columns:
            print(f"  Sample 'label' values: {df['label'].value_counts().head()}")
        if 'attack' in df.columns:
            print(f"  Sample 'attack' values: {df['attack'].value_counts().head()}")
        
        return df
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def standardize_bot_iot(df):
    """Standardize BoT-IoT dataset"""
    print("Standardizing BoT-IoT dataset")
    
    # Create a copy with only needed columns
    std_df = pd.DataFrame()
    
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
    std_df['binary_label'] = df['attack'].astype(int)  # This is already 0/1
    
    # Use the attack type from the 'label' field
    std_df['attack_type'] = df.apply(
        lambda x: 'Normal' if x['attack'] == 0 else ATTACK_TYPE_MAP.get(x['label'], 'Other'), 
        axis=1
    )
    
    # Add additional useful features
    std_df['flow_id'] = df.index
    
    # Add statistical features that might be useful
    for col in ['mean', 'stddev', 'sum', 'min', 'max']:
        if col in df.columns:
            std_df[f'stat_{col}'] = df[col]
    
    return std_df

def standardize_cic_ids2018(df):
    """Standardize CSE-CIC-IDS2018 dataset"""
    print("Standardizing CSE-CIC-IDS2018 dataset")
    
    # Create a new dataframe with standardized columns
    std_df = pd.DataFrame()
    
    # Source IP is not directly available - we'll use placeholders
    std_df['src_ip'] = "0.0.0.0"  # Placeholder
    std_df['src_port'] = 0  # Placeholder
    std_df['dst_ip'] = "0.0.0.0"  # Placeholder
    std_df['dst_port'] = df[' Destination Port'].astype(int)
    
    # Protocol 
    std_df['protocol'] = df[' Protocol'].astype(int)
    std_df['protocol_name'] = df[' Protocol'].apply(lambda x: PROTOCOL_MAP.get(x, "Unknown"))
    
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

def standardize_unsw_nb15(df):
    """Standardize UNSW-NB15 dataset"""
    print("Standardizing UNSW-NB15 dataset")
    
    # Create a new dataframe with standardized columns
    std_df = pd.DataFrame()
    
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

def standardize_nf_bot_iot(df):
    """Standardize NF-BoT-IoT dataset"""
    print("Standardizing NF-BoT-IoT dataset")
    
    # Create a new dataframe with standardized columns
    std_df = pd.DataFrame()
    
    # Basic flow identifiers
    std_df['src_ip'] = df['IPV4_SRC_ADDR']
    std_df['src_port'] = df['L4_SRC_PORT'].astype(int)
    std_df['dst_ip'] = df['IPV4_DST_ADDR']
    std_df['dst_port'] = df['L4_DST_PORT'].astype(int)
    
    # Protocol information
    std_df['protocol'] = df['PROTOCOL'].astype(int)
    std_df['protocol_name'] = df['PROTOCOL'].apply(lambda x: PROTOCOL_MAP.get(x, "Unknown"))
    
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

def standardize_nf_cic_ids2018(df):
    """Standardize NF-CSE-CIC-IDS2018 dataset"""
    print("Standardizing NF-CSE-CIC-IDS2018 dataset")
    
    # Create a new dataframe with standardized columns
    std_df = pd.DataFrame()
    
    # Basic flow identifiers
    std_df['src_ip'] = df['IPV4_SRC_ADDR']
    std_df['src_port'] = df['L4_SRC_PORT'].astype(int)
    std_df['dst_ip'] = df['IPV4_DST_ADDR']
    std_df['dst_port'] = df['L4_DST_PORT'].astype(int)
    
    # Protocol information
    std_df['protocol'] = df['PROTOCOL'].astype(int)
    std_df['protocol_name'] = df['PROTOCOL'].apply(lambda x: PROTOCOL_MAP.get(x, "Unknown"))
    
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

def standardize_nf_unsw_nb15(df):
    """Standardize NF-UNSW-NB15 dataset"""
    print("Standardizing NF-UNSW-NB15 dataset")
    
    # Create a new dataframe with standardized columns
    std_df = pd.DataFrame()
    
    # Basic flow identifiers
    std_df['src_ip'] = df['IPV4_SRC_ADDR']
    std_df['src_port'] = df['L4_SRC_PORT'].astype(int)
    std_df['dst_ip'] = df['IPV4_DST_ADDR']
    std_df['dst_port'] = df['L4_DST_PORT'].astype(int)
    
    # Protocol information
    std_df['protocol'] = df['PROTOCOL'].astype(int)
    std_df['protocol_name'] = df['PROTOCOL'].apply(lambda x: PROTOCOL_MAP.get(x, "Unknown"))
    
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

def create_graph_representation(df, dataset_name):
    """Create a graph representation of the dataset"""
    print(f"Creating graph representation for {dataset_name}")
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes (IP addresses)
    src_ips = df['src_ip'].unique()
    dst_ips = df['dst_ip'].unique()
    all_ips = list(set(src_ips) | set(dst_ips))
    
    # Add nodes with features
    for ip in all_ips:
        # Aggregate features for this IP
        src_flows = df[df['src_ip'] == ip]
        dst_flows = df[df['dst_ip'] == ip]
        all_flows = pd.concat([src_flows, dst_flows])
        
        if len(all_flows) > 0:
            # Node features - basic statistics
            features = {
                'total_flows': int(len(all_flows)),
                'attack_flows': int(all_flows['binary_label'].sum()),
                'normal_flows': int(len(all_flows) - all_flows['binary_label'].sum()),
                'is_source': bool(len(src_flows) > 0),
                'is_destination': bool(len(dst_flows) > 0),
                'avg_bytes_sent': float(src_flows['bytes_in'].mean() if len(src_flows) > 0 else 0),
                'avg_bytes_received': float(dst_flows['bytes_out'].mean() if len(dst_flows) > 0 else 0),
                'avg_packets_sent': float(src_flows['packets_in'].mean() if len(src_flows) > 0 else 0),
                'avg_packets_received': float(dst_flows['packets_out'].mean() if len(dst_flows) > 0 else 0),
            }
            
            # Label - majority class
            if len(all_flows) > 0:
                majority_label = int(1 if all_flows['binary_label'].mean() >= 0.5 else 0)
                majority_attack = str(all_flows['attack_type'].value_counts().index[0])
            else:
                majority_label = 0
                majority_attack = 'Normal'
                
            G.add_node(ip, **features, label=majority_label, attack_type=majority_attack)
    
    # Add edges (connections between IPs)
    for _, flow in df.iterrows():
        src = flow['src_ip']
        dst = flow['dst_ip']
        
        # Skip if source or destination is not in the graph
        if src not in G.nodes or dst not in G.nodes:
            continue
            
        # Convert numpy types to Python native types for networkx compatibility
        flow_id = int(flow['flow_id'])
        bytes_in = int(flow['bytes_in'])
        bytes_out = int(flow['bytes_out'])
        packets_in = int(flow['packets_in'])
        packets_out = int(flow['packets_out'])
        binary_label = int(flow['binary_label'])
            
        # If edge already exists, update its features
        if G.has_edge(src, dst):
            G[src][dst]['weight'] += 1
            G[src][dst]['flows'].append(flow_id)
            G[src][dst]['total_bytes'] += bytes_in + bytes_out
            G[src][dst]['total_packets'] += packets_in + packets_out
            
            # Update attack information
            if binary_label == 1:
                G[src][dst]['attack_flows'] += 1
        else:
            # Create new edge with features
            G.add_edge(src, dst, 
                      weight=1, 
                      flows=[flow_id], 
                      total_bytes=bytes_in + bytes_out,
                      total_packets=packets_in + packets_out,
                      attack_flows=1 if binary_label == 1 else 0)
    
    print(f"  Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G

def save_standardized_dataset(df, name, dataset_type, attack_mapping=None):
    """Save the standardized dataset to CSV"""
    output_path = os.path.join(OUTPUT_DIR, dataset_type, f"{name}_standardized.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved standardized dataset to {output_path}")
    
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
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
        'density': nx.density(G),
        'avg_degree': float(sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0),
    }
    
    metrics_path = os.path.join(OUTPUT_DIR, dataset_type, f"{name}_graph_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2, cls=NumpyEncoder)
    print(f"Saved graph metrics to {metrics_path}")

def process_dataset(name, path, dataset_type, standardize_func):
    """Process a dataset: load, standardize, encode, create graph and save"""
    # Load the dataset
    df = load_dataset(path, name)
    if df is None:
        return
    
    # Standardize
    std_df = standardize_func(df)
    
    # Encode attack types
    std_df, attack_mapping = encode_attack_types(std_df)
    
    # Save standardized dataset
    save_standardized_dataset(std_df, name, dataset_type, attack_mapping)
    
    # Create and save graph representation
    G = create_graph_representation(std_df, name)
    save_graph(G, name, dataset_type)
    
    return std_df

def main():
    """Main function to process all datasets"""
    print("Starting dataset standardization process")
    
    # Process standard datasets
    bot_iot_df = process_dataset('bot_iot', DATASETS['standard']['bot_iot'], 'standard', standardize_bot_iot)
    cic_ids2018_df = process_dataset('cic_ids2018', DATASETS['standard']['cic_ids2018'], 'standard', standardize_cic_ids2018)
    unsw_nb15_df = process_dataset('unsw_nb15', DATASETS['standard']['unsw_nb15'], 'standard', standardize_unsw_nb15)
    
    # Process netflow datasets
    nf_bot_iot_df = process_dataset('nf_bot_iot', DATASETS['netflow']['nf_bot_iot'], 'netflow', standardize_nf_bot_iot)
    nf_cic_ids2018_df = process_dataset('nf_cic_ids2018', DATASETS['netflow']['nf_cic_ids2018'], 'netflow', standardize_nf_cic_ids2018)
    nf_unsw_nb15_df = process_dataset('nf_unsw_nb15', DATASETS['netflow']['nf_unsw_nb15'], 'netflow', standardize_nf_unsw_nb15)
    
    print("Dataset standardization completed successfully")

if __name__ == "__main__":
    main() 