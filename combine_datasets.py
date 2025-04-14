#!/usr/bin/env python3
"""
Dataset Combination Script for Network Intrusion Detection GNN Models

This script combines the standardized network security datasets into unified datasets
for cross-model and cross-dataset evaluation of intrusion detection models.
"""

import os
import pandas as pd
import numpy as np
import json
import networkx as nx
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Base directory for standardized datasets
INPUT_DIR = "/media/ssd/test/standardized-datasets"
OUTPUT_DIR = "/media/ssd/test/standardized-datasets/combined"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define dataset paths
DATASETS = {
    "standard": {
        "bot_iot": os.path.join(INPUT_DIR, "standard", "bot_iot_standardized.csv"),
        "cic_ids2018": os.path.join(INPUT_DIR, "standard", "cic_ids2018_standardized.csv"),
        "unsw_nb15": os.path.join(INPUT_DIR, "standard", "unsw_nb15_standardized.csv")
    },
    "netflow": {
        "nf_bot_iot": os.path.join(INPUT_DIR, "netflow", "nf_bot_iot_standardized.csv"),
        "nf_cic_ids2018": os.path.join(INPUT_DIR, "netflow", "nf_cic_ids2018_standardized.csv"),
        "nf_unsw_nb15": os.path.join(INPUT_DIR, "netflow", "nf_unsw_nb15_standardized.csv")
    }
}

# Common columns to be included in the combined dataset
COMMON_COLUMNS = [
    'src_ip', 'src_port', 'dst_ip', 'dst_port', 'protocol', 'protocol_name',
    'duration_ms', 'bytes_in', 'bytes_out', 'packets_in', 'packets_out',
    'tcp_flags', 'binary_label', 'attack_type', 'attack_type_encoded'
]

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

def load_dataset(path, dataset_name):
    """Load a standardized dataset"""
    print(f"Loading {dataset_name} from {path}")
    try:
        df = pd.read_csv(path)
        print(f"  Loaded successfully with shape: {df.shape}")
        
        # Add dataset source column
        df['dataset_source'] = dataset_name
        
        return df
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def combine_standard_datasets():
    """Combine the standardized datasets into a single dataset"""
    print("Combining standard datasets...")
    
    datasets = []
    
    # Load all standard datasets
    for name, path in DATASETS["standard"].items():
        df = load_dataset(path, name)
        if df is not None:
            # Select only common columns and source column
            columns_to_use = COMMON_COLUMNS + ['dataset_source']
            # For columns that don't exist, add with default values
            for col in COMMON_COLUMNS:
                if col not in df.columns:
                    if col.endswith('_encoded'):
                        df[col] = 0
                    elif '_ip' in col:
                        df[col] = '0.0.0.0'
                    elif '_port' in col:
                        df[col] = 0
                    else:
                        df[col] = 'Unknown'
            
            # Select only needed columns
            df = df[columns_to_use]
            datasets.append(df)
    
    # Combine all datasets
    if datasets:
        combined_df = pd.concat(datasets, ignore_index=True)
        print(f"Combined standard dataset shape: {combined_df.shape}")
        
        # Re-encode attack types to have consistent encoding
        le = LabelEncoder()
        combined_df['attack_type_encoded'] = le.fit_transform(combined_df['attack_type'])
        
        # Save the attack type mapping
        attack_mapping = {str(k): int(v) for k, v in zip(le.classes_, le.transform(le.classes_))}
        mapping_path = os.path.join(OUTPUT_DIR, "combined_standard_attack_mapping.json")
        with open(mapping_path, 'w') as f:
            json.dump(attack_mapping, f, indent=2, cls=NumpyEncoder)
        print(f"Saved attack mapping to {mapping_path}")
        
        # Normalize numerical features
        numerical_cols = ['duration_ms', 'bytes_in', 'bytes_out', 'packets_in', 'packets_out']
        scaler = StandardScaler()
        combined_df[numerical_cols] = scaler.fit_transform(combined_df[numerical_cols])
        
        # Save the combined dataset
        output_path = os.path.join(OUTPUT_DIR, "combined_standard.csv")
        combined_df.to_csv(output_path, index=False)
        print(f"Saved combined standard dataset to {output_path}")
        
        return combined_df
    else:
        print("No standard datasets could be loaded for combination.")
        return None

def combine_netflow_datasets():
    """Combine the netflow datasets into a single dataset"""
    print("Combining netflow datasets...")
    
    datasets = []
    
    # Load all netflow datasets
    for name, path in DATASETS["netflow"].items():
        df = load_dataset(path, name)
        if df is not None:
            # Select only common columns and source column
            columns_to_use = COMMON_COLUMNS + ['dataset_source']
            
            # For columns that don't exist, add with default values
            for col in COMMON_COLUMNS:
                if col not in df.columns:
                    if col.endswith('_encoded'):
                        df[col] = 0
                    elif '_ip' in col:
                        df[col] = '0.0.0.0'
                    elif '_port' in col:
                        df[col] = 0
                    else:
                        df[col] = 'Unknown'
            
            # Select only needed columns
            df = df[columns_to_use]
            datasets.append(df)
    
    # Combine all datasets
    if datasets:
        combined_df = pd.concat(datasets, ignore_index=True)
        print(f"Combined netflow dataset shape: {combined_df.shape}")
        
        # Re-encode attack types to have consistent encoding
        le = LabelEncoder()
        combined_df['attack_type_encoded'] = le.fit_transform(combined_df['attack_type'])
        
        # Save the attack type mapping
        attack_mapping = {str(k): int(v) for k, v in zip(le.classes_, le.transform(le.classes_))}
        mapping_path = os.path.join(OUTPUT_DIR, "combined_netflow_attack_mapping.json")
        with open(mapping_path, 'w') as f:
            json.dump(attack_mapping, f, indent=2, cls=NumpyEncoder)
        print(f"Saved attack mapping to {mapping_path}")
        
        # Normalize numerical features
        numerical_cols = ['duration_ms', 'bytes_in', 'bytes_out', 'packets_in', 'packets_out']
        scaler = StandardScaler()
        combined_df[numerical_cols] = scaler.fit_transform(combined_df[numerical_cols])
        
        # Save the combined dataset
        output_path = os.path.join(OUTPUT_DIR, "combined_netflow.csv")
        combined_df.to_csv(output_path, index=False)
        print(f"Saved combined netflow dataset to {output_path}")
        
        return combined_df
    else:
        print("No netflow datasets could be loaded for combination.")
        return None

def create_combined_graph(standard_df, netflow_df):
    """Create a combined graph representation from both standard and netflow datasets"""
    print("Creating combined graph representation...")
    
    # Create a new dataframe with all data
    all_df = pd.concat([standard_df, netflow_df], ignore_index=True) if netflow_df is not None else standard_df
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes (IP addresses)
    src_ips = all_df['src_ip'].unique()
    dst_ips = all_df['dst_ip'].unique()
    all_ips = list(set(src_ips) | set(dst_ips))
    
    print(f"Creating graph with {len(all_ips)} unique IP addresses")
    
    # Add nodes with features
    for ip in all_ips:
        # Skip placeholder IPs
        if ip == '0.0.0.0':
            continue
            
        # Aggregate features for this IP
        src_flows = all_df[all_df['src_ip'] == ip]
        dst_flows = all_df[all_df['dst_ip'] == ip]
        all_flows = pd.concat([src_flows, dst_flows])
        
        if len(all_flows) > 0:
            # Node features - basic statistics
            features = {
                'total_flows': int(len(all_flows)),
                'attack_flows': int(all_flows['binary_label'].sum()),
                'normal_flows': int(len(all_flows) - all_flows['binary_label'].sum()),
                'is_source': bool(len(src_flows) > 0),
                'is_destination': bool(len(dst_flows) > 0),
                
                # Dataset sources - useful for tracking origin
                'from_bot_iot': int(sum(all_flows['dataset_source'] == 'bot_iot')),
                'from_cic_ids2018': int(sum(all_flows['dataset_source'] == 'cic_ids2018')),
                'from_unsw_nb15': int(sum(all_flows['dataset_source'] == 'unsw_nb15')),
                'from_nf_bot_iot': int(sum(all_flows['dataset_source'] == 'nf_bot_iot')),
                'from_nf_cic_ids2018': int(sum(all_flows['dataset_source'] == 'nf_cic_ids2018')),
                'from_nf_unsw_nb15': int(sum(all_flows['dataset_source'] == 'nf_unsw_nb15')),
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
    edge_count = 0
    for _, flow in all_df.iterrows():
        src = flow['src_ip']
        dst = flow['dst_ip']
        
        # Skip placeholder IPs
        if src == '0.0.0.0' or dst == '0.0.0.0':
            continue
            
        # Skip if source or destination is not in the graph
        if src not in G.nodes or dst not in G.nodes:
            continue
        
        # Convert values to Python native types
        bytes_in = int(flow['bytes_in'])
        bytes_out = int(flow['bytes_out'])
        packets_in = int(flow['packets_in'])
        packets_out = int(flow['packets_out'])
        binary_label = int(flow['binary_label'])
        dataset_source = str(flow['dataset_source'])
            
        # If edge already exists, update its features
        if G.has_edge(src, dst):
            G[src][dst]['weight'] += 1
            G[src][dst]['total_bytes'] += bytes_in + bytes_out
            G[src][dst]['total_packets'] += packets_in + packets_out
            
            # Update attack information
            if binary_label == 1:
                G[src][dst]['attack_flows'] += 1
                
            # Update dataset sources
            source_field = f"from_{dataset_source}"
            if source_field in G[src][dst]:
                G[src][dst][source_field] += 1
            else:
                G[src][dst][source_field] = 1
        else:
            # Create new edge with features
            edge_data = {
                'weight': 1,
                'total_bytes': bytes_in + bytes_out,
                'total_packets': packets_in + packets_out,
                'attack_flows': 1 if binary_label == 1 else 0,
                f"from_{dataset_source}": 1
            }
            G.add_edge(src, dst, **edge_data)
            edge_count += 1
    
    print(f"  Created combined graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Save the graph
    output_path = os.path.join(OUTPUT_DIR, "combined_graph.graphml")
    nx.write_graphml(G, output_path)
    print(f"Saved combined graph to {output_path}")
    
    # Save basic graph metrics
    metrics = {
        'nodes': int(G.number_of_nodes()),
        'edges': int(G.number_of_edges()),
        'density': float(nx.density(G)),
        'avg_degree': float(sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0),
    }
    
    metrics_path = os.path.join(OUTPUT_DIR, "combined_graph_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2, cls=NumpyEncoder)
    print(f"Saved combined graph metrics to {metrics_path}")
    
    return G

def main():
    """Main function to combine all standardized datasets"""
    print("Starting dataset combination process")
    
    # Combine standard datasets
    standard_df = combine_standard_datasets()
    
    # Combine netflow datasets
    netflow_df = combine_netflow_datasets()
    
    # Create combined graph representation if at least one dataset was combined
    if standard_df is not None or netflow_df is not None:
        create_combined_graph(standard_df, netflow_df)
    
    print("Dataset combination process completed successfully")

if __name__ == "__main__":
    main() 