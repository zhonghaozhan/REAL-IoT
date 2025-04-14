import torch
import numpy as np
import pandas as pd
import os
import pickle
import argparse
import random
import json
import re
import gc
import time
import logging
from typing import List, Dict, Tuple, Optional

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d, Dropout
import torch_geometric.utils as pyg_utils
from torch_geometric.data import Data

# Assuming openai library is installed: pip install openai
from openai import OpenAI

# --- Configuration ---
# Paths
DATASET_PATH = "/media/ssd/test/standardized-datasets/combined/combined_unsw_cicRed_botRed_netflow_10pct.csv"
FEATURE_LIST_PATH = "/media/ssd/test/GNN/Standardized Models/CAGN-GAT/cagn_gat_feature_list.pkl"
MODEL_PATH = "/media/ssd/test/GNN/Standardized Models/CAGN-GAT/best_cagn_model_Combined_10pct.pt"
BASE_OUTPUT_DIR = "GNN/LLM_Mitigation_Test"
RESULTS_DIR = os.path.join(BASE_OUTPUT_DIR, "Results")
LLM_DATA_DIR = os.path.join(BASE_OUTPUT_DIR, "LLM_Data")
PROCESSED_DATA_DIR = os.path.join(BASE_OUTPUT_DIR, "Processed_Data")

# Sampling and Injection
NUM_SAMPLES = 100 # Reverted sample size
NUM_INJECTED_NODES = 20 # Reverted injection count (20%)
INJECTION_K_CONNECTIONS = 5 # How many existing nodes each injected node connects to

# k-NN Graph Parameters (match CAGN training)
GRAPH_CONSTRUCTION_PARAMS = {
    'k_neighbors': 20,
    'threshold': 0.5, # Note: threshold wasn't strictly used in CAGN adaptation, k-NN was direct
    'metric': 'euclidean'
}

# Model Parameters (match CAGN training)
CAGN_HIDDEN_DIM = 64
CAGN_HEADS = 8
CAGN_DROPOUT = 0.6
OUTPUT_DIM = 1 # Binary classification

# LLM Configuration
# --- IMPORTANT: Use environment variables or a secure secrets manager for API keys! ---
# Example: export OPENAI_API_KEY='your_actual_key_here'
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY_PLACEHOLDER") # Use environment variable
OPENAI_API_KEY = "sk-XrKnoeDz1yHevBlpp9xeK4MtVBMvlwG-ZZx8YajdEET3BlbkFJ-MsTdBx8F7rnWLn2l5CYyDo4OanpK2Pom6sKl_xBwA" # Replace with your actual key OR use os.getenv above
LLM_MODEL = "gpt-4o"
CONFIDENCE_THRESHOLD = 0.8 # Minimum confidence score from LLM to flag a node for removal

# System Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Protocol Mapping ---
# Common network protocols by number
protocol_map = {
    1: 'ICMP', 6: 'TCP', 17: 'UDP',
    0: 'HOPOPT', 2: 'IGMP', 4: 'IPv4', 41: 'IPv6', 47: 'GRE',
    50: 'ESP', 51: 'AH', 58: 'IPv6-ICMP', 88: 'EIGRP', 89: 'OSPF',
    132: 'SCTP', 137: 'MPLS',
    # Add others if needed
    -1: 'Unknown' # Default for missing/unmapped
}

# --- Utility Functions ---
def ensure_dir_exists(path):
    """Creates a directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        logging.info(f"Created directory: {path}")

def save_data(data, path, description):
    """Saves data (graph, list, df, text) with logging."""
    try:
        ensure_dir_exists(os.path.dirname(path))
        if isinstance(data, Data):
            torch.save(data, path)
        elif isinstance(data, (list, dict)):
             with open(path, 'w') as f:
                # Handle potential non-serializable items like numpy floats
                json.dump(data, f, indent=4, default=lambda x: str(x) if isinstance(x, (np.float32, np.float64)) else x)
        elif isinstance(data, pd.DataFrame):
            data.to_csv(path, index=False)
        elif isinstance(data, str):
             with open(path, 'w') as f:
                f.write(data)
        else:
            with open(path, 'wb') as f:
                pickle.dump(data, f)
        logging.info(f"Saved {description} to {path}")
    except Exception as e:
        logging.error(f"Error saving {description} to {path}: {e}")


# --- Data Loading and Preprocessing ---
def load_and_preprocess_data(path):
    """Loads the dataset."""
    logging.info(f"Loading data from: {path}")
    try:
        df = pd.read_csv(path, low_memory=False)
        logging.info(f"Loaded full data: {len(df)} total samples.")
        # Basic cleaning example: fill NaNs in potential feature columns
        # Adapt based on actual dataset needs
        potential_feature_cols = df.select_dtypes(include=np.number).columns.tolist()
        df[potential_feature_cols] = df[potential_feature_cols].fillna(0)
        return df
    except FileNotFoundError:
        logging.error(f"Dataset file not found at {path}")
        raise
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def create_fixed_size_subset(df, n_samples, label_col='Label', random_state=42):
    """Creates a fixed-size subset, trying to preserve label distribution."""
    logging.info(f"Creating subset of size {n_samples}...")
    if n_samples >= len(df):
        logging.warning("Requested subset size is >= total data size. Returning full dataframe.")
        return df.copy()

    # Simple stratified sampling if possible, otherwise random sample
    try:
        if label_col in df.columns and df[label_col].nunique() > 1:
             # Ensure n_splits is not greater than the number of members in each class
             min_class_count = df[label_col].value_counts().min()
             if n_samples / df[label_col].nunique() > min_class_count:
                 logging.warning(f"Cannot guarantee stratification due to small class size ({min_class_count}). Falling back to random sampling.")
                 subset_df = df.sample(n=n_samples, random_state=random_state)
             else:
                # Using train_test_split for stratified sampling logic
                _, subset_df = train_test_split(
                    df, test_size=n_samples, stratify=df[label_col], random_state=random_state
                )
        else:
             logging.warning(f"Label column '{label_col}' not found or only one class present. Using random sampling.")
             subset_df = df.sample(n=n_samples, random_state=random_state)

    except Exception as e:
         logging.error(f"Error during stratified sampling, falling back to random sampling: {e}")
         subset_df = df.sample(n=n_samples, random_state=random_state)

    logging.info(f"Subset created with {len(subset_df)} samples.")
    logging.info(f"Subset label distribution:\n{subset_df[label_col].value_counts()}")
    # Add original_index for mapping raw features back
    subset_df['original_index'] = subset_df.index
    return subset_df.reset_index(drop=True)


def feature_engineer(df, node_features_expected_list, label_col='Label'):
    """Performs feature engineering similar to CAGN setup."""
    logging.info("Starting feature engineering...")
    df_subset = df.copy()

    # Identify actual numerical and categorical columns in the input DataFrame
    numerical_cols_raw = df_subset.select_dtypes(include=np.number).columns.tolist()
    if label_col in numerical_cols_raw:
        numerical_cols_raw.remove(label_col)
    if 'original_index' in numerical_cols_raw: # Exclude index if added
        numerical_cols_raw.remove('original_index')

    categorical_cols_raw = df_subset.select_dtypes(include=['object', 'category']).columns.tolist()
    # Assume some common categorical cols if not object/category type (adjust as needed)
    potential_cat_cols = ['PROTOCOL', 'L7_PROTO', 'TCP_FLAGS']
    for col in potential_cat_cols:
        if col in df_subset.columns and col not in categorical_cols_raw and col not in numerical_cols_raw:
             categorical_cols_raw.append(col)
             if col in numerical_cols_raw: # Should not happen based on above, but safety check
                 numerical_cols_raw.remove(col)

    # --- Process Numerical ---
    logging.info(f"Processing numerical features found in data: {numerical_cols_raw}")
    # Ensure numeric type and fill NaNs
    for col in numerical_cols_raw:
        df_subset[col] = pd.to_numeric(df_subset[col], errors='coerce').fillna(0)

    # Apply log1p transformation and scaling
    if numerical_cols_raw:
        log_transformed_features = np.log1p(df_subset[numerical_cols_raw].values)
        scaler = StandardScaler()
        scaled_numerical_features = scaler.fit_transform(log_transformed_features)
        scaled_numerical_df = pd.DataFrame(scaled_numerical_features, index=df_subset.index, columns=numerical_cols_raw)
    else:
        scaled_numerical_df = pd.DataFrame(index=df_subset.index)
        scaler = None # No scaler if no numeric features

    # --- Process Categorical ---
    logging.info(f"Processing categorical features found in data: {categorical_cols_raw}")
    # Fill NaNs and ensure string type for get_dummies
    df_subset[categorical_cols_raw] = df_subset[categorical_cols_raw].astype(str).fillna('missing')

    if categorical_cols_raw:
        categorical_encoded_df = pd.get_dummies(
            df_subset[categorical_cols_raw],
            columns=categorical_cols_raw,
            prefix=categorical_cols_raw,
            dummy_na=False, # Don't create NA dummy if we filled NA
            dtype=int
        )
    else:
        categorical_encoded_df = pd.DataFrame(index=df_subset.index)

    # --- Combine Features ---
    X_df_processed = pd.concat([scaled_numerical_df, categorical_encoded_df.set_index(scaled_numerical_df.index)], axis=1)
    logging.info(f"Combined processed features. Shape: {X_df_processed.shape}, Columns: {X_df_processed.columns[:10]}...")

    # --- Ensure feature consistency with the expected list ---
    logging.info(f"Aligning features with expected list ({len(node_features_expected_list)} features)...")
    current_features = set(X_df_processed.columns)
    expected_features = set(node_features_expected_list)

    # Add missing columns (those expected but not generated)
    missing_cols = list(expected_features - current_features)
    if missing_cols:
        logging.warning(f"Adding {len(missing_cols)} columns missing after processing: {missing_cols[:5]}...")
        for col in missing_cols:
            X_df_processed[col] = 0 # Add missing expected columns with value 0

    # Remove extra columns (those generated but not expected)
    extra_cols = list(current_features - expected_features)
    if extra_cols:
        logging.warning(f"Removing {len(extra_cols)} unexpected columns generated: {extra_cols[:5]}...")
        X_df_processed = X_df_processed.drop(columns=extra_cols)

    # Ensure final DataFrame has columns in the exact order of node_features_expected_list
    X_df_processed = X_df_processed[node_features_expected_list]

    logging.info(f"Feature engineering complete. Final X shape: {X_df_processed.shape}")
    X = X_df_processed.values.astype(np.float32)
    y = df_subset[label_col].values.astype(np.int64)

    return X, y, scaler, X_df_processed


def build_knn_graph(X, y, k, metric='euclidean'):
    """Builds a PyG Data object using k-NN similarity graph."""
    logging.info(f"Building k-NN similarity graph (k={k}, metric={metric})...")
    num_samples = X.shape[0]
    try:
        # Use 'connectivity' mode which returns 0/1, avoids distance threshold need
        knn_adj = kneighbors_graph(X, k, mode='connectivity', metric=metric, include_self=False, n_jobs=-1)
        logging.info(f"  Calculated k-NN graph. Shape: {knn_adj.shape}, NNZ: {knn_adj.nnz}.")

        final_adj = knn_adj
        final_adj_coo = final_adj.tocoo()
        edge_index = torch.tensor(np.vstack((final_adj_coo.row, final_adj_coo.col)), dtype=torch.long)

        features = torch.tensor(X, dtype=torch.float)
        labels = torch.tensor(y, dtype=torch.long)

        data = Data(x=features, edge_index=edge_index, y=labels)
        # Add node IDs for later mapping
        data.node_ids = torch.arange(num_samples)

        logging.info(f"Built graph: {data.num_nodes} nodes, {data.num_edges} edges")
        return data
    except MemoryError as e:
        logging.error(f"MemoryError during graph construction: {e}. Try reducing k or features.")
        raise
    except Exception as e:
        logging.error(f"Error during graph construction: {e}")
        raise


# --- Model Definition (CAGN) ---
class CAGN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=8, dropout=0.6):
        super(CAGN, self).__init__()
        self.dropout_rate = dropout
        self.nfeat = input_dim
        self.nclass = output_dim
        self.hidden_sizes = [hidden_dim * heads, hidden_dim]

        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=self.dropout_rate, edge_dim=None)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=self.dropout_rate, edge_dim=None)
        self.conv3 = GATConv(hidden_dim, output_dim, heads=1, concat=False, dropout=self.dropout_rate, edge_dim=None)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        z = self.conv2(x, edge_index)
        z = F.elu(z)
        z = F.dropout(z, p=self.dropout_rate, training=self.training)
        out_logits = self.conv3(z, edge_index)
        return out_logits

# --- Model Loading ---
def load_cagn_model(model_path, input_dim, output_dim, hidden_dim, heads, dropout, device):
    """Loads the pre-trained CAGN model state_dict."""
    logging.info(f"Loading CAGN model from {model_path}")
    if not os.path.exists(model_path):
        logging.error(f"Model file not found at {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        model = CAGN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, heads=heads, dropout=dropout)
        model_state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(model_state_dict)
        model.to(device)
        model.eval() # Set to evaluation mode
        logging.info(f"Loaded CAGN model to {device}")
        return model
    except Exception as e:
        logging.error(f"Error loading model state_dict: {e}")
        raise

# --- Attack Implementation ---
def inject_nodes_knn(data: Data, num_nodes_to_add: int, k_connections: int = 5, feature_strategy: str = 'mean') -> Tuple[Data, List[int]]:
    """Injects nodes into a k-NN graph Data object."""
    logging.info(f"Injecting {num_nodes_to_add} nodes (strategy: {feature_strategy}, connections: {k_connections})...")
    if num_nodes_to_add == 0:
        logging.warning("Skipping node injection (0 nodes to add).")
        return data.clone(), []

    original_num_nodes = data.num_nodes
    original_num_edges = data.num_edges
    num_features = data.num_features
    device = data.x.device

    perturbed_data = data.clone()

    # 1. Add new node features
    if feature_strategy == 'mean':
        new_features = data.x.mean(dim=0, keepdim=True).repeat(num_nodes_to_add, 1)
    elif feature_strategy == 'zeros':
        new_features = torch.zeros((num_nodes_to_add, num_features), dtype=data.x.dtype, device=device)
    else: # Default to mean if strategy unknown
        logging.warning(f"Unknown feature strategy '{feature_strategy}'. Using 'mean'.")
        new_features = data.x.mean(dim=0, keepdim=True).repeat(num_nodes_to_add, 1)

    perturbed_data.x = torch.cat([perturbed_data.x, new_features], dim=0)

    # 2. Add new edges (randomly connect new nodes to existing ones)
    new_node_indices = torch.arange(original_num_nodes, original_num_nodes + num_nodes_to_add, device=device)
    injected_node_ids = new_node_indices.tolist() # Keep track of the new node IDs

    # Ensure enough original nodes to connect to
    if original_num_nodes == 0:
         logging.warning("Cannot inject edges, no original nodes exist.")
         new_edges = torch.empty((2, 0), dtype=torch.long, device=device)
    elif k_connections > original_num_nodes:
         logging.warning(f"k_connections ({k_connections}) > original_num_nodes ({original_num_nodes}). Connecting to all original nodes.")
         k_connections = original_num_nodes

         # Create connections to all original nodes for each new node
         new_edge_sources = new_node_indices.repeat_interleave(k_connections)
         # Create targets by repeating range(original_num_nodes) for each new node
         new_edge_targets_list = [torch.arange(original_num_nodes, device=device) for _ in range(num_nodes_to_add)]
         new_edge_targets = torch.cat(new_edge_targets_list)

    else:
        # Randomly select k_connections targets for each new node
        new_edge_sources = new_node_indices.repeat_interleave(k_connections)
        # Sample without replacement for each new node if k <= original_num_nodes
        target_indices = torch.cat([
            torch.randperm(original_num_nodes, device=device)[:k_connections]
            for _ in range(num_nodes_to_add)
        ])
        new_edge_targets = target_indices


    if original_num_nodes > 0:
        new_edges = torch.stack([
            torch.cat([new_edge_sources, new_edge_targets]), # New -> Orig
            torch.cat([new_edge_targets, new_edge_sources])  # Orig -> New (add symmetric edges)
        ], dim=0)
        perturbed_data.edge_index = torch.cat([perturbed_data.edge_index, new_edges], dim=1)


    # 3. Add dummy labels for new nodes (assign benign label 0)
    # This is arbitrary; the LLM should identify them regardless of assigned label.
    new_labels = torch.zeros(num_nodes_to_add, dtype=data.y.dtype, device=device)
    perturbed_data.y = torch.cat([perturbed_data.y, new_labels], dim=0)

    # Update node count and node IDs attribute
    perturbed_data.num_nodes = original_num_nodes + num_nodes_to_add
    perturbed_data.node_ids = torch.arange(perturbed_data.num_nodes)


    logging.info(f"Node injection finished: Added {num_nodes_to_add} nodes.")
    logging.info(f"  New graph: {perturbed_data.num_nodes} nodes, {perturbed_data.num_edges} edges")
    return perturbed_data, injected_node_ids


# --- LLM Interaction Functions ---
def format_prompt_for_llm(df_raw_subset: pd.DataFrame, graph: Data, node_id_to_analyze: int, node_features_list: List[str]) -> str:
    """Formats the prompt for the LLM expert to analyze a SINGLE node within the graph."""
    logging.info(f"Formatting prompt for LLM analysis of node {node_id_to_analyze}...")
    prompt_lines = []

    num_total_nodes = graph.num_nodes
    num_original_nodes = df_raw_subset.shape[0]
    num_approx_original_edges = num_original_nodes * GRAPH_CONSTRUCTION_PARAMS['k_neighbors']

    # 1. Role Setting and Task Context
    prompt_lines.append("You are an expert cybersecurity analyst reviewing network flow data represented as a graph.")
    prompt_lines.append(f"The graph contains {num_total_nodes} nodes (flows) representing network activity.")
    prompt_lines.append(f"Some nodes may represent normal activity, while others could be anomalous or potentially malicious (e.g., part of DDoS, spoofing, reconnaissance) disguised as normal flows.")
    prompt_lines.append("\nYour task is to analyze the **single node** detailed below.")
    prompt_lines.append("Focus on comparing the node's **raw features** (if available) to its neighbors' **raw features** to assess consistency within its local neighborhood.")
    prompt_lines.append("Also consider if the node's own raw features (or processed features if raw are unavailable) seem inherently unusual.")
    prompt_lines.append("Provide a confidence score (0.0 = consistent/likely normal, 1.0 = inconsistent/highly likely anomalous) and a brief justification.")

    prompt_lines.append("\n**Key Analysis Points:**")
    prompt_lines.append("- **Raw Feature Consistency:** Does the target node's *raw* feature values align with its neighbors' *raw* values? (Primary focus)")
    prompt_lines.append("- **Unusual Own Features:** Does the target node have strange *raw* feature combinations (e.g., high packets for low-traffic protocol, impossible flags)?")
    prompt_lines.append("- **Connectivity:** Does the node connect to neighbors that seem unusually diverse or disconnected? (Secondary consideration)")

    # 2. Feature Descriptions
    prompt_lines.append("\nKey Raw Features (Used for Comparison):")
    feature_desc = {
        'IN_BYTES': "Incoming bytes", 'OUT_BYTES': "Outgoing bytes",
        'IN_PKTS': "Incoming packets", 'OUT_PKTS': "Outgoing packets",
        'FLOW_DURATION_MILLISECONDS': "Duration of the flow (ms)",
        'PROTOCOL': "Network protocol (e.g., 6=TCP, 17=UDP)",
        'L7_PROTO': "Application layer protocol (numeric code)",
        'TCP_FLAGS': "TCP flags set (numeric code)"
    }
    base_features_for_desc = set()
    for f in node_features_list:
        match = re.match(r"([a-zA-Z0-9_]+?)(_[0-9.\-]+)?(_scaled)?$", f)
        if match:
             base_features_for_desc.add(match.group(1))
        else:
             base_features_for_desc.add(f)
    manual_keys = ['IN_BYTES', 'OUT_BYTES', 'IN_PKTS', 'OUT_PKTS', 'FLOW_DURATION_MILLISECONDS', 'PROTOCOL', 'L7_PROTO', 'TCP_FLAGS']
    desc_keys = sorted(list(base_features_for_desc.intersection(set(manual_keys))))
    prompt_lines.append("(Showing descriptions for common base features)")
    displayed_count = 0
    for feat in desc_keys:
        desc = feature_desc.get(feat)
        if desc:
            prompt_lines.append(f"- {feat}: {desc}")
            displayed_count += 1
        if displayed_count >= 10: break
    if len(desc_keys) > displayed_count: prompt_lines.append("- ... (other base features exist)")

    # 3. Data for Analysis (Single Node)
    prompt_lines.append("\n--- Node Analysis --- ")
    prompt_lines.append(f"Analyze the following node (ID: {node_id_to_analyze}):")

    node_id = node_id_to_analyze
    prompt_lines.append(f"\n**Node ID: {node_id}**")

    # --- Attempt to display RAW features for the target node --- 
    target_node_raw_features_str = "N/A (Node was potentially added, no raw features)"
    if node_id < num_original_nodes:
        try:
            target_raw_features = df_raw_subset.iloc[node_id]
            key_features_display = []
            proto_num = target_raw_features.get('PROTOCOL', -1)
            proto_name = protocol_map.get(int(proto_num), 'Unknown')
            key_features_display.append(f"Protocol: {proto_name} ({proto_num})")
            l7 = target_raw_features.get('L7_PROTO', 'N/A')
            if l7 != 'N/A': key_features_display.append(f"L7_Proto: {l7}")
            flags = target_raw_features.get('TCP_FLAGS', 'N/A')
            if flags != 'N/A': key_features_display.append(f"TCP_Flags: {flags}")
            in_bytes = target_raw_features.get('IN_BYTES', 'N/A')
            if in_bytes != 'N/A': key_features_display.append(f"InBytes: {in_bytes}")
            out_bytes = target_raw_features.get('OUT_BYTES', 'N/A')
            if out_bytes != 'N/A': key_features_display.append(f"OutBytes: {out_bytes}")
            duration = target_raw_features.get('FLOW_DURATION_MILLISECONDS', 'N/A')
            if duration != 'N/A': key_features_display.append(f"Duration(ms): {duration}")
            target_node_raw_features_str = ", ".join(key_features_display)
        except Exception as e:
            target_node_raw_features_str = f"Error fetching raw features: {e}"
    prompt_lines.append(f"  Raw Features: {target_node_raw_features_str}")
    # --- End Raw Feature Display ---

    node_feature_vector = graph.x[node_id].cpu().numpy()
    prompt_lines.append(f"  Processed Features Snippet: {node_feature_vector[:10]} ... (Scaled/Encoded for GNN model)")

    # Find and display neighbors (logic remains the same)
    neighbors = graph.edge_index[1, graph.edge_index[0] == node_id]
    if neighbors.numel() > 0:
        prompt_lines.append(f"  Connects to {neighbors.numel()} other Nodes (Neighbors shown with their Raw Features):")
        displayed_neighbor_count = 0
        for neighbor_id_tensor in neighbors:
            neighbor_id = neighbor_id_tensor.item()
            if displayed_neighbor_count >= 5:
                 prompt_lines.append("      ...")
                 break
            if neighbor_id < num_original_nodes:
                try:
                    neighbor_raw_features = df_raw_subset.iloc[neighbor_id]
                    prompt_lines.append(f"    - Neighbor ID: {neighbor_id} (Original)")
                    key_features_display = []
                    proto_num = neighbor_raw_features.get('PROTOCOL', -1)
                    proto_name = protocol_map.get(int(proto_num), 'Unknown')
                    key_features_display.append(f"Protocol: {proto_name} ({proto_num})")
                    l7 = neighbor_raw_features.get('L7_PROTO', 'N/A')
                    if l7 != 'N/A': key_features_display.append(f"L7_Proto: {l7}")
                    flags = neighbor_raw_features.get('TCP_FLAGS', 'N/A')
                    if flags != 'N/A': key_features_display.append(f"TCP_Flags: {flags}")
                    in_bytes = neighbor_raw_features.get('IN_BYTES', 'N/A')
                    if in_bytes != 'N/A': key_features_display.append(f"InBytes: {in_bytes}")
                    out_bytes = neighbor_raw_features.get('OUT_BYTES', 'N/A')
                    if out_bytes != 'N/A': key_features_display.append(f"OutBytes: {out_bytes}")
                    duration = neighbor_raw_features.get('FLOW_DURATION_MILLISECONDS', 'N/A')
                    if duration != 'N/A': key_features_display.append(f"Duration(ms): {duration}")
                    features_str = ", ".join(key_features_display)
                    prompt_lines.append(f"      Raw Features: {features_str}")
                except Exception as e:
                    prompt_lines.append(f"    - Neighbor ID: {neighbor_id} (Original) - Error getting features: {e}")
            else:
                prompt_lines.append(f"    - Neighbor ID: {neighbor_id} (Potentially Added - No raw features)")
            displayed_neighbor_count += 1
        prompt_lines.append(f"\n  -> Assess Node {node_id}: Is it consistent with its neighbors based on **raw features**? Does it have unusual raw features itself?")
    else:
        prompt_lines.append("  Connects to 0 other Nodes.")
        prompt_lines.append(f"  -> Assess Node {node_id} based on its own features (raw if available, otherwise processed).")


    # 4. Output Format Instruction (Single Object)
    prompt_lines.append("\n--- Output Format --- ")
    prompt_lines.append("Please provide your analysis ONLY in the following JSON format (a single JSON object, NO list brackets):")
    prompt_lines.append("```json")
    prompt_lines.append("{ \"node_id\": <node_id>, \"confidence_score\": <score_0.0_to_1.0>, \"justification\": \"Your brief 1-sentence justification here.\" }")
    prompt_lines.append("```")
    prompt_lines.append(f"Ensure the node_id in the output is {node_id_to_analyze}.")

    return "\n".join(prompt_lines)


def call_openai_api(prompt: str, expected_node_id: int, api_key: str, model: str = "gpt-4o") -> Tuple[Optional[Dict], Optional[str], str]:
    """Calls the OpenAI API for a single node and parses the response.

    Returns:
        Tuple[Optional[Dict], Optional[str], str]:
            - Parsed result dict ({"node_id": ..., "confidence_score": ..., "justification": ...}) or dict with error key.
            - Raw response content string or None on failure.
            - The original prompt string.
    """
    logging.info(f"Calling OpenAI API (model: {model}) for node {expected_node_id}...")

    if not api_key or api_key == "YOUR_API_KEY_PLACEHOLDER":
        logging.error(f"OpenAI API key is not set or is a placeholder for node {expected_node_id}. Returning default score.")
        return ({"node_id": expected_node_id, "confidence_score": 0.0, "justification": "N/A", "error": "API Key Missing"}, None, prompt)

    client = OpenAI(api_key=api_key)

    parsed_result = None # Expect a single dict or None on failure
    raw_response_content = None
    error_info = None
    default_justification = "Parsing/Validation Failed"
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert cybersecurity analyst providing JSON output with node_id, confidence_score, and justification."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            response_format={ "type": "json_object" }
        )
        raw_response_content = response.choices[0].message.content
        logging.info(f"Received response from OpenAI API for node {expected_node_id}.")

        # --- Parsing Logic for Single JSON Object ---
        logging.info(f"Parsing LLM response for node {expected_node_id}...")
        json_str_to_parse = None
        parsed_data = None
        try:
            # 1. Strip leading/trailing whitespace
            content_to_parse = raw_response_content.strip()

            # 2. Check if it looks like a valid JSON object
            if content_to_parse.startswith('{') and content_to_parse.endswith('}'):
                json_str_to_parse = content_to_parse
                logging.info("Response looks like a JSON object.")
            elif content_to_parse.startswith('[') and content_to_parse.endswith(']'):
                 # Handle case where API incorrectly returned a list
                 logging.warning("Response unexpectedly contained a list, attempting to extract first object.")
                 temp_list = json.loads(content_to_parse)
                 if isinstance(temp_list, list) and len(temp_list) > 0 and isinstance(temp_list[0], dict):
                     parsed_data = temp_list[0] # Use the first object
                     logging.info(f"Extracted first object: {parsed_data}")
                 else:
                     logging.warning("Could not extract a valid object from the list.")
                     error_info = "Unexpected list format without valid object"
            else:
                # Maybe wrapped in ```json ... ``` block?
                match = re.search(r"```json\n(\{.*?\n\})\n```", content_to_parse, re.DOTALL)
                if match:
                    json_str_to_parse = match.group(1).strip()
                    logging.info("Extracted JSON object from markdown code block.")
                else:
                    logging.warning("Could not find well-formed JSON object structure in the response.")
                    error_info = "No JSON object structure found"

            # 3. Attempt to parse if not already parsed from list
            if json_str_to_parse and not parsed_data:
                logging.info(f"Attempting to parse JSON string: {json_str_to_parse[:200]}...") # Log snippet
                parsed_data = json.loads(json_str_to_parse)

            # 4. Validate the parsed data
            if isinstance(parsed_data, dict) and "node_id" in parsed_data and "confidence_score" in parsed_data and "justification" in parsed_data:
                try:
                    node_id = int(parsed_data["node_id"])
                    confidence = float(parsed_data["confidence_score"])
                    justification = str(parsed_data["justification"]) # Extract justification

                    if node_id != expected_node_id:
                        logging.warning(f"LLM returned score for wrong node ID! Expected {expected_node_id}, got {node_id}. Discarding.")
                        error_info = f"Mismatched node ID (expected {expected_node_id}, got {node_id})"
                        justification = "N/A - Mismatched Node ID"
                    else:
                        # Validate score range
                        if not (0.0 <= confidence <= 1.0):
                            logging.warning(f"LLM confidence score {confidence} out of range [0, 1] for node {node_id}. Clamping.")
                            confidence = max(0.0, min(1.0, confidence))
                        # Minimal validation for justification (is it a non-empty string?)
                        if not justification or not isinstance(justification, str):
                            logging.warning(f"LLM returned invalid justification: {justification}. Setting to 'N/A'.")
                            justification = "N/A - Invalid Format"
                        parsed_result = {"node_id": node_id, "confidence_score": confidence, "justification": justification}

                except (ValueError, TypeError) as conv_err:
                    logging.warning(f"Could not parse node_id/confidence/justification from object: {parsed_data}. Error: {conv_err}")
                    error_info = f"Type/Value error in parsing fields: {conv_err}"
                    default_justification = f"N/A - Error parsing fields: {conv_err}"
            elif parsed_data is not None:
                logging.warning(f"Invalid item format in LLM JSON response (missing keys?): {parsed_data}")
                error_info = error_info or "Invalid JSON object format (missing keys?)"
                default_justification = "N/A - Missing expected JSON keys"
            # else: error_info might have already been set if no JSON structure found

        except json.JSONDecodeError as json_err:
            logging.warning(f"Failed to parse JSON string: {json_err}. String was:\n\'{json_str_to_parse}\'") # Use repr
            error_info = f"JSONDecodeError: {json_err}"
            default_justification = f"N/A - JSON Decode Error: {json_err}"
        except Exception as parse_err:
            logging.error(f"Unexpected error during JSON parsing: {parse_err}")
            error_info = f"Unexpected parsing error: {parse_err}"
            default_justification = f"N/A - Unexpected parsing error: {parse_err}"

        # If parsing failed or validation failed, parsed_result is still None
        if not parsed_result:
             logging.warning(f"Could not parse valid result for node {expected_node_id}. Assigning default score 0.0.")
             # Return a dict structure with error info so the main loop gets consistent types
             parsed_result = {"node_id": expected_node_id, "confidence_score": 0.0, "justification": default_justification, "error": error_info or "Parsing/Validation Failed"}
             logging.info(f"Parsed result for node {expected_node_id}: {parsed_result}")


    except Exception as e:
        logging.error(f"Error calling OpenAI API or processing response for node {expected_node_id}: {e}")
        logging.warning(f"Assigning default score 0.0 to node {expected_node_id} due to API error.")
        error_info = f"API Call Error: {e}"
        parsed_result = {
            "node_id": expected_node_id,
            "confidence_score": 0.0,
            "justification": f"N/A - API Call Error: {e}",
            "error": error_info
        }
        # raw_response_content might be None here

    # Return parsed result (or default w/ error), raw response, and the original prompt
    return parsed_result, raw_response_content, prompt


def process_llm_results(llm_results_list: List[Dict], threshold: float) -> List[int]:
    """Filters LLM results based on confidence threshold."""
    logging.info(f"Filtering {len(llm_results_list)} LLM results with threshold >= {threshold}")
    flagged_nodes = []
    if not isinstance(llm_results_list, list):
        logging.error(f"Invalid LLM results format for processing: expected list, got {type(llm_results_list)}")
        return []

    for item in llm_results_list:
        # Check if item is a dict and has the required keys (ignore items with 'error' key added on failure)
        if isinstance(item, dict) and "node_id" in item and "confidence_score" in item and "error" not in item:
            try:
                node_id = int(item["node_id"])
                confidence = float(item["confidence_score"])
                if confidence >= threshold:
                    flagged_nodes.append(node_id)
            except (ValueError, TypeError):
                logging.warning(f"Skipping invalid item in LLM results during processing: {item}")
        elif isinstance(item, dict) and "error" in item:
             logging.warning(f"Skipping item with processing error: {item}")
        else:
             logging.warning(f"Skipping invalid item format in LLM results during processing: {item}")

    logging.info(f"Identified {len(flagged_nodes)} nodes to remove based on threshold.")
    return flagged_nodes

def remove_flagged_nodes(graph_data: Data, nodes_to_remove: List[int]) -> Data:
    """Removes flagged nodes from the graph."""
    logging.info(f"Removing {len(nodes_to_remove)} flagged nodes from the graph...")
    if not nodes_to_remove:
        logging.info("No nodes to remove.")
        return graph_data.clone()

    num_original_nodes = graph_data.num_nodes
    nodes_to_remove_tensor = torch.tensor(nodes_to_remove, dtype=torch.long, device=graph_data.x.device)

    # Create a mask for nodes to KEEP
    keep_mask = torch.ones(num_original_nodes, dtype=torch.bool, device=graph_data.x.device)
    # Ensure nodes_to_remove are within bounds
    valid_nodes_to_remove = nodes_to_remove_tensor[nodes_to_remove_tensor < num_original_nodes]
    if len(valid_nodes_to_remove) < len(nodes_to_remove):
        logging.warning("Some nodes flagged for removal were out of graph bounds.")

    if len(valid_nodes_to_remove) > 0:
        keep_mask[valid_nodes_to_remove] = False
    else:
        logging.info("No valid nodes to remove within graph bounds.")
        return graph_data.clone()


    # Use subgraph function which handles reindexing
    try:
        fixed_graph = graph_data.subgraph(keep_mask)

        # Update node_ids attribute if it exists, mapping old IDs to new sequential IDs
        if hasattr(graph_data, 'node_ids'):
            original_node_ids_kept = graph_data.node_ids[keep_mask]
            # Create a mapping from old ID to new sequential index
            old_to_new_id_map = {old_id.item(): new_id for new_id, old_id in enumerate(original_node_ids_kept)}
            fixed_graph.original_node_ids = original_node_ids_kept # Store the original IDs that were kept
            fixed_graph.node_ids = torch.arange(fixed_graph.num_nodes) # Assign new sequential IDs 0..N-1
            fixed_graph.id_map = old_to_new_id_map # Optional: store the mapping

        logging.info(f"Graph after node removal: {fixed_graph.num_nodes} nodes, {fixed_graph.num_edges} edges.")
        return fixed_graph
    except Exception as e:
        logging.error(f"Error using subgraph: {e}. Returning original graph.")
        import traceback
        traceback.print_exc()
        return graph_data.clone()


# --- Evaluation Function ---
def evaluate_model(model, data, description="Evaluation", device='cpu'):
    """Evaluates the model on the given data."""
    if model is None or data is None or data.num_nodes == 0:
        logging.warning(f"Skipping evaluation for {description} (model/data is None or data is empty)")
        return {}

    logging.info(f"Evaluating model on '{description}' data ({data.num_nodes} nodes, {data.num_edges} edges).")
    model.eval()
    try:
        data = data.to(device)

        with torch.no_grad():
            out = model(data.x, data.edge_index)

            if isinstance(out, tuple):
                out_logits = out[0]
            else:
                out_logits = out

            # Binary classification: expect [N, 1] or [N]
            if out_logits.ndim == 2 and out_logits.shape[1] == 1:
                out_logits = out_logits.squeeze(1)
            elif out_logits.ndim != 1:
                 logging.error(f"Unexpected model output shape: {out_logits.shape}. Cannot evaluate.")
                 return {}

            preds_proba = torch.sigmoid(out_logits).cpu()
            preds = (preds_proba > 0.5).long().numpy()
            y_true = data.y.cpu().numpy()
            preds_proba = preds_proba.numpy()

            accuracy = accuracy_score(y_true, preds)
            # Use macro average for precision/recall/f1 if multiclass in future, binary for now
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, preds, average='binary', zero_division=0)

            auc = 0.0
            try:
                if len(np.unique(y_true)) > 1:
                    auc = roc_auc_score(y_true, preds_proba)
                else:
                    logging.warning("Skipping AUC calculation: Only one class present in ground truth.")
            except Exception as e:
                logging.warning(f"Could not calculate AUC: {e}")

            metrics = {
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "AUC": auc,
                "Nodes": data.num_nodes,
                "Edges": data.num_edges
            }
            logging.info(f"{description} Results - Acc: {accuracy:.4f}, P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
            return metrics

    except Exception as e:
        logging.error(f"Error during evaluation for {description}: {e}")
        # Add traceback for debugging
        import traceback
        traceback.print_exc()
        return {}

# --- Main Execution ---
if __name__ == "__main__":
    logging.info("--- Starting LLM Mitigation Test (Analyzing All Nodes) ---")
    logging.info(f"Using device: {DEVICE}")
    ensure_dir_exists(RESULTS_DIR)
    ensure_dir_exists(LLM_DATA_DIR)
    ensure_dir_exists(PROCESSED_DATA_DIR)

    # Check for API Key early
    if not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_API_KEY_PLACEHOLDER":
         logging.warning("CRITICAL: OpenAI API key is missing or is a placeholder.")
         logging.warning("LLM calls will be skipped, resulting in default scores (0.0).")
         # exit() # Exit if API key is mandatory for the run

    final_all_results = {} # Changed variable name

    try:
        # 1. Load Feature List
        try:
            with open(FEATURE_LIST_PATH, 'rb') as f:
                NODE_FEATURES = pickle.load(f)
            logging.info(f"Successfully loaded {len(NODE_FEATURES)} feature names from {FEATURE_LIST_PATH}")
            input_dim = len(NODE_FEATURES)
        except FileNotFoundError:
            logging.error(f"Critical: Feature list file not found at {FEATURE_LIST_PATH}. Exiting.")
            exit()
        except Exception as e:
            logging.error(f"Critical: Error loading feature list: {e}. Exiting.")
            exit()

        # 2. Load Full Data
        df_full = load_and_preprocess_data(DATASET_PATH)

        # 3. Sample Subset
        df_subset = create_fixed_size_subset(df_full, n_samples=NUM_SAMPLES, label_col='Label')
        save_data(df_subset, os.path.join(PROCESSED_DATA_DIR, "raw_subset.csv"), "Raw Data Subset")
        del df_full
        gc.collect()

        # 4. Feature Engineer Subset
        X_subset, y_subset, scaler, X_df_processed = feature_engineer(df_subset, NODE_FEATURES, label_col='Label')
        save_data(X_df_processed, os.path.join(PROCESSED_DATA_DIR, "features_engineered_subset.csv"), "Engineered Features Subset")

        # 5. Build Clean Graph
        clean_graph = build_knn_graph(X_subset, y_subset, k=GRAPH_CONSTRUCTION_PARAMS['k_neighbors'], metric=GRAPH_CONSTRUCTION_PARAMS['metric'])
        # Add original indices to clean graph nodes for potential mapping later
        clean_graph.original_indices = torch.tensor(df_subset['original_index'].values, dtype=torch.long)
        save_data(clean_graph, os.path.join(PROCESSED_DATA_DIR, "clean_graph.pt"), "Clean Graph Subset")

        # 6. Inject Nodes
        injected_graph, injected_node_ids = inject_nodes_knn(clean_graph, NUM_INJECTED_NODES, k_connections=INJECTION_K_CONNECTIONS)
        # Also add original_indices attribute to injected graph for consistency, padding with -1 for injected nodes
        injected_graph.original_indices = torch.cat([
            clean_graph.original_indices,
            torch.full((NUM_INJECTED_NODES,), -1, dtype=torch.long)
        ])
        save_data(injected_graph, os.path.join(PROCESSED_DATA_DIR, "injected_graph.pt"), "Injected Graph Subset")
        save_data(injected_node_ids, os.path.join(PROCESSED_DATA_DIR, "injected_node_ids.json"), "Injected Node IDs") # Still useful to know which were injected for analysis

        # 7. Format Prompt & Call LLM API for *ALL* nodes in the injected graph
        logging.info(f"Starting LLM analysis for all {injected_graph.num_nodes} nodes in the graph...")
        all_prompts = []
        all_raw_responses = []
        all_llm_results = [] # This will store the parsed dicts {"node_id": ..., "confidence_score": ..., "justification": ...}

        nodes_to_analyze = injected_graph.node_ids.tolist() # Get all node IDs [0, 1, ..., N-1]

        start_time = time.time()
        for i, node_id_to_analyze in enumerate(nodes_to_analyze):
            if (i + 1) % 10 == 0:
                logging.info(f"  Analyzing node {i+1}/{len(nodes_to_analyze)}...")

            prompt_text = format_prompt_for_llm(df_subset, injected_graph, node_id_to_analyze, NODE_FEATURES)
            parsed_result, raw_response, prompt = call_openai_api(prompt_text, node_id_to_analyze, OPENAI_API_KEY, LLM_MODEL)

            all_prompts.append({"node_id": node_id_to_analyze, "prompt": prompt}) # Store prompt with node ID
            if raw_response is not None:
                 all_raw_responses.append({"node_id": node_id_to_analyze, "response": raw_response}) # Store raw response if available
            if parsed_result is not None: # Ensure we got a result (even if default/error)
                 all_llm_results.append(parsed_result) # Store parsed result

            # Optional: Add a small delay to avoid hitting rate limits
            # time.sleep(0.5) # Adjust delay as needed

        end_time = time.time()
        logging.info(f"Finished LLM analysis. Received {len(all_llm_results)} results for {len(nodes_to_analyze)} nodes. Time taken: {end_time - start_time:.2f} seconds.")

        # Save aggregated prompts and responses
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        save_data(all_prompts, os.path.join(LLM_DATA_DIR, f"all_prompts_{timestamp}.json"), "Aggregated LLM Prompts")
        save_data(all_raw_responses, os.path.join(LLM_DATA_DIR, f"all_raw_responses_{timestamp}.json"), "Aggregated LLM Raw Responses")
        save_data(all_llm_results, os.path.join(LLM_DATA_DIR, f"all_llm_results_parsed_{timestamp}.json"), "Aggregated LLM Parsed Results")

        # 8. Process Combined LLM Response
        nodes_flagged_for_removal = process_llm_results(all_llm_results, CONFIDENCE_THRESHOLD)
        save_data(nodes_flagged_for_removal, os.path.join(PROCESSED_DATA_DIR, "nodes_flagged_by_llm.json"), "Nodes Flagged by LLM")

        # 9. Create LLM-Fixed Graph
        llm_fixed_graph = remove_flagged_nodes(injected_graph, nodes_flagged_for_removal)
        save_data(llm_fixed_graph, os.path.join(PROCESSED_DATA_DIR, "llm_fixed_graph.pt"), "LLM-Fixed Graph")

        # 10. Load GNN Model (CAGN-GAT)
        model = load_cagn_model(MODEL_PATH, input_dim, OUTPUT_DIM, CAGN_HIDDEN_DIM, CAGN_HEADS, CAGN_DROPOUT, DEVICE)

        # 11. Evaluate
        logging.info("\n--- Starting GNN Evaluation ---")
        results_clean = evaluate_model(model, clean_graph, "CAGN on Clean Graph", DEVICE)
        results_injected = evaluate_model(model, injected_graph, f"CAGN on Injected Graph ({NUM_INJECTED_NODES} added nodes)", DEVICE)
        results_fixed = evaluate_model(model, llm_fixed_graph, f"CAGN on LLM-Fixed Graph ({len(nodes_flagged_for_removal)} removed)", DEVICE)

        # Analyze flagged nodes: How many were actually injected vs original?
        original_node_ids = set(range(df_subset.shape[0]))
        injected_node_ids_set = set(injected_node_ids)
        flagged_nodes_set = set(nodes_flagged_for_removal)

        correctly_flagged_injected = len(flagged_nodes_set.intersection(injected_node_ids_set))
        incorrectly_flagged_original = len(flagged_nodes_set.intersection(original_node_ids))
        missed_injected = len(injected_node_ids_set - flagged_nodes_set)


        final_results = { # Renamed variable to avoid confusion with previous 'all_results'
            "clean_graph_eval": results_clean,
            "injected_graph_eval": results_injected,
            "llm_fixed_graph_eval": results_fixed,
            "llm_mitigation_info": {
                 "model": LLM_MODEL,
                 "threshold": CONFIDENCE_THRESHOLD,
                 "nodes_analyzed_by_llm": injected_graph.num_nodes,
                 "nodes_injected_total": NUM_INJECTED_NODES,
                 "nodes_flagged_by_llm": len(nodes_flagged_for_removal),
                 "correctly_flagged_injected": correctly_flagged_injected, # True Positives (flagged injected)
                 "incorrectly_flagged_original": incorrectly_flagged_original, # False Positives (flagged original)
                 "missed_injected_nodes": missed_injected, # False Negatives (missed injected)
                 # Note: We don't store the full LLM output here to keep JSON manageable
                 # The full output is saved separately in the LLM_Data directory
            }
        }

        # 12. Save and Print Results
        # Create DataFrame for GNN eval metrics only
        results_df = pd.DataFrame({
             "Clean": results_clean,
             "Injected": results_injected,
             "LLM_Fixed": results_fixed
        }).T # Transpose for better readability
        # Fill missing metrics with NaN or 0 if evaluation failed for a graph
        results_df = results_df.fillna(0)
        results_df = results_df[['Nodes', 'Edges', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC']] # Reorder cols

        logging.info("\n--- GNN Evaluation Summary ---")
        print(results_df.to_string(float_format="%.4f")) # Use to_string for better console format

        logging.info("\n--- LLM Mitigation Summary ---")
        mitigation_info = final_results["llm_mitigation_info"]
        print(f"LLM Model: {mitigation_info['model']}")
        print(f"Confidence Threshold: {mitigation_info['threshold']}")
        print(f"Total Nodes Analyzed by LLM: {mitigation_info['nodes_analyzed_by_llm']}")
        print(f"Total Nodes Injected: {mitigation_info['nodes_injected_total']}")
        print(f"Total Nodes Flagged by LLM: {mitigation_info['nodes_flagged_by_llm']}")
        print(f"  - Correctly Flagged (Injected): {mitigation_info['correctly_flagged_injected']}")
        print(f"  - Incorrectly Flagged (Original): {mitigation_info['incorrectly_flagged_original']}")
        print(f"  - Missed Injected Nodes: {mitigation_info['missed_injected_nodes']}")

        # Save comprehensive results to JSON
        results_path_json = os.path.join(RESULTS_DIR, f"llm_mitigation_test_results_{timestamp}.json")
        save_data(final_results, results_path_json, "Full Mitigation Test Results (JSON)")
        # Save evaluation DataFrame to CSV
        results_path_csv = os.path.join(RESULTS_DIR, f"llm_mitigation_test_gnn_eval_{timestamp}.csv")
        results_df.to_csv(results_path_csv, float_format="%.6f")
        logging.info(f"Saved GNN evaluation summary table to {results_path_csv}")


    except FileNotFoundError as e:
         logging.error(f"File not found error: {e}")
    except ImportError:
         logging.error("OpenAI library not found. Please install it: pip install openai")
    except Exception as e:
        logging.error(f"An unexpected error occurred in the main workflow: {e}")
        import traceback
        traceback.print_exc()

    finally:
        logging.info("--- LLM Mitigation Test (Analyzing All Nodes) Finished ---")
