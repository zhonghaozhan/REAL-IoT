import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import networkx as nx
import category_encoders as ce
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer # Anomal-E uses Normalizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from pyod.models.cblof import CBLOF # Import CBLOF for the final detection step
import time
import os
import pickle
import math
import gc

# --- Constants ---
DATASET_PATH = '/media/ssd/test/standardized-datasets/combined/combined_netflow_reduced.csv'
# Assuming the model and preprocessing objects are relative to the AnomalE training directory
MODEL_DIR = '/media/ssd/test/GNN/Standardized Models/AnomalE/Combined/'
DGI_MODEL_FILENAME = 'best_dgi.pkl' # From the notebook cell 30
EVAL_DIR = '/media/ssd/test/GNN/Adversarial Evaluation/AnomalE_Evaluation/'
ATTACK_DATA_DIR = os.path.join(EVAL_DIR, 'Attacked_Data')
RESULTS_DIR = os.path.join(EVAL_DIR, 'Results')
RESULTS_FILE = os.path.join(RESULTS_DIR, 'adversarial_results.csv')
PREPROCESS_DIR = os.path.join(EVAL_DIR, 'Preprocess_Objects') # To save/load encoder/scaler

# Columns to keep and process based on the notebook
# Includes IP/Port initially, drops Port later
COLS_TO_KEEP = [
    'IPV4_SRC_ADDR', 'L4_SRC_PORT', 'IPV4_DST_ADDR', 'L4_DST_PORT', 'PROTOCOL', 'L7_PROTO',
    'IN_BYTES', 'OUT_BYTES', 'IN_PKTS', 'OUT_PKTS', 'TCP_FLAGS', 'FLOW_DURATION_MILLISECONDS',
    'Label', 'Attack',
    # Adding features present in the notebook's processing steps that weren't in EGraphSage list
    'CLIENT_TCP_FLAGS', 'SERVER_TCP_FLAGS', 'MIN_TTL', 'MAX_TTL',
    'LONGEST_FLOW_PKT', 'SHORTEST_FLOW_PKT', 'MIN_IP_PKT_LEN', 'MAX_IP_PKT_LEN',
    'RETRANSMITTED_IN_BYTES', 'RETRANSMITTED_IN_PKTS', 'RETRANSMITTED_OUT_BYTES',
    'RETRANSMITTED_OUT_PKTS', 'DURATION_IN', 'DURATION_OUT', 'TCP_WIN_MAX_IN',
    'TCP_WIN_MAX_OUT', 'ICMP_TYPE', 'ICMP_IPV4_TYPE', 'DNS_QUERY_ID',
    'DNS_QUERY_TYPE', 'DNS_TTL_ANSWER', 'FTP_COMMAND_RET_CODE', 'flow_id' # flow_id might be useful?
]

# Features used for Target Encoding in the notebook
CATEGORICAL_FEATURES = ['TCP_FLAGS','L7_PROTO','PROTOCOL',
                        'CLIENT_TCP_FLAGS','SERVER_TCP_FLAGS','ICMP_TYPE',
                        'ICMP_IPV4_TYPE','DNS_QUERY_ID','DNS_QUERY_TYPE',
                        'FTP_COMMAND_RET_CODE']

# Numerical features and categorical features (after encoding) get normalized
# The notebook normalizes all columns from index 2 onwards after encoding
COLS_TO_NORM = [] # Will be determined after target encoding

# PGD Attack Parameters (Targeting DGI Loss on Edge Features)
PGD_EPSILON_LIST = [0.01, 0.05, 0.1, 0.2] # Applied to NORMALIZED features - Reduced range
PGD_ITERATIONS = 60
PGD_ALPHA = 2.5 # Calculated relative to epsilon later

# Black-box Attack Parameters (IP:Port Graph)
EDGE_REMOVAL_RATES = [0.05, 0.10, 0.20, 0.30]
NODE_INJECTION_RATES = [0.05, 0.10, 0.20]
NODE_INJECTION_CONNECTIONS_PER_NODE = 5

# CBLOF Detector Parameters (From best run in notebook)
CBLOF_N_CLUSTERS = 8
CBLOF_CONTAMINATION = 0.01
CBLOF_ALPHA = 0.7
CBLOF_BETA = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Ensure output directories exist
os.makedirs(ATTACK_DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PREPROCESS_DIR, exist_ok=True)

# --- Model Definition (Copied from AnomalE Notebook) ---

class SAGELayer(nn.Module):
    def __init__(self, ndim_in, edims, ndim_out, activation):
      super(SAGELayer, self).__init__()
      # Note: The original notebook W_apply uses ndim_in + edims, which seems unusual for SAGE.
      # Typically it's ndim_in + ndim_out (input node feature + aggregated neighbor feature).
      # However, AnomalE DGI processes differently. Let's stick to the notebook's definition.
      # The input nfeats (ndim_in) and aggregated neighbors ('h_neigh', edims?) are concatenated.
      self.W_apply = nn.Linear(ndim_in + edims , ndim_out)
      self.activation = F.relu # Explicitly set, notebook used F.relu directly
      # This W_edge seems specific to DGI's way of getting edge embeddings *after* node updates
      self.W_edge = nn.Linear(128 * 2, 256) # Assumes ndim_out is 128
      self.reset_parameters()

    def reset_parameters(self):
      gain = nn.init.calculate_gain('relu')
      nn.init.xavier_uniform_(self.W_apply.weight, gain=gain)
      # Should W_edge also be initialized? Notebook doesn't show it.
      # Let's add initialization for W_edge as well for completeness.
      nn.init.xavier_uniform_(self.W_edge.weight, gain=gain)


    def message_func(self, edges):
      # DGI message function only passes edge features
      return {'m':  edges.data['h']}

    def forward(self, g_dgl, nfeats, efeats):
      with g_dgl.local_scope():
        g = g_dgl
        g.ndata['h'] = nfeats
        g.edata['h'] = efeats
        # The aggregation function seems to be mean of *edge features* ('m') incident to a node.
        g.update_all(self.message_func, fn.mean('m', 'h_neigh'))
        # The node update concatenates the original node feature 'h' and the aggregated edge features 'h_neigh'
        # This matches the W_apply input dims: ndim_in + edims
        g.ndata['h'] = self.activation(self.W_apply(torch.cat([g.ndata['h'], g.ndata['h_neigh']], 2)))

        # Compute edge embeddings *after* node embeddings are updated
        u, v = g.edges()
        # Uses the *updated* node features (h_u, h_v) to compute edge embeddings
        edge_features_out = self.W_edge(torch.cat((g.ndata['h'][u], g.ndata['h'][v]), 2))
        # Returns updated node features AND derived edge features
        return g.ndata['h'], edge_features_out


class SAGE(nn.Module):
    # This is the SAGE encoder used within DGI in the notebook
    def __init__(self, ndim_in, ndim_out, edim,  activation):
      super(SAGE, self).__init__()
      self.layers = nn.ModuleList()
      # Only one layer in the notebook implementation
      self.layers.append(SAGELayer(ndim_in, edim, 128, activation)) # ndim_out fixed to 128

    def forward(self, g, nfeats, efeats, corrupt=False):
      if corrupt:
        # Permute edge features for negative samples in DGI
        perm = torch.randperm(g.number_of_edges(), device=efeats.device)
        efeats = efeats[perm]
      # Node features are not corrupted in the notebook DGI implementation

      for i, layer in enumerate(self.layers):
          nfeats, efeats_out = layer(g, nfeats, efeats)

      # Notebook returns sum over the sequence dimension (dim 1)
      # Assumes input nfeats/efeats are [N/E, 1, D]
      return nfeats.sum(1), efeats_out.sum(1)

class Discriminator(nn.Module):
    # DGI discriminator
    def __init__(self, n_hidden):
      super(Discriminator, self).__init__()
      self.weight = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
      self.reset_parameters()

    def uniform(self, size, tensor):
      bound = 1.0 / math.sqrt(size)
      if tensor is not None:
        tensor.data.uniform_(-bound, bound)

    def reset_parameters(self):
      size = self.weight.size(0)
      self.uniform(size, self.weight)

    def forward(self, features, summary):
      # Bilinear scoring function
      features = torch.matmul(features, torch.matmul(self.weight, summary))
      return features

class DGI(nn.Module):
    def __init__(self, ndim_in, ndim_out, edim, activation):
      super(DGI, self).__init__()
      self.encoder = SAGE(ndim_in, ndim_out, edim, activation)
      self.discriminator = Discriminator(256) # Based on W_edge output dim in SAGELayer
      self.loss_fn = nn.BCEWithLogitsLoss() # Renamed from self.loss to avoid conflict

    def forward(self, g, n_features, e_features):
      # Positive samples (real graph structure)
      _, positive_edges = self.encoder(g, n_features, e_features, corrupt=False)
      # Negative samples (permuted edge features)
      _, negative_edges = self.encoder(g, n_features, e_features, corrupt=True)

      # Use edge features for discrimination
      positive = positive_edges
      negative = negative_edges

      # Create summary vector (sigmoid of mean of positive edge features)
      summary = torch.sigmoid(positive.mean(dim=0))

      # Score positive/negative samples against summary
      positive = self.discriminator(positive, summary)
      negative = self.discriminator(negative, summary)

      # Calculate BCE loss
      l1 = self.loss_fn(positive, torch.ones_like(positive))
      l2 = self.loss_fn(negative, torch.zeros_like(negative))

      return l1 + l2

# --- Data Loading and Preprocessing Function ---

def load_and_preprocess_data(dataset_path, preprocess_dir):
    """Loads, preprocesses (encoding, normalization), and splits the data."""
    print("Loading and preprocessing data...")
    data = pd.read_csv(dataset_path)
    print(f"Original data shape: {data.shape}")

    # Keep only necessary columns (ensure 'Label' and 'Attack' are present for split/encoding)
    # Filter COLS_TO_KEEP to only include columns present in the loaded dataframe
    cols_present = [col for col in COLS_TO_KEEP if col in data.columns]
    if len(cols_present) != len(COLS_TO_KEEP):
        print(f"Warning: Not all expected columns found in dataset. Using: {cols_present}")
    data = data[cols_present].copy()

    # Basic preprocessing from notebook
    data.rename(columns=lambda x: x.strip(), inplace=True)
    data['IPV4_SRC_ADDR'] = data["IPV4_SRC_ADDR"].astype(str)
    data['L4_SRC_PORT'] = data["L4_SRC_PORT"].astype(str) # Keep temporarily if needed for node naming? No, notebook drops it.
    data['IPV4_DST_ADDR'] = data["IPV4_DST_ADDR"].astype(str)
    data['L4_DST_PORT'] = data["L4_DST_PORT"].astype(str) # Keep temporarily? No, notebook drops it.
    data.drop(columns=["L4_SRC_PORT", "L4_DST_PORT"], inplace=True, errors='ignore')

    # Define features and labels
    X = data.drop(columns=["Attack", "Label"])
    y = data[["Attack", "Label"]] # Keep both for potential multi-class encoding/stratification

    # Split data (consistent with other evals: 70/30, stratified by binary Label)
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=13, stratify=y['Label']
    )
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Target Encoding (Fit ONLY on Training Data)
    print("Fitting TargetEncoder...")
    encoder = ce.TargetEncoder(cols=CATEGORICAL_FEATURES)
    # Use binary Label for fitting encoder target as done in notebook cell 14
    encoder.fit(X_train, y_train['Label'])
    encoder_path = os.path.join(preprocess_dir, 'anomale_target_encoder.pkl')
    with open(encoder_path, 'wb') as f:
        pickle.dump(encoder, f)
    print(f"TargetEncoder saved to {encoder_path}")

    print("Transforming data with TargetEncoder...")
    X_train_encoded = encoder.transform(X_train)
    X_test_encoded = encoder.transform(X_test)

    # Handle potential inf/-inf values created by encoder
    X_train_encoded.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test_encoded.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Fill NaNs with 0 as done in the notebook
    X_train_encoded.fillna(0, inplace=True)
    X_test_encoded.fillna(0, inplace=True)

    # Normalization (Fit ONLY on Training Data)
    # Identify columns to normalize (all except IPs)
    cols_to_norm = [col for col in X_train_encoded.columns if col not in ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR']]
    print(f"Columns to normalize: {cols_to_norm}")

    print("Fitting Normalizer...")
    scaler = Normalizer()
    # Pass numpy array to avoid feature name warnings
    scaler.fit(X_train_encoded[cols_to_norm].values)
    normalizer_path = os.path.join(preprocess_dir, 'anomale_normalizer.pkl')
    with open(normalizer_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Normalizer saved to {normalizer_path}")

    print("Transforming data with Normalizer...")
    X_train_scaled = X_train_encoded.copy()
    X_test_scaled = X_test_encoded.copy()
    X_train_scaled[cols_to_norm] = scaler.transform(X_train_encoded[cols_to_norm].values)
    X_test_scaled[cols_to_norm] = scaler.transform(X_test_encoded[cols_to_norm].values)

    # Add 'h' column (list of features) - exclude IPs
    print("Creating 'h' feature column...")
    X_train_scaled['h'] = X_train_scaled[cols_to_norm].values.tolist()
    X_test_scaled['h'] = X_test_scaled[cols_to_norm].values.tolist()

    print("Preprocessing finished.")
    return X_train_scaled, X_test_scaled, y_train, y_test, encoder, scaler, cols_to_norm

# --- Graph Building Function ---

def build_dgl_graph(df, y_df):
    """Builds a DGL graph from a pandas DataFrame."""
    print(f"Building DGL graph from dataframe with shape: {df.shape}")
    # Combine features and labels for graph construction
    graph_df = df[['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'h']].copy()
    graph_df['Label'] = y_df['Label'].values
    # Attack column might not be needed if not used as edge attribute, but include for consistency
    graph_df['Attack'] = y_df['Attack'].values

    # Create NetworkX graph
    G_nx = nx.from_pandas_edgelist(
        graph_df,
        "IPV4_SRC_ADDR",
        "IPV4_DST_ADDR",
        edge_attr=['h', 'Label'], # Include necessary attributes
        create_using=nx.MultiDiGraph() # Use MultiDiGraph as in notebook
    )
    print(f"NX graph created: {G_nx.number_of_nodes()} nodes, {G_nx.number_of_edges()} edges")

    # Convert to DGL graph
    g_dgl = dgl.from_networkx(G_nx, edge_attrs=['h', 'Label'])

    # Extract edge feature dimension from 'h' attribute
    # Need to handle potential errors if 'h' is empty or malformed
    try:
        edge_feature_dim = len(g_dgl.edata['h'][0]) if g_dgl.number_of_edges() > 0 else 0
        if edge_feature_dim == 0 and g_dgl.number_of_edges() > 0:
             print("Warning: Edge feature dimension calculated as 0. Check 'h' column.")
             # Attempt to infer dim from the df if possible
             if 'h' in df.columns and len(df['h']) > 0:
                 edge_feature_dim = len(df['h'].iloc[0])

        print(f"Edge feature dimension: {edge_feature_dim}")

        # Initialize node features (as ones with edge feature dimension, like notebook)
        node_feature_dim = edge_feature_dim # Notebook uses this convention
        g_dgl.ndata['h'] = torch.ones(g_dgl.num_nodes(), node_feature_dim)
        print(f"Initialized node features with shape: {g_dgl.ndata['h'].shape}")

        # Ensure edge features are tensors
        g_dgl.edata['h'] = torch.tensor(np.array(g_dgl.edata['h'].tolist()), dtype=torch.float32)
        g_dgl.edata['Label'] = torch.tensor(g_dgl.edata['Label'], dtype=torch.long)
        # Attack column is not directly used by model but keep it as tensor if needed later
        # If Attack label encoding is needed:
        # le_attack = preprocessing.LabelEncoder()
        # g_dgl.edata['Attack'] = torch.tensor(le_attack.fit_transform(g_dgl.edata['Attack']), dtype=torch.long)

        # Reshape features to [N/E, 1, D] as expected by the notebook's model
        g_dgl.ndata['h'] = g_dgl.ndata['h'].unsqueeze(1)
        g_dgl.edata['h'] = g_dgl.edata['h'].unsqueeze(1)

        print(f"Final DGL graph: {g_dgl.num_nodes()} nodes, {g_dgl.num_edges()} edges")
        print(f"Final Node feature shape: {g_dgl.ndata['h'].shape}")
        print(f"Final Edge feature shape: {g_dgl.edata['h'].shape}")

    except Exception as e:
        print(f"Error during DGL graph finalization: {e}")
        # Potentially return None or raise exception
        return None


    return g_dgl


# --- PGD Attack Function ---

def run_pgd_attack_anomale_dgi(dgi_model, graph, epsilon):
    """Performs PGD attack targeting the DGI loss on edge features."""
    print(f"Running PGD attack targeting DGI loss with epsilon={epsilon}")
    dgi_model.eval() # Keep DGI in eval mode but allow gradients

    attacked_graph = graph.clone().to(DEVICE)
    node_features = attacked_graph.ndata['h'].detach().clone() # Clean node features
    edge_features = attacked_graph.edata['h'].detach().clone() # Edge features to perturb
    edge_features_clean = edge_features.detach().clone()      # Reference clean features

    edge_features.requires_grad = True

    # Calculate step size relative to epsilon
    alpha = (PGD_ALPHA * epsilon) / PGD_ITERATIONS # Use constant PGD_ALPHA here

    for i in range(PGD_ITERATIONS):
        edge_features.requires_grad = True
        attacked_graph.edata['h'] = edge_features # Use current perturbed edge features

        # --- Calculate DGI Loss ---
        # We want to MAXIMIZE the DGI loss for the attack
        # So we calculate the gradient of the loss and *add* it (gradient ascent)
        # Or calculate loss = -dgi(...) and minimize that (gradient descent)
        loss = dgi_model(attacked_graph, node_features, edge_features)

        dgi_model.zero_grad()
        if edge_features.grad is not None:
            edge_features.grad.zero_()

        # Calculate gradients to maximize loss (or minimize negative loss)
        (-loss).backward() # Minimize negative loss == Maximize loss

        if edge_features.grad is None:
            print(f"Warning: Edge feature gradients are None on iteration {i}. Stopping attack.")
            break

        with torch.no_grad():
            # --- Apply PGD Update (Gradient Ascent Step) ---
            grad_edges = edge_features.grad.detach()
            if torch.isnan(grad_edges).any():
                print(f"Warning: NaN detected in edge feature gradients on iteration {i}. Skipping update.")
                continue
            # Update uses the sign of the gradient to find the direction of steepest ascent for the loss
            update = alpha * grad_edges.sign()
            perturbed_edge_features = edge_features + update # Add update to ascend loss

            # --- Clipping within Epsilon Ball (Edge Feature Space) ---
            eta = perturbed_edge_features - edge_features_clean
            eta = torch.clamp(eta, -epsilon, epsilon)
            perturbed_edge_features = edge_features_clean + eta

            # Update edge features for the next iteration
            edge_features = perturbed_edge_features.clone()

        if (i+1) % 10 == 0:
            print(f"PGD Iteration [{i+1}/{PGD_ITERATIONS}], DGI Loss: {loss.item():.4f}")

    # Store final perturbed edge features
    attacked_graph.edata['h_perturbed'] = edge_features.detach()
    # Restore clean node features
    attacked_graph.ndata['h'] = node_features.detach()
    # Copy clean features to 'h_perturbed' for nodes if needed
    if 'h_perturbed' not in attacked_graph.ndata:
         attacked_graph.ndata['h_perturbed'] = node_features.detach()

    print(f"PGD attack finished for epsilon={epsilon}.")
    return attacked_graph


# --- Evaluation Function ---

def evaluate_anomale(dgi_model, detector, graph_data, use_perturbed=False):
    """Evaluates Anomal-E performance using generated embeddings and a detector."""
    feature_key = 'h_perturbed' if use_perturbed else 'h'
    data_type = 'perturbed' if use_perturbed else 'clean'
    print(f"Evaluating Anomal-E on {data_type} graph using EDGE features '{feature_key}'...")

    if 'h' not in graph_data.ndata: # Need original node features
        print("Error: Node features 'h' not found in graph data.")
        return {'acc': np.nan, 'precision': np.nan, 'recall': np.nan, 'f1': np.nan, 'auc': np.nan}
    if feature_key not in graph_data.edata:
        print(f"Error: Edge features '{feature_key}' not found in graph data.")
        return {'acc': np.nan, 'precision': np.nan, 'recall': np.nan, 'f1': np.nan, 'auc': np.nan}

    dgi_model.eval() # Ensure DGI encoder is in eval mode
    with torch.no_grad():
        node_features = graph_data.ndata['h'].to(DEVICE)
        edge_features = graph_data.edata[feature_key].to(DEVICE)
        edge_labels_binary = graph_data.edata['Label'].to(DEVICE).long() # Ensure labels are long

        # Generate embeddings using the DGI encoder
        _, edge_embeddings = dgi_model.encoder(graph_data, node_features, edge_features)
        edge_embeddings_np = edge_embeddings.detach().cpu().numpy()

        # Predict using the pre-trained anomaly detector
        preds_binary = detector.predict(edge_embeddings_np) # Detector predicts 0 (inlier) or 1 (outlier/anomaly)

        # Ensure labels and preds are on CPU for sklearn metrics
        edge_labels_binary_cpu = edge_labels_binary.cpu().numpy()
        preds_binary_cpu = preds_binary # Already numpy array

        # Calculate metrics
        acc = accuracy_score(edge_labels_binary_cpu, preds_binary_cpu)
        precision = precision_score(edge_labels_binary_cpu, preds_binary_cpu, zero_division=0)
        recall = recall_score(edge_labels_binary_cpu, preds_binary_cpu, zero_division=0)
        f1 = f1_score(edge_labels_binary_cpu, preds_binary_cpu, zero_division=0)
        # AUC requires anomaly scores, not just labels. CBLOF predict_proba might work if available.
        # Using decision_function as a proxy for scores if predict_proba isn't available.
        try:
            if hasattr(detector, 'decision_function'):
                scores = detector.decision_function(edge_embeddings_np)
                auc = roc_auc_score(edge_labels_binary_cpu, scores)
            else:
                # If no decision_function, we can't calculate AUC this way
                print("Warning: Detector lacks decision_function. AUC cannot be calculated.")
                auc = np.nan
        # Handle cases where decision_function might return NaN or Inf or other errors
        except ValueError as ve:
            if 'Input contains NaN, infinity or a value too large' in str(ve):
                print("Warning: NaNs/Infs detected in detector scores. Setting AUC to NaN.")
                auc = np.nan
            else:
                print(f"Warning: ValueError during AUC calculation ({ve}).")
                auc = np.nan
        except Exception as e:
            print(f"Warning: Could not calculate AUC ({e}).")
            auc = np.nan

    print(f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
    return {'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1, 'auc': auc}


# --- Black-Box Attack Functions (Adapted for Anomal-E IP:Port Graph) ---

def run_edge_removal_attack_ipport_anomale(graph, rate):
    """
    Performs Edge Removal attack on an Anomal-E IP:Port DGL graph.
    Ensures operations are on CPU.

    Args:
        graph: The clean DGL graph (will be moved to CPU).
        rate: The fraction of edges to remove (e.g., 0.1 for 10%).

    Returns:
        A new DGL graph (on CPU) with the specified fraction of edges removed.
    """
    print(f"Running Edge Removal attack with rate={rate}")
    cpu_device = torch.device('cpu')
    graph_cpu = graph.to(cpu_device)

    num_edges_original = graph_cpu.num_edges()
    num_edges_to_remove = int(num_edges_original * rate)

    if num_edges_to_remove >= num_edges_original:
        print(f"Warning: Removal rate ({rate}) is too high. Removing all edges.")
        ids_to_remove = torch.arange(num_edges_original, device=cpu_device)
    elif num_edges_to_remove <= 0:
         print("Info: Removal rate resulted in 0 edges to remove. Returning original graph structure.")
         return graph_cpu.clone()
    else:
        edge_ids = torch.arange(num_edges_original)
        shuffled_ids = edge_ids[torch.randperm(num_edges_original)]
        ids_to_remove = shuffled_ids[:num_edges_to_remove].to(cpu_device)

    print(f"Removing {len(ids_to_remove)} edges out of {num_edges_original} (Rate: {rate})...")
    # Ensure graph is on CPU before removal
    attacked_graph = dgl.remove_edges(graph_cpu, ids_to_remove)
    print(f"Edge removal finished. New graph has {attacked_graph.num_edges()} edges.")

    return attacked_graph # Returns CPU graph

def run_node_injection_attack_ipport_anomale(graph, rate, connections_per_node):
    """
    Performs Node Injection attack on an Anomal-E IP:Port DGL graph.
    Ensures operations are on CPU.

    Args:
        graph: The clean DGL graph (will be moved to CPU).
        rate: The fraction of nodes to inject relative to original node count.
        connections_per_node: How many edges each new node should create.

    Returns:
        A new DGL graph (on CPU) with injected nodes and edges.
    """
    print(f"Running Node Injection attack with rate={rate}")
    cpu_device = torch.device('cpu')
    graph_cpu = graph.to(cpu_device)

    num_nodes_original = graph_cpu.num_nodes()
    num_nodes_to_inject = int(num_nodes_original * rate)

    if num_nodes_to_inject <= 0:
        print("Info: Injection rate resulted in 0 nodes to inject. Returning original graph.")
        return graph_cpu.clone()

    print(f"Injecting {num_nodes_to_inject} nodes (Rate: {rate})...")

    # --- 1. Add New Nodes --- #
    original_node_feat_shape = graph_cpu.ndata['h'].shape
    node_feature_dim = original_node_feat_shape[-1]
    new_node_features = torch.ones(num_nodes_to_inject, 1, node_feature_dim,
                                    dtype=graph_cpu.ndata['h'].dtype, device=cpu_device)

    attacked_graph = graph_cpu.clone()
    attacked_graph.add_nodes(num_nodes_to_inject, data={'h': new_node_features})
    num_nodes_total = attacked_graph.num_nodes()
    print(f"Nodes added. New graph node count: {num_nodes_total}")

    # --- 2. Prepare for Edge Injection --- #
    attack_edge_mask = graph_cpu.edata['Label'] == 1 # Use binary Label
    attack_edge_ids = torch.where(attack_edge_mask)[0].to(cpu_device)

    if len(attack_edge_ids) == 0:
        print("Warning: No attack edges found in the original graph to sample features from. Using mean features.")
        # Fallback: Use mean of all edge features if no attack edges found
        if graph_cpu.num_edges() > 0:
            # Calculate mean on CPU, handle potential empty graph
            mean_edge_features = graph_cpu.edata['h'].mean(dim=0, keepdim=True).to(cpu_device)
        else:
             print("Error: Cannot inject edges as graph has no edges to sample/average features from.")
             return attacked_graph # Return graph with only nodes added
    else:
        attack_edge_features = graph_cpu.edata['h'][attack_edge_ids]

    # --- 3. Generate and Add New Edges --- #
    new_edge_src_nodes = []
    new_edge_dst_nodes = []
    num_new_edges_to_add = num_nodes_to_inject * connections_per_node

    for i in range(num_nodes_to_inject):
        new_node_id = num_nodes_original + i
        possible_dst_nodes = torch.arange(num_nodes_original, device=cpu_device)
        k = min(connections_per_node, num_nodes_original)
        dst_nodes = possible_dst_nodes[torch.randperm(num_nodes_original, device=cpu_device)[:k]]

        new_edge_src_nodes.extend([new_node_id] * k)
        new_edge_dst_nodes.extend(dst_nodes.tolist())

    new_edge_src = torch.tensor(new_edge_src_nodes, device=cpu_device)
    new_edge_dst = torch.tensor(new_edge_dst_nodes, device=cpu_device)
    actual_new_edges_count = len(new_edge_src)

    if actual_new_edges_count == 0:
        print("Warning: No new edges were generated.")
        return attacked_graph

    print(f"Generating {actual_new_edges_count} new edges...")

    # --- 4. Generate Features and Labels for New Edges --- #
    if len(attack_edge_ids) == 0:
        # Use mean features if no attack edges were found
        new_edge_features = mean_edge_features.repeat(actual_new_edges_count, 1, 1)
    else:
        # Sample features from existing attack edges
        sample_indices = torch.randint(0, len(attack_edge_ids), (actual_new_edges_count,), device=cpu_device)
        new_edge_features = attack_edge_features[sample_indices]

    new_edge_labels_binary = torch.ones(actual_new_edges_count, dtype=torch.long, device=cpu_device)

    # --- 5. Add Edges to Graph --- #
    attacked_graph.add_edges(
        new_edge_src,
        new_edge_dst,
        data={
            'h': new_edge_features,
            'Label': new_edge_labels_binary # Use 'Label' key consistent with graph build
        }
    )

    print(f"Node injection finished. Final graph: {attacked_graph.num_nodes()} nodes, {attacked_graph.num_edges()} edges.")
    return attacked_graph # Returns CPU graph


# --- Main Execution Logic ---
if __name__ == "__main__":
    start_time = time.time()

    # 1. Load and Preprocess Data
    X_train_scaled, X_test_scaled, y_train, y_test, encoder, normalizer, cols_to_norm = load_and_preprocess_data(DATASET_PATH, PREPROCESS_DIR)

    if X_train_scaled is None:
        print("Data loading/preprocessing failed. Exiting.")
        exit()

    # 2. Build Graphs
    if not cols_to_norm:
         print("Error: Columns to normalize were not determined.")
         exit()
    edge_feature_dim = len(cols_to_norm)
    node_feature_dim = edge_feature_dim # Based on notebook's node init
    print(f"Determined Node/Edge Feature Dim: {node_feature_dim}") # Debug print

    # Build graphs only if they don't exist
    g_train_clean_path = os.path.join(ATTACK_DATA_DIR, 'train_data_clean_anomale.pt')
    g_test_clean_path = os.path.join(ATTACK_DATA_DIR, 'test_data_clean_anomale.pt')

    if not os.path.exists(g_train_clean_path) or not os.path.exists(g_test_clean_path):
        print("Building clean graphs...")
        print(f"\nBuilding training graph (Node dim: {node_feature_dim}, Edge dim: {edge_feature_dim})...")
        train_graph = build_dgl_graph(X_train_scaled, y_train)
        print(f"\nBuilding test graph (Node dim: {node_feature_dim}, Edge dim: {edge_feature_dim})...")
        test_graph = build_dgl_graph(X_test_scaled, y_test)

        if train_graph is None or test_graph is None:
            print("Graph building failed. Exiting.")
            exit()

        print(f"Saving clean training graph to {g_train_clean_path}...")
        torch.save(train_graph.cpu(), g_train_clean_path)
        print(f"Saving clean test graph to {g_test_clean_path}...")
        torch.save(test_graph.cpu(), g_test_clean_path)
        print("Clean graphs saved.")
        del train_graph, test_graph # Free memory after saving
    else:
        print("Clean graphs already exist. Skipping build.")

    del X_train_scaled, X_test_scaled, y_train, y_test, encoder, normalizer, cols_to_norm
    gc.collect()

    # 3. Load Pre-trained DGI Model (Prioritize original model)
    print("\nLoading DGI model...")
    dgi_ndim_in = node_feature_dim
    dgi_edim = edge_feature_dim
    dgi_ndim_out = 128 # Fixed in SAGE layer

    dgi_model = DGI(
        ndim_in=dgi_ndim_in,
        ndim_out=dgi_ndim_out,
        edim=dgi_edim,
        activation=F.relu
    )

    # Attempt to load the original pre-trained model first
    original_dgi_model_path = os.path.join(MODEL_DIR, DGI_MODEL_FILENAME)
    loaded_successfully = False
    if os.path.exists(original_dgi_model_path):
        print(f"Attempting to load pre-trained DGI model from {original_dgi_model_path}...")
        try:
            # Try loading with weights_only=True first for security
            dgi_model.load_state_dict(torch.load(original_dgi_model_path, map_location=DEVICE, weights_only=True))
            loaded_successfully = True
        except Exception as e_true:
            print(f"Failed loading DGI model with weights_only=True ({e_true}). Trying weights_only=False.")
            try:
                # Fallback to weights_only=False if needed (less secure)
                dgi_model.load_state_dict(torch.load(original_dgi_model_path, map_location=DEVICE, weights_only=False))
                loaded_successfully = True
            except Exception as e_false:
                 print(f"Warning: Failed loading DGI model from {original_dgi_model_path} even with weights_only=False: {e_false}")
    else:
        print(f"Warning: Original DGI model not found at {original_dgi_model_path}")

    if not loaded_successfully:
        # As a last resort, check if a locally trained one exists (e.g., from a previous eval run)
        new_dgi_model_filename = 'best_dgi_eval_script.pkl'
        new_dgi_model_path = os.path.join(MODEL_DIR, new_dgi_model_filename) # Check MODEL_DIR for it
        if os.path.exists(new_dgi_model_path):
            print(f"Attempting to load locally trained DGI model from {new_dgi_model_path}...")
            try:
                dgi_model.load_state_dict(torch.load(new_dgi_model_path, map_location=DEVICE, weights_only=True))
                loaded_successfully = True
            except Exception as e_true:
                 print(f"Failed loading locally trained DGI with weights_only=True ({e_true}). Trying weights_only=False.")
                 try:
                     dgi_model.load_state_dict(torch.load(new_dgi_model_path, map_location=DEVICE, weights_only=False))
                     loaded_successfully = True
                 except Exception as e_false:
                     print(f"ERROR: Failed loading locally trained DGI model from {new_dgi_model_path}: {e_false}")
                     exit()
        else:
            print(f"ERROR: No usable DGI model found at {original_dgi_model_path} or {new_dgi_model_path}. Cannot proceed.")
            exit()

    dgi_model.to(DEVICE)
    dgi_model.eval() # Ensure model is in evaluation mode
    print("DGI model loaded and set to evaluation mode.")


    # 4. Load Pre-trained Detector (or Train if necessary - changed logic)
    print("\nLoading anomaly detector...")
    # Prioritize detector saved alongside original DGI model
    original_detector_path = os.path.join(MODEL_DIR, DGI_MODEL_FILENAME)
    # Fallback path used by previous script version
    fallback_detector_path = os.path.join(PREPROCESS_DIR, 'cblof_detector.pkl')

    detector = None
    if os.path.exists(original_detector_path):
        print(f"Loading detector from primary path: {original_detector_path}...")
        try:
            with open(original_detector_path, 'rb') as f:
                detector = pickle.load(f)
            print("Detector loaded from primary path.")
        except Exception as e:
            print(f"Warning: Failed to load detector from {original_detector_path}: {e}")

    # If not loaded from primary path, try the fallback path
    if detector is None and os.path.exists(fallback_detector_path):
        print(f"Loading detector from fallback path: {fallback_detector_path}...")
        try:
            with open(fallback_detector_path, 'rb') as f:
                detector = pickle.load(f)
            print("Detector loaded from fallback path.")
        except Exception as e:
            print(f"Warning: Failed to load detector from {fallback_detector_path}: {e}")

    # Check if detector was loaded successfully from either path
    if detector is None:
        print(f"ERROR: Pre-trained detector not found at {original_detector_path} or {fallback_detector_path}. Cannot proceed.")
        exit() # Exit if no detector found
    # else:
        # print("Anomaly detector loaded successfully.") # Redundant, printed above


    # 5. Generate Attacked Data
    print("\n--- Generating Attacked Data ---")
    g_test_clean = torch.load(g_test_clean_path) # Load clean test graph (CPU)

    # PGD Attacks
    print("\nGenerating PGD Attacked Data...")
    for eps in PGD_EPSILON_LIST:
        attacked_graph_path = os.path.join(ATTACK_DATA_DIR, f'test_data_pgd_eps{eps}_AnomalE_DGI.pt')
        if not os.path.exists(attacked_graph_path):
            print(f"Generating PGD attack data for epsilon = {eps}")
            attacked_graph_pgd = run_pgd_attack_anomale_dgi(dgi_model, g_test_clean.clone().to(DEVICE), eps)
            # Save attacked graph (CPU)
            torch.save(attacked_graph_pgd.cpu(), attacked_graph_path)
            print(f"Attacked graph saved to {attacked_graph_path}")
            del attacked_graph_pgd
            gc.collect()
        else:
            print(f"PGD attack data for epsilon = {eps} already exists. Skipping generation.")

    # Edge Removal Attacks
    print("\nGenerating Edge Removal Attacked Data...")
    for rate in EDGE_REMOVAL_RATES:
        attacked_graph_path = os.path.join(ATTACK_DATA_DIR, f'test_data_edge_remove_{int(rate*100)}pct_anomale.pt')
        if not os.path.exists(attacked_graph_path):
            print(f"Generating Edge Removal attack data for rate = {rate}")
            attacked_graph_removed = run_edge_removal_attack_ipport_anomale(g_test_clean.clone(), rate)
            torch.save(attacked_graph_removed.cpu(), attacked_graph_path)
            print(f"Attacked graph saved to {attacked_graph_path}")
            del attacked_graph_removed
            gc.collect()
        else:
             print(f"Edge Removal attack data for rate = {rate} already exists. Skipping generation.")

    # Node Injection Attacks
    print("\nGenerating Node Injection Attacked Data...")
    for rate in NODE_INJECTION_RATES:
        attacked_graph_path = os.path.join(ATTACK_DATA_DIR, f'test_data_node_inject_{int(rate*100)}pct_anomale.pt')
        if not os.path.exists(attacked_graph_path):
            print(f"Generating Node Injection attack data for rate = {rate}")
            attacked_graph_injected = run_node_injection_attack_ipport_anomale(g_test_clean.clone(), rate, NODE_INJECTION_CONNECTIONS_PER_NODE)
            torch.save(attacked_graph_injected.cpu(), attacked_graph_path)
            print(f"Attacked graph saved to {attacked_graph_path}")
            del attacked_graph_injected
            gc.collect()
        else:
             print(f"Node Injection attack data for rate = {rate} already exists. Skipping generation.")

    del g_test_clean # Free memory after loop
    gc.collect()

    # 6. Evaluate Models
    print("\n--- Evaluating Model Performance ---")
    results = []
    # Ensure detector is loaded
    if 'detector' not in locals() or detector is None:
        print("ERROR: Detector not loaded. Cannot evaluate.")
        exit()

    # Evaluate on Clean Data
    print("\nEvaluating on Clean Data...")
    g_test_clean = torch.load(g_test_clean_path).to(DEVICE)
    clean_metrics = evaluate_anomale(dgi_model, detector, g_test_clean, use_perturbed=False)
    results.append({'attack_type': 'Clean', 'param': 0, **clean_metrics})
    del g_test_clean
    gc.collect()

    # Evaluate on PGD Attacked Data
    print("\nEvaluating on PGD Attacked Data...")
    for eps in PGD_EPSILON_LIST:
        attacked_graph_path = os.path.join(ATTACK_DATA_DIR, f'test_data_pgd_eps{eps}_AnomalE_DGI.pt')
        if os.path.exists(attacked_graph_path):
            print(f"\nLoading PGD attacked graph for epsilon = {eps}")
            attacked_graph_data = torch.load(attacked_graph_path).to(DEVICE)
            pgd_metrics = evaluate_anomale(dgi_model, detector, attacked_graph_data, use_perturbed=True)
            results.append({'attack_type': 'PGD_AnomalE_DGI', 'param': eps, **pgd_metrics})
            del attacked_graph_data
            gc.collect()
        else:
            print(f"Attacked graph file not found: {attacked_graph_path}")

    # Evaluate on Edge Removal Attacked Data
    print("\nEvaluating on Edge Removal Attacked Data...")
    for rate in EDGE_REMOVAL_RATES:
        attacked_graph_path = os.path.join(ATTACK_DATA_DIR, f'test_data_edge_remove_{int(rate*100)}pct_anomale.pt')
        if os.path.exists(attacked_graph_path):
            print(f"\nLoading Edge Removal graph for rate = {rate}")
            attacked_graph_data = torch.load(attacked_graph_path).to(DEVICE)
            # Evaluate using clean features (use_perturbed=False)
            edge_remove_metrics = evaluate_anomale(dgi_model, detector, attacked_graph_data, use_perturbed=False)
            results.append({'attack_type': 'EdgeRemove_AnomalE', 'param': rate, **edge_remove_metrics})
            del attacked_graph_data
            gc.collect()
        else:
            print(f"Attacked graph file not found: {attacked_graph_path}")

    # Evaluate on Node Injection Attacked Data
    print("\nEvaluating on Node Injection Attacked Data...")
    for rate in NODE_INJECTION_RATES:
        attacked_graph_path = os.path.join(ATTACK_DATA_DIR, f'test_data_node_inject_{int(rate*100)}pct_anomale.pt')
        if os.path.exists(attacked_graph_path):
            print(f"\nLoading Node Injection graph for rate = {rate}")
            attacked_graph_data = torch.load(attacked_graph_path).to(DEVICE)
            # Evaluate using clean features (use_perturbed=False)
            node_inject_metrics = evaluate_anomale(dgi_model, detector, attacked_graph_data, use_perturbed=False)
            results.append({'attack_type': 'NodeInject_AnomalE', 'param': rate, **node_inject_metrics})
            del attacked_graph_data
            gc.collect()
        else:
            print(f"Attacked graph file not found: {attacked_graph_path}")

    # 7. Save Results
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_FILE, index=False)
    print(f"\nResults saved to {RESULTS_FILE}")
    print(results_df)

    # 8. Sanity Check (Keep as is for PGD, might need adjustment if checking black-box)
    print("\n--- Sanity Checking Perturbations ---")
    try:
        g_clean_check = torch.load(g_test_clean_path, weights_only=False).to(DEVICE)
        max_eps = max(PGD_EPSILON_LIST)
        attacked_graph_check_path = os.path.join(ATTACK_DATA_DIR, f'test_data_pgd_eps{max_eps}_AnomalE_DGI.pt')
        g_attacked_check = torch.load(attacked_graph_check_path, weights_only=False).to(DEVICE)

        # --- Compare Input Edge Features ---
        print("\n--- Comparing INPUT Edge Features ---")
        clean_h_edge = g_clean_check.edata['h']
        perturbed_h_edge = g_attacked_check.edata.get('h_perturbed')

        if perturbed_h_edge is None:
            print("Error: Perturbed edge features 'h_perturbed' not found in attacked graph.")
        elif clean_h_edge.shape != perturbed_h_edge.shape:
             print(f"Error: Shape mismatch between clean ({clean_h_edge.shape}) and perturbed ({perturbed_h_edge.shape}) edge features.")
        else:
            diff = (clean_h_edge - perturbed_h_edge).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            print(f"Max difference between clean edge 'h' and perturbed edge 'h' (eps={max_eps}): {max_diff:.6f}")
            print(f"Mean absolute difference for edge 'h': {mean_diff:.6f}")

            if max_diff < 1e-5:
                print("WARNING: Perturbed edge features seem identical to clean edge features!")
            elif torch.isnan(diff).any():
                print("WARNING: NaNs detected in the difference between clean and perturbed edge features.")
            else:
                print("Perturbed edge features ARE different from clean edge features.")

        # --- Compare OUTPUT Embeddings ---
        print("\n--- Comparing OUTPUT Edge Embeddings --- ")
        if perturbed_h_edge is not None and clean_h_edge.shape == perturbed_h_edge.shape:
            with torch.no_grad():
                # Ensure models are on device and in eval mode
                dgi_model.to(DEVICE)
                dgi_model.eval()

                # Clean embeddings
                clean_nodes = g_clean_check.ndata['h'].to(DEVICE)
                clean_edges_input = g_clean_check.edata['h'].to(DEVICE)
                _, clean_embeddings = dgi_model.encoder(g_clean_check, clean_nodes, clean_edges_input)

                # Perturbed embeddings
                pert_nodes = g_attacked_check.ndata['h'].to(DEVICE)
                pert_edges_input = g_attacked_check.edata['h_perturbed'].to(DEVICE)
                _, perturbed_embeddings = dgi_model.encoder(g_attacked_check, pert_nodes, pert_edges_input)

                # Compare
                if clean_embeddings.shape != perturbed_embeddings.shape:
                     print(f"Error: Shape mismatch between clean ({clean_embeddings.shape}) and perturbed ({perturbed_embeddings.shape}) OUTPUT embeddings.")
                else:
                    emb_diff = (clean_embeddings - perturbed_embeddings).abs()
                    max_emb_diff = emb_diff.max().item()
                    mean_emb_diff = emb_diff.mean().item()
                    print(f"Max difference between clean and perturbed OUTPUT embeddings (eps={max_eps}): {max_emb_diff:.6f}")
                    print(f"Mean absolute difference for OUTPUT embeddings: {mean_emb_diff:.6f}")

                    if max_emb_diff < 1e-5:
                        print("WARNING: Perturbed OUTPUT embeddings seem identical to clean OUTPUT embeddings! Encoder is robust.")
                    elif torch.isnan(emb_diff).any():
                        print("WARNING: NaNs detected in the difference between clean and perturbed OUTPUT embeddings.")
                    else:
                        print("Perturbed OUTPUT embeddings ARE different from clean OUTPUT embeddings.")
        else:
            print("Skipping output embedding comparison due to issues with input features.")


        del g_clean_check, g_attacked_check
        gc.collect()
    except FileNotFoundError:
        print("Error: Could not load graph files for sanity check.")
    except KeyError as e:
        print(f"Error: Missing key during sanity check ({e}).")
    except Exception as e:
        print(f"An error occurred during sanity check: {e}")


    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
    print("Anomal-E Adversarial Evaluation Script Finished.") 