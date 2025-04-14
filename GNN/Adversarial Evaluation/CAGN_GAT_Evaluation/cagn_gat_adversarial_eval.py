import torch
import numpy as np
import pandas as pd
import os
import pickle
import argparse # Added
from deeprobust.graph.data import Dataset # Keep for potential future use
from deeprobust.graph.global_attack import PGDAttack # Keep for PGD integration
# from deeprobust.graph.targeted_attack import PGD # Original incorrect path
# from deeprobust.graph.global_attack import MetaAttack, DICE, Random # Keep for potential future use
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch_geometric.transforms as T
from torch_geometric.data import Data
from sklearn.neighbors import kneighbors_graph # Moved import for build_graph
from sklearn.preprocessing import StandardScaler # Moved import for build_graph
import torch.nn.functional as F # Added for model definitions
from torch_geometric.nn import GATConv # Added for CAGN definition
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d, Dropout # Added for CAGN
import math # Added for AnomalE Discriminator
import torch_geometric.utils as pyg_utils # Added for sparse_to_dense_adj
import gc # Added for garbage collection
import logging # Added for logging in sampling function
import torch.nn as nn # Added for manual PGD loss


# Assuming model definitions are in a separate file or accessible here
# from ..Standardized_Models.CAGN_GAT.models import CAGN_GAT # Example import
# from ..Standardized_Models.EGraphSage.models import EGraphSage # Example import

# --- Configuration ---

# Get the absolute path of the directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = "/media/ssd/test/standardized-datasets/combined/combined_unsw_cicRed_botRed_netflow_10pct.csv"
MODEL_SAVE_DIR = "/media/ssd/test/GNN/Standardized Models/" # Adjust as needed
# Define paths relative to the script location using absolute paths
ATTACK_SAVE_DIR = os.path.join(script_dir, "Attacked_Data") # Changed to absolute path
FEATURE_LIST_PATH = "/media/ssd/test/GNN/Standardized Models/CAGN-GAT/cagn_gat_feature_list.pkl" 
RESULTS_SAVE_DIR = os.path.join(script_dir, "Results") # Changed to absolute path

TARGET_MODELS = {
    "CAGN_GAT": "CAGN-GAT/best_cagn_model_Combined_10pct.pt", # Path relative to MODEL_SAVE_DIR, ensures compatibility with k-NN graph
    # "EGraphSage": "EGraphSage/best_model.pt", # Removed - Incompatible graph structure (trained on flow graph)
    # "Anomal_E": "AnomalE/Combined/best_dgi.pkl", # Removed - Incompatible graph structure
    # Add other models trained on k-NN similarity graph as needed
}

# Define numerical features vulnerable to perturbation (ORIGINAL names before transform)
FEATURES_TO_PERTURB = [
    'IN_BYTES', 'OUT_BYTES', 'IN_PKTS', 'OUT_PKTS', 'FLOW_DURATION_MILLISECONDS'
]

# NODE_FEATURES will be loaded from FEATURE_LIST_PATH
NODE_FEATURES = None # Placeholder, loaded later
LABEL_COLUMN = 'Label' # Changed to match CAGN notebook's processing
GRAPH_CONSTRUCTION_PARAMS = { # Parameters used to build the graph originally
    'k_neighbors': 20, # Changed to match CAGN notebook cell 3
    'threshold': 0.5, # Changed to match CAGN notebook cell 3
    'metric': 'euclidean' # Added to match CAGN notebook cell 3
}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Attack Parameters
PGD_EPSILON_LIST = [0.3, 0.5, 0.7] # Increased epsilon values significantly
PGD_ITERATIONS = 100 # Increased iterations
# PGD_ALPHA calculated dynamically based on epsilon

BLACK_BOX_NODE_INJECTION_PCT = [0.01, 0.05, 0.10] # Percentage of nodes to add
BLACK_BOX_EDGE_REMOVAL_PCT = [0.01, 0.05, 0.10, 0.20] # Percentage of edges to remove

# --- Utility Functions ---
def ensure_dir_exists(path):
    """Creates a directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

# --- Data Loading and Preprocessing ---
def load_and_preprocess_data(path, label_col):
    """Loads the dataset."""
    print(f"Loading data from: {path}")
    df = pd.read_csv(path, low_memory=False)

    # Basic check
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataframe.")

    print(f"Loaded full data: {len(df)} total samples.")
    return df # Return the full dataframe

def feature_engineer(df, node_features_expected_list):
    """Performs feature engineering (scaling, encoding) like CAGN Cell 2."""
    print("Starting feature engineering...")
    numerical_cols = FEATURES_TO_PERTURB # Use the defined list
    categorical_cols = ['PROTOCOL', 'L7_PROTO', 'TCP_FLAGS'] # From CAGN notebook
    
    columns_to_process = numerical_cols + categorical_cols # Label not needed for X
    flows_df_processed = df[columns_to_process].copy()

    # --- Process Numerical ---
    print(f"Processing numerical features: {numerical_cols}")
    for col in numerical_cols:
        flows_df_processed[col] = pd.to_numeric(flows_df_processed[col], errors='coerce').fillna(0)
    log_transformed_features = np.log1p(flows_df_processed[numerical_cols].values)
    scaler = StandardScaler()
    scaled_numerical_features = scaler.fit_transform(log_transformed_features)
    scaled_numerical_df = pd.DataFrame(scaled_numerical_features, index=flows_df_processed.index, columns=numerical_cols)

    # --- Process Categorical ---
    print(f"Processing categorical features: {categorical_cols}")
    flows_df_processed[categorical_cols] = flows_df_processed[categorical_cols].astype(str).fillna('missing')
    categorical_encoded_df = pd.get_dummies(
        flows_df_processed[categorical_cols],
        columns=categorical_cols,
        prefix=categorical_cols,
        dummy_na=False,
        dtype=int
    )
    
    # --- Combine Features ---
    X_df = pd.concat([scaled_numerical_df, categorical_encoded_df.set_index(scaled_numerical_df.index)], axis=1)

    # --- Ensure feature consistency ---
    # Reindex X_df to match the expected feature list from training
    # Add missing columns (if any) with 0, and drop extra columns
    missing_cols = list(set(node_features_expected_list) - set(X_df.columns))
    for col in missing_cols:
        X_df[col] = 0
    extra_cols = list(set(X_df.columns) - set(node_features_expected_list))
    X_df = X_df.drop(columns=extra_cols)
    X_df = X_df[node_features_expected_list] # Ensure correct order
    
    print(f"Feature engineering complete. X shape: {X_df.shape}")
    X = X_df.values
    y = df[LABEL_COLUMN].values.astype(np.int64)
    
    return X, y, scaler # Return X, y, and scaler


def build_graph(X, y, k, threshold, metric):
    """Builds a PyG Data object using k-NN similarity graph (like CAGN Cell 3)."""
    print(f"Building k-NN similarity graph (k={k}, threshold={threshold}, metric={metric})...")
    
    # Using adaptive graph construction logic from CAGN notebook
    num_samples = X.shape[0]
    try:
        knn_adj = kneighbors_graph(X, k, mode='connectivity', metric=metric, include_self=False, n_jobs=-1)
        print(f"  Calculated k-NN graph. Shape: {knn_adj.shape}, NNZ: {knn_adj.nnz}.")

        # Simplified thresholding: Using KNN graph directly, no separate distance calc
        # This assumes k-NN already captures sufficient similarity
        # If distance threshold is strictly needed, uncomment and adapt distance calc part
        # print("  Calculating pairwise distances and applying threshold...")
        # distances = pairwise_distances(X.astype(np.float32), metric=metric, n_jobs=-1)
        # dist_adj_mask = (distances < threshold)
        # np.fill_diagonal(dist_adj_mask, False)
        # dist_adj_sparse = sparse.csr_matrix(dist_adj_mask)
        # del distances, dist_adj_mask
        # gc.collect()
        # print("  Intersecting k-NN and distance graphs...")
        # final_adj = knn_adj.multiply(dist_adj_sparse)

        final_adj = knn_adj # Using k-NN graph directly
        print(f"  Using k-NN graph directly. Final Adj NNZ: {final_adj.nnz}")

        final_adj_coo = final_adj.tocoo()
        edge_index = torch.tensor(np.vstack((final_adj_coo.row, final_adj_coo.col)), dtype=torch.long)
        
        features = torch.tensor(X, dtype=torch.float32)
        labels = torch.tensor(y, dtype=torch.long)

        data = Data(x=features, edge_index=edge_index, y=labels)
        print(f"Built graph: {data.num_nodes} nodes, {data.num_edges} edges")
        return data
    except MemoryError as e:
        print(f"MemoryError during graph construction: {e}. Try reducing k or features.")
        raise
    except Exception as e:
        print(f"Error during graph construction: {e}")
        raise


# --- Model Class Definitions (Moved Up) ---

# == CAGN Model Definition (from CAGN-GAT Cell 7) ==
class CAGN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=2, dropout=0.6): # Default heads=2
        super(CAGN, self).__init__()
        self.dropout_rate = dropout
        # Add attributes expected by DeepRobust
        self.nfeat = input_dim # Number of input features
        self.nclass = output_dim # Number of output classes (or 1 for binary logits)
        # Define hidden_sizes based on intermediate layer dimensions
        self.hidden_sizes = [hidden_dim * heads, hidden_dim]
        
        # If output_dim is 1 (binary logits), DeepRobust PGDAttack seems to work with nclass=1 or nclass=2
        # Let's stick with nclass=output_dim for consistency, DeepRobust might handle binary internally.
        # If issues persist, try nclass=2 for binary case.
        # self.nclass = output_dim 
        # Use edge_dim=None since the similarity graph doesn't have edge features
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=self.dropout_rate, edge_dim=None)
        # Input to conv2 is hidden_dim * heads because concat=True by default in GATConv
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=self.dropout_rate, edge_dim=None)
        # Input to conv3 is the output dimension of conv2, which is hidden_dim
        self.conv3 = GATConv(hidden_dim, output_dim, heads=1, concat=False, dropout=self.dropout_rate, edge_dim=None)

        self.contrastive_loss_weight = 0.5  # Keep for potential use, though not needed for inference

    def forward(self, x, edge_index):
        # Simplified forward pass for inference (no contrastive loss needed)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        z = self.conv2(x, edge_index)
        z = F.elu(z)
        z = F.dropout(z, p=self.dropout_rate, training=self.training)
        out_logits = self.conv3(z, edge_index)
        # Return only logits for evaluation/gradient purposes
        # If embeddings 'z' are needed for some attack, modify return
        return out_logits #, z

# --- End Model Class Definitions (Moved Up) ---


# --- Model Loading ---
def load_model(model_name, model_relative_path, model_base_dir, input_dim=None, output_dim=1):
    """Loads a pre-trained PyTorch model (state_dict)."""
    # Removed unused edim parameter

    model_path = os.path.join(model_base_dir, model_relative_path)
    if not os.path.exists(model_path):
        print(f"Warning: Model file not found at {model_path}")
        return None
    
    # Define fixed hyperparameters based on notebooks (adjust if necessary)
    cagn_hidden_dim = 64
    cagn_heads = 8
    cagn_dropout = 0.6
    anomal_e_hidden_dim = 128 # This is ndim_out for DGI
    anomal_e_activation = F.relu

    model = None # Initialize model to None

    try:
        # Adjust loading based on how models were saved (.pt vs .pkl)
        if model_path.endswith('.pt'):
            # Assume saved state_dict for PyTorch models like CAGN
            if model_name == "CAGN_GAT":
                if input_dim is None:
                    print(f"Error: input_dim required to instantiate CAGN_GAT model.")
                    return None
                print(f"Instantiating CAGN model (Input: {input_dim}, Hidden: {cagn_hidden_dim}, Output: {output_dim}, Heads: {cagn_heads})")
                model = CAGN(input_dim=input_dim, hidden_dim=cagn_hidden_dim, output_dim=output_dim, heads=cagn_heads, dropout=cagn_dropout)
                model_state_dict = torch.load(model_path, map_location=DEVICE)
                model.load_state_dict(model_state_dict)
                print(f"Loaded state_dict for {model_name} from {model_path}")
            elif model_name == "EGraphSage": # Added case for E-GraphSage
                 print(f"Loading entire E-GraphSage model object from {model_path}")
                 # E-GraphSage notebook saves the entire model object
                 model = torch.load(model_path, map_location=DEVICE)
                 print(f"Loaded entire E-GraphSage model object.")
                 # Ensure the loaded model has the necessary forward method
                 if not hasattr(model, 'forward'):
                     print(f"Error: Loaded E-GraphSage model from {model_path} does not have a 'forward' method.")
                     return None
            else:
                print(f"Warning: Unhandled .pt model: {model_name}. Attempting to load as entire model object.")
                try:
                     model = torch.load(model_path, map_location=DEVICE)
                     print(f"Loaded entire model object for {model_name} from {model_path}")
                     if not hasattr(model, 'forward'):
                         print(f"Error: Loaded model object from {model_path} does not have a 'forward' method.")
                         model = None
                except Exception as load_err:
                     print(f"Failed to load {model_path} as entire model object: {load_err}. Cannot load without architecture.")
                     model = None
        elif model_path.endswith('.pkl'):
             # Removed Anomal-E .pkl handling
              print(f"Warning: Unknown model file extension for {model_path}")
              return None
              
        if model: # If model was successfully instantiated and loaded
            model.to(DEVICE)
            model.eval() # Set to evaluation mode
            print(f"Loaded and prepared model '{model_name}' on {DEVICE}")
        return model # Return the loaded model or None if failed
    except Exception as e:
        print(f"Error loading model {model_name} from {model_path}: {e}")
        return None

# --- Evaluation Function ---
def evaluate_model(model, data, description="Evaluation"):
    """Evaluates the model on the given data."""
    if model is None or data is None:
        print(f"Skipping evaluation for {description} (model or data is None)")
        return {}
        
    print(f"Evaluating model on '{description}' data.")
    model.eval() # Ensure model is in evaluation mode
    data = data.to(DEVICE) # Move data to the correct device

    try:
        with torch.no_grad(): # Disable gradient calculation for evaluation
            # Perform forward pass - Assuming model takes x and edge_index
            out = model(data.x, data.edge_index)

            # --- Interpret output --- 
            # Check if model returns tuple (like CAGN might have with embeddings)
            if isinstance(out, tuple):
                out_logits = out[0] # Assume logits are the first element
            else:
                out_logits = out # Assume output is just logits

            # --- Get predictions (Binary classification) ---
            # Check output shape - should be [num_nodes, 1] for binary logits
            if out_logits.ndim == 2 and out_logits.shape[1] == 1:
                out_logits = out_logits.squeeze(1) # Remove the last dimension
            elif out_logits.ndim != 1:
                 print(f"Warning: Unexpected model output shape: {out_logits.shape}. Assuming binary logits.")
                 # Attempt to handle potential issues, may need adjustment based on actual output
                 if out_logits.shape[1] > 1:
                     out_logits = out_logits[:, 0] # Take first column if multiple outputs
                 out_logits = out_logits.squeeze() # Squeeze remaining dimensions

            preds_proba = torch.sigmoid(out_logits) # Probabilities for binary
            preds = (preds_proba > 0.5).long()      # Threshold at 0.5

            # --- Calculate metrics ---
            y_true = data.y.cpu().numpy()
            y_pred = preds.cpu().numpy()

            accuracy = accuracy_score(y_true, y_pred)
            # Use average='binary' for binary classification
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)

            # Optional: Calculate AUC
            auc = 0.0
            try:
                 if len(np.unique(y_true)) > 1: # Check if both classes are present
                     from sklearn.metrics import roc_auc_score # Import locally if not at top
                     auc = roc_auc_score(y_true, preds_proba.cpu().numpy()) # Use probabilities for AUC
                 else:
                     print("Skipping AUC calculation: Only one class present in true labels.")
            except ImportError:
                 print("Skipping AUC calculation: roc_auc_score not available.")
            except Exception as auc_err:
                 print(f"Could not calculate AUC: {auc_err}")

            print(f"{description} Results - Acc: {accuracy:.4f}, P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
            metrics = {
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "AUC": auc
            }
            return metrics

    except Exception as e:
        print(f"Error during evaluation for {description}: {e}")
        return {}

# --- Attack Implementations ---

# == White-Box Attacks (Feature Perturbation) ==

# REMOVED: run_pgd_attack using DeepRobust
# def run_pgd_attack(...):
#    ...

def run_manual_pgd_attack(model, data, epsilon, iterations, features_to_perturb_indices):
    """Runs a manual PGD feature attack using PyTorch AutoGrad."""
    if model is None or data is None:
        print("Skipping Manual PGD attack (model or data is None)")
        return None
    if not features_to_perturb_indices:
        print("Skipping Manual PGD attack: No features_to_perturb_indices specified.")
        return None

    print(f"  Running Manual PGD attack (eps={epsilon}, iterations={iterations})...")
    device = data.x.device
    n_nodes = data.num_nodes
    n_features = data.num_features

    # Ensure model is on the right device and in eval mode
    model.to(device)
    model.eval() # Keep model in eval mode for consistent behavior, but track gradients

    # --- Prepare features and labels ---
    original_features = data.x.clone().detach()
    perturbed_features = data.x.clone().detach()
    labels = data.y.to(device)

    # --- Setup Loss (Binary classification assumed) ---
    # Calculate pos_weight for the subgraph
    num_positives = (labels == 1).sum().item()
    num_negatives = (labels == 0).sum().item()
    pos_weight = None
    if num_positives > 0 and num_negatives > 0:
        pos_weight_value = num_negatives / num_positives
        pos_weight = torch.tensor([pos_weight_value], device=device)
        print(f"    Manual PGD using pos_weight: {pos_weight_value:.4f}")
    else:
        print("    Manual PGD warning: Cannot calculate pos_weight for subgraph.")
    criterion_cls = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # --- PGD Parameters ---
    # Simple step size rule, can be adapted
    alpha = epsilon / iterations * 1.25 # Slightly larger step based on common practice

    # --- Create mask for perturbable features ---
    perturb_mask = torch.zeros_like(perturbed_features)
    if features_to_perturb_indices:
        perturb_mask[:, features_to_perturb_indices] = 1.0
    else:
        print("    Manual PGD warning: No feature indices to perturb provided.")
        return data.clone() # Return original data if no features specified

    # --- PGD Attack Loop ---
    for i in range(iterations):
        perturbed_features.requires_grad = True # Enable gradient tracking for this step
        model.zero_grad() # Zero gradients before forward pass

        # Forward pass - handle potential tuple output from CAGN
        output = model(perturbed_features, data.edge_index)
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output
        
        # Squeeze logits if needed for binary classification loss
        if logits.ndim > 1 and logits.shape[1] == 1:
            logits = logits.squeeze(1)

        # Calculate loss
        loss = criterion_cls(logits, labels.float())

        # Backward pass to get gradients w.r.t features
        loss.backward()

        with torch.no_grad(): # Operations after gradient calculation
            # Get gradient and apply mask
            grad = perturbed_features.grad.data
            masked_grad = grad * perturb_mask # Apply mask to gradient

            # PGD step: update features using the sign of the masked gradient
            adv_features = perturbed_features + alpha * masked_grad.sign()

            # Projection step: Clip perturbations to epsilon ball around original features
            # Apply projection only to the features that were allowed to be perturbed
            eta = torch.clamp(adv_features - original_features, min=-epsilon, max=epsilon)
            perturbed_features = original_features + eta * perturb_mask # Apply mask to the perturbation delta

            # Detach for the next iteration
            perturbed_features = perturbed_features.detach()
        
        # Optional: Print progress
        # if (i + 1) % 10 == 0:
        #     print(f"    PGD Iteration [{i+1}/{iterations}], Loss: {loss.item():.4f}")

    # --- Create final perturbed data object ---
    perturbed_data = data.clone()
    perturbed_data.x = perturbed_features

    print(f"Manual PGD attack generation finished for eps={epsilon}.")
    return perturbed_data


# == Black-Box Attacks (Structural Perturbation) ==
# (Functions run_node_injection_attack and run_edge_removal_attack remain largely the same as before)
def run_node_injection_attack(data, injection_rate, feature_strategy='mean'):
    """Simulates random node injection."""
    print(f"Running Node Injection: rate={injection_rate}, strategy={feature_strategy}")
    num_nodes_to_add = int(data.num_nodes * injection_rate)
    if num_nodes_to_add == 0:
        print("Skipping node injection (0 nodes to add).")
        return data.clone()

    original_num_nodes = data.num_nodes
    original_num_edges = data.num_edges
    num_features = data.num_features
    device = data.x.device # Get device from data

    perturbed_data = data.clone()

    # 1. Add new node features
    new_features = None
    if feature_strategy == 'zeros':
        new_features = torch.zeros((num_nodes_to_add, num_features), dtype=data.x.dtype, device=device)
    elif feature_strategy == 'mean':
        new_features = data.x.mean(dim=0, keepdim=True).repeat(num_nodes_to_add, 1)
    # Add other strategies: sample from benign/malicious, etc.
    else:
        raise ValueError(f"Unknown feature strategy: {feature_strategy}")

    perturbed_data.x = torch.cat([perturbed_data.x, new_features], dim=0)

    # 2. Add new edges (randomly connect new nodes to existing ones)
    k_connections = 5
    new_edge_sources = torch.arange(original_num_nodes, original_num_nodes + num_nodes_to_add, device=device).repeat_interleave(k_connections)
    new_edge_targets = torch.randint(0, original_num_nodes, (num_nodes_to_add * k_connections,), device=device)

    new_edges = torch.stack([
        torch.cat([new_edge_sources, new_edge_targets]),
        torch.cat([new_edge_targets, new_edge_sources]) # Add edges in both directions
    ], dim=0)

    perturbed_data.edge_index = torch.cat([perturbed_data.edge_index, new_edges], dim=1)

    # 3. Add dummy labels for new nodes (assign benign label 0)
    new_labels = torch.zeros(num_nodes_to_add, dtype=data.y.dtype, device=device)
    perturbed_data.y = torch.cat([perturbed_data.y, new_labels], dim=0)

    perturbed_data.num_nodes = original_num_nodes + num_nodes_to_add

    print(f"Node injection finished: Added {num_nodes_to_add} nodes, {perturbed_data.num_edges - original_num_edges} edges.")
    return perturbed_data

def run_edge_removal_attack(data, removal_rate):
    """Simulates random edge removal."""
    print(f"Running Edge Removal: rate={removal_rate}")
    num_edges_to_remove = int(data.num_edges * removal_rate)
    if num_edges_to_remove == 0:
        print("Skipping edge removal (0 edges to remove).")
        return data.clone()

    perturbed_data = data.clone()
    perm = torch.randperm(data.num_edges)
    edges_to_keep_indices = perm[num_edges_to_remove:]
    perturbed_data.edge_index = perturbed_data.edge_index[:, edges_to_keep_indices]

    print(f"Edge removal finished: Removed {num_edges_to_remove} edges. Remaining: {perturbed_data.num_edges}")
    return perturbed_data

# --- Generation Function ---
def generate_and_save_attacks(sampled_graph_data, node_features_list, input_dim, output_dim): # Removed scaler
    """Generates attacked datasets on the test subgraph and saves them."""
    print("--- Starting Attack Generation Phase ---")
    ensure_dir_exists(ATTACK_SAVE_DIR)
    feature_indices_to_perturb = []
    if node_features_list:
        try:
            feature_indices_to_perturb = [node_features_list.index(f) for f in FEATURES_TO_PERTURB if f in node_features_list]
            print(f"Identified indices for perturbation: {feature_indices_to_perturb}")
        except ValueError as e:
            print(f"Error finding feature indices: {e}")
    else:
        print("Warning: NODE_FEATURES list not loaded, cannot determine perturbation indices.")

    # Extract the test subgraph
    if not hasattr(sampled_graph_data, 'test_mask') or sampled_graph_data.test_mask is None:
        print("Error: Sampled graph data does not have a 'test_mask'. Cannot extract test subgraph.")
        return
        
    print(f"Extracting test subgraph using test_mask (Num test nodes: {sampled_graph_data.test_mask.sum()})")
    test_nodes = sampled_graph_data.test_mask.nonzero(as_tuple=False).view(-1)
    test_data_subgraph = sampled_graph_data.subgraph(test_nodes)
    print(f"Test subgraph created: {test_data_subgraph.num_nodes} nodes, {test_data_subgraph.num_edges} edges")

    # Save clean test subgraph data
    clean_save_path = os.path.join(ATTACK_SAVE_DIR, "test_data_clean.pt")
    torch.save(test_data_subgraph, clean_save_path)
    print(f"Saved clean test subgraph data to {clean_save_path}")

    # --- Generate White-Box PGD Attacks (Per Model) using Manual PGD ---
    for model_name, model_file in TARGET_MODELS.items():
        print(f"\nGenerating Manual PGD attacks for model: {model_name} on the test subgraph")
        # Load the actual model, passing dimensions
        model = load_model(model_name, model_file, MODEL_SAVE_DIR, input_dim=input_dim, output_dim=output_dim)
        if model is None:
             print(f"Skipping Manual PGD for {model_name} as model loading failed.")
             continue
             
        for eps in PGD_EPSILON_LIST:
            # Call the manual PGD attack function
            pgd_data = run_manual_pgd_attack(
                model,
                test_data_subgraph, 
                eps, 
                PGD_ITERATIONS, # Use the globally defined iterations
                feature_indices_to_perturb
            )
            if pgd_data:
                save_path = os.path.join(ATTACK_SAVE_DIR, f"test_data_pgd_eps{eps}_{model_name}.pt")
                torch.save(pgd_data, save_path)
                print(f"Saved Manual PGD (eps={eps}) attacked test subgraph data for {model_name} to {save_path}")
            else:
                 print(f"Manual PGD attack failed for eps={eps}, model={model_name}")
        del model # Clean up model
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # --- Generate Black-Box Attacks (Once) on the Test Subgraph ---
    print("\nGenerating Black-Box attacks on the test subgraph...")
    # Node Injection
    for rate in BLACK_BOX_NODE_INJECTION_PCT:
        attack_name = f"node_inject_{int(rate*100)}pct"
        injected_data = run_node_injection_attack(test_data_subgraph, rate, feature_strategy='mean')
        save_path = os.path.join(ATTACK_SAVE_DIR, f"test_data_{attack_name}.pt")
        torch.save(injected_data, save_path)
        print(f"Saved {attack_name} data to {save_path}")

    # Edge Removal
    for rate in BLACK_BOX_EDGE_REMOVAL_PCT:
        attack_name = f"edge_remove_{int(rate*100)}pct"
        removed_data = run_edge_removal_attack(test_data_subgraph, rate)
        save_path = os.path.join(ATTACK_SAVE_DIR, f"test_data_{attack_name}.pt")
        torch.save(removed_data, save_path)
        print(f"Saved {attack_name} data to {save_path}")

    print("--- Attack Generation Phase Complete ---")

# --- Evaluation Function ---
def run_evaluation(input_dim, output_dim):
    """Loads attacked datasets (test subgraphs) and evaluates models."""
    print("--- Starting Evaluation Phase ---")
    ensure_dir_exists(RESULTS_SAVE_DIR)
    results = {}
    
    # --- Load Attacked Data ---
    print("Loading attacked datasets...")
    attacked_datasets = {}
    try:
        for filename in os.listdir(ATTACK_SAVE_DIR):
            if filename.startswith("test_data_") and filename.endswith(".pt"):
                attack_name = filename.replace("test_data_", "").replace(".pt", "")
                load_path = os.path.join(ATTACK_SAVE_DIR, filename)
                attacked_datasets[attack_name] = torch.load(load_path, map_location=DEVICE)
                print(f"  Loaded: {attack_name}")
    except FileNotFoundError:
        print(f"Error: Attack data directory not found: {ATTACK_SAVE_DIR}")
        print("Please run the script in 'generate' mode first.")
        return
    except Exception as e:
        print(f"Error loading attacked data: {e}")
        return
        
    if not attacked_datasets:
         print("No attacked datasets found to evaluate.")
         return

    # --- Evaluate Models ---
    for model_name, model_file in TARGET_MODELS.items():
        print(f"\n--- Evaluating Model: {model_name} ---")
        results[model_name] = {}
        
        # Load the actual model, passing dimensions
        model = load_model(model_name, model_file, MODEL_SAVE_DIR, input_dim=input_dim, output_dim=output_dim)
        if model is None:
             print(f"Skipping evaluation for {model_name} as model loading failed.")
             continue

        # Evaluate on clean data
        clean_data = attacked_datasets.get("clean")
        if clean_data:
             baseline_metrics = evaluate_model(model, clean_data, description=f"{model_name} Clean")
             results[model_name]["clean"] = baseline_metrics
        else:
            print("Clean test data not found, skipping baseline evaluation.")
            
        # Evaluate on attacked data
        for attack_name, data in attacked_datasets.items():
             if attack_name == "clean": continue # Already evaluated
             
             # Only evaluate PGD attacks on the model they were generated for
             if "pgd" in attack_name and not attack_name.endswith(model_name):
                 continue
                 
             print(f"  Evaluating on: {attack_name}")
             attack_metrics = evaluate_model(model, data, description=f"{model_name} {attack_name}")
             results[model_name][attack_name] = attack_metrics
        
        del model # Clean up model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        
    # --- Save Results ---
    print("\n--- Final Results Summary ---")
    results_df = pd.DataFrame.from_dict({(m, atk): res for m, attacks in results.items()
                                      for atk, res in attacks.items() if res}, orient='index') # Filter out empty results
    if not results_df.empty:
        print(results_df)
        results_file = os.path.join(RESULTS_SAVE_DIR, "adversarial_results.csv")
        results_df.to_csv(results_file)
        print(f"\nSaved evaluation results to {results_file}")
    else:
        print("No evaluation results generated.")

    print("--- Evaluation Phase Complete ---")

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Adversarial Attack Generation or Evaluation.")
    parser.add_argument('mode', choices=['generate', 'evaluate'], help="Choose 'generate' to create attacked data or 'evaluate' to run evaluations on existing data.")
    args = parser.parse_args()

    print(f"Starting Adversarial Script in '{args.mode}' mode")
    print(f"Using device: {DEVICE}")

    # Load the expected node features list
    try:
        with open(FEATURE_LIST_PATH, 'rb') as f:
            NODE_FEATURES = pickle.load(f)
        print(f"Successfully loaded {len(NODE_FEATURES)} feature names from {FEATURE_LIST_PATH}")
    except FileNotFoundError:
        print(f"Error: Feature list file not found at {FEATURE_LIST_PATH}")
        print("Please ensure the CAGN-GAT notebook was run to generate this file.")
        NODE_FEATURES = None # Set to None to indicate failure
        if args.mode == 'generate': 
            print("Exiting as feature list is required for generation.")
            exit()
    except Exception as e:
        print(f"Error loading feature list: {e}")
        NODE_FEATURES = None
        if args.mode == 'generate': 
             print("Exiting as feature list is required for generation.")
             exit()
             
    # Calculate dimensions if features loaded
    input_dim = len(NODE_FEATURES) if NODE_FEATURES else None
    output_dim = 1 # Binary classification
             
    if args.mode == 'generate':
        if not NODE_FEATURES:
            print("Cannot proceed with generation without the NODE_FEATURES list.")
        else:
            # --- New Data Pipeline ---
            # 1. Load full data
            full_df = load_and_preprocess_data(DATASET_PATH, LABEL_COLUMN)

            # 2. Sample data (Using actual function from notebook)
            # Add sampling constants near the top (already defined above)
            SAMPLED_SIZE_LARGE_CLASSES = 50000 # Example value (Ensure this matches notebook or desired value)
            MIN_LARGE_CLASS_SIZE = 1000     # Example value (Ensure this matches notebook or desired value)

            # Actual sampling function from CAGN-GAT notebook
            def create_imbalanced_subset(df, target_col, new_dataset_size_large_classes, min_large_class_size):
                """
                Create a smaller dataset while preserving class imbalance, focusing on reducing majority classes.

                Args:
                    df (pd.DataFrame): Original dataset.
                    target_col (str): Name of the target label column.
                    new_dataset_size_large_classes (int): Target total number of samples from classes exceeding min_large_class_size.
                    min_large_class_size (int): Minimum number of samples for a class to be considered 'large'.

                Returns:
                    pd.DataFrame: A reduced dataset.
                """
                logging.info(f"Starting imbalanced sampling. Target size for large classes: {new_dataset_size_large_classes}")
                class_counts = df[target_col].value_counts()
                logging.info(f"Original class distribution:\\n{class_counts}")

                large_classes = class_counts[class_counts >= min_large_class_size]
                small_classes = class_counts[class_counts < min_large_class_size]

                total_large_samples_original = large_classes.sum()
                num_large_classes = len(large_classes)

                sampled_data = []

                if num_large_classes > 0 and total_large_samples_original > 0:
                    logging.info(f"Found {num_large_classes} large classes (>= {min_large_class_size} samples).")
                    # Calculate scaling factor based on the sum of large classes
                    scaling_factor = min(1.0, new_dataset_size_large_classes / total_large_samples_original)
                    logging.info(f"Scaling factor for large classes: {scaling_factor:.4f}")

                    # Sample from large classes proportionally
                    for class_label, original_count in large_classes.items():
                        # Calculate proportional target size
                        target_size = int(original_count * scaling_factor)
                        # Ensure we don't sample more than available and respect min_large_class_size if scaled size is too small
                        sample_size = max(1, min(target_size, original_count)) # Ensure at least 1 sample, don't exceed original count
                        logging.info(f"  Sampling class '{class_label}': target size={target_size}, final sample size={sample_size}")
                        sampled_class_df = df[df[target_col] == class_label].sample(n=sample_size, random_state=42, replace=False)
                        sampled_data.append(sampled_class_df)
                else:
                    logging.info("No large classes found or large classes sum to zero.")

                # Keep all samples from small classes
                if not small_classes.empty:
                    logging.info(f"Keeping all samples for {len(small_classes)} small classes (< {min_large_class_size} samples).")
                    small_class_df = df[df[target_col].isin(small_classes.index)]
                    sampled_data.append(small_class_df)

                if not sampled_data:
                    logging.warning("Sampling resulted in an empty dataset.")
                    return pd.DataFrame(columns=df.columns)

                # Concatenate sampled dataframes
                df_sampled = pd.concat(sampled_data).sample(frac=1, random_state=42).reset_index(drop=True) # Shuffle
                logging.info(f"Finished sampling. New dataset size: {len(df_sampled)}")
                logging.info(f"New class distribution:\\n{df_sampled[target_col].value_counts()}")

                return df_sampled

            df_sampled = create_imbalanced_subset(full_df, LABEL_COLUMN, SAMPLED_SIZE_LARGE_CLASSES, MIN_LARGE_CLASS_SIZE)
            del full_df # Free memory
            gc.collect() # Explicit garbage collection

            # --- Check if sampling was successful ---
            if df_sampled is None or df_sampled.empty:
                print("Sampling failed or resulted in an empty DataFrame. Exiting.")
                exit()
            else:
                 print(f"Sampling successful. Sampled DataFrame shape: {df_sampled.shape}")

            # 3. Feature engineer sampled data
            X_sampled, y_sampled, scaler = feature_engineer(df_sampled, NODE_FEATURES)
            
            # 4. Build graph on sampled data
            sampled_graph_data = build_graph(X_sampled, y_sampled, 
                                             k=GRAPH_CONSTRUCTION_PARAMS['k_neighbors'],
                                             threshold=GRAPH_CONSTRUCTION_PARAMS['threshold'],
                                             metric=GRAPH_CONSTRUCTION_PARAMS['metric'])

            # 5. Add train/test masks to the sampled graph data
            # Assuming a similar split ratio as before (80/20) for the *sampled* data
            num_nodes = sampled_graph_data.num_nodes
            indices = torch.randperm(num_nodes)
            train_size = int(0.8 * num_nodes)
            train_indices = indices[:train_size]
            test_indices = indices[train_size:]
            
            sampled_graph_data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            sampled_graph_data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            sampled_graph_data.train_mask[train_indices] = True
            sampled_graph_data.test_mask[test_indices] = True
            print(f"Added train_mask ({sampled_graph_data.train_mask.sum()} nodes) and test_mask ({sampled_graph_data.test_mask.sum()} nodes) to sampled graph.")

            # 6. Generate and save attacks using the sampled graph (with masks)
            generate_and_save_attacks(sampled_graph_data, NODE_FEATURES, input_dim, output_dim)
            
    elif args.mode == 'evaluate':
        if not NODE_FEATURES:
            print("Cannot proceed with evaluation without the NODE_FEATURES list.")
        else:
        # Run evaluation (loads attacked test subgraphs from ATTACK_SAVE_DIR)
            run_evaluation(input_dim, output_dim)

    print("Adversarial script finished.") 

# --- End Model Class Definitions --- 