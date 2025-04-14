import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import networkx as nx
import pandas as pd
import numpy as np
import socket
import struct
import random
import pickle
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import category_encoders as ce
# Potentially need DeepRobust or similar for PGD
# from deeprobust.graph.targeted_attack import PGD # Or other relevant attacker

# --- Constants ---
DATASET_PATH = '/media/ssd/test/standardized-datasets/combined/combined_unsw_cicRed_botRed_netflow_10pct.csv'
# Assuming the model is saved in the E-GraphSage training directory relative to the GNN folder
MODEL_DIR = '/media/ssd/test/GNN/Standardized Models/E-GraphSage/'
MODEL_FILENAME = 'best_model.pt' # From the notebook
EVAL_DIR = '/media/ssd/test/GNN/Adversarial Evaluation/EGraphSage_Evaluation/'
ATTACK_DATA_DIR = os.path.join(EVAL_DIR, 'Attacked_Data')
RESULTS_DIR = os.path.join(EVAL_DIR, 'Results')
RESULTS_FILE = os.path.join(RESULTS_DIR, 'adversarial_results.csv')

# Columns to keep and process based on the notebook
COLUMNS_TO_KEEP = ['IPV4_SRC_ADDR', 'L4_SRC_PORT', 'IPV4_DST_ADDR', 'L4_DST_PORT', 'PROTOCOL', 'L7_PROTO',
                   'IN_BYTES', 'OUT_BYTES', 'IN_PKTS', 'OUT_PKTS', 'TCP_FLAGS', 'FLOW_DURATION_MILLISECONDS',
                   'Label', 'Attack'] # Keep original Label/Attack for potential multi-class eval later if needed

# Numerical features targeted for PGD (original scale)
NUMERICAL_FEATURES_PGD = ['IN_BYTES', 'OUT_BYTES', 'IN_PKTS', 'OUT_PKTS', 'FLOW_DURATION_MILLISECONDS']
CATEGORICAL_FEATURES = ['TCP_FLAGS', 'L7_PROTO', 'PROTOCOL']

# All features that get scaled (numerical + target encoded categorical)
# Will be determined after target encoding in the preprocessing step
COLS_TO_NORM = []


# PGD Attack Parameters (from plan)
PGD_EPSILON_LIST = [0.3, 0.5, 0.7]
EDGE_REMOVAL_RATES = [0.05, 0.10, 0.20, 0.30] # Add edge removal rates
NODE_INJECTION_RATES = [0.05, 0.10, 0.20] # Add node injection rates
NODE_INJECTION_CONNECTIONS_PER_NODE = 5 # Number of connections each new node makes
PGD_ITERATIONS = 60
# Alpha (step size) often set relative to epsilon, e.g., eps/10 or 2.5 * eps / steps
PGD_ALPHA = 2.5 # Placeholder, will calculate relative alpha later

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Ensure output directories exist
os.makedirs(ATTACK_DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Model Definition (Copied from E-GraphSage Notebook) ---
class SAGELayer(nn.Module):
    def __init__(self, ndim_in, edims, ndim_out, activation):
        super(SAGELayer, self).__init__()
        self.W_msg = nn.Linear(ndim_in + edims, ndim_out)
        self.W_apply = nn.Linear(ndim_in + ndim_out, ndim_out)
        self.activation = activation

    def message_func(self, edges):
        # Note the dimension adjustment for message function if input has extra dim
        # Original notebook reshaped features, assuming features are [num_edges, feature_dim] here
        # If features are [num_edges, 1, feature_dim], adjust concat dim
        # Assuming edge.src['h'] is [batch_size, ndim_in], edge.data['h'] is [batch_size, edims]
        # Check dimensions during runtime if errors occur
        # For PGD, ensure gradients flow back correctly through this concat
        # The original notebook reshapes node/edge features to [N, 1, dim]
        # We'll handle this shape before passing to the model if needed
        msg_input = torch.cat([edges.src['h'], edges.data['h']], dim=-1)
        return {'m': self.W_msg(msg_input)}

    def forward(self, g_dgl, nfeats, efeats):
        with g_dgl.local_scope():
            g = g_dgl
            # Handle potential extra dimension if needed based on input shape
            g.ndata['h'] = nfeats.squeeze(1) if nfeats.ndim == 3 else nfeats
            g.edata['h'] = efeats.squeeze(1) if efeats.ndim == 3 else efeats

            g.update_all(self.message_func, fn.mean('m', 'h_neigh'))

            h_neigh = g.ndata['h_neigh']
            h_self = g.ndata['h']

            # Ensure dimensions match for concatenation
            # If h_self is [N, ndim_in] and h_neigh is [N, ndim_out], this won't work directly
            # The original notebook applies W_apply to [h_self, h_neigh] concatenated
            # This implies h_neigh should be aggregated result of messages (ndim_out)
            # And h_self is the input node feature (ndim_in)
            apply_input = torch.cat([h_self, h_neigh], dim=-1)
            g.ndata['h_out'] = F.relu(self.W_apply(apply_input)) # Use 'h_out' to avoid overwriting 'h' needed later?
            # Check if the notebook's SAGE model returns h or h_out
            # Notebook seems to overwrite g.ndata['h']
            g.ndata['h'] = g.ndata['h_out']
            # Add the dimension back if the model expects it
            return g.ndata['h'].unsqueeze(1) if nfeats.ndim == 3 else g.ndata['h']


class SAGE(nn.Module):
    def __init__(self, ndim_in, ndim_out, edim, activation, dropout):
        super(SAGE, self).__init__()
        self.layers = nn.ModuleList()
        # Assuming node features are initialized (e.g., ones) and the first layer transforms them
        # The notebook initializes node features to ones of size edim, this seems unusual.
        # Let's assume ndim_in is the initial node feature dim, edim is edge feature dim
        # Layer 1: ndim_in -> 128
        self.layers.append(SAGELayer(ndim_in, edim, 128, activation))
        # Layer 2: 128 -> ndim_out (e.g., 128)
        self.layers.append(SAGELayer(128, edim, ndim_out, activation))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, nfeats, efeats):
        # Handle the [N, 1, dim] shape from notebook if present
        nfeats_internal = nfeats.squeeze(1) if nfeats.ndim == 3 else nfeats
        efeats_internal = efeats.squeeze(1) if efeats.ndim == 3 else efeats

        for i, layer in enumerate(self.layers):
            if i != 0:
                nfeats_internal = self.dropout(nfeats_internal)

            # The layer forward needs adjustment based on how node/edge features are used
            # Revisit SAGELayer forward: it uses edges.src['h'] (node feat) and edges.data['h'] (edge feat)
            # Input nfeats corresponds to edges.src['h'], efeats to edges.data['h']
            # Layer output becomes the new node features
            nfeats_internal = layer(g, nfeats_internal, efeats_internal)
            # Remove the extra dimension added by the layer
            nfeats_internal = nfeats_internal.squeeze(1) if nfeats_internal.ndim == 3 else nfeats_internal


        # The notebook does nfeats.sum(1) - this sums over the sequence dimension added?
        # If input was [N, 1, dim], output of last layer is [N, 1, ndim_out], sum(1) -> [N, ndim_out]
        # If input was [N, dim], output is [N, ndim_out], sum(1) doesn't make sense.
        # Assuming we stick to [N, dim], we just return the final node features
        return nfeats_internal


class MLPPredictor(nn.Module):
    def __init__(self, in_features, out_classes):
        super().__init__()
        # in_features here should be the output dim of the SAGE model (node embedding dim)
        self.W = nn.Linear(in_features * 2, out_classes) # Predicts on edges using src/dst node embeddings

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        score = self.W(torch.cat([h_u, h_v], 1))
        return {'score': score}

    def forward(self, graph, h):
        # h should be the node embeddings output by SAGE [N, ndim_out]
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(self.apply_edges)
            return graph.edata['score'] # Return edge scores [E, out_classes]

class Model(nn.Module):
    def __init__(self, ndim_in, ndim_out, edim, activation, dropout, n_classes):
        super().__init__()
        # ndim_in: initial node feature dim (e.g., edge feature dim if initialized like notebook)
        # ndim_out: node embedding dim output by SAGE (e.g., 128)
        # edim: edge feature dim (input flow features)
        self.gnn = SAGE(ndim_in, ndim_out, edim, activation, dropout)
        self.pred = MLPPredictor(ndim_out, n_classes) # ndim_out is the input feature size for MLP

    def forward(self, g, nfeats, efeats):
        # nfeats: initial node features [N, ndim_in] or [N, 1, ndim_in]
        # efeats: edge features [E, edim] or [E, 1, edim]
        h = self.gnn(g, nfeats, efeats) # Output node embeddings [N, ndim_out]
        return self.pred(g, h) # Output edge predictions [E, n_classes]

# --- Load Data ---
print("Loading data...")
data_df = pd.read_csv(DATASET_PATH)
print(f"Original data shape: {data_df.shape}")

# --- Preprocessing ---
print("Preprocessing data...")
# Keep only necessary columns
data_df = data_df[COLUMNS_TO_KEEP].copy()

# Convert IP/Port to node identifiers
data_df['IPV4_SRC_ADDR_str'] = data_df['IPV4_SRC_ADDR'].astype(str)
data_df['L4_SRC_PORT_str'] = data_df['L4_SRC_PORT'].astype(str)
data_df['IPV4_DST_ADDR_str'] = data_df['IPV4_DST_ADDR'].astype(str)
data_df['L4_DST_PORT_str'] = data_df['L4_DST_PORT'].astype(str)
data_df['IPV4_SRC_ADDR'] = data_df['IPV4_SRC_ADDR_str'] + ':' + data_df['L4_SRC_PORT_str']
data_df['IPV4_DST_ADDR'] = data_df['IPV4_DST_ADDR_str'] + ':' + data_df['L4_DST_PORT_str']
data_df.drop(columns=['L4_SRC_PORT','L4_DST_PORT',
                        'IPV4_SRC_ADDR_str', 'L4_SRC_PORT_str',
                        'IPV4_DST_ADDR_str', 'L4_DST_PORT_str'], inplace=True)

# Handle Labels (binary for PGD loss calculation, keep original 'Attack' for potential later use)
# Use 'Label' column (0 for benign, 1 for attack)
# Need LabelEncoder for potential multi-class evaluation later from 'Attack' column
le = LabelEncoder()
data_df['label_encoded'] = le.fit_transform(data_df['Attack'])
N_CLASSES = len(le.classes_)
print(f"Number of classes: {N_CLASSES}")
# Keep binary label for PGD loss targeting binary classification
data_df['binary_label'] = data_df['Label']

# Define features to be scaled
COLS_TO_NORM = CATEGORICAL_FEATURES + NUMERICAL_FEATURES_PGD

# Split data (stratify by binary label for consistency)
X = data_df.drop(columns=['Label', 'Attack', 'label_encoded', 'binary_label'])
y_binary = data_df['binary_label']
y_multi = data_df['label_encoded']

# Use a fixed random state for reproducibility
X_train, X_test, y_train_binary, y_test_binary, y_train_multi, y_test_multi = train_test_split(
    X, y_binary, y_multi, test_size=0.3, random_state=123, stratify=y_binary
)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# --- Fit Encoders/Scalers on Training Data ONLY ---
print("Fitting target encoder and scaler...")
# Target Encoder for categorical features
encoder = ce.TargetEncoder(cols=CATEGORICAL_FEATURES)
# Use multi-class labels for fitting target encoder as in notebook
encoder.fit(X_train, y_train_multi)
X_train_encoded = encoder.transform(X_train)
X_test_encoded = encoder.transform(X_test)

# Scaler for numerical + encoded categorical features
scaler = StandardScaler()
X_train_scaled = X_train_encoded.copy()
X_test_scaled = X_test_encoded.copy()
# Pass numpy array to avoid feature name warnings later
X_train_scaled[COLS_TO_NORM] = scaler.fit_transform(X_train_encoded[COLS_TO_NORM].values)
X_test_scaled[COLS_TO_NORM] = scaler.transform(X_test_encoded[COLS_TO_NORM].values)

# Save the scaler for PGD attack
scaler_filename = os.path.join(MODEL_DIR, 'egraphsage_scaler.pkl')
with open(scaler_filename, 'wb') as f:
    pickle.dump(scaler, f)
print(f"Scaler saved to {scaler_filename}")

# Save target encoder as well? May not be needed for PGD if not targeting these features
encoder_filename = os.path.join(MODEL_DIR, 'egraphsage_target_encoder.pkl')
with open(encoder_filename, 'wb') as f:
    pickle.dump(encoder, f)
print(f"TargetEncoder saved to {encoder_filename}")


# --- Prepare DataFrames for Graph Construction ---
# Add labels back for graph construction
X_train_processed = X_train_scaled.copy()
X_train_processed['label'] = y_train_multi # Use multi-class label as edge attribute like notebook
X_train_processed['binary_label'] = y_train_binary # Add binary label too if needed

X_test_processed = X_test_scaled.copy()
X_test_processed['label'] = y_test_multi
X_test_processed['binary_label'] = y_test_binary

# --- Build Clean Test Graph ---
print("Building clean test graph...")
# Features for edges will be the scaled features
edge_feature_cols = COLS_TO_NORM
X_test_processed['h'] = X_test_processed[edge_feature_cols].values.tolist()
EDGE_FEATURE_DIM = len(edge_feature_cols)

# Create NetworkX graph from test data edges
# Use only columns needed for graph structure and edge features/labels
test_graph_df = X_test_processed[['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'h', 'label', 'binary_label']].copy()

G_nx_test = nx.from_pandas_edgelist(
    test_graph_df,
    "IPV4_SRC_ADDR",
    "IPV4_DST_ADDR",
    edge_attr=['h', 'label', 'binary_label'], # Include binary label if needed by loss
    create_using=nx.MultiDiGraph() # Use DiGraph or MultiDiGraph as needed
)
print(f"NX test graph: {G_nx_test.number_of_nodes()} nodes, {G_nx_test.number_of_edges()} edges")

# Convert to DGL graph
# Need edge attributes: 'h' (features), 'label' (multi-class), 'binary_label'
g_test_dgl = dgl.from_networkx(G_nx_test, edge_attrs=['h', 'label', 'binary_label'])

# Initialize node features (as done in notebook: ones with edge feature dimension)
# This seems counter-intuitive, maybe node features aren't used directly?
# The SAGE layer uses edges.src['h'], which requires node features.
# Let's initialize them as ones with a potentially different dimension if needed, e.g., 128?
# Or maybe the notebook meant initial node embeddings are size of edge features?
# Let's stick to notebook: size is EDGE_FEATURE_DIM
NODE_FEATURE_DIM = EDGE_FEATURE_DIM # Based on notebook cell 30
g_test_dgl.ndata['h'] = torch.ones(g_test_dgl.num_nodes(), NODE_FEATURE_DIM)
# Add the sequence dimension [N, 1, dim] as used in notebook model training
g_test_dgl.ndata['h'] = g_test_dgl.ndata['h'].unsqueeze(1)
g_test_dgl.edata['h'] = torch.tensor(np.array(g_test_dgl.edata['h'].tolist()), dtype=torch.float32).unsqueeze(1)
g_test_dgl.edata['label'] = torch.tensor(g_test_dgl.edata['label'], dtype=torch.long)
g_test_dgl.edata['binary_label'] = torch.tensor(g_test_dgl.edata['binary_label'], dtype=torch.long)


print(f"DGL test graph created with {g_test_dgl.num_nodes()} nodes and {g_test_dgl.num_edges()} edges.")
print(f"Node feature shape: {g_test_dgl.ndata['h'].shape}")
print(f"Edge feature shape: {g_test_dgl.edata['h'].shape}")

# Move graph to device
g_test_dgl = g_test_dgl.to(DEVICE)
g_test_clean_path = os.path.join(ATTACK_DATA_DIR, 'test_data_clean_ipport.pt')
torch.save(g_test_dgl, g_test_clean_path)
print(f"Clean test graph saved to {g_test_clean_path}")

# --- Load Pre-trained EGraphSage Model ---
print("Loading pre-trained EGraphSage model...")
# Determine model parameters based on notebook training cell 46
# ndim_in = NODE_FEATURE_DIM, ndim_out = 128, edim = EDGE_FEATURE_DIM, n_classes = N_CLASSES
model = Model(
    ndim_in=NODE_FEATURE_DIM,
    ndim_out=128, # Output node embedding dim from notebook
    edim=EDGE_FEATURE_DIM,
    activation=F.relu,
    dropout=0.2, # From notebook training cell
    n_classes=N_CLASSES # Number of original classes for prediction head
)
model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval() # Set model to evaluation mode
    print(f"Model loaded successfully from {model_path}")
else:
    print(f"ERROR: Model file not found at {model_path}")
    exit()

# --- PGD Attack Function (Targeting Edge Features) ---
def run_pgd_attack_egraphsage_edge(model, graph, epsilon):
    """
    Performs PGD attack on the EGraphSage model by perturbing edge features.

    Args:
        model: The pre-trained EGraphSage model (in eval mode).
        graph: The clean DGL graph (test set).
        epsilon: The perturbation budget (L-infinity norm for edge features).

    Returns:
        A new DGL graph with perturbed edge features in graph.edata['h_perturbed'].
    """
    print(f"Running PGD attack targeting EDGE features with epsilon={epsilon}")
    model.eval() # Ensure model is in eval mode

    attacked_graph = graph.clone().to(DEVICE) # Work on a copy on the correct device

    # Get initial node features and clean edge features
    node_features = attacked_graph.ndata['h'].detach().clone()       # Clean Node features [N, 1, D_node]
    edge_features = attacked_graph.edata['h'].detach().clone()       # Edge features [E, 1, D_edge]
    edge_features_clean = edge_features.detach().clone() # Keep clean original features for clipping

    # Ensure node features require gradients for attack
    # node_features.requires_grad = True
    edge_features.requires_grad = True

    # Calculate step size (alpha) relative to edge feature epsilon
    alpha = (2.5 * epsilon) / PGD_ITERATIONS

    # --- PGD Iterations ---
    for i in range(PGD_ITERATIONS):
        # Necessary to allow gradient computation in the loop
        # node_features.requires_grad = True
        # attacked_graph.ndata['h'] = node_features # Use current perturbed node features
        edge_features.requires_grad = True
        attacked_graph.edata['h'] = edge_features # Use current perturbed edge features

        # Forward pass to get predictions using perturbed nodes and CLEAN edges
        # Forward pass to get predictions using CLEAN nodes and PERTURBED edges
        logits = model(attacked_graph, node_features, edge_features) # Shape: [E, N_CLASSES] - Pass clean nodes, perturbed edges

        # --- Calculate Loss ---
        # Use binary labels for loss calculation (target: flip benign to attack, attack to benign)
        labels_binary = attacked_graph.edata['binary_label'].to(DEVICE)
        target_labels = 1 - labels_binary # Target the opposite class
        loss = F.cross_entropy(logits, target_labels) # Maximize error -> minimize loss towards wrong label

        # Zero gradients, backward pass, get gradients w.r.t. node features
        # Zero gradients, backward pass, get gradients w.r.t. edge features
        model.zero_grad()
        # Check if graph requires grad? No, only node_features should.
        # if node_features.grad is not None:
        #      node_features.grad.zero_()
        if edge_features.grad is not None:
             edge_features.grad.zero_()

        loss.backward()

        # if node_features.grad is None:
        #      print(f"Warning: Node feature gradients are None on iteration {i}. Stopping attack.")
        #      break # Stop if gradients become None
        if edge_features.grad is None:
             print(f"Warning: Edge feature gradients are None on iteration {i}. Stopping attack.")
             break # Stop if gradients become None

        # --- Apply PGD Update to Node Features ---
        # grad_nodes = node_features.grad.detach() # [N, 1, D_node]
        # update = alpha * grad_nodes.sign()
        # perturbed_node_features = node_features.detach() + update # Apply update
        # --- Apply PGD Update to Edge Features ---
        grad_edges = edge_features.grad.detach() # [E, 1, D_edge]
        update = alpha * grad_edges.sign() #
        perturbed_edge_features = edge_features.detach() + update # Apply update

        # --- Clipping within Epsilon Ball (Node Feature Space) ---
        # Ensure perturbation doesn't exceed epsilon relative to original node features (ones)
        # eta = perturbed_node_features - node_features_clean
        # eta = torch.clamp(eta, -epsilon, epsilon)
        # perturbed_node_features = node_features_clean + eta
        # --- Clipping within Epsilon Ball (Edge Feature Space) ---
        # Ensure perturbation doesn't exceed epsilon relative to original edge features
        eta = perturbed_edge_features - edge_features_clean
        eta = torch.clamp(eta, -epsilon, epsilon)
        perturbed_edge_features = edge_features_clean + eta

        # --- Domain Constraints (Optional for node features, e.g., non-negativity?) ---
        # Since they start as ones, simple clipping might be okay.
        # perturbed_node_features = torch.clamp(perturbed_node_features, min=0.0) # Example constraint
        # --- Domain Constraints (Optional for edge features, e.g., based on scaled range?) ---
        # Clipping within the epsilon ball of the original scaled features is often sufficient.
        # If needed, could clamp based on min/max of scaled training data.

        # Update node features for the next iteration
        # node_features = perturbed_node_features.detach().clone()
        # Update edge features for the next iteration
        edge_features = perturbed_edge_features.detach().clone()

        if (i+1) % 10 == 0:
            print(f"PGD Iteration [{i+1}/{PGD_ITERATIONS}], Loss: {loss.item():.4f}")


    # Store final perturbed node features
    # attacked_graph.ndata['h_perturbed'] = node_features.detach()
    # Restore original clean edge features (as they weren't perturbed)
    # attacked_graph.edata['h'] = edge_features.detach()
    #  # Add perturbed edge features key even if not used, maybe for compatibility? Or remove entirely.
    # if 'h_perturbed' not in attacked_graph.edata:
    #     attacked_graph.edata['h_perturbed'] = edge_features.detach() # just copy clean ones
    # Store final perturbed edge features
    attacked_graph.edata['h_perturbed'] = edge_features.detach()
    # Restore original clean node features (as they weren't perturbed)
    attacked_graph.ndata['h'] = node_features.detach()
    # Add perturbed node features key even if not used, maybe for compatibility? Or remove entirely.
    if 'h_perturbed' not in attacked_graph.ndata:
        attacked_graph.ndata['h_perturbed'] = node_features.detach() # just copy clean ones


    print(f"PGD attack finished for epsilon={epsilon}.")
    return attacked_graph


# --- Black-Box Attack Function (Edge Removal for IP:Port Graph) ---
def run_edge_removal_attack_ipport(graph, rate):
    """
    Performs Edge Removal attack on an IP:Port DGL graph.

    Args:
        graph: The clean DGL graph.
        rate: The fraction of edges to remove (e.g., 0.1 for 10%).

    Returns:
        A new DGL graph with the specified fraction of edges removed.
    """
    print(f"Running Edge Removal attack with rate={rate}")
    graph_device = graph.device # Get the device of the input graph
    num_edges_original = graph.num_edges()
    num_edges_to_remove = int(num_edges_original * rate)

    if num_edges_to_remove >= num_edges_original:
        print(f"Warning: Removal rate ({rate}) is too high. Removing all edges.")
        ids_to_remove = torch.arange(num_edges_original, device=graph_device) # Create on correct device
    elif num_edges_to_remove <= 0:
         print("Info: Removal rate resulted in 0 edges to remove. Returning original graph structure.")
         return graph.clone() # Return a clone to be safe
    else:
        edge_ids = torch.arange(num_edges_original)
        # Shuffle edge IDs using torch.randperm for efficiency
        shuffled_ids = edge_ids[torch.randperm(num_edges_original)]
        ids_to_remove = shuffled_ids[:num_edges_to_remove].to(graph_device) # Move IDs to graph device

    print(f"Removing {len(ids_to_remove)} edges out of {num_edges_original} (Rate: {rate})...")
    attacked_graph = dgl.remove_edges(graph, ids_to_remove)
    print(f"Edge removal finished. New graph has {attacked_graph.num_edges()} edges.")

    # Important: Check if nodes became isolated. DGL keeps nodes by default.
    # We might want to log this.
    # num_isolated_nodes = (attacked_graph.in_degrees() == 0) & (attacked_graph.out_degrees() == 0)
    # print(f"Number of isolated nodes after removal: {num_isolated_nodes.sum().item()}")

    return attacked_graph


# --- Black-Box Attack Function (Node Injection for IP:Port Graph) ---
def run_node_injection_attack_ipport(graph, rate, connections_per_node, le):
    """
    Performs Node Injection attack on an IP:Port DGL graph.

    Args:
        graph: The clean DGL graph (will be moved to CPU if not already).
        rate: The fraction of nodes to inject relative to original node count.
        connections_per_node: How many edges each new node should create.
        le: Fitted LabelEncoder to find an attack label index.

    Returns:
        A new DGL graph (on CPU) with injected nodes and edges.
    """
    print(f"Running Node Injection attack with rate={rate}")
    # --- Ensure graph is on CPU for manipulation --- #
    cpu_device = torch.device('cpu')
    graph_cpu = graph.to(cpu_device)

    num_nodes_original = graph_cpu.num_nodes()
    num_nodes_to_inject = int(num_nodes_original * rate)

    if num_nodes_to_inject <= 0:
        print("Info: Injection rate resulted in 0 nodes to inject. Returning original graph.")
        return graph_cpu.clone()

    print(f"Injecting {num_nodes_to_inject} nodes (Rate: {rate})...")

    # --- 1. Add New Nodes --- #
    # Determine initial features for new nodes (using the same method as original nodes)
    # Assuming original nodes have features of shape [N, 1, D]
    original_node_feat_shape = graph_cpu.ndata['h'].shape
    node_feature_dim = original_node_feat_shape[-1]
    # Create new features explicitly on CPU
    new_node_features = torch.ones(num_nodes_to_inject, 1, node_feature_dim,
                                    dtype=graph_cpu.ndata['h'].dtype, device=cpu_device)

    # Create a copy to modify (already on CPU)
    attacked_graph = graph_cpu.clone()
    attacked_graph.add_nodes(num_nodes_to_inject, data={'h': new_node_features})
    num_nodes_total = attacked_graph.num_nodes()
    print(f"Nodes added. New graph node count: {num_nodes_total}")

    # --- 2. Prepare for Edge Injection --- #
    # Identify existing attack edges to sample features from (on CPU graph)
    attack_edge_mask = graph_cpu.edata['binary_label'] == 1
    attack_edge_ids = torch.where(attack_edge_mask)[0].to(cpu_device)

    if len(attack_edge_ids) == 0:
        print("Warning: No attack edges found in the original graph to sample features from. Cannot inject edges.")
        return attacked_graph # Return graph with only nodes added

    # Get features and labels of existing attack edges (on CPU)
    attack_edge_features = graph_cpu.edata['h'][attack_edge_ids]
    attack_label_multi = graph_cpu.edata['label'][attack_edge_ids[0]].item()

    # --- 3. Generate and Add New Edges --- #
    new_edge_src_nodes = []
    new_edge_dst_nodes = []
    num_new_edges_to_add = num_nodes_to_inject * connections_per_node

    for i in range(num_nodes_to_inject):
        new_node_id = num_nodes_original + i
        # Select random existing nodes as destinations (on CPU)
        possible_dst_nodes = torch.arange(num_nodes_original, device=cpu_device)
        k = min(connections_per_node, num_nodes_original)
        dst_nodes = possible_dst_nodes[torch.randperm(num_nodes_original, device=cpu_device)[:k]]

        new_edge_src_nodes.extend([new_node_id] * k)
        new_edge_dst_nodes.extend(dst_nodes.tolist())

    # Create tensors explicitly on CPU
    new_edge_src = torch.tensor(new_edge_src_nodes, device=cpu_device)
    new_edge_dst = torch.tensor(new_edge_dst_nodes, device=cpu_device)
    actual_new_edges_count = len(new_edge_src)

    if actual_new_edges_count == 0:
        print("Warning: No new edges were generated.")
        return attacked_graph

    print(f"Generating {actual_new_edges_count} new edges...")

    # --- 4. Generate Features and Labels for New Edges (on CPU) --- #
    sample_indices = torch.randint(0, len(attack_edge_ids), (actual_new_edges_count,), device=cpu_device)
    new_edge_features = attack_edge_features[sample_indices]
    new_edge_labels_binary = torch.ones(actual_new_edges_count, dtype=torch.long, device=cpu_device)
    new_edge_labels_multi = torch.full((actual_new_edges_count,), fill_value=attack_label_multi, dtype=torch.long, device=cpu_device)

    # --- 5. Add Edges to Graph (already on CPU) --- #
    attacked_graph.add_edges(
        new_edge_src,
        new_edge_dst,
        data={
            'h': new_edge_features,
            'binary_label': new_edge_labels_binary,
            'label': new_edge_labels_multi
        }
    )

    print(f"Node injection finished. Final graph: {attacked_graph.num_nodes()} nodes, {attacked_graph.num_edges()} edges.")
    # Return the attacked graph, which is on the CPU
    return attacked_graph


# --- Evaluation Function ---
def evaluate_model(model, graph_data, le, use_perturbed=False):
    """
    Evaluates the EGraphSage model on the given graph data.

    Args:
        model: The pre-trained EGraphSage model (in eval mode).
        graph_data: The DGL graph (clean or attacked).
        le: The fitted LabelEncoder for mapping 'Attack' strings to integers.
        use_perturbed: Boolean flag to use perturbed features ('h_perturbed') if True.

    Returns:
        A dictionary containing evaluation metrics (acc, precision, recall, f1, auc).
    """
    # feature_key = 'h_perturbed' if use_perturbed else 'h'
    node_feature_key = 'h' # Always use clean node features
    edge_feature_key = 'h_perturbed' if use_perturbed else 'h' # Use perturbed edge features if flag is set
    data_type = 'perturbed (edge attack)' if use_perturbed else 'clean' #
    # Updated print statement
    # print(f"Evaluating model on {data_type} graph using NODE features '{feature_key}'...")
    print(f"Evaluating model on {data_type} graph using NODE features '{node_feature_key}' and EDGE features '{edge_feature_key}'...")

    # Check for node features
    # if feature_key not in graph_data.ndata:
    #     print(f"Error: Node features '{feature_key}' not found in graph data.")
    #     return {'acc': np.nan, 'precision': np.nan, 'recall': np.nan, 'f1': np.nan, 'auc': np.nan}
    # Check for necessary features
    if node_feature_key not in graph_data.ndata:
        print(f"Error: Node features '{node_feature_key}' not found in graph data.")
        return {'acc': np.nan, 'precision': np.nan, 'recall': np.nan, 'f1': np.nan, 'auc': np.nan}
    if edge_feature_key not in graph_data.edata:
        print(f"Error: Edge features '{edge_feature_key}' not found in graph data.")
        return {'acc': np.nan, 'precision': np.nan, 'recall': np.nan, 'f1': np.nan, 'auc': np.nan}
    if 'h' not in graph_data.edata:
        print(f"Error: Edge features 'h' not found in graph data.")
        return {'acc': np.nan, 'precision': np.nan, 'recall': np.nan, 'f1': np.nan, 'auc': np.nan}


    model.eval()
    with torch.no_grad():
        # Updated feature loading
        # node_features = graph_data.ndata[feature_key].to(DEVICE) if use_perturbed else graph_data.ndata['h'].to(DEVICE)
        # edge_features = graph_data.edata['h'].to(DEVICE) # Always use clean edge features for evaluation
        node_features = graph_data.ndata[node_feature_key].to(DEVICE) #
        edge_features = graph_data.edata[edge_feature_key].to(DEVICE) #

        edge_labels_binary = graph_data.edata['binary_label'].to(DEVICE) # Binary labels (0/1)
        edge_labels_multi = graph_data.edata['label'].to(DEVICE)      # Multi-class labels (encoded)

        # Check for NaNs in features
        if torch.isnan(node_features).any() or torch.isnan(edge_features).any():
             print(f"Warning: NaNs detected in input features for {data_type} graph.")
             # Option: return NaNs or try to proceed if model handles them (unlikely)
             # return {'acc': np.nan, 'precision': np.nan, 'recall': np.nan, 'f1': np.nan, 'auc': np.nan}


        # Get edge predictions - Updated model call
        logits = model(graph_data, node_features, edge_features) # Shape: [E, N_CLASSES]

        # Check for NaNs in logits
        if torch.isnan(logits).any():
            print(f"Warning: NaNs detected in model logits for {data_type} graph.")
            return {'acc': np.nan, 'precision': np.nan, 'recall': np.nan, 'f1': np.nan, 'auc': np.nan}

        preds_multi = logits.argmax(1)
        probs = F.softmax(logits, dim=1) # Get probabilities for AUC

        # Calculate metrics using binary labels for consistency with CAGN eval
        # Map multi-class preds back to binary (attack vs benign)
        try:
            # Ensure 'Benign' is actually in the learned classes by LabelEncoder
            if 'Benign' in le.classes_:
                 benign_label_encoded = le.transform(['Benign'])[0]
                 print(f"Benign label encoded as: {benign_label_encoded}")
                 # Convert multi-class predictions to binary (1 if not benign, 0 if benign)
                 preds_binary = (preds_multi != benign_label_encoded).long()

                 # Calculate binary metrics
                 acc = accuracy_score(edge_labels_binary.cpu(), preds_binary.cpu())
                 precision = precision_score(edge_labels_binary.cpu(), preds_binary.cpu(), zero_division=0)
                 recall = recall_score(edge_labels_binary.cpu(), preds_binary.cpu(), zero_division=0)
                 f1 = f1_score(edge_labels_binary.cpu(), preds_binary.cpu(), zero_division=0)

                 # Calculate AUC using probability of the positive class (attack=1)
                 # P(attack) = 1 - P(benign)
                 prob_benign = probs[:, benign_label_encoded]
                 prob_attack = 1.0 - prob_benign
                 auc = roc_auc_score(edge_labels_binary.cpu(), prob_attack.cpu())

            else:
                 print("Warning: 'Benign' class not found in LabelEncoder classes during evaluation. Cannot calculate binary metrics.")
                 acc, precision, recall, f1, auc = np.nan, np.nan, np.nan, np.nan, np.nan

        except ValueError as e:
            print(f"Error during binary metric calculation (ValueError: {e}). Check LabelEncoder setup.")
            acc, precision, recall, f1, auc = np.nan, np.nan, np.nan, np.nan, np.nan
        except IndexError as e:
             print(f"Error during binary metric calculation (IndexError: {e}). Check probs shape vs benign label index.")
             acc, precision, recall, f1, auc = np.nan, np.nan, np.nan, np.nan, np.nan


        print(f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        return {'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1, 'auc': auc}


# --- Main Execution Logic ---
if __name__ == "__main__":
    start_time = time.time()

    # --- Remove Unused Variable Calculation ---
    # Indices are no longer needed for the node-based attack
    # target_feature_indices = [COLS_TO_NORM.index(f) for f in NUMERICAL_FEATURES_PGD]
    # print(f"Indices of features to perturb in edge features 'h': {target_feature_indices}")
    # Indices are not directly needed when perturbing the entire scaled edge feature vector


    # --- Generate Attacked Data ---
    print("\n--- Generating Attacked Data ---")

    # PGD Attacks (Existing)
    print("\nGenerating PGD Attack Data...")
    for eps in PGD_EPSILON_LIST:
        print(f"\nGenerating PGD attack data for epsilon = {eps}")
        # Load clean graph data each time
        g_test_clean = torch.load(g_test_clean_path).to(DEVICE)

        # Run attack - Updated Function Call
        attacked_graph = run_pgd_attack_egraphsage_edge(model, g_test_clean, eps) #

        # Save attacked graph
        attacked_graph_path = os.path.join(ATTACK_DATA_DIR, f'test_data_pgd_eps{eps}_EGraphSage_EdgeAttack.pt') #
        # Ensure perturbed edge features exist before removing gradients
        if 'h_perturbed' in attacked_graph.edata and attacked_graph.edata['h_perturbed'].grad is not None:
            del attacked_graph.edata['h_perturbed'].grad #
        torch.save(attacked_graph.cpu(), attacked_graph_path) # Save on CPU
        print(f"Attacked graph saved to {attacked_graph_path}")
        del attacked_graph, g_test_clean # Free memory

    # Black-Box Edge Removal Attacks (Existing)
    print("\nGenerating Edge Removal Attack Data...")
    g_test_clean_cpu = torch.load(g_test_clean_path) # Load clean graph once on CPU
    for rate in EDGE_REMOVAL_RATES:
        print(f"\nGenerating Edge Removal attack data for rate = {rate}")
        # No need to move to DEVICE for removal, dgl handles it
        attacked_graph_removed = run_edge_removal_attack_ipport(g_test_clean_cpu, rate)

        # Define save path
        attacked_graph_path = os.path.join(ATTACK_DATA_DIR, f'test_data_edge_remove_{int(rate*100)}pct_ipport.pt')
        torch.save(attacked_graph_removed, attacked_graph_path) # Save the CPU graph
        print(f"Attacked graph saved to {attacked_graph_path}")
        del attacked_graph_removed # Free memory

    del g_test_clean_cpu # Free memory

    # Black-Box Node Injection Attacks (New)
    print("\nGenerating Node Injection Attack Data...")
    g_test_clean_cpu = torch.load(g_test_clean_path) # Load clean graph once on CPU
    for rate in NODE_INJECTION_RATES:
        print(f"\nGenerating Node Injection attack data for rate = {rate}")
        # Pass the label encoder 'le' needed for determining attack labels
        attacked_graph_injected = run_node_injection_attack_ipport(g_test_clean_cpu, rate, NODE_INJECTION_CONNECTIONS_PER_NODE, le)

        # Define save path
        attacked_graph_path = os.path.join(ATTACK_DATA_DIR, f'test_data_node_inject_{int(rate*100)}pct_ipport.pt')
        torch.save(attacked_graph_injected, attacked_graph_path) # Save the CPU graph
        print(f"Attacked graph saved to {attacked_graph_path}")
        del attacked_graph_injected # Free memory

    del g_test_clean_cpu # Free memory

    # --- Evaluate Models ---
    print("\n--- Evaluating Model Performance ---")
    results = []

    # Evaluate on Clean Data (Existing)
    print("\nEvaluating on Clean Data...")
    g_test_clean = torch.load(g_test_clean_path).to(DEVICE)
    clean_metrics = evaluate_model(model, g_test_clean, le, use_perturbed=False)
    results.append({'attack_type': 'Clean', 'param': 0, **clean_metrics})
    del g_test_clean

    # Evaluate on PGD Attacked Data (Existing)
    print("\nEvaluating on PGD Attacked Data...")
    for eps in PGD_EPSILON_LIST:
        attacked_graph_path = os.path.join(ATTACK_DATA_DIR, f'test_data_pgd_eps{eps}_EGraphSage_EdgeAttack.pt') #
        if os.path.exists(attacked_graph_path):
            print(f"\nLoading attacked graph for epsilon = {eps}")
            attacked_graph_data = torch.load(attacked_graph_path).to(DEVICE)
            # Pass le to evaluate_model
            pgd_metrics = evaluate_model(model, attacked_graph_data, le, use_perturbed=True)
            results.append({'attack_type': 'PGD_EGraphSage_EdgeAttack', 'param': eps, **pgd_metrics}) # Indicate edge attack
            del attacked_graph_data
        else:
            print(f"Attacked graph file not found: {attacked_graph_path}")

    # Evaluate on Edge Removal Attacked Data (Existing)
    print("\nEvaluating on Edge Removal Attacked Data...")
    for rate in EDGE_REMOVAL_RATES:
        attacked_graph_path = os.path.join(ATTACK_DATA_DIR, f'test_data_edge_remove_{int(rate*100)}pct_ipport.pt')
        if os.path.exists(attacked_graph_path):
            print(f"\nLoading edge removal graph for rate = {rate}")
            attacked_graph_data = torch.load(attacked_graph_path).to(DEVICE) # Load and move to device
            # Evaluate using the standard function. Features ('h') are clean.
            edge_remove_metrics = evaluate_model(model, attacked_graph_data, le, use_perturbed=False)
            results.append({'attack_type': 'EdgeRemove_IPPort', 'param': rate, **edge_remove_metrics})
            del attacked_graph_data
        else:
            print(f"Attacked graph file not found: {attacked_graph_path}")

    # Evaluate on Node Injection Attacked Data (New)
    print("\nEvaluating on Node Injection Attacked Data...")
    for rate in NODE_INJECTION_RATES:
        attacked_graph_path = os.path.join(ATTACK_DATA_DIR, f'test_data_node_inject_{int(rate*100)}pct_ipport.pt')
        if os.path.exists(attacked_graph_path):
            print(f"\nLoading node injection graph for rate = {rate}")
            attacked_graph_data = torch.load(attacked_graph_path).to(DEVICE) # Load and move to device
            # Evaluate using the standard function. Features ('h') are clean for original nodes/edges.
            # Perturbation is the addition of new nodes/edges.
            node_inject_metrics = evaluate_model(model, attacked_graph_data, le, use_perturbed=False)
            results.append({'attack_type': 'NodeInject_IPPort', 'param': rate, **node_inject_metrics})
            del attacked_graph_data
        else:
            print(f"Attacked graph file not found: {attacked_graph_path}")


    # --- Save Results ---
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_FILE, index=False)
    print(f"\nResults saved to {RESULTS_FILE}")
    print(results_df)

    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")

    # --- Sanity Check: Compare Clean vs Perturbed Edge Features ---
    print("\n--- Sanity Checking Perturbations (eps=0.7) ---")
    try:
        g_clean_check = torch.load(g_test_clean_path).to(DEVICE)
        # Use the largest epsilon value for the check
        max_eps = max(PGD_EPSILON_LIST)
        # attacked_graph_check_path = os.path.join(ATTACK_DATA_DIR, f'test_data_pgd_eps{max_eps}_EGraphSage.pt')
        attacked_graph_check_path = os.path.join(ATTACK_DATA_DIR, f'test_data_pgd_eps{max_eps}_EGraphSage_EdgeAttack.pt') #
        g_attacked_check = torch.load(attacked_graph_check_path).to(DEVICE)

        # Updated comparison for edge features
        # clean_h_node = g_clean_check.ndata['h']
        # perturbed_h_node = g_attacked_check.ndata['h_perturbed']
        clean_h_edge = g_clean_check.edata['h'] #
        perturbed_h_edge = g_attacked_check.edata['h_perturbed'] #

        # if 'h' not in g_clean_check.ndata or 'h_perturbed' not in g_attacked_check.ndata:
        #     print("Error: Node feature keys not found for comparison.")
        # else:
        #     diff = (clean_h_node - perturbed_h_node).abs()
        #     max_diff = diff.max().item()
        #     mean_diff = diff.mean().item()
        #     # Updated print statements for NODE features
        #     print(f"Max difference between clean node 'h' and perturbed node 'h' (eps={max_eps}): {max_diff:.6f}")
        #     print(f"Mean absolute difference for node 'h': {mean_diff:.6f}")

        #     if max_diff < 1e-5:
        #         print("WARNING: Perturbed node features seem identical to clean node features!")
        #     else:
        #         print("Perturbed node features ARE different from clean node features.")
        if 'h' not in g_clean_check.edata or 'h_perturbed' not in g_attacked_check.edata: #
            print("Error: Edge feature keys ('h', 'h_perturbed') not found for comparison.") #
        else: #
            diff = (clean_h_edge - perturbed_h_edge).abs() #
            max_diff = diff.max().item() #
            mean_diff = diff.mean().item() #
            # Updated print statements for EDGE features #
            print(f"Max difference between clean edge 'h' and perturbed edge 'h' (eps={max_eps}): {max_diff:.6f}") #
            print(f"Mean absolute difference for edge 'h': {mean_diff:.6f}") #

            if max_diff < 1e-5: #
                print("WARNING: Perturbed edge features seem identical to clean edge features!") #
            else: #
                print("Perturbed edge features ARE different from clean edge features.") #

        del g_clean_check
        del g_attacked_check
    except FileNotFoundError:
        print("Error: Could not load graph files for sanity check.")
    except Exception as e:
        print(f"An error occurred during sanity check: {e}") 