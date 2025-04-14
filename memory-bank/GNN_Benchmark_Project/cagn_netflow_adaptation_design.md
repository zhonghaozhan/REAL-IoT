## Design Document: Adapting CAGN Methodology for Standardized NetFlow Data (with Sampling)

**1. Objective:**

*   Revise the `CAGN-GAT Standardized.ipynb` notebook to use the standardized NetFlow dataset (`combined_unsw_cicRed_botRed_netflow_10pct.csv`).
*   Adapt the **CAGN model methodology** from the `CAGN-GAT Fusion.ipynb` notebook to process this standardized NetFlow data for benchmarking purposes.
*   The core idea is to replicate the original notebook's process: **Sample Data -> Tabular Features -> Similarity Graph Construction -> GNN (CAGN) Training**.

**2. Input Data:**

*   Primary Dataset: `/media/ssd/test/standardized-datasets/combined/combined_unsw_cicRed_botRed_netflow_10pct.csv`
    *   Contains standardized flow features, `Label`, `Attack`, etc.

**3. Core Challenge: Scalability of Original Methodology**

The original CAGN notebook processed tabular data by first engineering features and then constructing a graph where nodes represent samples and edges represent feature similarity (`adaptive_graph_construction`). However, this graph construction method (specifically pairwise distance calculation) does not scale to the size of our full standardized dataset (~655k samples). Therefore, **sampling** is required first.

**4. Proposed Solution Strategy:**

The adaptation will involve the following key steps within the notebook:

*   **Data Loading:** Load the full standardized CSV.
*   **Sampling:** Apply targeted sampling (like `create_imbalanced_subset`) to reduce the dataset size while preserving class distribution, especially for minority classes.
*   **Feature Engineering:** Engineer a feature matrix `X` (samples x features) and label vector `y` from the *sampled* data.
*   **Graph Construction (Similarity-Based):** Use the engineered features `X` from the sampled data to construct a graph where nodes are samples and edges connect similar samples (replicating `adaptive_graph_construction`).
*   **Model Input Formatting:** Structure the resulting graph and features into a PyTorch Geometric `Data` object (`data.x`, `data.edge_index`, `data.y`).
*   **Model Adaptation:** Use the original `CAGN` model definition.
*   **Training & Evaluation:** Implement training loop using the combined classification and contrastive loss from the original `CAGN` model. Evaluate using relevant metrics.

**5. Detailed Design:**

*   **5.1. Data Loading:**
    *   Load `/media/ssd/test/standardized-datasets/combined/combined_unsw_cicRed_botRed_netflow_10pct.csv` into a pandas DataFrame `df`.

*   **5.2. Sampling:**
    *   Define and apply a sampling function (e.g., `create_imbalanced_subset` from the original notebook) to `df`.
    *   This function should reduce the number of samples from majority classes while keeping minority classes.
    *   The output is a smaller DataFrame `df_sampled`.
    *   Adjust the target size (e.g., `SAMPLED_SIZE_LARGE_CLASSES`) based on memory constraints.

*   **5.3. Feature Engineering (on `df_sampled`):**
    *   Select relevant columns from `df_sampled`. Drop IP/Port columns initially.
    *   **Flow Features (`X`):** Create the feature matrix from `df_sampled`.
        *   *Numerical Features:* (`IN_BYTES`, `OUT_BYTES`, `IN_PKTS`, `OUT_PKTS`, `FLOW_DURATION_MILLISECONDS`) - Apply Log transformation + StandardScaler.
        *   *Categorical Features:* (`PROTOCOL`, `L7_PROTO`, `TCP_FLAGS`) - Use **OneHotEncoder** (e.g., via `pd.get_dummies`) to align with original notebook's processing of UNSW-NB15 features. **Note:** Memory usage for this step depends on the sampled size and cardinality. If OOM occurs even after sampling, the fallback is LabelEncoder + modify CAGN model.
        *   Concatenate processed features into the final feature matrix `X`.
    *   **Labels (`y`):** Use the `Label` column from `df_sampled`.

*   **5.4. Graph Construction (Similarity-Based, on sampled `X`, `y`):**
    *   Implement or replicate the `adaptive_graph_construction` function.
    *   **Nodes:** Sampled flows (rows of `X`).
    *   **Node Features (`data.x`):** The feature matrix `X` from the previous step.
    *   **Edges (`data.edge_index`):** Connect nodes `i` and `j` based on k-NN intersection with distance threshold applied to `X`.
    *   **Edge Features (`data.edge_attr`):** None.
    *   Output: A PyG `Data` object.

*   **5.5. Model Input Formatting (PyTorch Geometric):**
    *   Use the `Data` object created in the previous step.
    *   `data.x`: Node features (shape `[num_sampled_flows, num_features]`).
    *   `data.edge_index`: Similarity edge connectivity.
    *   `data.edge_attr`: None.
    *   `data.y`: Labels (shape `[num_sampled_flows]`).

*   **5.6. Model Adaptation:**
    *   **CAGN:** Copy the `CAGN` class definition. Ensure its input layer dimension matches `num_features` in `X`. Align hyperparameters (e.g., `hidden_dim=64`, `heads=8`) with the original/fusion notebook where feasible and documented.
    *   **Baseline Models:** Include definitions for baseline GNNs (GCN, GAT, GIN, GraphSAGE) for comparative benchmarking, adapting their input/output dimensions as needed.

*   **5.7. Training & Evaluation:**
    *   Implement a training loop compatible with the `CAGN` model and the `Data` object.
    *   Use the combined loss function (classification + contrastive) from `CAGN`.
    *   Address class imbalance if necessary (the sampling helps, but weights might still be useful).
    *   Evaluate using metrics: Accuracy, Precision, Recall, F1-Score, AUC.
    *   Split data into train/validation/test sets *before* graph construction (split the *sampled* tabular data `X`, `y`), then build separate graphs or use masks on a single graph built from the full sampled data. Using masks is generally preferred. Create masks based on the indices of the sampled nodes.

**6. Notebook Structure:**

1.  Setup & Imports
2.  Configuration (Paths, Parameters)
3.  Data Loading (Full dataset)
4.  **Data Sampling (Apply `create_imbalanced_subset`)**
5.  Feature Engineering (On sampled data -> Create `X`, `y`)
6.  Adaptive Graph Construction (On sampled `X`, `y` -> Create PyG `Data`)
7.  Data Splitting (Create train/val/test masks for the `Data` object)
8.  Model Definition (Copy `CAGN` class, add baseline GNN classes)
9.  Training Loop Implementation (Combined loss for CAGN, standard loss for baselines)
10. Evaluation Function Implementation (Separate for CAGN and baselines if needed)
11. Model Training & Evaluation Execution (Loop through CAGN and baselines)
12. Results Analysis & Comparison
13. Save Models & Results

**7. Memory & Scalability:**

*   Sampling significantly reduces the N^2 complexity of graph construction.
*   Memory usage now depends on the *sampled* dataset size, the feature dimension (OneHot), and the resulting graph density.
*   The `adaptive_graph_construction` step should now be feasible.
*   Full graph training on the sampled graph should be possible.

**8. Potential Research/Refinement:**

*   Tune the sampling parameters (`new_dataset_size_large_classes`, `min_large_class_size`).
*   Experiment with graph construction parameters (`k`, `threshold`, `metric`).
*   If OneHot still causes issues on the sampled data, implement LabelEncoder + internal CAGN embeddings.
*   Investigate incorporating IP/Port features into the node features `X`.

**9. Next Steps:**

1.  Implement the changes (sampling, feature eng., graph const., model def., training loop) in the `CAGN-GAT Standardized.ipynb` notebook.
2.  Run the notebook to test feasibility and performance. 