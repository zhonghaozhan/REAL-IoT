# Adversarial Attack Evaluation Plan

## 1. Goal
Evaluate the robustness of the trained Graph Neural Network (GNN) models developed for network intrusion detection against representative white-box and black-box adversarial attacks. This evaluation aims to understand model vulnerabilities and performance degradation under attack conditions.

## 2. Rationale: White-box vs. Black-box Attacks
Adversarial attacks test the resilience of machine learning models. In the context of Network Intrusion Detection Systems (NIDS):
- **White-box attacks** assume the attacker possesses full knowledge of the NIDS model (architecture, parameters, training data). While useful for identifying theoretical worst-case vulnerabilities, this level of access is often unrealistic in real-world deployment scenarios.
- **Black-box attacks** assume limited or no knowledge of the target model, often relying on model queries or transferability from surrogate models. These represent more plausible attack scenarios where adversaries interact with the NIDS from the outside.
This evaluation will include both types to provide a comprehensive robustness assessment, acknowledging the higher practical relevance of black-box scenarios.

## 3. Target Models (for this script/evaluation)
The following GNN model, trained for network intrusion detection, will be evaluated **using the script located at `GNN/Adversarial Evaluation/CAGN_GAT_Evaluation/cagn_gat_adversarial_eval.py`** due to its compatibility with the chosen graph structure:
- CAGN-GAT (Using k-NN similarity graph)

**Models requiring separate evaluation setups (due to different graph structures and/or feature processing):**
- E-GraphSage (Uses IP:Port node graph) - White-box PGD attack plan detailed below (Section 5.1.2). Requires dedicated script.
- Anomal-E (Uses IP:Port node graph and expects different feature input format) - Attack plan TBD.
- Baseline Models (GCN, GAT, GIN, GraphSAGE - Would require retraining on the k-NN graph for direct comparison within the CAGN-GAT script, or separate evaluation on their native graphs).

## 4. Target Dataset (for this script/evaluation)
The evaluation **using the `cagn_gat_adversarial_eval.py` script** will use the **test split** derived from the standardized, combined NetFlow dataset:
- `/media/ssd/test/standardized-datasets/combined/combined_unsw_cicRed_botRed_netflow_10pct.csv`
- **Graph Structure:** k-NN similarity graph built on processed features (Log1p+Scale for numerical, OHE for categorical), compatible with CAGN-GAT.

## 5. Selected Attacks
Attacks will primarily focus on perturbing node features for white-box and graph structure for black-box, applied to the k-NN graph for CAGN-GAT and the IP:Port graph for EGraphSage.

### 5.1 White-Box (Feature Perturbation)

#### 5.1.1 PGD Attack on CAGN-GAT (k-NN Graph)
    - **Target Model:** `CAGN-GAT`
    - **Target Graph:** k-NN similarity graph.
    - **Target Features:** Perturbations will be applied to the original scale of the following **numerical features** *before* any scaling/transformations used in model input:\n        - `IN_BYTES`\n        - `OUT_BYTES`\n        - `IN_PKTS`\n        - `OUT_PKTS`\n        - `FLOW_DURATION_MILLISECONDS`
    - **Method:** Projected Gradient Descent (PGD).
    - **Mechanism:** Leverage full knowledge of the target model (`CAGN-GAT`) to calculate the gradient of the loss with respect to the *interpretable input node features* (\\( \\nabla_x L(\\theta, x, y) \\)). PGD takes multiple smaller steps with projection: \\( x^{(t+1)} = \\text{Proj}_{\\epsilon} (x^{(t)} + \\alpha \\cdot \\text{sign}(\\nabla_x L(\\theta, x^{(t)}, y))) \\).\n    - **Perturbation Budget:** Define `epsilon` values (max L-infinity norm), e.g., \\( \\epsilon = [0.05, 0.1, 0.2] \\).\n    - **PGD Iterations:** Use a fixed number of iterations, e.g., **40 iterations**.
    - **PGD Step Size (\\(\\alpha\\)):** Use a step size relative to epsilon, e.g., \\( \\alpha = \\epsilon / 10 \\).\n    - **Data/Domain Constraints:** Apply constraints *after* calculating perturbations on the original feature scale:\n        - Ensure non-negativity for numerical features.\n        - Round packet counts (`IN_PKTS`, `OUT_PKTS`) to the nearest integer (if deemed necessary, currently handled by non-negativity).\n        - Clip final perturbed values within the `epsilon`-ball (handled by PGD projection).\n    - **Scaling:** Re-apply necessary scaling (Log1p + StandardScaler) specific to CAGN-GAT to the constrained, perturbed features before feeding them to the model.

#### 5.1.2 PGD Attack on EGraphSage (IP:Port Graph) - **Revised Approach (Edge Feature Attack)**
    - **Status:** Implemented in `EGraphSage_Evaluation/egraphsage_adversarial_eval.py`. Evaluation shows **expected performance degradation** at higher epsilon values.
    - **Target Model:** `EGraphSage`
    - **Target Graph:** IP:Port graph constructed from the standardized test dataset split using `nx.from_pandas_edgelist`, where edges represent flows.
    - **Original Target (Node Features - Ineffective):** Initial attempts targeted perturbing the initial node features (`graph.ndata['h']`), which were initialized as `torch.ones`. This proved ineffective as the model relies more heavily on edge features.
    - **Revised Target (Edge Features - Effective):** The PGD attack was modified to target the **edge features** (`graph.edata['h']`). These features are the **scaled** numerical and target-encoded categorical features from the preprocessing step.
    - **Method:** Projected Gradient Descent (PGD).
    - **Mechanism:** Leverage full knowledge of the `EGraphSage` model to calculate gradients of the edge prediction loss w.r.t. the **scaled edge features**. Apply standard PGD update rule directly to these features.
    - **Perturbation Budget:** Define `epsilon` values (max L-infinity norm) for **scaled edge feature** perturbations, e.g., \\\\( \\\\epsilon = [0.3, 0.5, 0.7] \\\\).
    - **PGD Iterations:** Use a sufficient number of iterations, e.g., **60 iterations**.
    - **PGD Step Size (\\\\(\\\\alpha\\\\)):** Use a step size relative to epsilon, e.g., \\\\( \\\\alpha = 2.5 * \\\\epsilon / \\\\text{iterations} \\\\).\n    - **Data/Domain Constraints:** Apply constraints *after* calculating perturbations on scaled edge features:\n        - Clip final perturbed values within the `epsilon`-ball around the **original scaled edge features**.\n    - **Scaling:** Perturbations are applied directly to the **already scaled edge features** used by the GNN layers. No further scaling/inverse scaling is required within the attack itself.\n    - **Results:** Attack successfully degraded model performance (Acc/F1/AUC) for \\\\(\\epsilon=0.5\\\\) and \\\\(\\epsilon=0.7\\\\). An unexpected slight performance *increase* was observed for \\\\(\\epsilon=0.3\\\\), potentially due to regularization effects.\n

#### 5.1.3 PGD Attack on Anomal-E (IP:Port Graph)
    - **Status:** Planned.
    - **Target Model:** `Anomal-E`, specifically targeting the DGI encoder.
    - **Target Graph:** IP:Port graph constructed from the standardized test dataset split (using `Normalizer` for feature scaling).
    - **Target Features:** Perturbations will be applied to the **normalized edge features** (`graph.edata['h']`) used as input to the DGI encoder.
    - **Method:** Projected Gradient Descent (PGD).
    - **Mechanism:** Leverage full knowledge of the pre-trained DGI model (`best_dgi.pkl`). The attack aims to **maximize** the DGI contrastive loss function with respect to the input edge features. This indirectly aims to corrupt the embeddings produced by the encoder, which are subsequently used by the classical anomaly detector (e.g., CBLOF).
    - **Perturbation Budget:** Define `epsilon` values (max L-infinity norm) for **normalized edge feature** perturbations, e.g., \\( \\epsilon = [0.3, 0.5, 0.7] \\).
    - **PGD Iterations:** Use a sufficient number of iterations, e.g., **60 iterations**.
    - **PGD Step Size (\\(\\alpha\\)):** Use a step size relative to epsilon, e.g., \\( \\alpha = 2.5 * \\epsilon / \\text{iterations} \\).
    - **Data/Domain Constraints:** Apply constraints *after* calculating perturbations on normalized edge features:
        - Clip final perturbed values within the `epsilon`-ball around the **original normalized edge features**.
    - **Scaling:** Perturbations are applied directly to the **already normalized edge features**. No further scaling/inverse scaling is required within the attack itself.
    - **Evaluation Strategy:** Generate perturbed edge features, pass clean and perturbed features through the frozen DGI encoder to get embeddings, then evaluate the performance of the chosen anomaly detector (e.g., CBLOF) on both sets of embeddings.
    - **Results:** Attack successfully degraded model performance. Evaluation using CBLOF (n_clusters=8, cont=0.01) showed minimal impact at \\(\\epsilon=0.01\\) (Acc ~0.807 vs ~0.807 clean) but a sharp drop and saturation at \\(\\epsilon=0.05\\) (Acc ~0.212), \\(\\epsilon=0.10\\) (Acc ~0.214), and \\(\\epsilon=0.20\\) (Acc ~0.211). This indicates a sharp vulnerability threshold for this attack vector and detector combination.

### 5.2 Black-Box (Structural Modification) - **Applied to k-NN Graph for CAGN-GAT Evaluation**
    - **Rationale:** Evaluate robustness against attacks where the adversary has no knowledge of the target model internals. Focus on direct structural manipulations of the k-NN graph (relevant for CAGN-GAT evaluation). Black-box attacks on the IP:Port graph for EGraphSage TBD.

### 5.3 Black-Box (Structural Modification) - Applied to IP:Port Graph (EGraphSage, Anomal-E) - **Implemented & Evaluated (EGraphSage), Planned (Anomal-E)**
    - **Rationale:** Evaluate robustness against attacks where the adversary has limited knowledge of the target model, focusing on manipulations relevant to the IP:Port graph structure. This represents scenarios like traffic dropping/jamming (Edge Removal) or IP spoofing/Sybil attacks (Node Injection).
    - **Target Models:** `EGraphSage`, `Anomal-E`.
    - **Target Graph:** IP:Port graph constructed from the standardized test dataset split.
    - **Methods:**
        - **Edge Removal (EGraphSage):**
            - **Mechanism:** Randomly remove a specified percentage (`rate`) of edges (flows) from the clean test graph. This simulates network disruptions or data loss.
            - **Parameters:** `rate` values = `[0.05, 0.10, 0.20, 0.30]`.
            - **Implementation:** Implemented in `EGraphSage_Evaluation/egraphsage_adversarial_eval.py`.
            - **Results (EGraphSage):** Showed **minor performance degradation**, suggesting EGraphSage is relatively robust to random edge drops on this graph structure.
        - **Node Injection (EGraphSage):**
            - **Mechanism:** Add a specified percentage (`rate`) of new nodes (representing fake IPs) to the graph. Connect these new nodes by injecting new edges (flows) to existing nodes. Features for injected nodes/edges were sampled from existing attack instances.
            - **Parameters:** `rate` values = `[0.05, 0.10, 0.20]`, `connections_per_node = 5`.
            - **Implementation:** Implemented in `EGraphSage_Evaluation/egraphsage_adversarial_eval.py`. *Note: Feature generation strategy uses sampling from existing attack edges.*
            - **Results (EGraphSage):** Showed **significant and roughly linear performance degradation** as the injection rate increased, indicating a vulnerability to this type of structural attack.
        - **Edge Removal (Anomal-E):** Planned.
        - **Node Injection (Anomal-E):** Planned.

## 6. Implementation Tool
- **Library:** DeepRobust (for PGD), standard libraries (`torch`, `numpy`, `sklearn`, `torch_geometric`) for structural attacks and data handling.
- **Backend:** PyTorch
- **Integration:** Utilize DeepRobust's PGD implementation. Implement structural attacks via graph manipulation on PyG `Data` objects.

## 7. Methodology (Decoupled by Model/Graph Type)

### 7.1 CAGN-GAT Evaluation (Using `CAGN_GAT_Evaluation/cagn_gat_adversarial_eval.py`)

**(Status: Generation and Evaluation for CAGN-GAT Completed Successfully)**

    - **Implementation Details:**
        - The script `cagn_gat_adversarial_eval.py` was finalized to include data sampling (using the `create_imbalanced_subset` logic from the CAGN-GAT training notebook) and train/test masking on the sampled k-NN graph.
        - Attacks were performed on the test subgraph extracted using the test mask.
        - **PGD Attack:** Due to persistent compatibility issues with `deeprobust.graph.global_attack.PGDAttack`, a manual PGD feature attack was implemented using PyTorch AutoGrad. This attack targeted the specified numerical features (`FEATURES_TO_PERTURB`) directly on their scaled (log1p + StandardScaler) representation. Final parameters: `iterations=100`, `epsilon=[0.3, 0.5, 0.7]`.
        - **Black-Box Attacks:** Node Injection and Edge Removal were performed on the k-NN test subgraph.
    - **Results:**
        - **PGD:** The attack showed minimal impact at lower epsilon values (e.g., eps=0.1, 0.2). Higher values significantly degraded performance (F1 dropping from ~0.83 clean to ~0.69 at eps=0.5 and ~0.62 at eps=0.7), demonstrating vulnerability to strong feature perturbations.
        - **Node Injection:** Showed moderate impact, reducing F1 score progressively with higher injection rates.
        - **Edge Removal:** Showed relatively minor impact, suggesting some robustness to random edge drops in the k-NN graph.
    - **Outputs:** Attacked test subgraphs were saved to `GNN/Adversarial Evaluation/CAGN_GAT_Evaluation/Attacked_Data/` and evaluation results to `GNN/Adversarial Evaluation/CAGN_GAT_Evaluation/Results/adversarial_results.csv`.

**Phase 1: Attack Generation (k-NN Graph)**
1.  **Load Data & Preprocess:** Load the full dataset, apply sampling (if used in training), split into train/test.
2.  **Feature Engineering:** Apply Log1p+Scale and OHE to the test set features based on the CAGN-GAT training. Load the full feature list (`NODE_FEATURES`) from `CAGN_GAT_Evaluation/cagn_gat_feature_list.pkl`. Save the StandardScaler used.
3.  **Build Clean Test Graph:** Construct the k-NN similarity graph (`test_data_clean.pt`) using specified parameters (k=20, metric=euclidean). Save in `CAGN_GAT_Evaluation/Attacked_Data/`.
4.  **Generate & Save White-Box Data (for CAGN-GAT):**
    - Load the pre-trained `CAGN-GAT` model (`best_cagn_model_{dataset}.pt`).
    - For each `epsilon` in `PGD_EPSILON_LIST`:
        - Run `run_pgd_attack` using the `CAGN-GAT` model, `test_data_clean`, `epsilon`, feature indices, and the `scaler`.
        - Apply constraints (non-negativity) within the attack function, then re-apply scaling.
        - Save the resulting perturbed `Data` object in `CAGN_GAT_Evaluation/Attacked_Data/` (e.g., `test_data_pgd_eps{eps}_CAGN_GAT.pt`).
5.  **Generate & Save Black-Box Data (on k-NN graph):**
    - **Node Injection:** For each `rate`, run `run_node_injection_attack` on `test_data_clean` and save in `CAGN_GAT_Evaluation/Attacked_Data/` (e.g., `test_data_node_inject_{rate*100}pct.pt`).
    - **Edge Removal:** For each `rate`, run `run_edge_removal_attack` on `test_data_clean` and save in `CAGN_GAT_Evaluation/Attacked_Data/` (e.g., `test_data_edge_remove_{rate*100}pct.pt`).

**Phase 2: Evaluation (CAGN-GAT Model)**
1.  **Load Attacked Data:** Load all saved `.pt` files from `CAGN_GAT_Evaluation/Attacked_Data/` (`clean`, `pgd_CAGN_GAT*`, `node_inject_*`, `edge_remove_*`).
2.  **Evaluate CAGN-GAT Model:**
    - Load the pre-trained `CAGN-GAT` model.
    - Evaluate on `test_data_clean` (baseline).
    - Evaluate on its corresponding PGD attacked datasets (`test_data_pgd_eps{eps}_CAGN_GAT.pt`).
    - Evaluate on all black-box attacked datasets (`node_inject_*`, `edge_remove_*`).
3.  **Collect & Report Metrics:**
    - **Performance:** Accuracy, Precision, Recall, F1-Score (binary average), AUC. Compare performance on clean vs. adversarial data for the CAGN-GAT model against different attack types/parameters. Save results to `CAGN_GAT_Evaluation/Results/adversarial_results.csv`.

### 7.2 EGraphSage Evaluation (Requires Dedicated Script/Process) - **Implemented (Edge Attack)**

**(Status: PGD Edge Attack, Edge Removal, Node Injection Implemented & Evaluated)**

**Phase 1: Attack Generation (IP:Port Graph, Targeting Edge Features)**
1.  **Load Data & Preprocess:** Load the standardized test dataset split. Fit `TargetEncoder` and `StandardScaler` on train split.
2.  **Build Clean Test Graph:** Construct the IP:Port node graph (`test_data_clean_ipport.pt`) using `nx.from_pandas_edgelist`. Initialize node features (`g.ndata[\'h\']`) as `torch.ones`. Store scaled edge features (`g.edata[\'h\']`). Save clean graph.
3.  **Generate & Save White-Box Data (PGD Edge Attack):**
    - Load the pre-trained `EGraphSage` model (`best_model.pt`).
    - For each `epsilon` in `PGD_EPSILON_LIST`:\n        - Run `run_pgd_attack_egraphsage_edge` using the `EGraphSage` model, the clean IP:Port graph, and `epsilon`.\n        - The attack function calculates loss based on edge predictions, computes gradients w.r.t *scaled edge features*, applies iterative updates with clipping to edge features.\n        - Save the resulting graph object containing original node features (`g.ndata[\'h\']`) and *perturbed edge features* (`g.edata[\'h_perturbed\']`) in `EGraphSage_Evaluation/Attacked_Data/` (e.g., `test_data_pgd_eps{eps}_EGraphSage_EdgeAttack.pt`).\n4.  **Generate & Save Black-Box Data (for EGraphSage):**
    - **Edge Removal:** For each `rate` in `EDGE_REMOVAL_RATES`, run `run_edge_removal_attack_ipport` on the clean IP:Port graph and save in `EGraphSage_Evaluation/Attacked_Data/` (e.g., `test_data_edge_remove_{int(rate*100)}pct_ipport.pt`).
    - **Node Injection:** For each `rate` in `NODE_INJECTION_RATES`, run `run_node_injection_attack_ipport` on the clean IP:Port graph and save in `EGraphSage_Evaluation/Attacked_Data/` (e.g., `test_data_node_inject_{int(rate*100)}pct_ipport.pt`).

**Phase 2: Evaluation (EGraphSage Model)**
1.  **Load Attacked Data:** Load saved `.pt` files from `EGraphSage_Evaluation/Attacked_Data/` (`clean_ipport`, `pgd_EGraphSage_EdgeAttack*`, `edge_remove_*`, `node_inject_*`).
2.  **Evaluate EGraphSage Model:**
    - Load the pre-trained `EGraphSage` model.
    - Evaluate on the clean IP:Port graph (using original node and edge features).
    - Evaluate on the PGD attacked datasets (using original node features `g.ndata[\'h\']` and *perturbed edge features* `g.edata[\'h_perturbed\']`).
    - Evaluate on the Edge Removal attacked datasets (using original features on the reduced graph).
    - Evaluate on the Node Injection attacked datasets (using original features on the augmented graph).
3.  **Collect & Report Metrics:**
    - **Performance:** Accuracy, Precision, Recall, F1-Score (binary average), AUC.
    - Compare performance on clean vs. PGD, Edge Removal, and Node Injection adversarial data for the EGraphSage model.
    - Save results to a dedicated file (`EGraphSage_Evaluation/Results/adversarial_results.csv`).

### 7.3 Anomal-E Evaluation (Requires Dedicated Script/Process) - **Planned**

**Phase 1: Attack Generation (IP:Port Graph, Targeting Edge Features via DGI Loss)**
1.  **Load Data & Preprocess:** Load the standardized test dataset split. Fit `TargetEncoder` and `Normalizer` on train split.
2.  **Build Clean Test Graph:** Construct the IP:Port node graph (`test_data_clean_anomale.pt`) using `nx.from_pandas_edgelist`. Initialize node features (`g.ndata['h']`) as `torch.ones`. Store normalized edge features (`g.edata['h']`). Save clean graph.
3.  **Generate & Save White-Box Data (PGD on DGI Loss):**
    - Load the pre-trained DGI model (`best_dgi.pkl`).
    - For each `epsilon` in `PGD_EPSILON_LIST`:
        - Implement and run a PGD attack function targeting the DGI loss.
        - The attack function calculates the DGI loss, computes gradients w.r.t *normalized edge features*, applies iterative updates with clipping to edge features.
        - Save the resulting graph object containing original node features (`g.ndata['h']`) and *perturbed edge features* (`g.edata['h_perturbed']`) in `AnomalE_Evaluation/Attacked_Data/` (e.g., `test_data_pgd_eps{eps}_AnomalE_DGI.pt`).
4.  **(Optional) Generate & Save Black-Box Data:** Plan and implement black-box attacks suitable for the IP:Port graph if desired later.

**Phase 2: Evaluation (Anomal-E: DGI Embeddings + Anomaly Detector)**
1.  **Load DGI Encoder & Attacked Data:** Load the pre-trained DGI model (`best_dgi.pkl`). Load saved `.pt` files (`clean_anomale`, `pgd_AnomalE_DGI*`).
2.  **Generate Embeddings:**
    - Pass the clean graph through the DGI encoder (`dgi.encoder`) to get clean test embeddings.
    - Pass each attacked graph through the DGI encoder to get perturbed test embeddings.
    - Load the *clean training embeddings* generated during the original Anomal-E training run (or regenerate them).
3.  **Train & Evaluate Anomaly Detector:**
    - Select the best-performing anomaly detector configuration from the Anomal-E notebook (e.g., CBLOF with n_clusters=8, contamination=0.01).
    - Train the detector on the *clean training embeddings*.
    - Evaluate the trained detector on the *clean test embeddings* (baseline performance).
    - Evaluate the trained detector on each set of *perturbed test embeddings*.
4.  **Collect & Report Metrics:**
    - **Performance:** Accuracy, Precision, Recall, F1-Score (binary average), AUC.
    - Compare performance on clean vs. adversarial embeddings.
    - Save results to a dedicated file (e.g., `AnomalE_Evaluation/Results/adversarial_results.csv`).

## 8. Expected Outcomes
- Quantitative assessment of the robustness of the **CAGN-GAT** model against specific adversarial attacks on the **k-NN similarity graph**.
- Measurement of performance degradation under white-box (feature perturbation) and black-box (structural) scenarios for CAGN-GAT.
- Identification of potential CAGN-GAT vulnerabilities related to specific NetFlow features targeted by PGD.
- Insights specific to CAGN-GAT's robustness on this graph type, which may inform defenses.
- **(Completed)** Quantitative assessment of the robustness of the **EGraphSage** model against white-box PGD attacks (targeting **edge features**) on the **IP:Port graph**.\n- **(Completed)** Measurement of EGraphSage performance degradation under white-box feature perturbation.\n- Identification of potential EGraphSage vulnerabilities related to its reliance on specific scaled edge features targeted by PGD on its native graph structure.\n- (Note: Direct comparison between CAGN-GAT and EGraphSage requires careful consideration due to different graph structures and potentially different feature processing).\n- **(Planned)** Quantitative assessment of the robustness of the **Anomal-E** approach (DGI encoder + anomaly detector) against white-box PGD attacks targeting the **DGI encoder\'s loss function** on the **IP:Port graph**.\n- **(Planned)** Measurement of the downstream anomaly detector\'s performance degradation when operating on perturbed embeddings.\n- **(Completed for EGraphSage, Planned for Anomal-E)** Quantitative assessment of the robustness against black-box structural attacks (Edge Removal, Node Injection) on the **IP:Port graph**. EGraphSage showed minor vulnerability to edge removal but significant vulnerability to node injection.
