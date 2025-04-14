# Design Document: LLM-Based Mitigation for Node Injection Attacks

**Author:** AI Assistant (Pair Programming)
**Date:** 2024-07-26
**Version:** 1.0

## 1. Goal

To develop and evaluate a defense mechanism against node injection attacks on Graph Neural Networks (GNNs) trained on network flow data. This mechanism utilizes a Large Language Model (LLM) acting as a cybersecurity expert to identify and flag artificially injected nodes within a graph representation of the network traffic.

## 2. Methodology

The process involves sampling data, constructing a graph, injecting nodes, using an LLM for detection, mitigating the attack by removing flagged nodes, and evaluating the impact on GNN performance.

### 2.1. Data Sampling and Preprocessing

*   **Source Dataset:** `/media/ssd/test/standardized-datasets/combined/combined_unsw_cicRed_botRed_netflow_10pct.csv`
*   **Sampling:** Extract a small, fixed-size subset (e.g., 100-500 samples) from the source dataset. This is necessary for manageable LLM processing time. Stratified sampling based on the 'Label' column will be attempted to preserve the original class distribution, falling back to random sampling if stratification isn't feasible (e.g., due to small class sizes).
*   **Feature Engineering:** Apply the same feature engineering steps used for training the target GNN model (CAGN). This includes handling numerical (log1p transformation, scaling) and categorical features (one-hot encoding), ensuring consistency with the expected feature list (`cagn_gat_feature_list.pkl`).

### 2.2. Graph Construction

*   **Technique:** Build a k-Nearest Neighbors (k-NN) similarity graph based on the engineered features of the sampled data points (nodes).
*   **Parameters:** Use parameters consistent with the original CAGN model training (e.g., `k_neighbors=20`, `metric='euclidean'`). Edges represent high similarity between network flows.

### 2.3. Attack Simulation: Node Injection

*   **Attack Type:** Node Injection.
*   **Injection Rate:** Inject a predefined number of synthetic nodes (e.g., 20 nodes, representing 20% of a 100-node subset).
*   **Injected Node Features:** Assign features to injected nodes using a simple strategy (e.g., the mean feature vector of the original nodes in the subset).
*   **Injected Node Connectivity:** Connect each injected node to a small, fixed number (`k_connections`, e.g., 5) of randomly selected original nodes in the graph. Add symmetric edges (Original -> Injected).
*   **Injected Node Labels:** Assign a benign label (e.g., 0) to injected nodes. The LLM's task is to identify them based on features and context, not the assigned label.

### 2.4. LLM-Based Detection and Mitigation

*   **LLM Role:** The LLM acts as a cybersecurity analyst.
*   **Prompting Strategy:** Construct a detailed prompt **for each node in the graph subset (original and injected)** containing:
    *   **Role & Task Definition:** Clearly instruct the LLM on its role, the context (graph of network flows, potential anomalies), and the task (identify nodes that appear anomalous or inconsistent within the graph context, provide confidence scores). **Crucially, do not label nodes as 'injected' or 'synthetic' in the prompt.**
    *   **Feature Context:** Provide descriptions of key network flow features (e.g., `IN_BYTES`, `PROTOCOL`, `FLOW_DURATION_MILLISECONDS`).
    *   **Node Analysis Section:** For *each* node being analyzed:
        *   Provide its unique ID within the graph.
        *   Present its **raw feature vector** (if available, i.e., for original nodes) and its **processed feature vector snippet** (scaled/encoded).
        *   List its connections (edges) to neighbors within the graph subset.
        *   For a small sample of these neighbors (e.g., 3-5), display their node IDs and their **raw feature values** (like Protocol Name, Flags, Bytes, Duration) to give the LLM concrete traffic context for comparison.
    *   **Analysis Focus:** Explicitly instruct the LLM to primarily compare the **raw features** of the target node (if available) against the **raw features** of its neighbors to assess consistency within the local neighborhood. Mention that processed features are for GNN model use and might differ.
    *   **Output Format Specification:** Explicitly request the output as a single JSON object containing `node_id`, `confidence_score` (0.0-1.0, defining 0 as normal/consistent and 1 as anomalous/inconsistent), **and a brief textual `justification`** for the score.
*   **LLM Interaction:**
    *   Use an API call (ideally using API keys from environment variables).
    *   Specify JSON output format if the API supports it.
    *   **Aggregate Prompts:** Collect all generated prompts into a single file for review.
    *   **Aggregate Responses:** Collect all raw and parsed JSON responses from the LLM into single files.
    *   Implement robust parsing for the expected JSON structure from each call. Handle potential errors, malformed responses, or missing scores gracefully (e.g., assign a default score of 0.0 if analysis fails for a node).
*   **Mitigation:**
    *   Define a `CONFIDENCE_THRESHOLD` (e.g., 0.6).
    *   Identify **any node (original or injected)** whose `confidence_score` from the LLM meets or exceeds this threshold.
    *   Create a "fixed" graph by removing the flagged nodes and their associated edges from the graph subset.

### 2.5. Evaluation

*   **Model:** Load the pre-trained CAGN GNN model (`best_cagn_model_Combined_10pct.pt`).
*   **Graphs:** Evaluate the model's performance on three graph instances:
    1.  `Clean Graph`: The original graph built from the data subset before injection.
    2.  `Injected Graph`: The graph after node injection.
    3.  `LLM-Fixed Graph`: The graph after LLM-identified nodes have been removed.
*   **Metrics:** Calculate standard classification metrics (Accuracy, Precision, Recall, F1-Score, AUC) for each graph.
*   **Goal:** Compare metrics across the three graphs to assess:
    *   The impact of the node injection attack on GNN performance.
    *   The effectiveness of the LLM-based mitigation strategy in restoring performance.

## 3. Implementation Details

*   **Script:** `GNN/LLM_Mitigation_Test/llm_cagn_basic_test.py` (To be generalized to support multiple models).
*   **Model Paths:**
    *   CAGN: `/media/ssd/test/GNN/Standardized Models/CAGN-GAT/best_cagn_model_Combined_10pct.pt`
    *   Anomal-E: `/media/ssd/test/GNN/Standardized Models/AnomalE/Combined/best_dgi_eval_script.pkl` (Note: `.pkl` format, likely includes DGI model state and potentially the detector).
    *   EGraphSage: `/media/ssd/test/GNN/Standardized Models/E-GraphSage/best_model.pt`
*   **Feature Lists (Engineered):**
    *   Analysis of the training notebooks revealed that CAGN, Anomal-E, and EGraphSage likely use **different sets of engineered features**.
    *   These specific lists (columns after model-specific preprocessing like target encoding, but before scaling/normalization if applied) must be saved as separate `.pkl` files (e.g., `cagn_gat_feature_list.pkl`, `anomale_feature_list.pkl`, `egraphsage_feature_list.pkl`).
    *   **Crucially, the generalized script must load the correct feature list corresponding to the selected model type.**
*   **Generalization Plan:**
    *   Refactor the script to accept command-line arguments (`argparse`) for `model_type`, `model_path`, and `feature_list_path`.
    *   Implement logic to load the appropriate model architecture and state dictionary based on `model_type`.
    *   Ensure the `feature_engineer` function uses the feature list specified by `feature_list_path`.
    *   Update output file naming to include the `model_type`.
*   **Key Parameters (Configurable - potentially via args):**
    *   `NUM_SAMPLES`: Size of the data subset.
    *   `NUM_INJECTED_NODES`: Number of nodes to inject.
    *   `INJECTION_K_CONNECTIONS`: Number of connections for each injected node.
    *   `LLM_MODEL`: Identifier for the LLM to be used (e.g., "gpt-4o").
    *   `OPENAI_API_KEY`: API key (use secure management in production).
    *   `CONFIDENCE_THRESHOLD`: Threshold for flagging nodes based on LLM score.
*   **Output:** Save evaluation metrics, LLM prompts/responses, and processed graphs for analysis.

## 4. Discussion Points

*   **LLM Prompt Effectiveness:** Is the current prompt structure (`format_prompt_for_llm`) providing the right level of detail **for identifying general anomalies**? Does the LLM benefit significantly from raw neighbor features vs. just graph structure when it lacks the "injected" hint?
*   **Injected Feature Strategy:** Using 'mean' features for injected nodes is simple. Would a different strategy (e.g., sampling features from specific classes, perturbing existing malicious node features) create more realistic/challenging injected nodes **that are harder for the LLM to distinguish without explicit labels**?
*   **Scalability:** The current approach analyzes every node in the subset via LLM. How could this be adapted for larger graphs (e.g., pre-filtering potentially anomalous nodes using GNN metrics or other heuristics before sending to the LLM)?
*   **Cost/Latency:** LLM API calls can be slow and costly, especially when analyzing every node. This limits real-time application but is acceptable for research/evaluation.
*   **False Positives:** Analyzing all nodes increases the risk of the LLM flagging original nodes as anomalous (false positives). The evaluation showed this occurred (14 false positives with a 0.8 threshold in the CAGN test), highlighting the trade-off between detecting all injected nodes and minimizing disruption to benign ones.
*   **Prompt Refinement Success:** The strategy of providing raw features for both the target node and its neighbors, and explicitly asking the LLM to compare raw-vs-raw for local consistency, proved much more effective than comparing processed vs. raw features.
*   **Injected Node Detectability:** With the refined prompt, the LLM consistently assigned high confidence scores (>=0.8) to injected nodes created using the 'mean feature' strategy, suggesting they are clearly distinguishable from their neighbors by the LLM, allowing for effective threshold-based removal.
*   **Feature Set Consistency:** It's critical to use the specific set of engineered features that each GNN model (CAGN, Anomal-E, EGraphSage) was trained on. Using mismatched feature sets during evaluation or mitigation would lead to invalid results. 