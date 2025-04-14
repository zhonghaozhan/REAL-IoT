# Project Progress

## Completed Tasks
- ✅ Environment setup and dataset acquisition
- ✅ Initial project structure established
- ✅ Dataset placement in appropriate directories
- ✅ Model checkpoints saved from previous runs
- ✅ Successfully set up gnn_cuda_env environment for GPU acceleration
- ✅ Successfully implemented the three main models:
  - E-GraphSage
  - CAGN-GAT
  - Anomal E (with DGI as its trained model)
- ✅ Comprehensive evaluation of E-GraphSage model:
  - Successfully tested on UNSW-NB15 dataset 
  - Successfully tested on CIC-IDS2018 dataset
  - Modified model architecture to handle 10 classes for multi-dataset compatibility
- ✅ Dataset combination and reduction:
  - Created an intelligent sampling approach that preserves attack distributions
  - Developed a script that reduces combined datasets while maintaining attack patterns
  - Successfully combined BoT-IoT, CIC-IDS, and UNSW-NB15 datasets with stratified sampling
- ✅ Anomal-E implementation and testing:
  - Successfully tested on UNSW-NB15 dataset
  - Successfully tested on BoT-IoT dataset
  - Successfully tested on combined dataset (with parameter adjustments for CBLOF)
- ✅ CAGN-GAT testing on the combined dataset
- ✅ CAGN-GAT Benchmark Training Completed (on combined dataset).
- ✅ Adversarial Evaluation for CAGN-GAT Completed:
  - Used data sampling and k-NN graph built on the test subgraph.
  - Implemented *manual* PGD feature attack (due to DeepRobust issues) with `iterations=100`, `epsilon=[0.3, 0.5, 0.7]` on scaled features.
  - Implemented Node Injection and Edge Removal attacks.
  - PGD (high epsilon) and Node Injection significantly degraded F1 score; Edge Removal had minor impact.
  - Results saved to `GNN/Adversarial Evaluation/CAGN_GAT_Evaluation/Results/adversarial_results.csv`.
- ✅ Adversarial Evaluation for EGraphSage Completed (Initial Attempt - Node Attack):
  - Implemented PGD attack targeting initial (ineffective) node features.
- ✅ Adversarial Evaluation for EGraphSage Completed (Revised - Edge Attack):
  - Implemented PGD attack targeting scaled edge features (effective).
  - Generated attacked data for epsilon = [0.3, 0.5, 0.7].
  - Evaluated EGraphSage performance on clean and edge-attacked data.
  - Saved results to `EGraphSage_Evaluation/Results/adversarial_results.csv`.
- ✅ Adversarial Evaluation for Anomal-E Completed:
  - Implemented PGD attack targeting DGI loss on edge features (epsilon=[0.01, 0.05, 0.1, 0.2]).
  - Trained a compatible DGI model using script's preprocessing.
  - Trained CBLOF detector on clean training embeddings.
  - Evaluated CBLOF on clean and attacked test embeddings.
  - Observed sharp performance drop at epsilon=0.05 (Acc ~0.81 -> ~0.21), indicating vulnerability.
  - Results saved to `AnomalE_Evaluation/Results/adversarial_results.csv`.

## In Progress
- 🔄 Memory optimization for handling larger graph structures
- 🔄 Evaluating model robustness against adversarial attacks (White-box Completed, Planning Black-box?)

## Pending Tasks
- ⏳ Implement black-box adversarial attacks (e.g., structural attacks on IP:Port graph?)?
- ⏳ Complete data preprocessing pipeline for ToN IoT dataset
- ⏳ Benchmark testing for all models (E-GraphSage, CAGN-GAT, Anomal E) on unified dataset (post-robustness evaluation)
- ⏳ Analysis of model generalization capabilities across different attack types
- ⏳ Document findings and insights from cross-dataset evaluation
- ⏳ Compare performance metrics across different architectures

## Known Issues
- 🔴 Path resolution in notebook code (using absolute vs. relative paths)
- 🔴 Notebook editing access via SSH
- 🟡 Need to ensure consistent preprocessing across different runs
- 🟡 CUDA memory limitations when processing large graphs (implemented sampling solution)
- 🟡 CBLOF clustering requires parameter adjustments for combined datasets (solved by increasing n_clusters and adjusting alpha/beta)

## Results So Far
- E-GraphSage model successfully tested on multiple datasets showing promising results
- Successfully adapted model to handle multi-class classification (10 classes)
- Developed efficient sampling strategy for handling combined datasets while preserving attack distributions
- Framework established for cross-dataset evaluation
- Model checkpoints and results stored appropriately
- Anomal-E implementation successfully tested on individual and combined datasets
- Identified and resolved CBLOF clustering issues for heterogeneous data distributions
- CAGN-GAT implementation successfully tested on combined dataset

## Future Directions
- Compare performance across the three GNN architectures
- Test model generalization capabilities across unified dataset
- Evaluate models on different types of network intrusion data
- Explore hybrid approaches combining GNNs with other techniques
- Investigate explainability methods for GNN-based intrusion detection
- Assess model robustness and potential defense strategies 