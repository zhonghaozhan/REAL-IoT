# Active Workflow

## Current Focus
- 🔄 **Evaluate GNN Robustness:** Assess the resilience of trained models (CAGN-GAT, E-GraphSage, Anomal-E) against adversarial attacks using the `combined_unsw_cicRed_botRed_netflow_10pct.csv` test set, following the plan in `adversarial_attack_plan.md`.
- Implement White-Box (PGD feature perturbation) and Black-Box (Direct Random Node Injection/Edge Removal) attacks.
- Install and configure DeepRobust library for attack implementation support.
- ✅ **Evaluate GNN Robustness (CAGN-GAT):** Completed evaluation against manual PGD feature attacks and structural attacks (Node Injection, Edge Removal) on the k-NN test subgraph. Results in `adversarial_attack_plan.md` and `progress.md`.
- 🔄 **Evaluate GNN Robustness (Anomal-E):** Assess the resilience of the Anomal-E model against adversarial attacks using the `combined_unsw_cicRed_botRed_netflow_10pct.csv` test set, following the plan in `adversarial_attack_plan.md`.
- Implement White-Box PGD attack targeting the DGI loss.
- Implement Black-Box Node Injection and Edge Removal attacks.

## Standard Dataset Standardization
- Using standard format datasets:
  - BoT-IoT: `/media/ssd/test/GNN/kaggle/input/BoT-IoT/bot_reduced.csv`
  - CSE-CIC-IDS2018: `/media/ssd/test/GNN/kaggle/input/CSE-CIC-IDS2018/CSE-CIC-ID2018-Converted.csv`
  - UNSW-NB15: Training and testing sets at
    - `/media/ssd/test/GNN/kaggle/input/unsw-nb15/UNSW_NB15_training-set.csv`
    - `/media/ssd/test/GNN/kaggle/input/unsw-nb15/UNSW_NB15_testing-set.csv`

## NetFlow Dataset Standardization
- Using NetFlow format datasets:
  - NF-BoT-IoT v1: `/media/ssd/test/GNN/kaggle/input/NF-BoT-IoT/NF-BoT-IoT.csv`
  - NF-BoT-IoT v2: `/media/ssd/test/GNN/kaggle/input/NF-BoT-IoT/NF-BoT-IoT-v2.csv`
  - NF-CSE-CIC-IDS2018: `/media/ssd/test/GNN/kaggle/input/NF-CSE-CIC-IDS2018-v2/NF-CSE-CIC-IDS2018-v2.csv`
  - NF-UNSW-NB15: `/media/ssd/test/GNN/kaggle/input/nf-unsw-nb15-v2/NF-UNSW-NB15-v2.csv`

## Dataset Processing Workflow
The correct workflow for processing datasets consists of these separate, sequential steps:

1. **Dataset Reduction** - Using dedicated scripts:
   - `reduce_bot_iot_v2.py`: Reduces BoT-IoT v2 dataset with stratified sampling
   - `reduce_cic_ids2018.py`: Reduces CIC-IDS2018 dataset
   - `reduce_unsw_nb15.py`: Reduces UNSW-NB15 dataset

2. **Dataset Standardization** - Using the common standardization script:
   - `standardize_datasets.py`: Standardizes both full and reduced datasets
   - Ensures consistent column names, data types, and structure
   - No additional filtering should occur during standardization

3. **Dataset Combination** - Using the dataset combining script:
   - `combine_datasets.py`: Combines standardized datasets with options:
     - `--reduced`: Use reduced versions of datasets 
     - `--use-v2`: Use BoT-IoT v2 instead of v1

## Standardized Datasets
- All datasets successfully standardized in `/media/ssd/test/standardized-datasets/netflow/`
  - `nf_bot_iot_standardized.csv`
  - `nf_bot_iot_v2_standardized.csv`
  - `nf_bot_iot_v2_reduced_standardized.csv`
  - `nf_cic_ids2018_standardized.csv`
  - `nf_cic_ids2018_reduced_standardized.csv`
  - `nf_unsw_nb15_standardized.csv`
- Combined datasets available in `/media/ssd/test/standardized-datasets/combined/`:
  - `combined_netflow.csv`: Combined full datasets
  - `combined_netflow_reduced.csv`: Combined reduced datasets (BoT-IoT v2, CIC reduced, UNSW)

## Recent Progress
- Successfully tested E-GraphSage on UNSW-NB15 dataset
- Successfully tested E-GraphSage on CIC-IDS2018 dataset
- Successfully tested Anomal-E on UNSW-NB15, BoT-IoT, and combined datasets
- Resolved CBLOF clustering issues for combined datasets by parameter tuning
- Adapted E-GraphSage model to handle 10 classes for multi-dataset compatibility
- Created an intelligent dataset sampling approach that preserves attack distributions
- Developed script to combine and reduce datasets while maintaining attack patterns
- Successfully combined BoT-IoT v2, CIC-IDS, and UNSW-NB15 datasets with stratified sampling

## Recent Achievements
- Successfully completed Adversarial Evaluation for EGraphSage (PGD Edge Attack, Node Injection, Edge Removal). Found significant vulnerability to Node Injection.
- Successfully replicated and verified three advanced models:
  - E-GraphSage (tested on multiple datasets including combined dataset)
  - Anomal-E (tested on multiple datasets including combined dataset)
  - CAGN-GAT (implementation in progress for combined dataset)
- Created memory bank for persistent context
- Set up gnn_cuda_env environment for GPU acceleration
- Developed advanced dataset combination strategy for cross-dataset evaluation

## Active Issues
1. **CUDA Memory Limitations**
   - Problem: Large graphs consume excessive CUDA memory
   - Solution: Implemented intelligent sampling strategy that preserves attack distributions
   - Status: Successfully addressed, monitoring for further optimizations

2. **Path Resolution**
   - Problem: Code using absolute paths (`/kaggle/input/...`) fails
   - Solution: Update to relative paths (`kaggle/input/...`)
   - Status: Partially implemented, ongoing

3. **Dataset Standardization Consistency**
   - Problem: Inconsistent processing of BoT-IoT v2 dataset
   - Solution: Clarified the proper workflow separating reduction and standardization
   - Status: Resolved, using correct workflow going forward

4. **CBLOF Clustering for Combined Datasets**
   - Problem: CBLOF algorithm in Anomal-E fails to form valid cluster separation on combined dataset
   - Solution: Increased n_clusters values (8,12,15,20,25,30) and adjusted alpha/beta parameters (0.7/3) 
   - Status: Resolved, successfully applied to combined dataset

## Next Steps
1. Analyze and document robustness results across all evaluated models.
2. (Optional/Secondary) Implement and evaluate transfer attacks.

**Completed for EGraphSage:** Attack generation & evaluation (PGD Edge, Node Inj., Edge Rem.).
**Completed for CAGN-GAT:** Attack generation & evaluation (Manual PGD Feature, Node Inj., Edge Rem.).
**Completed for Anomal-E:** Attack generation & evaluation (PGD DGI Loss).

## Current Environment State
- Running on remote server via SSH
- Using gnn_cuda_env for GPU acceleration
- Three advanced models: E-GraphSage and Anomal-E fully tested; CAGN-GAT in progress
- All standardized datasets ready for evaluation
- Combined dataset available with consistent processing
- Memory optimization strategies implemented 