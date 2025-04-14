# Dataset Context

## Standard Datasets
- **BoT-IoT Dataset**
  - **Location**: `/media/ssd/test/GNN/kaggle/input/BoT-IoT/bot_reduced.csv`
  - **Format**: Standard form (reduced version)
  - **Description**: Network traffic data with various IoT-based attacks

- **CSE-CIC-IDS2018 Dataset**
  - **Location**: `/media/ssd/test/GNN/kaggle/input/CSE-CIC-IDS2018/CSE-CIC-ID2018-Converted.csv`
  - **Format**: Standard form
  - **Description**: Network traffic data with modern attack scenarios

- **UNSW-NB15 Dataset**
  - **Training Location**: `/media/ssd/test/GNN/kaggle/input/unsw-nb15/UNSW_NB15_training-set.csv`
  - **Testing Location**: `/media/ssd/test/GNN/kaggle/input/unsw-nb15/UNSW_NB15_testing-set.csv`
  - **Format**: Standard form (split into training and testing sets)
  - **Description**: Network traffic data with various attack types

## NetFlow Datasets
- **NF-BoT-IoT v1 Dataset**
  - **Location**: `/media/ssd/test/GNN/kaggle/input/NF-BoT-IoT/NF-BoT-IoT.csv`
  - **Format**: NetFlow format
  - **Description**: NetFlow version of the BoT-IoT dataset

- **NF-BoT-IoT v2 Dataset**
  - **Location**: `/media/ssd/test/GNN/kaggle/input/NF-BoT-IoT/NF-BoT-IoT-v2.csv`
  - **Format**: NetFlow format
  - **Description**: NetFlow version 2 of the BoT-IoT dataset with more features
  - **Size**: Much larger than v1, requires reduction for efficient processing

- **NF-CSE-CIC-IDS2018 Dataset**
  - **Location**: `/media/ssd/test/GNN/kaggle/input/NF-CSE-CIC-IDS2018-v2/NF-CSE-CIC-IDS2018-v2.csv`
  - **Format**: NetFlow format
  - **Description**: NetFlow version of the CSE-CIC-IDS2018 dataset

- **NF-UNSW-NB15 Dataset**
  - **Location**: `/media/ssd/test/GNN/kaggle/input/nf-unsw-nb15-v2/NF-UNSW-NB15-v2.csv`
  - **Format**: NetFlow format
  - **Description**: NetFlow version of the UNSW-NB15 dataset

## Processed Datasets

### Reduced Datasets
- **Location**: `/media/ssd/test/standardized-datasets/netflow/`
- **Key Files**:
  - `nf_bot_iot_v2_reduced.csv` - BoT-IoT v2 reduced with sampling (53MB)
  - `nf_cic_ids2018_reduced.csv` - CIC-IDS2018 reduced with sampling
- **Creation Process**: 
  - Generated using dedicated reduction scripts:
    - `reduce_bot_iot_v2.py`
    - `reduce_cic_ids2018.py`
    - `reduce_unsw_nb15.py`
  - Each script applies intelligent sampling that preserves attack distributions

### Standardized Datasets
- **Location**: `/media/ssd/test/standardized-datasets/netflow/`
- **Key Files**:
  - `nf_bot_iot_standardized.csv` - Standardized BoT-IoT v1 dataset
  - `nf_bot_iot_v2_standardized.csv` - Standardized BoT-IoT v2 dataset (full)
  - `nf_bot_iot_v2_reduced_standardized.csv` - Standardized BoT-IoT v2 reduced dataset (15MB)
  - `nf_cic_ids2018_standardized.csv` - Standardized CIC-IDS2018 dataset (full)
  - `nf_cic_ids2018_reduced_standardized.csv` - Standardized CIC-IDS2018 reduced dataset
  - `nf_unsw_nb15_standardized.csv` - Standardized UNSW-NB15 dataset (473MB)
- **Format**: Standardized CSV with consistent feature naming and preprocessing
- **Description**: Unified format compatible with all GNN models
- **Creation Process**: Generated using the common `standardize_datasets.py` script

### Combined Datasets
- **Location**: `/media/ssd/test/standardized-datasets/combined/`
- **Key Files**:
  - `combined_netflow.csv` - Combined full standardized datasets
  - `combined_netflow_reduced.csv` - Combined reduced standardized datasets (886MB)
    - Contains BoT-IoT v2 (reduced), CIC-IDS2018 (reduced), and UNSW-NB15 (not reduced)
- **Format**: Standardized CSV with dataset source identifier
- **Creation Process**: Generated using `combine_datasets.py` with appropriate flags

## Dataset Processing Workflow

The correct workflow for processing datasets follows these steps:

1. **Dataset Reduction** (Step 1):
   - Use dedicated reduction scripts for each dataset
   - Each script applies appropriate sampling strategies
   - Preserves attack distribution while reducing size
   - No standardization should occur during this step

2. **Dataset Standardization** (Step 2):
   - Use the common `standardize_datasets.py` for all datasets
   - Standardize both full datasets and reduced datasets
   - Only applies format standardization, not additional filtering
   - Adds necessary metadata (flow_id, dataset_source)

3. **Dataset Combination** (Step 3):
   - Use `combine_datasets.py` to merge standardized datasets
   - Can combine full datasets or reduced versions with `--reduced` flag
   - Use `--use-v2` flag to use BoT-IoT v2 instead of v1

## Standardization Process
- All standardization work is stored in the `/media/ssd/test/standardized-datasets` folder
- The standardized data is compatible with all three GNN models
- Dataset combination and intelligent sampling implemented for memory efficiency
- Source tracking added to identify dataset origins in the combined dataset

## NSL-KDD Dataset
- **Location**: `/media/ssd/test/GNN/kaggle/input/nslkdd/`
- **Key files**: 
  - `KDDTrain+.txt` - Full training set
  - `KDDTest+.txt` - Full test set
  - `KDDTrain+_20Percent.txt` - 20% subset for faster iterations
- **Features**: 41 features including basic, content, traffic, and host-based features
- **Classes**: 5 main attack categories (DoS, Probe, R2L, U2R, and Normal)
- **Advantages**: Improved version of KDD'99 with removed redundant records

## UNSW-NB15 Dataset
- **Location**: `/media/ssd/test/GNN/kaggle/input/unsw-nb15/`
- **Features**: 49 features representing flow-based and content features
- **Classes**: 9 attack types and normal traffic
- **Advantages**: More recent and realistic attack scenarios

## CICIDS2017 Dataset
- **Location**: `/media/ssd/test/GNN/kaggle/input/cicids2017/`
- **Key files**: CSVs containing network flows for different days
- **Features**: 78 network flow features
- **Classes**: Various attack types including DoS, DDoS, Brute Force, XSS, SQL Injection
- **Advantages**: Modern attack scenarios with full packet captures

## Data Preprocessing Steps
1. Feature scaling/normalization
2. Categorical encoding 
3. Graph representation conversion:
   - Nodes: Network connections or entities (IP addresses, ports)
   - Edges: Based on similarity, temporal, or communication relationships
   - Features: Connection attributes

## Cross-Dataset Preprocessing Plan
- Unified preprocessing pipeline implemented for consistent graph construction
- Standardized feature sets across datasets
- Comparable evaluation metrics for cross-dataset performance analysis
- Dataset-specific characteristics preserved for accurate modeling

## Dataset Combination Strategy
- Intelligent sampling to preserve attack distributions
- Stratification by both attack type and dataset source
- Higher sampling rates for rare attack classes
- Consistent tracking of data source throughout preprocessing

## Feature Engineering
- Feature selection based on domain knowledge
- Graph structure design decisions impact model performance
- Consider temporal aspects of network connections
- Consistent feature representation across datasets for fair comparison 