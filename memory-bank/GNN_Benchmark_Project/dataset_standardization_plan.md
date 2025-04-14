# Dataset Standardization Plan for Cross-Model Testing

## Overview
This plan outlines the standardization process for testing network intrusion detection models (Anomal_E, CAGN-GAT, and E-GraphSAGE) across different NetFlow datasets.

### NetFlow Datasets
- NF-BoT-IoT v1: `/media/ssd/test/GNN/kaggle/input/NF-BoT-IoT/NF-BoT-IoT.csv`
- NF-BoT-IoT v2: `/media/ssd/test/GNN/kaggle/input/NF-BoT-IoT/NF-BoT-IoT-v2.csv`
- NF-CSE-CIC-IDS2018: `/media/ssd/test/GNN/kaggle/input/NF-CSE-CIC-IDS2018-v2/NF-CSE-CIC-IDS2018-v2.csv`
- NF-UNSW-NB15: `/media/ssd/test/GNN/kaggle/input/nf-unsw-nb15-v2/NF-UNSW-NB15-v2.csv`

## Important Guidelines
- All standardization work must be contained in the `/media/ssd/test/standardized-datasets/` folder
- The standardized datasets should be compatible with all three models
- **IMPORTANT CHANGE**: We are now focusing exclusively on NetFlow data and dropping all work on standard format datasets
- **CRUCIAL WORKFLOW CLARIFICATION**: Dataset reduction and standardization should be separate, sequential steps:
  1. First reduce datasets using dedicated reduction scripts (e.g., `reduce_bot_iot_v2.py`)
  2. Then standardize all datasets (both full and reduced) using the common `standardize_datasets.py` script

## Data Standardization Requirements

Based on the examination of the GNN models (Anomal_E, CAGN-GAT, and E-GraphSAGE), the standardization will focus on:

1. **Common Feature Set**: Identify and standardize common NetFlow features across all datasets
2. **Graph Construction**: Ensure consistent approach for creating network graphs from NetFlow data
3. **Label Standardization**: Create unified labeling scheme across datasets
4. **Data Scaling/Normalization**: Apply consistent preprocessing
5. **Feature Engineering**: Extract common derived features where needed

## Model-Dataset Feature Alignment Analysis

### Core Features Required by All Models

The standardized NetFlow format requires these essential features:
- `IPV4_SRC_ADDR`, `L4_SRC_PORT`: Source IP address and port
- `IPV4_DST_ADDR`, `L4_DST_PORT`: Destination IP address and port
- `PROTOCOL`: Protocol number
- `L7_PROTO`: Layer 7 protocol information
- `IN_BYTES`, `OUT_BYTES`: Bytes transferred in each direction
- `IN_PKTS`, `OUT_PKTS`: Packet counts in each direction
- `TCP_FLAGS`: TCP flags if applicable
- `FLOW_DURATION_MILLISECONDS`: Flow duration
- `Label`: Binary classification (attack/normal)
- `Attack`: Attack classification
- `flow_id`: Unique identifier for each flow (to be generated)
- `dataset_source`: Source dataset identifier (to be added)

### NetFlow Data Compatibility Analysis

After analyzing the NetFlow datasets, we've identified excellent compatibility across all three datasets:

1. **NetFlow Format Consistency**:
   - BoT-IoT uses NetFlow_v1 (12 core features)
   - CSE-CIC-IDS2018 and UNSW-NB15 use NetFlow_v2 (44 features that include all v1 features)
   - NetFlow_v2 is a superset of NetFlow_v1, providing backward compatibility

2. **Common Core Features Across All NetFlow Datasets**:
   - Source/destination identifiers (IPV4_SRC_ADDR, IPV4_DST_ADDR, L4_SRC_PORT, L4_DST_PORT)
   - Protocol information (PROTOCOL, L7_PROTO)
   - Flow statistics (IN_BYTES, OUT_BYTES, IN_PKTS, OUT_PKTS, FLOW_DURATION_MILLISECONDS)
   - TCP flags (TCP_FLAGS)

3. **Advantage of NetFlow Standardization**:
   - Consistent feature naming across all datasets
   - Same data types and formats
   - No missing critical data fields
   - Common source for data extraction

### Model Requirements with NetFlow Data

1. **Anomal_E Model**:
   - Successfully works with CSE-CIC-IDS2018 NetFlow data
   - Uses source/destination IP addresses for graph construction
   - All required features are available in all NetFlow datasets
   - Can work with the core set of NetFlow features common to all datasets

2. **E-GraphSage Model**:
   - Already uses BoT-IoT NetFlow data
   - Creates nodes representing IP:port combinations
   - Creates edges representing flows between nodes
   - All required features are available in all NetFlow datasets

3. **CAGN-GAT Model**:
   - Does not have native support for NetFlow data
   - Special handling required for NetFlow data preprocessing
   - May require model adaptation to work with NetFlow format
   - If standardization is insufficient, we may need to modify the model itself

### NetFlow Standardization Strategy

Based on the compatibility analysis, we can standardize on NetFlow data across all models by:

1. **Feature Set**: Use the common subset of features present in all datasets (NetFlow_v1 features as the baseline)
2. **Graph Construction**: Create nodes representing IP:port combinations and edges representing flows between them
3. **Feature Engineering**: Generate additional features from the basic set for models that require them
4. **Label Standardization**: Create unified attack type mapping across datasets
5. **Data Processing**: Apply consistent preprocessing (scaling, normalization) across datasets
6. **CAGN-GAT Adaptation**: Develop special preprocessing for CAGN-GAT or modify the model as needed

### Implementation Steps for NetFlow Standardization

1. **Data Reduction Step**:
   - Use dedicated scripts for each dataset:
     - `reduce_bot_iot_v2.py`: Reduces BoT-IoT v2 dataset
     - `reduce_cic_ids2018.py`: Reduces CIC-IDS2018 dataset
     - `reduce_unsw_nb15.py`: Reduces UNSW-NB15 dataset
   - These scripts preserve attack distribution while sampling
   - The reduction step should not perform any standardization

2. **Data Standardization Step**:
   - Use common `standardize_datasets.py` script for all datasets
   - This standardizes both full and reduced datasets
   - Standardization includes:
     - Ensuring consistent column names and data types
     - Adding required columns like flow_id and dataset_source
     - Formatting data according to common structure
   - No additional filtering or sampling during standardization

3. **Dataset Combining Step**:
   - Use `combine_datasets.py` to merge standardized datasets
   - Can combine full datasets or reduced datasets
   - Maintains consistent format across all datasets

4. **Create Unified Graph Representation**:
   - IP:port combinations as nodes
   - Flows as edges between nodes
   - Flow statistics as edge attributes
   - Protocol information as additional node/edge features

5. **Standardize Labels**:
   - Create binary labels (attack/normal)
   - Standardize attack type categories across datasets
   - Create numeric encoding for attack types

6. **Dataset Processing Pipeline**:
   - Load any NetFlow dataset
   - Extract common features
   - Apply appropriate scaling/normalization
   - Create graph representation
   - Format for model input

7. **CAGN-GAT Special Handling**:
   - Develop custom preprocessing for CAGN-GAT to handle NetFlow data
   - If necessary, adapt the CAGN-GAT model architecture to work with NetFlow features
   - Test and validate CAGN-GAT with modified NetFlow inputs

## Expected Outputs
- Standardized NetFlow datasets in `/media/ssd/test/standardized-datasets/netflow/`
- Combined NetFlow dataset in `/media/ssd/test/standardized-datasets/combined/`
- Documentation of attack type mappings and schema
- CAGN-GAT adaptation guidance or modified model (if needed)

## Status
- Initial analysis of NetFlow compatibility completed
- NetFlow standardization in progress
- Combined NetFlow dataset planned
- CAGN-GAT adaptation for NetFlow data pending

## Outcome
- Standardized NetFlow datasets stored in `/media/ssd/test/standardized-datasets/`
- Each dataset will have a consistent format with the same feature set
- Documentation of transformation process and guidelines for usage
- All models working with NetFlow data format

## Future Considerations
- Ability to combine datasets for transfer learning
- Extension to additional NetFlow datasets
- Comparative analysis of performance across different datasets
- Further refinement of CAGN-GAT for NetFlow data 