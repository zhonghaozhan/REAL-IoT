# Network Intrusion Detection - Cross-Dataset GNN Evaluation

This project focuses on standardizing and evaluating network intrusion detection datasets for cross-model and cross-dataset testing using Graph Neural Networks (GNNs).

## Overview

We standardize and evaluate multiple cybersecurity datasets to enable cross-dataset testing of GNN models for network intrusion detection. The project implements:

1. **Data Standardization**: Conversion of diverse datasets to a common format
2. **Graph Construction**: Creation of graph representations for GNN models
3. **Cross-Dataset Evaluation**: Framework for testing models across different datasets

## Datasets

### Standard Datasets
- **BoT-IoT**: Network traffic data from IoT devices with various attack types
- **CSE-CIC-IDS2018**: Comprehensive network traffic with modern attack types
- **UNSW-NB15**: Network traffic with a diverse range of attack categories

### NetFlow Datasets
- **NF-BoT-IoT**: Network flow version of BoT-IoT
- **NF-CSE-CIC-IDS2018**: Network flow version of CSE-CIC-IDS2018
- **NF-UNSW-NB15**: Network flow version of UNSW-NB15

## Models

We evaluate the following GNN architectures:
- **Anormal-E** 
- **E-GraphSAGE** 
- **CAGN-GAT Fusion** 

## Directory Structure

```
.
├── GNN/                           # GNN model implementations
│   ├── CAGN-GAT Fusion.ipynb      # Fusion model implementation
│   ├── E-GraphSAGE-BoT-IoT-mean-agg_multiclass.ipynb
│   ├── GAT_UNSW_NB15.ipynb        # GAT model for UNSW-NB15
│   ├── GCN_UNSW_NB15.ipynb        # GCN model for UNSW-NB15
│   └── dgl_batch_helpers/         # Helper functions for DGL
│
├── standardized-datasets/         # Standardized datasets
│   ├── standard/                  # Standard datasets
│   ├── netflow/                   # NetFlow datasets
│   ├── combined/                  # Combined datasets
│   └── README.md                  # Dataset documentation
│
├── memory-bank/                   # Documentation and plans
│   └── dataset_standardization_plan.md  # Plan for dataset standardization
│
├── standardize_datasets.py        # Dataset standardization script
├── optimize_standardize_datasets.py # Optimized standardization script
├── combine_datasets.py            # Dataset combination script
└── README.md                      # This file
```

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch
- DGL (Deep Graph Library)
- Pandas, NumPy, Scikit-learn
- NetworkX

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd <repository-name>

# Install dependencies
pip install torch dgl pandas numpy scikit-learn networkx matplotlib seaborn
```

### Data Preparation

```bash
# Run the standardization script
python optimize_standardize_datasets.py

# Combine standardized datasets
python combine_datasets.py
```

### Running Models

Each model is implemented as a Jupyter notebook in the `GNN/` directory.

```bash
# Start Jupyter notebook
jupyter notebook

# Navigate to GNN/ directory and open the desired model notebook
```

## Standardization Process

The standardization process consists of:

1. **Feature Selection**: Common features across all datasets
2. **Schema Alignment**: Unified column naming and data types
3. **Graph Construction**: Creating graph representations for GNN models
4. **Label Standardization**: Consistent attack type categorization

## Common Schema

Each standardized dataset follows this schema:

| Column | Type | Description |
|--------|------|-------------|
| src_ip | str | Source IP address |
| src_port | int | Source port |
| dst_ip | str | Destination IP address |
| dst_port | int | Destination port |
| protocol | int | Protocol number |
| protocol_name | str | Protocol name (e.g., TCP, UDP) |
| duration_ms | float | Flow duration in milliseconds |
| bytes_in | int | Bytes from source to destination |
| bytes_out | int | Bytes from destination to source |
| packets_in | int | Packets from source to destination |
| packets_out | int | Packets from destination to source |
| tcp_flags | int | TCP flags (if applicable) |
| binary_label | int | Binary label (0=normal, 1=attack) |
| attack_type | str | Attack category |
| attack_type_encoded | int | Numerically encoded attack type |

## Cross-Dataset Testing

To evaluate models across different datasets:

1. Train a model on one standardized dataset
2. Test the trained model on other standardized datasets
3. Evaluate performance metrics

## Results and Analysis

TBD - Will be added after comprehensive testing.

## References

- BoT-IoT Dataset: Koroniotis, N., Moustafa, N., Sitnikova, E., & Turnbull, B. (2019). Towards the development of realistic botnet dataset in the Internet of Things for network forensic analytics: Bot-IoT dataset. Future Generation Computer Systems, 100, 779-796.
- CSE-CIC-IDS2018 Dataset: Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). Toward generating a new intrusion detection dataset and intrusion traffic characterization. In Proceedings of the 4th International Conference on Information Systems Security and Privacy (ICISSP 2018) (pp. 108-116).
- UNSW-NB15 Dataset: Moustafa, N., & Slay, J. (2015). UNSW-NB15: a comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set). In Military Communications and Information Systems Conference (MilCIS), 2015 (pp. 1-6). IEEE.
- GNN Models: Kipf, T. N., & Welling, M. (2016). Semi-supervised classification with graph convolutional networks. arXiv preprint arXiv:1609.02907. 
