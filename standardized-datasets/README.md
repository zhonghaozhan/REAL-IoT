# Network Intrusion Detection Dataset Standardization

This directory contains standardized versions of multiple cybersecurity datasets for cross-model and cross-dataset evaluation of network intrusion detection models using GNNs (GCN, GAT, and GraphSAGE).

## Directory Structure

```
standardized-datasets/
├── standard/              # Standardized versions of standard datasets
│   ├── bot_iot_standardized.csv
│   ├── cic_ids2018_standardized.csv
│   ├── unsw_nb15_standardized.csv
│   └── *_attack_mapping.json # Attack type mapping files
├── netflow/               # Standardized versions of netflow datasets
│   ├── nf_bot_iot_standardized.csv
│   ├── nf_cic_ids2018_standardized.csv
│   ├── nf_unsw_nb15_standardized.csv
│   └── *_attack_mapping.json # Attack type mapping files
├── combined/              # Combined datasets
│   ├── combined_standard.csv
│   ├── combined_netflow.csv
│   └── combined_*_attack_mapping.json
├── standardize_datasets.py  # Script to standardize datasets
└── combine_datasets.py      # Script to combine standardized datasets
```

## Datasets

### Original Datasets

1. **Standard Datasets:**
   - **BoT-IoT**: Network traffic data from IoT devices with various attack types
   - **CSE-CIC-IDS2018**: Comprehensive network traffic with modern attack types
   - **UNSW-NB15**: Network traffic with a diverse range of attack categories

2. **NetFlow Datasets:**
   - Network flow versions of the same datasets, with more focus on flow statistics

### Standardization Process

The standardization process involves:

1. **Feature Selection**: Common features across all datasets
2. **Schema Alignment**: Unified column naming and data types
3. **Label Standardization**: Consistent attack type categorization (preserved in both original and encoded formats)
4. **Data Combination**: Unified versions for cross-dataset evaluation

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
| flow_id | int | Unique flow identifier |

## Usage Instructions

### Standardizing Datasets

To standardize your datasets, place the original CSV files in an input directory and run:

```bash
python standardize_datasets.py --input_dir /path/to/input/datasets --standard_output_dir standard --netflow_output_dir netflow
```

Optional arguments:
- `--dataset`: Choose specific dataset to standardize ('bot_iot', 'cic_ids2018', 'unsw_nb15', or 'all')

The script expects the following files in the input directory:
- `bot_iot.csv`: BoT-IoT dataset
- `cic_ids2018.csv`: CSE-CIC-IDS2018 dataset
- `unsw_nb15.csv`: UNSW-NB15 dataset
- `nf_bot_iot.csv`: NetFlow version of BoT-IoT dataset
- `nf_cic_ids2018.csv`: NetFlow version of CSE-CIC-IDS2018 dataset
- `nf_unsw_nb15.csv`: NetFlow version of UNSW-NB15 dataset

### Combining Datasets

To combine standardized datasets into unified versions:

```bash
python combine_datasets.py --standard_input_dir standard --netflow_input_dir netflow --output_dir combined
```

### Using the Standardized Datasets

Load a standardized dataset:

```python
import pandas as pd

# Load standardized dataset
df = pd.read_csv('standard/bot_iot_standardized.csv')

# Extract features and labels
X = df.drop(['binary_label', 'attack_type', 'attack_type_encoded'], axis=1)
y_binary = df['binary_label']
y_multiclass = df['attack_type_encoded']
```

### Using Combined Datasets

For cross-dataset evaluation:

```python
import pandas as pd

# Load combined dataset
df = pd.read_csv('combined/combined_standard.csv')

# Filter by dataset source if needed
bot_iot_subset = df[df['dataset_source'] == 'bot_iot']
```

## Important Notes

- All standardized datasets preserve both the original attack type labels (`attack_type`) and numerically encoded versions (`attack_type_encoded`)
- Each dataset includes its own attack mapping file (`*_attack_mapping.json`) for decoding numerical labels
- The combined datasets include a unified attack mapping across all datasets
- Binary classification labels (0=normal, 1=attack) are preserved in all datasets
- If you're using these datasets for deep learning models, you may need to preprocess them further based on your specific model requirements

## Available Datasets

### NetFlow Datasets
- BoT-IoT (v1): Original NetFlow format
- BoT-IoT v2: Enhanced NetFlow format with more features
- CIC-IDS2018: NetFlow format
- UNSW-NB15: NetFlow format

## Dataset Processing Scripts

### Preprocessing Scripts
- `standardize_datasets.py`: Standardizes NetFlow datasets for cross-model compatibility
- `reduce_bot_iot_v2.py`: Reduces BoT-IoT v2 dataset while preserving attack distribution
- `reduce_cic_ids2018.py`: Reduces CIC-IDS2018 dataset while preserving attack distribution
- `reduce_unsw_nb15.py`: Reduces UNSW-NB15 dataset while preserving attack distribution
- `combine_datasets.py`: Combines standardized datasets into a single dataset
- `combine_and_reduce_datasets.py`: Combines and reduces datasets with intelligent sampling
- `process_all_datasets.py`: Runs the entire pipeline (reduce, standardize, combine) in one command

## Workflow for Processing Datasets

### Manual Step-by-Step Workflow

The proper workflow for processing the datasets is:

1. **Reduce large datasets** (if needed)
   ```bash
   # For BoT-IoT v2 dataset
   python reduce_bot_iot_v2.py --rate 0.01
   
   # For CIC-IDS2018 dataset
   python reduce_cic_ids2018.py --rate 0.1
   
   # For UNSW-NB15 dataset
   python reduce_unsw_nb15.py --rate 0.1
   ```
   This produces files like `nf_bot_iot_v2_reduced.csv`.

2. **Standardize datasets** (including reduced ones)
   ```bash
   python standardize_datasets.py
   ```
   This produces files like:
   - `nf_bot_iot_standardized.csv` (from original)
   - `nf_bot_iot_v2_reduced_standardized.csv` (from reduced)

3. **Combine standardized datasets**
   ```bash
   python combine_datasets.py --reduced
   ```
   - Without `--reduced`: Uses files with pattern `nf_*_standardized.csv`
   - With `--reduced`: Prefers files with pattern `nf_*_reduced_standardized.csv`

### Automated Pipeline

To run the entire pipeline in one command, use the `process_all_datasets.py` script:

```bash
python process_all_datasets.py
```

This script will:
1. Reduce all datasets (BoT-IoT v2, CIC-IDS2018, UNSW-NB15)
2. Standardize all datasets (including the reduced versions)
3. Combine the datasets (both full and reduced versions)

Available options:
- `--bot-iot-v2-rate 0.01`: Set the reduction rate for BoT-IoT v2 (default: 0.01)
- `--cic-ids-rate 0.1`: Set the reduction rate for CIC-IDS2018 (default: 0.1)
- `--unsw-rate 0.1`: Set the reduction rate for UNSW-NB15 (default: 0.1)
- `--skip-reduce`: Skip the dataset reduction step
- `--skip-standardize`: Skip the dataset standardization step
- `--skip-combine`: Skip the dataset combination step
- `--use-v2`: Use BoT-IoT v2 instead of v1

Example:
```bash
# Run just the standardization and combination steps, using BoT-IoT v2
python process_all_datasets.py --skip-reduce --use-v2

# Run the entire pipeline with custom reduction rates
python process_all_datasets.py --bot-iot-v2-rate 0.02 --cic-ids-rate 0.15 --unsw-rate 0.05
```

## File Naming Conventions

- **Original standardized datasets**: `nf_dataset_standardized.csv`
- **Reduced datasets**: `nf_dataset_reduced.csv`
- **Standardized reduced datasets**: `nf_dataset_reduced_standardized.csv`
- **Combined datasets**: `combined_netflow.csv` or `combined_netflow_reduced.csv`

## Usage Examples

### Reducing BoT-IoT v2 Dataset
The BoT-IoT v2 dataset is very large (6GB). To make it more manageable:

```bash
python reduce_bot_iot_v2.py --rate 0.01
```

This creates a reduced version at `netflow/nf_bot_iot_v2_reduced.csv` containing approximately 1% of the original data while preserving attack distribution.

### Standardizing Datasets
To standardize all datasets (including reduced versions if they exist):

```bash
python standardize_datasets.py
```

To standardize BoT-IoT v2 instead of v1:

```bash
python standardize_datasets.py --use-v2
```

To skip specific datasets:

```bash
python standardize_datasets.py --skip-bot-iot --skip-cic-ids
```

### Combining Datasets
To combine all standardized datasets:

```bash
python combine_datasets.py
```

To combine using BoT-IoT v2 instead of v1:

```bash
python combine_datasets.py --use-v2
```

To use standardized reduced versions when available:

```bash
python combine_datasets.py --reduced
```

## File Structure

- `netflow/`: Contains standardized NetFlow datasets
  - `nf_bot_iot_standardized.csv`: Standardized BoT-IoT dataset
  - `nf_bot_iot_v2_reduced.csv`: Reduced BoT-IoT v2 dataset
  - `nf_bot_iot_v2_reduced_standardized.csv`: Standardized version of reduced BoT-IoT v2
  - `nf_bot_iot_v2_standardized.csv`: Standardized BoT-IoT v2 (if full dataset was processed)
  - `nf_cic_ids2018_reduced.csv`: Reduced CIC-IDS2018 dataset
  - `nf_cic_ids2018_reduced_standardized.csv`: Standardized version of reduced CIC-IDS2018
  - `nf_cic_ids2018_standardized.csv`: Standardized CIC-IDS2018
  - `nf_unsw_nb15_standardized.csv`: Standardized UNSW-NB15

- `combined/`: Contains combined datasets
  - `combined_netflow.csv`: Combined full dataset
  - `combined_netflow_reduced.csv`: Combined reduced dataset

## Notes

- The standardization process ensures compatibility with all three GNN models: Anomal_E, CAGN-GAT, and E-GraphSage
- The reduction processes preserve the distribution of attack types while making the datasets more manageable
- The BoT-IoT v2 dataset provides enhanced features compared to v1, making it more compatible with CIC-IDS2018 and UNSW-NB15
- Always standardize reduced datasets before combining them
- The combination script automatically handles file selection based on availability, prioritizing standardized files 