# Model Architecture

## Graph Neural Network Models

### E-GraphSage
- **Implementation**: Using DGL and PyTorch
- **Architecture**:
  - Enhanced GraphSAGE approach for network intrusion detection
  - Aggregates features from a node's local neighborhood using mean aggregation
  - Optimized for network flow data analysis
  - Multi-class classification capability (supports 10 attack classes)
  - Custom sampling approach for handling large graphs
- **Key Files**: 
  - `E-GraphSage_NetFlow_bot_iot_multiclass_mean_agg.ipynb` (UNSW-NB15 implementation)
  - E-GraphSage implementations for CIC-IDS2018
- **Status**: 
  - Model implemented and verified on multiple datasets
  - Successfully tested on UNSW-NB15 dataset
  - Successfully tested on CIC-IDS2018 dataset
  - Adapted for multi-dataset compatibility
- **Memory Optimization**:
  - Implemented intelligent sampling strategies for large graphs
  - Support for stratified sampling to maintain attack distribution
- **Adversarial Robustness Insights**: 
  - Initial PGD attacks targeting initialized node features were ineffective.
  - Subsequent PGD attacks targeting **scaled edge features** successfully degraded performance, indicating the model's strong reliance on edge information (flow features) for predictions.

### CAGN-GAT
- **Implementation**: Using PyTorch Geometric library
- **Architecture**:
  - Context-Aware Graph Neural Network with Graph Attention mechanisms
  - Multi-head attention for better feature learning
  - Specialized for capturing contextual information in network traffic
- **Key Files**: Related GAT implementation notebooks
- **Saved Models**: Related model checkpoint files

### Anomal E
- **Implementation**: Using PyTorch and pyod
- **Architecture**:
  - Specialized for network intrusion detection
  - Uses Deep Graph Infomax (DGI) as its trained model (encoder) for unsupervised learning of graph embeddings.
  - Combines graph embeddings with anomaly detection techniques: A separate classical anomaly detector (e.g., CBLOF, IF, PCA from `pyod`) is trained/applied on the DGI-generated embeddings to classify nodes/edges.
  - Processes netflow data into graph representations (IP:Port graph).
  - Input features are target-encoded and then normalized (using `sklearn.preprocessing.Normalizer`).
- **Key Files**: `Anomal_E_Combined.ipynb` (references `Anomal_E_cicids2017.ipynb`)
- **Status**: Successfully replicated and tested on combined dataset.
- **Adversarial Attack Implications**:
  - White-box attacks targeting the final anomaly prediction are difficult due to the likely non-differentiability of the classical anomaly detector.
  - Attacks should likely target the DGI encoder's objective function (maximizing DGI loss) to corrupt the embeddings fed into the detector.

## Dataset Integration
- **Combined Datasets**:
  - Successfully unified BoT-IoT, CIC-IDS2018, and UNSW-NB15 datasets
  - Implemented intelligent sampling to preserve attack distributions
  - Support for multi-dataset evaluation with consistent preprocessing
- **Sampling Strategy**:
  - Stratified sampling by attack type and dataset source
  - Higher sampling rates for rare attack classes
  - Custom sampling rates based on attack prevalence
  - Memory-efficient processing of large graph structures
- **Key Files**:
  - `combine_and_reduce_datasets.py` - Advanced dataset combination with stratified sampling

## Benchmark Testing Plan
- Cross-dataset evaluation of all models (E-GraphSage, CAGN-GAT, Anomal E)
- Datasets for evaluation: CIC-IDS2018, UNSW-NB15, and BoT-IoT
- Combined dataset for generalization testing with unified preprocessing
- Performance comparison using standard metrics (accuracy, F1, precision, recall)
- Analysis of generalization capabilities across datasets
- Resource efficiency measurements (training time, memory usage)

## Data Flow
```
Raw Network Data → Preprocessing → Graph Construction → 
GNN Model Training → Model Evaluation → Results Analysis
```

## Model Training Parameters
- Learning rate: Typically 0.001
- Epochs: 100-200 depending on dataset size
- Optimizer: Adam
- Loss function: Cross-entropy for classification
- Class weights: Custom weights based on class distribution
- Early stopping: Based on validation loss

## Evaluation Framework
- k-fold cross-validation
- Train/Validation/Test splits
- Confusion matrix analysis
- Performance metrics per attack category
- Generalization metrics across datasets
- Multi-dataset performance comparison

## Model Persistence
- Save best models during training
- Models stored in PyTorch format (.pt or .pkl)
- Results saved in pickle format for later analysis 