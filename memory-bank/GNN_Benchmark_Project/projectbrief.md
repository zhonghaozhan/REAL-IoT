# Project Brief: Network Intrusion Detection with Graph Neural Networks

## Project Goal
Develop, evaluate, and benchmark graph-based deep learning models for network intrusion detection across multiple cybersecurity datasets to assess their generalization capabilities.

## Core Requirements
1. Process network traffic data into graph representations
2. Implement and evaluate various Graph Neural Network architectures:
   - Basic models: GCN, GAT, GraphSAGE
   - Advanced models: E-GraphSage, CAGN-GAT, Anomal-E (with DGI)
3. Evaluate models across multiple standard network intrusion datasets
4. Compare performance with baseline methods
5. Identify optimal model configurations for different attack types
6. Assess model generalization capabilities

## Current Status
- ✅ E-GraphSage implemented and tested on all datasets (UNSW-NB15, CIC-IDS2018, BoT-IoT, combined)
- ✅ Anomal-E implemented and tested on all datasets (with parameter adjustments for combined dataset)
- 🔄 CAGN-GAT implementation in progress for the combined dataset
- ✅ Dataset standardization, reduction, and combination completed successfully
- ✅ Graph construction and feature engineering pipeline established

## Key Metrics
- Accuracy
- Precision/Recall
- F1 Score
- Detection Rate per attack type
- False Alarm Rate
- Generalization performance across datasets
- Training efficiency and resource requirements

## Datasets
- UNSW-NB15 dataset 
- CIC-IDS2018 dataset
- BoT-IoT v2 dataset
- Combined dataset (UNSW + CIC + BoT-IoT)
- ToN-IoT dataset (pending)

## Expected Outcomes
- Working GNN models for intrusion detection
- Performance benchmarks across different attack types and datasets
- Insights into model generalization capabilities
- Comparative analysis of basic vs. advanced GNN architectures
- Recommendations for optimal model selection based on dataset characteristics
- Insights into the effectiveness of graph-based approaches for cybersecurity
- Analysis of model performance on heterogeneous combined datasets 