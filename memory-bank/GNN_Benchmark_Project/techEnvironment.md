# Technical Environment

## Hardware Configuration
- Remote server with 125GB RAM
- Swap space: 15GB
- Storage location: `/media/ssd/test/`
- GPU acceleration available

## Development Environment
- Remote SSH access via VS Code/Cursor
- Python virtual environment: gnn_cuda_env (with CUDA support)
- Jupyter notebooks for experiments and analysis

## Key Dependencies
- PyTorch (with CUDA support)
- DGL (Deep Graph Library)
- PyTorch Geometric (for GNN implementations)
- pandas (for data handling)
- scikit-learn (for evaluation metrics)
- pyod (for anomaly detection algorithms)
- matplotlib/seaborn (for visualization)

## Virtual Environment Specifics

- **`adversarial_env`:**
    - Primary environment for most GNN training and evaluation (e.g., CAGN-GAT).
    - Uses `dgl==2.1.0+cu118`.
    - May require OpenSSL 3.x due to dependencies like `torchdata`.

- **`egraphsage_eval_venv`:**
    - **Purpose:** Created specifically for EGraphSage adversarial evaluation due to library conflicts.
    - **Reason:** The primary `adversarial_env` required OpenSSL 3.x (via `torchdata` from `dgl>=2.x`), which was unavailable on the system without sudo access.
    - **Setup:**
        - Based on system Python 3.8.10.
        - Uses `dgl==1.1.2+cu118` (or `1.1.3+cu118`) to avoid OpenSSL 3 dependency.
        - Uses `networkx<3` because newer NetworkX versions require Python >= 3.10.
    - **Location:** `/media/ssd/test/egraphsage_eval_venv/`
    - **Activation:** `source /media/ssd/test/egraphsage_eval_venv/bin/activate`

## File Structure
```
/media/ssd/test/
├── GNN/
│   ├── network_intrusion_detection.ipynb
│   ├── GAT_UNSW_NB15.ipynb
│   ├── GCN_UNSW_NB15.ipynb
│   ├── Anomal_E_cicids2017.ipynb
│   ├── best_dgi.pkl
│   ├── best_gat_model.pt
│   ├── best_gcn_model.pt
│   ├── gat_results.pkl
│   ├── gcn_results.pkl
│   └── kaggle/
│       └── input/
│           ├── nslkdd/
│           ├── unsw-nb15/
│           ├── cicids2017/
│           └── kdd-cup-1999-data/
├── venv/
├── gnn_cuda_env/
└── memory-bank/
```

## Replicated Models
- **Deep Graph Infomax (DGI)**: Self-supervised learning approach
- **GraphSAGE**: Scalable neighborhood aggregation method
- **Anomal_E_cicids2017**: Network intrusion detection with graph embeddings and anomaly detection

## Development Challenges
- Jupyter notebook edit mode access issues via SSH
- Path issues with dataset access (absolute vs. relative paths)
- Need to adapt Kaggle notebook code for local environment

## Best Practices
- Use relative paths for file access
- Save intermediate results to avoid recomputation
- Document model parameters and results
- Use version control for tracking changes
- Ensure consistent preprocessing across datasets for fair comparison 