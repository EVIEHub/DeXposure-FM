---
license: apache-2.0
language:
  - en
tags:
  - graph-neural-network
  - defi
  - financial-networks
  - link-prediction
  - time-series
  - credit-exposure
  - blockchain
  - pytorch
  - transformer
  - tabular
library_name: pytorch
pipeline_tag: graph-ml
datasets:
  - custom
metrics:
  - auprc
  - auroc
  - mae
  - rmse
model-index:
  - name: DeXposure-FM
    results:
      - task:
          type: link-prediction
          name: Link Prediction
        metrics:
          - type: auprc
            value: 0.978
            name: AUPRC (h1)
          - type: auroc
            value: 0.996
            name: AUROC (h1)
---

# DeXposure-FM

**DeXposure-FM** is a graph-tabular foundation model for forecasting credit exposure networks in Decentralized Finance (DeFi). Built on [GraphPFN](https://arxiv.org/abs/2501.xxxxx) and fine-tuned on weekly DeFi exposure snapshots, it predicts:

- 🔗 **Edge existence**: Will a credit exposure link form between two protocols?
- ⚖️ **Edge weight**: What will be the magnitude of the exposure?
- 📈 **Node TVL change**: How will protocol Total Value Locked evolve?

## Model Description

| Checkpoint | Horizon | Task | AUPRC | AUROC | 
|------------|---------|------|-------|-------|
| `dexposure-fm-h1.pt` | 1 week | Link Prediction | **0.978** | **0.996** |
| `dexposure-fm-h4.pt` | 4 weeks | Link Prediction | 0.973 | 0.995 |
| `dexposure-fm-h8-h12.pt` | 8-12 weeks | Link Prediction | 0.967 | 0.993 |
| `graphpfn-frozen-all-horizons.pt` | All | Link Prediction | 0.936-0.940 | 0.986-0.988 |

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/EVIEHub/graph-dexposure.git
cd graph-dexposure

# Install dependencies
pip install -r requirements.txt

# Download model weights
huggingface-cli download EVIEHub/DeXposure-FM --local-dir checkpoints/
```

### Inference

```python
import torch
from huggingface_hub import hf_hub_download
from lib.deep import GraphPFNLinkPredictor
from lib.graphpfn.model import GraphPFN

# Download checkpoint
checkpoint_path = hf_hub_download(
    repo_id="EVIEHub/DeXposure-FM",
    filename="dexposure-fm-h1.pt"
)

# Initialize model
model = GraphPFNLinkPredictor(
    graphpfn_checkpoint="checkpoints/graphpfn-v1.ckpt",
    limix_checkpoint="checkpoints/LimiX-16M.ckpt",
    hidden_dim=64,
    num_heads=4,
    finetune=True
)

# Load fine-tuned weights
checkpoint = torch.load(checkpoint_path, map_location="cpu")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Prepare input data
# node_features: [N, F] tensor of node features
# edge_index: [2, E] tensor of edge indices
# edge_attr: [E, D] tensor of edge attributes (optional)

# Run inference
with torch.no_grad():
    # Get node embeddings
    node_embeddings = model.encode(node_features, edge_index, edge_attr)
    
    # Predict link existence for all node pairs
    src_nodes = torch.tensor([0, 1, 2])  # source protocol indices
    dst_nodes = torch.tensor([3, 4, 5])  # target protocol indices
    
    link_probs, edge_weights = model.predict_links(
        node_embeddings, src_nodes, dst_nodes
    )
    
    print(f"Link probabilities: {link_probs}")
    print(f"Predicted edge weights: {edge_weights}")
```

### Full Pipeline Example

```python
from lib.data import load_defi_network, prepare_link_prediction_data

# Load DeFi network snapshot
graph_data = load_defi_network("data/network_data/", date="2025-01-01")

# Prepare data for link prediction
train_data, val_data, test_data = prepare_link_prediction_data(
    graph_data,
    horizon=1,  # 1-week forecast
    neg_ratio=1.0
)

# Run model
model.eval()
with torch.no_grad():
    predictions = model(
        test_data.x,
        test_data.edge_index,
        test_data.edge_attr
    )
```

## Input Format

### Node Features (Tabular)

| Feature | Description | Type |
|---------|-------------|------|
| `log_tvl` | Log-scaled Total Value Locked: $\log(1 + \text{TVL})$ | float |
| `n_token_types` | Number of distinct token types held | int |
| `max_token_share` | Largest token concentration ratio | float |
| `token_entropy` | Shannon entropy of token distribution | float |
| `category` | Protocol category (one-hot, 15 classes) | categorical |
| `tvl_change` | Previous week TVL change (log-scale) | float |
| `in_degree_norm` | Normalized incoming edge count | float |
| `out_degree_norm` | Normalized outgoing edge count | float |

### Graph Structure

- **Nodes**: DeFi protocols (lending, DEX, bridges, etc.)
- **Edges**: Directed credit exposure links
- **Edge weights**: Log-scaled exposure magnitude

## Model Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Inputs                               │
│  ┌──────────────────┐    ┌──────────────────────────────┐   │
│  │ Tabular Features │    │ Graph Snapshot G_τ           │   │
│  │ x^tab_p          │    │ Nodes: w_τ(p), Edges: w_τ(e) │   │
│  └────────┬─────────┘    └──────────────┬───────────────┘   │
│           │                              │                   │
│           └──────────────┬───────────────┘                   │
│                          ▼                                   │
│           ┌──────────────────────────────┐                   │
│           │   GraphPFN Encoder (LiMiX)   │                   │
│           │   Pre-trained Transformer    │                   │
│           └──────────────┬───────────────┘                   │
│                          ▼                                   │
│           ┌──────────────────────────────┐                   │
│           │   Node Embeddings h_p,τ      │                   │
│           │   Mean-pooled over features  │                   │
│           └──────────────┬───────────────┘                   │
│                          ▼                                   │
│    ┌─────────────────────┴─────────────────────┐            │
│    ▼                                           ▼            │
│ ┌──────────────────┐              ┌──────────────────────┐  │
│ │ Link Prediction  │              │ Node TVL Prediction  │  │
│ │ MLP Head         │              │ MLP Head             │  │
│ └──────────────────┘              └──────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Training Details

| Parameter | Value |
|-----------|-------|
| Base Model | GraphPFN + LiMiX-16M |
| Training Data | DeFi exposure networks (Mar 2020 - Jan 2025) |
| Training Weeks | 104 weeks |
| Validation Weeks | 12 weeks |
| Test Weeks | 8 weeks |
| Optimizer | AdamW |
| Learning Rate | 1e-4 (head), 1e-5 (backbone) |
| Epochs | 20 per horizon |
| Batch Size | Full graph (in-context learning) |
| Loss | BCE (existence) + MSE (weight) + MSE (node) |

## Evaluation Results

### Link Prediction (Edge Existence)

| Model | h=1 | h=4 | h=8 | h=12 |
|-------|-----|-----|-----|------|
| Persistence Baseline | 0.912 | 0.891 | 0.873 | 0.856 |
| GraphPFN-Frozen | 0.938 | 0.940 | 0.938 | 0.936 |
| **DeXposure-FM** | **0.978** | **0.973** | **0.967** | **0.967** |

### Edge Weight Regression (MAE)

| Model | h=1 | h=4 | h=8 | h=12 |
|-------|-----|-----|-----|------|
| GraphPFN-Frozen | 3.26 | 3.19 | 3.17 | 3.14 |
| **DeXposure-FM** | **2.09** | **2.49** | **2.55** | **2.65** |

## Intended Use

### Primary Use Cases

- **Risk monitoring**: Track systemic risk in DeFi ecosystems
- **Stress testing**: Simulate contagion under shock scenarios  
- **Portfolio management**: Assess counterparty exposure risks
- **Regulatory analysis**: Monitor concentration and interconnectedness

### Out-of-Scope

- Real-time trading decisions (model trained on weekly snapshots)
- Price prediction (model predicts network structure, not prices)
- Non-DeFi financial networks (trained specifically on DeFi protocols)

## Limitations

- **Temporal granularity**: Weekly snapshots may miss intra-week dynamics
- **Protocol coverage**: Limited to protocols with on-chain data
- **Category distribution**: ~58% of protocols lack category labels
- **Market regime**: Trained primarily on 2020-2025 market conditions

## Citation

```bibtex
@article{huang2026dexposurefm,
  title={DeXposure-FM: A Graph-Tabular Foundation Model for DeFi Credit Exposure Forecasting},
  author={Huang, Fangzhou and others},
  journal={arXiv preprint},
  year={2026}
}
```

## License

This model is released under the [Apache 2.0 License](LICENSE).

## Acknowledgments

- [GraphPFN](https://github.com/graph-pfn/graphpfn) for the pre-trained graph-tabular encoder
- [LiMiX](https://github.com/limix/limix) for the tabular transformer backbone
- DefiLlama for DeFi protocol data

## Contact

- **GitHub**: [EVIEHub/graph-dexposure](https://github.com/EVIEHub/graph-dexposure)
- **Paper**: [arXiv:2026.xxxxx](https://arxiv.org/abs/2026.xxxxx)
