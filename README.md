# FA-UNIFEWS: Feature-Aware Unified Few-Shot Edge Weight Sharing for Graph Neural Networks

Official implementation of FA-UNIFEWS — extending UNIFEWS with node-adaptive feature-aware gating for improved graph sparsification, especially on heterophilic graphs.

## Overview

FA-UNIFEWS introduces a feature-aware scoring mechanism into the UNIFEWS pruning framework. Instead of relying solely on structural message magnitudes for edge pruning, FA-UNIFEWS blends structural and feature-based (cosine similarity) signals — either with a static ratio or via a learned per-node adaptive gate.

**Key idea:** `score = α · ‖msg‖ + (1−α) · cos_sim(x_i, x_j)`

| Mode | `fa_alpha` | Behavior |
|------|-----------|----------|
| UNIFEWS (original) | `1.0` | Structure-only pruning |
| FA-static | `0.0–0.99` | Fixed blend of structure + feature similarity |
| FA-adaptive | `-1.0` | Learned per-node α via MLP + Sigmoid |

## Project Structure

```
├── Unifews/                    # Core library
│   ├── run_fb.py               # Main runner (full-batch training)
│   ├── run_mb.py               # Mini-batch runner
│   ├── run_ablation.py         # Ablation study runner
│   ├── archs/
│   │   ├── layers.py           # GCNConvThr, ConvThr (FA scoring), etc.
│   │   ├── models.py           # GNNThr, MLP, SandwitchThr
│   │   ├── prunes.py           # Threshold pruning utilities
│   │   └── transform.py        # Graph transforms
│   ├── config/                 # JSON configs per dataset
│   │   ├── cora.json
│   │   ├── chameleon.json
│   │   └── ...
│   ├── data/                   # Datasets (auto-downloaded)
│   └── utils/                  # Data loading, metrics, logging
├── run_missing_experiments.ipynb  # Full experiment pipeline
├── run_colab_v2.ipynb            # Colab-ready visualization
├── save/                         # Experiment results (CSVs)
└── figures/                      # Generated plots
```

## Installation

### Option 1: Conda (recommended)

```bash
conda create -n fa-unifews python=3.11
conda activate fa-unifews
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install torch_geometric torch_scatter torch_sparse torch_cluster
pip install ptflops powerlaw dotmap numpy scipy scikit-learn matplotlib pandas
```

### Option 2: pip only

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric torch_scatter torch_sparse torch_cluster \
    -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install ptflops powerlaw dotmap numpy scipy scikit-learn matplotlib pandas
```

### Option 3: Google Colab

The notebooks auto-detect Colab and install dependencies:

```python
!pip install -q torch_geometric ptflops powerlaw dotmap
```

## Quick Start

### Single experiment

```bash
cd Unifews

# Run FA-UNIFEWS (static, α=0.5) on Cora
python run_fb.py -c cora -m gcn_thr -f 42 -a 0.7 -w 0.5 --fa_alpha 0.5

# Run original UNIFEWS on Cora
python run_fb.py -c cora -m gcn_thr -f 42 -a 0.7 -w 0.5 --fa_alpha 1.0

# Run FA-adaptive on Chameleon
python run_fb.py -c chameleon -m gcn_thr -f 42 -a 0.7 -w 0.5 --fa_alpha -1.0

# Run MLP baseline
python run_fb.py -c cora -m mlp -f 42

# Run Dense GCN (no pruning)
python run_fb.py -c cora -m gcn -f 42
```

### CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `-c` / `--config` | str | `cora` | Config name (loads `config/<name>.json`) |
| `-m` / `--algo` | str | — | Algorithm: `gcn_thr`, `mlp`, `gcn`, `gcn_rnd`, etc. |
| `-f` / `--seed` | int | `11` | Random seed |
| `-a` / `--thr_a` | float | — | Adjacency pruning threshold |
| `-w` / `--thr_w` | float | — | Weight pruning threshold |
| `--fa_alpha` | float | `0.5` | FA blending: `1.0`=UNIFEWS, `0.5`=FA-static, `-1.0`=FA-adaptive |
| `-v` / `--dev` | int | `0` | GPU device id |
| `-l` / `--layer` | int | — | Number of GNN layers |
| `-n` / `--suffix` | str | `''` | Checkpoint save suffix |

### Run all experiments (notebook)

Open `run_missing_experiments.ipynb` in Colab or locally and execute cells sequentially:

1. **Part A** — Random baseline (`gcn_rnd`) on all datasets
2. **Part B** — Missing datasets (citeseer, pubmed, texas) × 6 methods
3. **Part C/D** — Dense GCN baseline + cleanup
4. **Part E** — 2-phase hyperparameter tuning on problem datasets
5. **Part F** — Fine-grained sweep
6. **Part G** — Additional heterophilic datasets (Squirrel, Actor)

## Methods

| Method | `--algo` | `--fa_alpha` | `--thr_a` | `--thr_w` |
|--------|----------|-------------|-----------|-----------|
| Dense GCN | `gcn` | 1.0 | 0.0 | 0.0 |
| MLP | `mlp` | 1.0 | 0.0 | 0.0 |
| Random (DropEdge) | `gcn_rnd` | 1.0 | 0.7 | 0.5 |
| UNIFEWS | `gcn_thr` | 1.0 | 0.7 | 0.5 |
| FA-static | `gcn_thr` | 0.5 | 0.7 | 0.5 |
| FA-adaptive | `gcn_thr` | -1.0 | 0.7 | 0.5 |

## Datasets

Supported datasets (stored in `Unifews/data/`):

| Dataset | Type | Nodes | Homophily |
|---------|------|-------|-----------|
| Cora | Homophilic | 2,708 | 0.81 |
| Citeseer | Homophilic | 3,327 | 0.74 |
| PubMed | Homophilic | 19,717 | 0.80 |
| Computers | Homophilic | 13,752 | 0.78 |
| CS | Homophilic | 18,333 | 0.81 |
| Chameleon | Heterophilic | 2,277 | 0.23 |
| Cornell | Heterophilic | 183 | 0.30 |
| Texas | Heterophilic | 183 | 0.11 |
| Wisconsin | Heterophilic | 251 | 0.21 |
| Squirrel | Heterophilic | 5,201 | 0.22 |
| Actor | Heterophilic | 7,600 | 0.22 |

### Adding new datasets

1. Prepare files in `Unifews/data/<name>/`:
   - `adj.npz` — Sparse CSR adjacency (scipy, int8, no self-loops)
   - `feats.npy` — Node features (numpy, float32)
   - `labels.npz` — Contains `labels`, `idx_train`, `idx_val`, `idx_test`
   - `degree.npz` — Degree array (optional)

2. Create `Unifews/config/<name>.json`:
```json
{
    "data": "<name>",
    "path": "./data/",
    "algo": "gcn_thr",
    "epochs": 200,
    "patience": 20,
    "lr": 0.001,
    "weight_decay": 1e-5,
    "layer": 2,
    "hidden": 512,
    "dropout": 0.5,
    "thr_a": 0.5,
    "thr_w": 0.5,
    "inductive": false,
    "multil": false
}
```

3. Run: `python run_fb.py -c <name> -m gcn_thr --fa_alpha 0.5`

## Config File Format

Each dataset has a JSON config in `Unifews/config/`:

| Field | Description |
|-------|-------------|
| `data` | Dataset directory name |
| `path` | Parent data directory |
| `algo` | Default algorithm |
| `epochs` | Max training epochs |
| `patience` | Early stopping patience |
| `lr` | Learning rate |
| `weight_decay` | L2 regularization |
| `layer` | Number of GNN layers |
| `hidden` | Hidden dimension |
| `dropout` | Dropout rate |
| `thr_a` / `thr_w` | Default pruning thresholds |
| `inductive` | Transductive (false) or inductive (true) |
| `multil` | Multi-label classification |

## Supported Backbones

| Backbone | Dense | UNIFEWS/FA | Random |
|----------|-------|------------|--------|
| GCN | `gcn` | `gcn_thr` | `gcn_rnd` |
| GATv2 | `gat` | `gat_thr` | `gat_rnd` |
| GCNII | `gcn2` | `gcn2_thr` | — |
| GraphSAGE | `gsage` | `gsage_thr` | — |
| GIN | `gin` | — | — |
| MLP | `mlp` | — | — |

> **Note:** GCN backbone (`gcn_thr`) is recommended. GAT/SAGE may have compatibility issues with newer PyG versions.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{fa-unifews2026,
  title={FA-UNIFEWS: Feature-Aware Unified Few-Shot Edge Weight Sharing for Graph Neural Networks},
  author={...},
  booktitle={NeurIPS},
  year={2026}
}
```
