# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

**所有代码必须在 `e320root` conda 环境中运行。** 运行任何 Python 脚本或测试时，必须使用以下格式：

```bash
conda run -n e320root python xxx.py
conda run -n e320root pytest test/ -v
```

不要使用 `conda activate e320root` 后直接运行，始终通过 `conda run -n e320root` 执行所有命令。

## Data Path Configuration

All data paths are resolved through `src/config.py`. The resolution order is:

1. `DATA_PATH` environment variable
2. `.env` file in the project root (key=value format)
3. Default: `~/hep/data_Run502`

To configure for a different environment, copy `.env.example` to `.env` and set `DATA_PATH`:

```bash
cp .env.example .env
# Edit .env and set:
# DATA_PATH=/your/path/to/data_Run502
```

Or set the environment variable directly:

```bash
export DATA_PATH=/your/path/to/data_Run502
```

The `.env` file is git-ignored. The key paths exported from `src/config.py` are:
- `DATA_ROOT` — base data directory
- `SIM_DIR` — `DATA_ROOT/simulation`
- `RUNS_DIR` — `DATA_ROOT/runs`
- `OUTPUTS_DIR` — `DATA_ROOT/outputs`
- `HIT_LEVEL_PARQUET` — `DATA_ROOT/hit_level.parquet`
- `HIT_LEVEL_PROCESSED` — `DATA_ROOT/processed/hit_level.parquet`

## Common Commands

### Testing

```bash
# Run all tests
pytest test/ -v

# Run a single test file
pytest test/test_geometry.py -v

# Run a specific test function
pytest test/test_geometry.py::TestAlpideSpec::test_width_mm -v

# Show stdout for passing tests
pytest -s test/test_embedder_smoke.py
```

### Running Algorithms

```bash
# Evaluate baseline (slope-window + chain seeding)
python scripts/run_baseline.py

# Evaluate Hough transform
python scripts/run_hough.py

# Train a GNN/ML model
python -m src.train --task edge --clusters sim_clusters_train.parquet \
  --model gnn --epochs 50 --batch-size 256

# Inference with a trained model checkpoint
python scripts/run_model.py --checkpoint runs/exp_gnn_v1/best_model.pt \
  --clusters sim_clusters_test.parquet

# Full benchmark across all methods
python scripts/run_benchmark.py --device mps --epochs 200 --workers 8

# Sweep GNN threshold and plot efficiency
python scripts/grid_search_gnn.py
```

### Data Generation

```python
from src.simulator import Simulator, SimConfig
sim = Simulator(SimConfig(background_mode="data"))
clusters_df, tracks_df = sim.generate(n_events=1000)
```

## Architecture Overview

The project implements a full track reconstruction pipeline for the E320 prototype ALPIDE detector (5 layers).

### Data Flow

```
Simulator → clusters_df / tracks_df (Polars)
                ↓
         build_labeled_edges_from_sim() (utils.py)
                ↓
          edges_df with truth labels
                ↓
      train.py / train_embedder.py  ←→  models.py
                ↓
        Trained checkpoint (.pt)
                ↓
      scripts/run_model.py → reco_df
                ↓
        compute_metrics() → efficiency, fake rate
```

### Key Source Files (`src/`)

| File | Role |
|------|------|
| `geometry.py` | ALPIDE sensor specs, 5-layer geometry, coordinate transforms (pixel → chip-local → TRK → LAB) |
| `simulator.py` | Cluster-level fast simulator; `SimConfig` controls signal rate, noise, cluster size model |
| `baseline.py` | Slope-window edge building + greedy chain seeding + 3D line fit; `BaselineConfig` |
| `hough_baseline.py` | Hough transform tracker in (θ, ρ) space; `HoughConfig` |
| `models.py` | 11 model classes: `EdgeMLP`, `InteractionNet`, `ResGNN`, `MPNN`, `AGNN`, `TransformerEdgeClassifier`, `TrackFormerSeed`, `EggNet`, `HierarchicalGNN`, `Embedder`, `TransformerEmbedder` |
| `layers.py` | Reusable building blocks: `MLP()` factory, `PositionalEncoding3D`, `MultiHeadAttention`, transformer layers |
| `losses.py` | `FocalLoss` (for class imbalance) and `HingeLoss` (for metric learning) |
| `train.py` | Unified training loop for all edge-classification models; `TrainConfig` |
| `train_embedder.py` | Metric-learning embedder training + KDTree-based neighbor inference; `EmbedderTrainConfig` |
| `utils.py` | Tensor construction, node/edge feature column definitions, `build_labeled_edges_from_sim()` |

### Node and Edge Feature Schemas

All ML models share a standard feature schema:
- **Node features (7-dim):** `[layer_id, x_trk_mm, y_trk_mm, z_trk_mm, size_x, size_y, size]`
- **Edge features (6-dim):** `[dx_mm, dy_mm, dz_mm, dr_mm, slope_x, slope_y]`
- **Output:** Edge score ∈ [0,1] (probability of belonging to the same track)

### Reconstruction Output Schema

All algorithms (baseline, Hough, ML) produce the same output:
```
event_id | candidate_id | node_ids | n_layers | ax | bx | ay | by | chi2 | rms | is_kept
```

### Configuration

All components use dataclasses for configuration:
- `SimConfig` — simulator parameters
- `BaselineConfig` — slope windows, KNN k, chain extension thresholds
- `HoughConfig` — Hough space bounds, binning strategy
- `TrainConfig` — model type, hidden dims, learning rate, focal loss params
- `EmbedderTrainConfig` — embedding dim, hinge margin, pair sampling

### Data Format

Polars DataFrames serialized to Parquet. Real Run 502 background data lives at `/Users/IvanTang/hep/data_Run502/`.

### Device Handling

Scripts auto-detect the best available device: MPS (Apple Silicon) > CUDA > CPU.
