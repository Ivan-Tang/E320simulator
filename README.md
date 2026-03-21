# E320simulator

Simulator and track reconstruction algorithms for the E320 prototype ALPIDE tracker — a 5-layer silicon pixel detector measuring positrons produced in electron–laser collisions.

## Overview

This project implements a full track reconstruction pipeline that compares classical and machine-learning-based seeding methods. The goal is to push tracking efficiency ≥ 95% with fake rate ≤ 5% and inference time ≤ 10 ms/event.

**Detector:** 5-layer ALPIDE sensor array
**Signal:** sparse positrons (signal fraction ~0.02%) against a heavy background
**Platform:** Weizmann ATLAS Grid Cluster (PBS, NVIDIA RTX 3090)

## Repository Structure

```
E320simulator/
├── src/                    # Core library
│   ├── geometry.py         # ALPIDE sensor geometry & coordinate transforms
│   ├── simulator.py        # Fast cluster-level event simulator (SimConfig)
│   ├── baseline.py         # Slope-window + greedy chain seeding (BaselineConfig)
│   ├── hough_baseline.py   # Hough transform tracker (HoughConfig)
│   ├── models.py           # ML models: EdgeMLP, InteractionNet, ResGNN,
│   │                       #   EggNet, HierarchicalGNN, TransformerEdgeClassifier,
│   │                       #   Embedder, TransformerEmbedder
│   ├── layers.py           # Building blocks: MLP, PositionalEncoding3D,
│   │                       #   MultiHeadAttention, transformer layers
│   ├── losses.py           # FocalLoss, HingeLoss
│   ├── train.py            # Unified edge-classification training loop (TrainConfig)
│   ├── train_embedder.py   # Metric-learning embedder training (EmbedderTrainConfig)
│   ├── train_trackformer.py# Transformer end-to-end track finding (TrackFormerConfig)
│   ├── train_hit_filter.py # Hit-level noise filter training
│   ├── ddp.py              # Multi-GPU DistributedDataParallel utilities
│   ├── utils.py            # Edge building, node/edge feature schemas
│   └── config.py           # Data path resolution (env var / .env / default)
├── scripts/                # Entry-point scripts
│   ├── run_baseline.py     # Evaluate baseline algorithm
│   ├── run_hough.py        # Evaluate Hough tracker
│   ├── run_model.py        # Inference with a trained checkpoint
│   ├── run_benchmark.py    # Full benchmark across all methods
│   ├── grid_search_baseline.py
│   ├── grid_search_ml.py
│   ├── compare_scaling.py  # Sweep background / signal density
│   ├── compare_reco.py
│   ├── compare_timing.py
│   └── plot_complexity.py
├── test/                   # pytest suite (182 tests, 8 modules)
├── explore/                # Jupyter notebooks for data exploration
├── requirements.txt
├── pytest.ini
├── progress.md             # Experiment log (updated each session)
└── research_proposal_ML_seeding.md  # Research directions: EvoHierGNN & TrackFormer-Seed
```

## Installation

```bash
# Create and activate the conda environment
conda create -n e320root python=3.10
conda activate e320root
pip install -r requirements.txt
pip install -e .   # if setup.py / pyproject.toml is present
```

All Python commands below assume the `e320root` environment is active.

## Data Path

Data paths are resolved from `src/config.py` in this order:

1. `DATA_PATH` environment variable
2. `.env` file in the project root (`DATA_PATH=/path/to/data`)
3. Default: `~/hep/data_Run502`

Key exports: `DATA_ROOT`, `SIM_DIR`, `RUNS_DIR`, `OUTPUTS_DIR`, `HIT_LEVEL_PARQUET`, `HIT_LEVEL_PROCESSED`.

## Usage

### Run Tests

```bash
conda run -n e320root pytest test/ -v
```

### Generate Simulated Data

```python
from src.simulator import Simulator, SimConfig
sim = Simulator(SimConfig(background_mode="data"))
clusters_df, tracks_df = sim.generate(n_events=1000)
```

### Evaluate Algorithms

```bash
# Slope-window + chain seeding baseline
conda run -n e320root python scripts/run_baseline.py

# Hough transform tracker
conda run -n e320root python scripts/run_hough.py

# Inference with a trained ML model
conda run -n e320root python scripts/run_model.py \
  --checkpoint runs/exp_interactionnet_v1/best_model.pt \
  --clusters sim_clusters_test.parquet
```

### Train ML Models

```bash
# Edge-classification GNN (e.g. InteractionNet)
conda run -n e320root python -m src.train \
  --task edge \
  --clusters sim_clusters_train.parquet \
  --model interaction_net \
  --epochs 50 \
  --batch-size 256

# Full benchmark (submit as batch job for long runs)
conda run -n e320root python scripts/run_benchmark.py \
  --device cuda --epochs 200 --workers 8

# Sweep ML thresholds
conda run -n e320root python scripts/grid_search_ml.py
```

## Feature Schemas

All ML models share a unified schema:

| Type | Dimensions | Features |
|------|-----------|---------|
| Node | 7 | `layer_id, x_trk_mm, y_trk_mm, z_trk_mm, size_x, size_y, size` |
| Edge | 6 | `dx_mm, dy_mm, dz_mm, dr_mm, slope_x, slope_y` |
| Output | 1 | Edge score ∈ [0, 1] |

## Reconstruction Output Schema

All algorithms produce the same output columns:

```
event_id | candidate_id | node_ids | n_layers | ax | bx | ay | by | chi2 | rms | is_kept
```

## Benchmark Results (10k test events)

| Method | Track Eff. | Fake Rate | Mean RMS | Inference Time |
|--------|-----------|-----------|----------|----------------|
| Baseline | 74.3% | 42.5% | 4.73 µm | 341 s |
| Hough | 82.7% | 46.1% | 13.33 µm | 2880 s |
| EdgeMLP | 74.3% | 40.4% | 4.67 µm | 2447 s |
| ResGNN | 51.4% | 43.5% | 4.78 µm | 2270 s |
| **InteractionNet** | **70.6%** | **14.3%** | **4.09 µm** | **767 s** |
| EggNet | 1.4% | 10.5% | 4.95 µm | 713 s |
| HierarchicalGNN | 14.2% | 10.2% | 4.14 µm | 712 s |
| Transformer | 2.2% | 99.7% | 4726 µm | 151 s |

**Target:** efficiency ≥ 95%, fake rate ≤ 5%, ≤ 10 ms/event
**Best current model:** InteractionNet — lowest fake rate (14.3%) and best position resolution (4.09 µm).

## Research Directions

Two parallel ML seeding approaches under investigation:

- **EvoHierGNN** — Evolving Hierarchical Graph Neural Network: dynamic k-NN graph construction + hierarchical pooling + iterative graph evolution. O(E) complexity.
- **TrackFormer-Seed** — Transformer encoder–decoder: learnable seed queries over the full hit cloud. End-to-end, fully parallel. O(N²) with sparse attention.

See [`research_proposal_ML_seeding.md`](research_proposal_ML_seeding.md) for full details.

## Cluster Usage (Weizmann ATLAS Grid)

```bash
# Submit batch job
qsub subs/benchmark.sh

# Check job status
qstat -u yiwen

# View logs
tail -f logs/<job_id>.out
```

See [`CLAUDE.md`](CLAUDE.md) for full cluster and session workflow instructions.

## References

1. Choma et al. (2020). "Track Seeding and Labelling with Embedded-space Graph Neural Networks." arXiv:2007.00149
2. Liu et al. (2023). "Hierarchical Graph Neural Networks for Particle Track Reconstruction." arXiv:2303.01640
3. Calafiura et al. (2024). "EggNet: An Evolving Graph-based Graph Attention Network." arXiv:2407.13925
4. Stroud et al. (2024). "Transformers for Charged Particle Track Reconstruction in High Energy Physics."
5. Borysov et al. "Preliminary experience with the E320 Prototype Tracker."
