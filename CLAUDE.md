# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 会话工作规范

每次会话开始时，必须先阅读：
1. `research_proposal_ML_seeding.md` — 了解研究方向和目标
2. `progress.md` — 了解当前进度和下一步计划

每次会话结束时，必须更新：
- `progress.md` — 记录本次完成的工作、实验结果、遇到的问题、下一步计划

## Cluster Environment

This runs on the **Weizmann ATLAS Grid Cluster** (PBS/OpenPBS batch system). The current login/analysis node is `wipp-an1` or `wipp-an2`. Heavy computation must be submitted as batch jobs to the `gpu` queue (routed to `gpuE`). Do not run CPU-heavy jobs interactively for more than an hour or two.

### Git Sync Discipline

**Before running any code, always pull the latest changes:**
```bash
cd ~/E320simulator && git pull
```
**After modifying code, always push to remote before submitting batch jobs** (worker nodes pull from git, not from your local working directory):
```bash
git add ... && git commit -m "..." && git push
```

### Conda Environment

All Python code must run inside the `e320root` environment. On the **analysis node** (interactive):
```bash
conda run -n e320root python scripts/run_baseline.py
conda run -n e320root pytest test/ -v
```

Inside **batch scripts**, use the explicit init pattern instead (conda run doesn't work well in PBS):
```bash
source /usr/wipp/conda/24.5.0/etc/profile.d/conda.sh
conda activate e320root
python scripts/run_baseline.py
```

Conda environments are stored on Lustre at `/storage/agrp/yiwen/conda_envs/` (set via `.condarc`).

### Submitting Batch Jobs

Job scripts live in `~/subs/`. Submit from the home directory (PBS sets `$PBS_O_WORKDIR`):
```bash
qsub subs/benchmark.sh        # submit a job
qstat -u yiwen                 # check your running jobs
qtop -f                        # detailed performance view (running and finished)
qdel <job_id>                  # cancel a job
```

Key PBS directives used in this project:
```
#PBS -q N              # submit to queue N (routes to gpuE for GPU jobs)
#PBS -l ngpus=1        # request 1 GPU (NVIDIA RTX 3090, 24 GB VRAM)
#PBS -l ncpus=8        # CPU cores
#PBS -l mem=64gb       # RAM
#PBS -l io=1           # I/O rate in MB/s (required for disk-intensive jobs)
#PBS -l walltime=12:00:00
#PBS -o logs/job.out   # stdout (relative to $PBS_O_WORKDIR)
#PBS -e logs/job.err   # stderr
#PBS -m n              # disable email (or -m ae for start/end notification)
```

Logs from batch jobs go to `~/logs/`. Worker nodes are `wipp-wn*` (do not SSH in directly).

### Storage

| Location | Purpose | Backed up? | Quota |
|---|---|---|---|
| `/srv01/agrp/yiwen/` (home) | Source code, scripts | Yes | 30 GB / 50k files |
| `/storage/agrp/yiwen/` (Lustre) | Data, model checkpoints, conda envs | **No** | 2 TB / 150k files |

Check quotas: `quota` (home), `lquota` (Lustre).

## Data Path Configuration

All data paths are resolved through `src/config.py`. The resolution order is:

1. `DATA_PATH` environment variable
2. `.env` file in the project root (key=value format, git-ignored)
3. Default: `~/hep/data_Run502`

The current deployment uses `DATA_PATH=/storage/agrp/yiwen/data_Run502` (set in `.env`).

Key paths exported from `src/config.py`:
- `DATA_ROOT` — base data directory (`/storage/agrp/yiwen/data_Run502`)
- `SIM_DIR` — `DATA_ROOT/simulation`
- `RUNS_DIR` — `DATA_ROOT/runs`
- `OUTPUTS_DIR` — `DATA_ROOT/outputs`
- `HIT_LEVEL_PARQUET` — `DATA_ROOT/hit_level.parquet`
- `HIT_LEVEL_PROCESSED` — `DATA_ROOT/processed/hit_level.parquet`

## Common Commands (Interactive)

### Testing

```bash
conda run -n e320root pytest test/ -v
conda run -n e320root pytest test/test_geometry.py -v
conda run -n e320root pytest test/test_geometry.py::TestAlpideSpec::test_width_mm -v
conda run -n e320root pytest -s test/test_embedder_smoke.py
```

### Running Algorithms

```bash
# Evaluate baseline (slope-window + chain seeding)
conda run -n e320root python scripts/run_baseline.py

# Evaluate Hough transform
conda run -n e320root python scripts/run_hough.py

# Train a GNN/ML model
conda run -n e320root python -m src.train --task edge --clusters sim_clusters_train.parquet \
  --model gnn --epochs 50 --batch-size 256

# Inference with a trained model checkpoint
conda run -n e320root python scripts/run_model.py --checkpoint runs/exp_gnn_v1/best_model.pt \
  --clusters sim_clusters_test.parquet

# Full benchmark across all methods (use batch job for long runs)
conda run -n e320root python scripts/run_benchmark.py --device cuda --epochs 200 --workers 8

# Sweep ML model thresholds
conda run -n e320root python scripts/grid_search_ml.py
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
| `models.py` | Model classes: `EdgeMLP`, `InteractionNet`, `ResGNN`, `EggNet`, `HierarchicalGNN`, `TransformerEdgeClassifier`, `Embedder`, `TransformerEmbedder` |
| `layers.py` | Reusable building blocks: `MLP()` factory, `PositionalEncoding3D`, `MultiHeadAttention`, transformer layers |
| `losses.py` | `FocalLoss` (class imbalance) and `HingeLoss` (metric learning) |
| `train.py` | Unified training loop for all edge-classification models; `TrainConfig` |
| `train_embedder.py` | Metric-learning embedder training + KDTree-based neighbor inference; `EmbedderTrainConfig` |
| `train_trackformer.py` | Transformer-based end-to-end track finding; `TrackFormerConfig` |
| `utils.py` | Tensor construction, node/edge feature column definitions, `build_labeled_edges_from_sim()` |

### Feature Schemas

All ML models share a standard feature schema:
- **Node features (7-dim):** `[layer_id, x_trk_mm, y_trk_mm, z_trk_mm, size_x, size_y, size]`
- **Edge features (6-dim):** `[dx_mm, dy_mm, dz_mm, dr_mm, slope_x, slope_y]`
- **Output:** Edge score ∈ [0,1] (probability of belonging to the same track)

### Reconstruction Output Schema

All algorithms (baseline, Hough, ML) produce identical output columns:
```
event_id | candidate_id | node_ids | n_layers | ax | bx | ay | by | chi2 | rms | is_kept
```

### Configuration Dataclasses

- `SimConfig` — simulator parameters
- `BaselineConfig` — slope windows, KNN k, chain extension thresholds
- `HoughConfig` — Hough space bounds, binning strategy
- `TrainConfig` — model type, hidden dims, learning rate, focal loss params
- `EmbedderTrainConfig` — embedding dim, hinge margin, pair sampling

### Data Format

Polars DataFrames serialized to Parquet. Use Polars idioms throughout (not pandas).

### Device Handling

Scripts auto-detect the best available device: CUDA > CPU. GPU nodes have NVIDIA RTX 3090 (24 GB VRAM, CUDA 13.0). MPS/Apple Silicon is not applicable on this Linux cluster.
