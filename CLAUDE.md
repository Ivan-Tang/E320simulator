# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 会话工作规范

每次会话开始时，必须先阅读：
1. `research_proposal_ML_seeding.md` — 了解研究方向和目标
2. `progress.md` — 了解当前进度和下一步计划

每次会话结束时，必须按顺序执行：
1. 更新 `progress.md` — 记录本次完成的工作、实验结果、遇到的问题、下一步计划
2. **Git commit + push** — 将本次所有修改提交并推送到远端（worker 节点从 git 拉取代码，不推送则批处理作业看不到更改）：
   ```bash
   cd ~/E320simulator
   git add <changed files>
   git commit -m "描述本次修改"
   git push
   ```

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

---

## Autonomous Research Mode

### 什么是 Autonomous Research Mode

当 Claude Code 以 `claude --print --dangerously-skip-permissions` 被 `autonomous_watcher.sh` 自动调用时，进入 **Autonomous Research Mode**。

在此模式下，Claude 作为自主研究员：以 `research_goal.md` 中的目标为导向，分析代码和实验结果，提出假设，修改模型架构，提交 PBS 实验，读取结果，形成新假设，持续循环。

### 进入判断

会话开始时检查：读取 `experiment_state.json`，若 `loop_status` 为 `idle/submitted/waiting`，且存在 `research_goal.md`，则以 Autonomous Research Mode 执行 `autonomous_loop_prompt.md` 中的步骤。

### 代码改动白名单（Autonomous Mode 下）

| 允许修改 | 禁止修改 |
|---------|---------|
| `src/models.py` | `src/geometry.py` |
| `src/layers.py` | `src/simulator.py` |
| `src/train.py` | `src/utils.py` |
| `src/losses.py` | `src/config.py` |
| `experiment_state.json` | `CLAUDE.md`（本文件）|
| `research_log.md` | `autonomous_loop_prompt.md` |
| `status.md` | `autonomous_watcher.sh` |
| `~/subs/auto_loop*.sh`（生成） | `~/subs/auto_eval_template.sh`（模板）|

### Git 分支规则

- 所有 autonomous mode 的代码改动必须在 `auto-research-*` 分支
- **不得直接 commit 到 master**（`settings.json` 的 deny 规则会阻止）
- 每次循环至少一次 `git push`（在 qsub 之前）
- 用户决定何时 merge 到 master

### 测试门槛

```bash
conda run -n e320root pytest test/ -v
```

pytest 必须通过才能 commit 代码。测试失败 → 修复代码 → 重跑。
若无法修复 → 设置 `error_state` → 退出（不提交 PBS 作业）。

### 关键文件说明

| 文件 | 角色 | 由谁维护 |
|------|------|---------|
| `research_goal.md` | 研究目标和约束 | **用户**（每 session 更新）|
| `research_log.md` | 假设+变更+结果的研究日志 | **Claude**（每轮追加）|
| `experiment_state.json` | 循环控制状态 | **Claude**（每轮更新）|
| `status.md` | 实时监控 | **Claude**（每轮覆盖）|
| `autonomous_loop_prompt.md` | Claude 的执行指令 | **用户**（修改循环行为）|

### 启动新研究 Session

```bash
# 1. 填写研究目标
vim ~/E320simulator/research_goal.md

# 2. 创建研究分支
cd ~/E320simulator
git checkout -b auto-research-{目标缩写}
git push -u origin auto-research-{目标缩写}

# 3. 初始化状态文件（填入 loop_start_time, research_branch, max_loops）
vim ~/E320simulator/experiment_state.json

# 4. 启动 watcher（后台守护进程）
nohup bash ~/subs/autonomous_watcher.sh > ~/logs/watcher.log 2>&1 &
echo "Watcher PID: $!"

# 5. 手动触发第一次循环（冷启动）
cd ~/E320simulator
claude --print --dangerously-skip-permissions \
  "$(cat autonomous_loop_prompt.md)" \
  >> ~/logs/claude_init.log 2>&1 &
```

### 停止与恢复

**查看状态**: `cat ~/E320simulator/status.md`
**查看日志**: `tail -50 ~/logs/watcher.log`

**优雅停止**（等当前作业完成后停止）：
```bash
python3 -c "
import json
with open('~/E320simulator/experiment_state.json') as f: s = json.load(f)
s['stop_requested'] = True
with open('~/E320simulator/experiment_state.json', 'w') as f: json.dump(s, f, indent=2)
"
touch ~/E320simulator/.stop_watcher
```

**紧急停止**：
```bash
touch ~/E320simulator/.stop_watcher
kill $(cat ~/E320simulator/.watcher.pid 2>/dev/null) 2>/dev/null
qdel $(python3 -c "import json; print(json.load(open('experiment_state.json')).get('current_pbs_job_id',''))")
```

**恢复**（清除错误后重启）：
```bash
python3 -c "
import json
with open('experiment_state.json') as f: s = json.load(f)
s['error_state'] = None; s['stop_requested'] = False; s['loop_status'] = 'idle'
with open('experiment_state.json', 'w') as f: json.dump(s, f, indent=2)
"
rm -f ~/E320simulator/.stop_watcher
nohup bash ~/subs/autonomous_watcher.sh > ~/logs/watcher.log 2>&1 &
```
