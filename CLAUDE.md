# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 会话工作规范

每次会话开始时，必须先阅读：
1. `research_proposal_ML_seeding.md` — 了解研究方向和目标
2. `progress.md` — 了解当前进度和下一步计划

每次会话结束时，必须按顺序执行：
1. 更新 `progress.md` — 记录本次完成的工作、实验结果、遇到的问题、下一步计划；同时更新顶部的"最近变更"表格
2. **Git commit + push** — 将本次所有修改提交并推送到远端（worker 节点从 git 拉取代码，不推送则批处理作业看不到更改）：
   ```bash
   cd ~/E320simulator
   git add <changed files>
   git commit -m "描述本次修改"
   git push
   ```

## PBS Job 跟进规则

提交 PBS job 后，**必须**设置跟进，禁止"提交即忘"：

1. **提交时**：用 `CronCreate` 设置跟进时间（预估 walltime + 30 分钟缓冲）
2. **跟进内容**：读 job 输出日志（`~/logs/`），解析关键指标（efficiency / fake_rate / loss），汇报给用户
3. **Job 仍在运行**：延长跟进时间，再次检查
4. **Job 失败**：分析 stderr 日志，诊断根因，向用户汇报
5. **禁止**：提交 job 后不设置跟进，或只说"我稍后检查"却不实际设置

跟进示例：
```
CronCreate: "30 15 13 4 *" (one-shot, 约 walltime 后)
Prompt: "检查 PBS job <ID> 是否完成：读 ~/logs/ 对应日志，解析最终 metrics，
与上次最优结果（82.01% / 11.66%）对比，汇报结论。"
```

## 计划 vs 直接执行

以下场景**必须先规划**（使用 `EnterPlanMode`）：
- 修改 `src/models.py` 中的模型架构
- 修改训练流程（`src/train.py`、`src/train_embedder.py`、`src/train_trackformer.py`）
- 涉及 3 个以上源文件的改动
- 存在多种可行方案需要你做选择的场景（如"用哪种图构建策略"）
- 修改后处理流程（chi2 过滤、NMS）

以下场景**直接执行**，不需要规划：
- 修 bug、加 log / debug 输出
- 写新脚本（`scripts/` 目录）
- 修改测试文件
- 改配置参数、环境变量
- 文档更新（`progress.md`、`research_log.md`）

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

### 架构说明（Worktree 模式）

```
/srv01/agrp/yiwen/E320simulator/      ← 主仓库（永远在 master，不做研究改动）
/srv01/agrp/yiwen/research/
    <slug-1>/                          ← session 1 worktree (branch: auto-research-<slug-1>)
        experiment_state.json          ← 该 session 的循环状态
        research_log.md
        status.md
        src/ ...                       ← 代码改动在此 worktree
    <slug-2>/                          ← session 2 worktree（可同时运行）
        ...
```

多个 session 可并发运行，互不干扰。

### Git 分支规则

- 所有 autonomous mode 的代码改动在对应 session 的 worktree（`auto-research-*` 分支）
- **主仓库 master 不做改动**（worktree 与主仓库完全分离）
- 每次循环至少一次 `git push`（在 qsub 之前，从 worktree 目录执行）
- 用户决定何时 merge 到 master

### 测试门槛

```bash
# 在 worktree 目录中运行
conda run -n e320root pytest test/ -v
```

pytest 必须通过才能 commit 代码。测试失败 → 修复代码 → 重跑。
若无法修复 → 设置 `error_state` → 退出（不提交 PBS 作业）。

### 关键文件说明

| 文件 | 位置 | 角色 | 由谁维护 |
|------|------|------|---------|
| `research_goal.md` | 主仓库（模板）/ worktree（各 session）| 研究目标和约束 | **用户** |
| `research_log.md` | 各 session worktree | 假设+变更+结果日志 | **Claude**（每轮追加）|
| `experiment_state.json` | 各 session worktree | 循环控制状态 | **Claude**（每轮更新）|
| `status.md` | 各 session worktree | 实时监控 | **Claude**（每轮覆盖）|
| `autonomous_loop_prompt.md` | 各 session worktree | Claude 的执行指令 | **用户** |
| `cluster_agent/session_watcher.sh` | 主仓库 | Per-session watcher | **用户** |

### 启动新研究 Session

```bash
# 1. 填写研究目标（主仓库模板，会被复制到新 session）
vim ~/E320simulator/research_goal.md

# 2. 一键启动（建 worktree + 初始化状态 + 启动 watcher + 触发首次循环）
cd ~/E320simulator
bash start_research.sh "目标描述"           # 默认最大 15 轮
bash start_research.sh "目标描述" 20        # 指定最大循环次数

# 3. 通过 Slack 启动（更便捷）
# !start "目标描述" [N]
```

### 查看所有 Session 状态

```bash
# 列出所有 session
ls ~/research/

# 查看某个 session
cat ~/research/<slug>/status.md
tail -50 ~/logs/watcher_<slug>.log

# 通过 Slack
# !sessions              列出所有
# !status <slug>         查看 status.md
# !wlog <slug>           查看 watcher 日志
# !result <slug>         查看 eval 数字
```

### 停止与恢复

**优雅停止**（等当前 PBS 作业完成后停止）：
```bash
bash ~/E320simulator/stop_research.sh <session-slug> grace
# 或 Slack: !stop <slug>
```

**紧急停止**（立即终止）：
```bash
bash ~/E320simulator/stop_research.sh <session-slug>
# 或 Slack: !kill <slug>
```

**恢复**（清除错误后重启）：
```bash
# 直接重启（start_research.sh 会复用已有 worktree）
bash ~/E320simulator/start_research.sh "<同一目标描述>"
```

**清理 worktree**（彻底删除某个 session 的本地副本）：
```bash
cd ~/E320simulator && git worktree remove ~/research/<slug> --force
```

---

## 下一步研究方向（2026-04-01 更新）

> **当前最优**：TransformerEdgeClassifier 82.95% / 11.66%（in-graph TPR=88.45%）
>
> **任务类型说明**：失效诊断和 DDP 修复适合 session 内直接做，**不适合套 auto-research loop**。
> auto-research 以「提交 PBS 训练 job → 等结果 → 改模型」为循环单位，调试/分析类任务不匹配。

### 🔴 高优先级（session 内直接做，非 auto-research）

1. **【前置必做】失效案例诊断脚本**
   - 将 18% 效率损失拆解为三类：
     - ① 真实边不在图里（kNN 覆盖率不足）→ 根因：图构建
     - ② 真实边在图里但模型打分低 → 根因：模型
     - ③ 边打分正确但后处理丢失 → 根因：chi2/NMS
   - 写 `scripts/diagnose_failures.py`，提交单个 PBS job，读输出后再决定优化方向
   - **所有后续优化方向都依赖此结论**

2. **DDP 多卡训练修复**
   - 修改 `src/train.py` 支持 `DistributedDataParallel`
   - 先在两卡小规模验证，再提交完整训练 job

### 🟡 中优先级（依赖诊断结果，可用 auto-research）

3. **改进图构建**（若诊断①占主导）
   - 增大 kNN k；或改用层间优先的 physics-informed 图构建

4. **TransformerEdgeClassifier 深度优化**（若诊断②占主导）
   - Hard negative mining、更多物理特征（击中密度、曲率估计）、更大模型

5. **后处理改进**（若诊断③占主导）
   - 用小 MLP 学习径迹质量分数替代硬截断 chi2；加入 NMS 消歧

### 🟢 低优先级（长期探索）

6. **重新训练 EggNet**：当前效率仅 1.4%，需重训或放弃
7. **TrackFormer-Seed 端到端**：跳过边分类，直接从击中点输出径迹（DETR 风格）；作为并行探索分支
