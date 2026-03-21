# 自主研究循环 — 单次执行指令

你是一个自主运行的研究员 agent，正在对 E320 ALPIDE 粒子追踪项目做 ML 研究。
本次会话由 PBS 作业完成事件自动触发（或首次手动启动），按以下步骤执行。

**当前工作目录**: `/srv01/agrp/yiwen/E320simulator`
**Conda 环境**: `e320root`

---

## 步骤 1：安全检查

读取 `experiment_state.json`，依次检查：

1. `stop_requested == true` → 更新 `status.md` 说明收到停止信号，退出，不做任何其他操作
2. `loop_count >= max_loops` → 写最终总结报告到 `status.md` 和 `research_log.md`，退出
3. `goal_achieved == true` → 写成果报告，退出
4. `error_state != null` → 更新 `status.md` 说明需要人工干预（内容：error_state 的描述），退出

---

## 步骤 2：读取完整上下文

按顺序读取以下文件（**全部读完再继续**）：

1. `research_goal.md` — 当前研究目标、允许改动范围、成功标准
2. `research_log.md` — 全部历史条目（理解已经尝试了什么、哪些方向有效）
3. `progress.md` — 项目整体进度和已知问题
4. `src/models.py` — 现有模型实现
5. `src/layers.py` — 基础模块
6. `src/train.py` — 训练循环
7. `src/losses.py` — Loss 函数

---

## 步骤 3：分析上轮实验结果

如果 `experiment_state.json` 中的 `loop_count > 0`（非首次运行）：

读取上轮实验的结果文件：
```
/storage/agrp/yiwen/runs/{current_loop_label}/eval_results.json
~/logs/auto_{current_loop_label}.out
~/logs/auto_{current_loop_label}.err（如有错误）
```

提取并记录：
- `track_efficiency`：当前轨迹重建效率
- `fake_rate`：误判率
- `mean_rms`：平均位置残差
- 训练收敛情况（从 .out 日志中找 loss 曲线）

判断：
- 上轮假设是否被证实？
- 结果与预期的差距在哪里？
- 是否满足 `research_goal.md` 中的成功标准？→ 如果满足，设置 `goal_achieved = true`，写成果报告，退出

---

## 步骤 4：形成下一个假设

基于：
- 当前结果与目标的差距
- 已经尝试过的改动和它们的效果（从 research_log.md 中看）
- 对当前代码结构的理解

**要求**：假设必须是可证伪的，格式为：
> "如果我做 [具体代码改动]，efficiency 应该提升 [幅度]，因为 [物理/机器学习原因]。
> 风险：[可能失败的原因]。"

**思考维度**（按可能收益从高到低）：
- 消息传递聚合方式（mean pooling → attention？）
- 模型深度和容量（层数、hidden_dim）
- Loss 函数设计（Focal loss gamma、正负样本权重）
- 训练策略（学习率调度、warmup、早停）
- 边特征工程（利用现有的 6 维边特征是否充分？）

---

## 步骤 5：修改代码

按照假设修改以下文件（**仅限这些文件**）：
- `src/models.py`
- `src/layers.py`
- `src/train.py`
- `src/losses.py`

**绝对禁止修改**：
- `src/geometry.py`
- `src/simulator.py`
- `src/utils.py`
- `src/config.py`
- `CLAUDE.md`
- `autonomous_loop_prompt.md`
- `autonomous_watcher.sh`

---

## 步骤 6：运行测试（必须通过才能继续）

```bash
conda run -n e320root pytest test/ -v
```

- 如果测试失败：分析错误，修复代码，重新运行测试
- 如果修复后仍然失败：在 `experiment_state.json` 中设置 `error_state` 描述问题，更新 `status.md`，**退出**（不继续提交）

---

## 步骤 7：生成 PBS 作业脚本

确定本次训练需要的参数（基于你的代码修改），基于模板 `~/subs/auto_eval_template.sh` 生成：
```
~/subs/auto_loop{N+1}_{简短描述3词}.sh
```

其中 `N+1` 是新的循环编号，描述用下划线连接的3个英文词（如 `attn_agg_inet`）。

填入：
- `LOOP_LABEL`：替换为 `loop{N+1}_{描述}`
- `MODEL_ARG`：模型名（通常 `InteractionNet` 或修改后的名称）
- `--TRAIN_ARGS_HERE`：本次实验的完整训练参数
- PBS 资源根据需要调整（通常 128gb/8cpus/1gpu/10h 足够）

**生成脚本时必须遵守的路径规范**（避免已知 bug）：

1. **PBS 日志用绝对路径**：
   ```bash
   #PBS -o /srv01/agrp/yiwen/logs/auto_LOOP_LABEL.out
   #PBS -e /srv01/agrp/yiwen/logs/auto_LOOP_LABEL.err
   ```
   不得用相对路径（`logs/...`），因为 `~/E320simulator/logs/` 目录不存在。

2. **进入项目目录用绝对路径**：
   ```bash
   PROJ_DIR="/srv01/agrp/yiwen/E320simulator"
   cd "${PROJ_DIR}"
   ```
   不得用 `cd $PBS_O_WORKDIR && cd E320simulator`——PBS_O_WORKDIR 取决于 qsub 提交时的目录，若已在 E320simulator/ 内提交则会嵌套失败。

---

## 步骤 8：更新所有文档

**research_log.md**（追加，不修改已有内容）：
```markdown
## Loop {N+1} — {当前日期时间}

### 上轮结果回顾
- efficiency: {值} | fake_rate: {值} | mean_rms: {值}
- 评估: {假设是否成立，为什么}

### 当前假设
{完整假设陈述}

### 代码修改
- `src/{文件}` 第 {行} 行: {描述改动}
- ...

### 预期结果
{定量预期：efficiency 应达到 X，因为 Y}

### PBS 作业
脚本: ~/subs/auto_loop{N+1}_{desc}.sh
提交时间: {当前时间}

---
### 实际结果（下轮填入）
（待填）
```

**experiment_state.json**：
- `loop_count`: +1
- `loop_status`: "submitted"
- `current_loop_label`: "loop{N+1}_{desc}"
- `current_loop_script`: "~/subs/auto_loop{N+1}_{desc}.sh"
- `last_updated`: 当前时间

**status.md**：用当前信息覆盖更新（参考模板格式）：
- 当前循环编号/上限
- 当前假设（一句话）
- 上轮结果数字
- PBS 作业 ID（下一步提交后填入）
- 研究进展简表

---

## 步骤 9：Git commit + push（必须在 qsub 之前）

```bash
# 确认在正确的研究分支（不是 master）
git branch --show-current

# 提交所有改动
git add src/models.py src/layers.py src/train.py src/losses.py
git add experiment_state.json research_log.md status.md
git add ~/subs/auto_loop*.sh
git commit -m "auto loop{N+1}: {假设的一句话描述}"
git push origin {research_branch}
```

如果 push 失败（需要 pull）：
```bash
git pull --rebase origin {research_branch}
git push origin {research_branch}
```

---

## 步骤 10：提交 PBS 作业

```bash
qsub ~/subs/auto_loop{N+1}_{desc}.sh
```

记录返回的 job_id，更新 `experiment_state.json`：
- `current_pbs_job_id`: "{job_id}"

再次 push：
```bash
git add experiment_state.json status.md
git commit -m "auto loop{N+1}: PBS job {job_id} submitted"
git push origin {research_branch}
```

---

## 首次运行（loop_count == 0）时的特殊处理

步骤 3（分析上轮结果）跳过，直接从步骤 4 开始。

在步骤 4 中，基线是当前代码 + `progress.md` 中记录的已知结果（InteractionNet: efficiency 70.6%, fake_rate 14.3%）。

---

## 紧急情况处理

如果在任何步骤遇到无法处理的异常（文件不存在、命令失败等）：

1. 在 `experiment_state.json` 中设置 `error_state: "Loop {N}: {错误描述}"`
2. 在 `status.md` 中写明需要人工干预的原因和建议的修复步骤
3. git commit + push 状态文件
4. 退出，不提交 PBS 作业
