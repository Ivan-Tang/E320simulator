# Agent Memory
_由 agent 自动维护，跨重启持久。用 `!memories` 查看，`!remember` 添加，`!forget` 删除。_

## 操作规则（必须遵守，防止超时）

- **禁止内联轮询 PBS 任务**：提交 `qsub` 后立即返回 job ID，不要用 `qstat` 轮询等待结果。任务完成时 agent 监控线程会自动通知。
- **禁止内联等待长时间运算**：任何预计超过 5 分钟的计算（数据生成、边构建、模型训练）必须通过 PBS 批处理，不要在 Claude 会话内直接运行并等待。
- **超时上限**：Claude 会话硬超时 3600s，内联等待必然导致超时且用户看不到任何输出。

## 集群状态
<!-- 记录节点问题、配置特殊情况等 -->
- **gwn247（A6000）节点故障**：调度到该节点的 HGNN job 立即 SIGSEGV 崩溃，无任何输出。gwn246 也曾导致 combined benchmark 全部 SIGSEGV。遇到不明 SIGSEGV 时优先检查是否在 gwn246/247 上。
  - 修复：在 PBS 脚本加 `#PBS -l gputype=A5000`，路由到 gwn243/244（A5000，已验证正常）。
  - 节点 GPU 对照：gwn241/242=RTX3090，gwn243/244=A5000，gwn245/246/247=A6000（247已确认故障）

## 用户偏好
<!-- 记录用户习惯、偏好的模型/参数等 -->

## 关键结论
<!-- 记录实验发现、重要决定、已证实/证伪的假设 -->
- [2026-04-16] **balanced_sampling=True 效果确认（Job 4040917，200 epochs，gwn244）**：
  - GNN 效率 51.4%→**92.84%**（暴涨），但 fake_rate 也升至 79.48%，F1 仅 33.6%——模型偏保守→激进，需调阈值或后处理
  - EggNet 效率 1.4%→**67.95%**，完全恢复，证明原失效根因为 balanced_sampling 关闭
  - HGNN 效率 14.2%→**46.04%**，同步改善
  - **InteractionNet 效率 70.6%→52.0% 下降**，需排查是否与 balanced_sampling 相互作用或 checkpoint 差异有关
  - MLP 74.3%→79.45%，小幅改善
  - 全局最优仍是 TransformerEdgeClassifier（82.01% / 11.66%，auto-research loop 7，未参与本次 benchmark）

## 进行中的问题
<!-- 记录未解决的问题、待跟进的事项 -->
- [2026-04-09] agent.py monitor_pbs_jobs() 有个 bug：qstat 瞬态返回空时会误报所有已知作业"已结束"，已在 ebeeb00 修复（current空+known非空时跳过本轮）
- [2026-04-13] A6000节点gwn245/246/247均会导致SIGSEGV（不只是gwn247），所有训练job需加`#PBS -l gputype=A5000`路由到gwn243/244
- [2026-04-16] GNN fake_rate 79.48% 过高（效率 92.84%）：需要调低 edge-threshold 或加强后处理，找效率-纯度最优点
- [2026-04-16] InteractionNet 效率意外下降（70.6%→52%）：可能是本次 checkpoint 质量差，或 balanced_sampling 对 InteractionNet 反而有负面影响，待诊断
- [2026-04-16] 下一步：运行 diagnose_failures.py 分析 18% 效率损失的三类根因（图构建/模型/后处理），再决定优化方向
