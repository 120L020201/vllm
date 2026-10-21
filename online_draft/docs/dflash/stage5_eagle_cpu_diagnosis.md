# 阶段 5：EAGLE3 对照与 CPU 瓶颈定位

2026-10-16，在相同工作站与 `eagle3&dflash` worktree 上继续实验。目标是降低在线
训练对生成和请求结束的成本，保留请求内 W0 reset、真实安装划分 epoch、混合历史
KV、完整非因果 block，以及现有冻结和 KL 设置。

## EAGLE3 短请求对照

复用同一个 smoke 脚本和数学解释 prompt，target 为 Qwen3-8B，eager target/draft，
4 个 CPU 线程、`OMP_WAIT_POLICY=PASSIVE`，greedy、seed=17、1024 输出 token。
EAGLE3 使用其原有实验的 K=7；DFlash 仍为 K=15，所以这是匹配服务条件的对照，
不是相同模型或相同 query 数的算力比较。EAGLE3 自有 lm_head 继续参与训练。

每种模式 4 次请求，排除第一条后：

| 模式 | 平均生成吞吐 token/s | 请求内 GPU 安装次数 | 结束等待 s |
| --- | ---: | ---: | --- |
| EAGLE3 baseline | 139.20 | 0 | 约 0.001 |
| EAGLE3 online | 113.30 | 13–15 | 0.177–0.680 |

基线单次有明显抖动，因此不把约 18.6% 的净下降视为精确开销归因，但该结果足以
否定“EAGLE3 在所有场景都没有在线成本”的解释。前次 AIME 长请求的接受收益与本次
短请求表现不能直接等同。此次 EAGLE3 窗口约 20–38 轮，训练通常 0.4–0.7 秒；
它能在请求内使用多次更新，DFlash 原始实现则通常只有 W1。

原始数据：`/tmp/dflash-tuning/eagle-baseline-k7.json`、
`/tmp/dflash-tuning/eagle-online-k7.json`。两次引擎均正常结束。

## DFlash CPU 分阶段时间线

新增 `benchmark_dflash_cpu.py`，只在 CPU 上使用实际 checkpoint 权重与连续的合成
H/teacher 窗口定位计算成本，不将其 loss 作为模型质量指标。标准样本为 prompt=55、
完整 query=16、每轮 6 个确认位置，bootstrap 后训练 96 轮。真实服务收益另行测量。

4 线程 CPU profiler 结果：

| 阶段 | 时间 ms |
| --- | ---: |
| forward（含 head） | 1525.2 |
| loss | 592.7 |
| backward | 3854.6 |
| optimizer | 327.9 |
| 新增 KV 投影 | 67.0 |

窗口合计约 6.44 秒。矩阵乘法 self time 约 3.30 秒；CPU Flash attention backward
约 0.65 秒。代码确实使用 `_scaled_dot_product_flash_attention_for_cpu`，没有落到
手写普通 attention。仅增加 CPU 并行度的初筛中，8/16 线程窗口约 3.82/2.73 秒，
但它们是不同运行且未开 profiler，仅用于选候选配置，不当作严格加速比。
之后 benchmark 固定用单线程生成合成输入，再恢复训练线程数，避免随机输入随
线程数量变化。

## 初步工程结论

只做 CPU 核隔离并使用 16 个训练线程，eager 生成仍约 3.33 秒，虽然结束等待
降至约 1.35 秒。说明“CPU 训练更快”不保证“GPU 推理更快”，不能只优化 CPU
单次更新的时间。

双 slot CUDA Graph 是更有效的方向：每个固定参数地址集合捕获一份 draft graph，
安装时选择对应 graph，仍由 event.query 决定是否切换，不等待上传，也不在请求内
重新 capture。配合 8 核训练隔离，初筛 3 组热请求中 baseline 2.408 秒、online
2.457 秒，结束等待约 1.76 秒。该初筛结果将在后续阶段扩大样本并验证 slot 重用、
reset、长请求和数值一致性，不在本阶段宣称最终达到零干扰。

已新增并通过实际 CUDA graph 回归：分别捕获两个 slot，连续安装 3 个明显不同的
版本，每次核对 graph 输出与当前参数 eager 输出一致，并核对 reset 后恢复 W0
输出。相关 weight-slot suite 共 8 项通过。
