# 阶段 17：GQA 重排接入与完整 CPU 训练基准

2026-10-19，`eagle3&dflash`，base `3979838d2d`，未提交工作树。

## 结果

在真实 Qwen3-8B-DFlash-b16 权重、16000-token 合成历史和关闭 context 梯度的
CPU 训练基准中，GQA query 重排将 N16 热更新从 **1123.22 ms 降到 969.04 ms**，
减少 13.73%。进一步减少每步完整训练 block，N8/N4/N2 分别为
806.25/596.90/567.90 ms。N2 相对 N4 的额外收益只有约 4.86%。

| 布局 | 完整训练 block | 热更新中位数 ms | 相对原 N16 耗时减少 | 进程峰值 RSS GiB |
| --- | ---: | ---: | ---: | ---: |
| K/V 展开，原实现 | 16 | 1123.22 | — | 15.80 |
| query 重排 | 16 | 969.04 | 13.73% | 15.41 |
| query 重排 | 8 | 806.25 | 28.22% | 14.97 |
| query 重排 | 4 | 596.90 | 46.86% | 14.88 |
| query 重排 | 2 | 567.90 | 49.44% | 14.97 |

每组独立进程、串行运行，无 profiler。先 bootstrap，再重复训练同一个固定窗口
5 次，舍弃第一次窗口计时，表中取后 4 次中位数。固定 CPU 0–15、16 线程，
BF16、fused AdamW、PASSIVE、glibc 分配器参数。prompt=16000，窗口 96 轮，
每轮确认 3 token，teacher 为合成均匀分布，学习率沿用脚本默认 1e-5。
窗口训练读取到的历史最多 16288 token，新增持久历史至 16291 token。

这是全 CPU 模型训练成本，包括投影、五层 query、head、loss、backward、更新
和新 KV 追加，但不包含 capture、snapshot、H2D 或 GPU 推理。CPU 基准不能
评价在线学习效果；真实服务仍使用此前选定的 1e-4，系统结果另见阶段 18。

## 实现和不变量

`Qwen3DFlashConfig.grouped_query_attention` 控制 CPU attention 布局；worker
通过 `additional_config.online_draft.grouped_query_attention` 传入，两套实验
脚本增加 `--grouped-query-attention`。关闭时保留原 K/V 展开路径，便于受控对照。

开启时将共享一个 KV head 的 query head 合并到 query 维，显式 mask 重复相应
次数，SDPA 之后恢复原 head/query 布局。完整非因果 block、可见历史、绝对位置
RoPE、context KV 的版本与梯度范围不变。没有修改 GPU 推理 attention、双 slot、
capture、异步调度或权重安装。BF16 归并顺序差异见阶段 16，不能保证逐位梯度相同。

N8/N4/N2 使用现有 `max_train_rounds`，不改变 GPU 实际安装关闭窗口的语义。
所有 round 的历史仍然连续处理，只对最近 N 个完整 block 做 loss 和 query
forward/backward。它们是训练样本量消融，不能当成数学等价优化。

## 验证

扩展已有 `test_dflash_training.py`，用最便宜的 tiny FP32 模型验证具体行为：

- 两种布局、1/2 个 KV head、2:1/4:1 GQA 的输出、参数和 context 梯度，均与
  显式 attention 数学参考一致。
- 重排后的 packed window 与原布局逐 block 参考的 KL、Adam 一阶状态一致；
  覆盖整窗口、每 chunk 1/2 个 block，以及 context 梯度开启/关闭。
- 复用已有连续历史、首拒绝位置、未监督 suffix、请求 reset、双 slot 安装、
  transport 和失败传播测试。

相关完整测试 **145 passed，14 个原有 JIT 弃用 warning**，pre-commit 通过。
测试日志和命令保存在实验目录。未把真实服务验证尚未完成的候选改为默认策略。

为后续服务实验增加可选 `--queue-metrics`，只在 benchmark RPC 中安装轻量
callback，记录 capture/train 提交时队列深度及已准备 round 在 CPU FIFO 中的
等待秒数；不逐轮打印，不改变队列和训练调度。等待时间从 CPU batch 准备完成
到训练线程取出该 round，不含此前 D2H，也不包含该 round 进入 epoch 后等安装
关闭窗口的时间。因此它不能独自代表完整样本年龄。

## 追加消融：最后一层只输出监督位置

增加 `selective_last_layer` / `--selective-last-layer` 候选：前四层仍计算全部
query；最后一层的 Q/K/V 投影仍来自完整 block，K/V 不裁减，只对最终需要监督
的行执行 attention 输出、输出投影、残差、MLP 和最终 norm。LM head 原本就
只处理这些行。这与在前四层删除未监督 query 不同：最后一层的其他输出没有
后续消费者，保留它们的 K/V 即保留了监督行所需的非因果依赖。

在既有梯度测试中加入开关，覆盖 chunk=整窗/1/2、context_grad=True/False；
与原布局逐 block 全输出参考的 loss 和参数梯度一致。完整回归增加至
**151 passed，14 个原有 warning**，pre-commit 和 mypy-3.12 通过。
真实 checkpoint 的 startup 数学对照还检查了监督行在 W0 和非零参数安装后的
GPU/CPU logits，其 KL 分别约 0.000787 / 0.001015，均通过原有阈值。

| CPU 候选 | N16 热更新 ms | N8 热更新 ms |
| --- | ---: | ---: |
| 仅 GQA 重排 | 969.04 | 806.25 |
| GQA + 最后一层选行 | 948.78 | 796.31 |

相对仅 GQA 分别只减少 2.09% / 1.23%，不能把最后一层 query 行数减少比例
当成完整训练收益。旧历史的 K/V backward 等剩余成本仍然存在。单次服务筛选中
N16、lr=1e-4 的 decode 也只从 206.88 到 207.04 token/s，GPU 安装均为 74 次，
没有可靠的服务收益证据；完整重复和最终配置选择以阶段 18 为准。

本追加消融的 CPU 命令见 `cpu-selective-commands.json`，结果为
`cpu-selective-n16.json` / `cpu-selective-n8.json`。对应正确性和检查日志为
`final-tests.log`、`selective-precommit.log`、`selective-mypy312.log`。

## 复现与产物

[实验目录](../../results/experiments/dflash_system_optimization_20261019/) 保存
`cpu-commands.json`、五组 `cpu-*.json/.log`、`tests.log` 和 `precommit.log`。

```bash
OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OMP_WAIT_POLICY=PASSIVE \
MALLOC_MMAP_THRESHOLD_=1073741824 MALLOC_TRIM_THRESHOLD_=-1 \
taskset -c 0-15 .venv/bin/python online_draft/scripts/benchmark_dflash_cpu.py \
  --threads 16 --rounds 96 --prompt-tokens 16000 --confirmed-tokens 3 \
  --max-train-rounds 4 --no-context-grad --grouped-query-attention \
  --repeat 5 --output /tmp/dflash-cpu-gqa-n4.json
```
