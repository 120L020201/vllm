# 阶段 7：真实长请求暴露的大窗口成本

阶段 6 的短请求已经把生成开销降到约 2%，但 AIME 测量显示结束等待仍可能很长。
本阶段继续优化 CPU attention，不改变训练窗口或请求结束策略。最终代码将
`attention_blocks_per_chunk` 从阶段 6 的 8 调整为 **64**。阶段 8 汇总最终模型
测量及检查；阶段 6 的数字保留为真实的中间版本结果。

## AIME 初筛

本地 `/data/benchmarks/aime2026.jsonl` 前四题，使用原题 prompt，Qwen thinking
开启，greedy seed=17，最多 8192 输出 token、上下文 12288、K=15、FULL graph。
online 使用 16 核隔离、学习率 1e-5、attention 分组 8。baseline 是完全关闭
online 的独立进程。没有把第一道不同题型的请求当成可交换预热样本而删除。

| 题号 | 输出 token | baseline 生成 s | online 生成 s | online 结束等待 s |
| --- | ---: | ---: | ---: | ---: |
| 1 | 3521 | 10.664 | 11.066 | 8.275 |
| 2 | 8192 | 27.361 | 28.150 | 12.234 |
| 3 | 8192 | 31.852 | 32.822 | 71.556 |
| 4 | 8192 | 31.781 | 32.733 | 10.076 |

四题的 baseline/online token IDs 分别完全一致。第一题正常结束且答案 277 正确；
后三题达到 token 上限，没有最终正确答案，不能将此称为完整 AIME 准确率评估。
GPU 分别安装 3/4/5/4 个版本，验证了重复使用两个 graph slot 与每请求 reset。

第三题最后启动的窗口积累 1066 轮，CPU 训练约 77.06 秒，生成结束后仍需等待。
这不是 H2D 同步：窗口训练已经启动，当前 finish 约定要求完成它，再恢复 W0。
GPU 比 CPU 更快地产生 round，窗口又由实际安装决定，因而会形成越来越大的窗口。
短请求的几秒实验不足以揭示此问题。

## 大窗口 attention 分组对照

实际 checkpoint 的 CPU benchmark，固定单线程生成同一组随机输入，随后恢复
16 线程，并固定 affinity 0–15。prompt 2048，先做 bootstrap，再训练 384 轮，
每轮完整 query=16、确认位置=3、词表=151936；teacher 为合成均匀分布。因此这里只
比较计算/内存成本。峰值是整个进程的 `ru_maxrss`，包括模型和输入，不是单独激活。

| 每组 round 数 | 窗口 s | 进程峰值 GiB |
| --- | ---: | ---: |
| 8 | 13.578 | 35.49 |
| 32 | 10.491 | 27.72 |
| 64 | 9.651 | 26.58 |
| 128 | 10.797 | 26.07 |
| 整窗口 | 12.383 | 25.87 |

64 组相对 8 组减少约 28.9% 窗口时间和 25.1% 进程峰值。分组太小会重复构建和
保存 context K/V，放大复制和 backward 累积；分组太大则计算更多必然被 mask 的
跨轮 query 关系。当前硬件实测中 64 是这组候选的最佳折中，并非所有上下文长度
的数学最优值。BF16 loss 变化约 0.00033，FP32 前向/参数及 context 梯度对照通过。

## 撤回 GQA query 重排

曾尝试把共享一个 KV head 的多个 query head 折入 query 行，用普通 SDPA 避免
`repeat_interleave` 复制 KV。数学上仍是相同的非因果 attention；含多 KV head 的
前向和参数/context 梯度对照通过。但该 PyTorch CPU kernel 上分组 8/32/64 的
窗口时间变成 14.407/11.679/11.657 秒，均慢于对应原布局，因此删除实现。
最终继续使用清楚直接的标准 GQA 展开；多 KV head 梯度回归保留。

## 16 与 18 个训练核

64 分组、同样输入，交替重复：16 核 10.551/10.377 秒，18 核 11.190/10.959 秒。
新增核心没有带来收益，保留长请求 16 核配置。不同 CPU 并行度下 BF16 归约顺序
会改变数值，因此这不是逐 bit 等价实验；没有以合成 loss 判断训练优劣。

## 复现与证据

```bash
OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OMP_WAIT_POLICY=PASSIVE \
taskset -c 0-15 .venv/bin/python online_draft/scripts/benchmark_dflash_cpu.py \
  --threads 16 --rounds 384 --prompt-tokens 2048 --confirmed-tokens 3 \
  --attention-blocks 64 --output /tmp/dflash-tuning/cpu-large-blocks64.json
```

原始数据：`cpu-large-blocks*.json`、`cpu-gqa-blocks*.json`、`cpu-threads*.json`，
以及 `aime-baseline.json`/`aime-online.json`，都在 `/tmp/dflash-tuning/`。
`test_dflash_training.py` 此阶段 10 项通过，覆盖 bootstrap 梯度、旧 KV detach、
连续历史、新增段版本、整窗口与分组 mask，以及多 KV head 的显式数学参考。

CPU 分组调优能降低单个窗口成本，但无法保证更短的完整请求时间：更早安装也可能
触发下一次更大窗口。因此最终同时报告生成、结束等待、CPU/GPU 版本和窗口轮数，
不以一次 CPU microbenchmark 的加速比代替服务端收益。
