# 阶段 24：hidden、FC 输出、GPU KV 的完整系统消融

## 结论

关闭 context grad 后，直接传 GPU context KV 确实省掉 CPU 投影，三次 16K
请求的平均 CPU 更新时间从 **906.37 ms 降到 836.48 ms（-7.71%）**，
CPU 更新次数从 68–69 增至 73–74。但 decode 吞吐仅从 **227.08 增至
228.94 token/s（+0.82%）**。

传 FC 输出的 CPU 更新时间为 **864.98 ms（-4.57%）**，decode 为
**228.76 token/s（+0.74%）**。它的 context 数据量最小，并保留 CPU 当前权重
构建新 KV 的策略。两种压缩格式的 decode 相差约 0.08%，本实验不支持宣布
哪一种具有稳定的端到端性能优势。

FC 第三次 16K 请求仅截断处最后一个 token 与原始基线不同，尚未定位实际
verification 中的原因。因此本阶段保留 `hidden` 默认，交付三种可复现消融路径，
不将 FC 或 KV 宣称为已证明无损的默认替代。

## 配置与测量口径

- RTX PRO 6000 Blackwell，Intel Xeon w7-3445；Qwen3-8B + Qwen3-8B-DFlash-b16。
- K=15，单 GPU/单请求，target/draft FULL graph，FLASHINFER，关闭 prefix cache。
- CPU 训练 16 线程、核 0–15；推理核 18，capture 核 19。
- context grad 关闭，GQA 优化、最后层监督行计算、BF16 fused AdamW，lr=2e-4，
  最近 16 个完整 block 训练，位置衰减 KL gamma=7。
- 延续 EAGLE3 handoff：训练线程发布 snapshot，推理线程提交异步 H2D 并 query 安装。
- 三组均移除 FC/hidden_norm 的无效上传，capture 位置和 epoch marker 顺序一致。

AIME Q15 的 prompt 为 286 token，自然生成 16098 token，到总上下文 16384；
没有补齐 prompt，也没有强迫自然提前结束的题目继续生成。这一题在长度上限
截断，没有最终 boxed 答案，仅用作长请求性能测量。

无 profiler 的顺序为 hidden→fc→kv、kv→fc→hidden、fc→hidden→kv，共 9 次。
另跑每种格式的 AIME Q1/Q2 自然结束回归，以及各一份 early/last trace。
所有 GPU 运行串行；trace 吞吐不进入性能表。decode 使用 engine 的首/末 token
时间戳；合计吞吐为总 decode token / 总 decode 时间。完整请求包含 prefill、
finish/reset，不含引擎初始化。多请求的 draft/accepted 计数先做累计值差分。

## 16K 无 profiler 结果

| 格式 | 三次 decode token/s | 合计 decode | 完整请求 token/s | 平均 GPU round ms | 接受长度 |
| --- | --- | ---: | ---: | ---: | ---: |
| 多层 hidden | 227.21 / 225.48 / 228.55 | 227.08 | 224.48 | 14.503 | 3.2940 |
| FC 输出 | 228.01 / 230.32 / 227.95 | 228.76 | 226.14 | 14.511 | 3.3196 |
| GPU KV | 228.98 / 228.61 / 229.22 | 228.94 | 225.39 | 14.509 | 3.3218 |

| 格式 | CPU 更新次数 | GPU 安装次数 | 平均 CPU 更新 ms | 最后 10 次更新均值 ms | 最后 10 次每更新覆盖 GPU round |
| --- | --- | --- | ---: | ---: | ---: |
| 多层 hidden | 69 / 68 / 69 | 68 / 67 / 68 | 906.37 | 1075.21 | 78.97 |
| FC 输出 | 71 / 71 / 71 | 70 / 70 / 70 | 864.98 | 1051.65 | 77.67 |
| GPU KV | 74 / 73 / 73 | 73 / 72 / 72 | 836.48 | 1029.48 | 76.73 |

CPU 更新时间来自未启用 profiler 的 train_rounds 范围，不含 snapshot/H2D。
每次更新的历史长度、监督行数和窗口边界随异步运行变化，表格不是同一批样本的
算子对照。三组 GPU round 时间接近，吞吐差异主要对应接受长度的小幅变化。
三次重复的范围存在重叠，不能把亚百分比差异当作统计显著的最优方案。

训练队列峰值分别为 **86 / 80 / 79** 个 round，capture 队列峰值为 **10 / 8 / 10**。
压缩格式缓解了训练积压，但没有消除长上下文 query attention 的前反向成本。
CPU 更新与 GPU 安装次数分开统计，所有请求结束后的 CPU/GPU version 均恢复 0。

阶段 22 的 225.87 token/s 是历史结果。本轮 hidden 已同时减少上传参数、移动
capture hook，不能把相对阶段 22 的全部变化归因于 payload 格式。

## CPU 投影的独立微基准

真实 checkpoint、16 CPU 线程、相同输入与位置，正反顺序交错执行，预热 2 轮，
记录 10 轮中位数；每次结果与同版本完整投影逐张量比对。这里只计一次 context
物化，不包括历史拼接、query forward/backward、optimizer 或数据传输。

| 新增 context token | 多层 hidden ms | FC 输出 ms | 已有 KV 解包 ms |
| --- | ---: | ---: | ---: |
| 64 | 8.246 | 5.614 | 0.046 |
| 256 | 14.681 | 10.470 | 0.045 |
| 512 | 20.318 | 13.389 | 0.049 |

hidden/fc 仍在 forward 前构建窗口 context，并在 optimizer 后投影新增段。
KV 两次均解包已有数据，保留 GPU 版本。减少投影不等于消除整个 CPU 更新，
也不等于消除历史 KV 拼接；last trace 中 KV 的 rebuild_cache 中位仍约 9.93 ms。
isolated 微基准与并发在线测量的缓存、带宽和窗口不同，不能直接相减解释整机收益。

## 数据量与 overlap

每次 H2D 从 **2097252864 降为 1929472512 字节（-8.00%）**，三组相同。
减少的是 FC 160 MiB 和 hidden_norm 8 KiB；共享 K/V 和 k_norm 继续训练、上传。
last trace 单次 H2D DMA 约 **34.57 / 34.64 / 34.60 ms**。更新次数增加也会
增加整次请求的上传总量，不能仅以单次 bytes 解释系统收益。

稳态每轮仍传输 16 行 target 数据，在 capture worker 去掉拒绝行；没有在 GPU
主线程回读拒绝数。teacher logits 和 metadata 保持不变。

| 格式 | context KiB/token | 完整 D2H MB/轮 | 相对完整 D2H 减少 |
| --- | ---: | ---: | ---: |
| 多层 hidden | 40 | 5.517712 | — |
| FC 输出 | 8 | 4.993424 | 9.50% |
| GPU KV | 20 | 5.190032 | 5.94% |

teacher logits 每轮为 4.861952 MB，是 D2H 的主要部分。因此 context 缩小
50%/80%，不会让整轮 D2H 缩小同样比例。

| 格式/阶段 | H2D overlap | D2H overlap | 推理 stream kernel 时间占比 | 采集 GPU 工作 μs/轮 |
| --- | ---: | ---: | ---: | ---: |
| hidden early | 99.09% | 98.80% | 99.01% | 10.79 |
| fc early | 99.09% | 98.68% | 98.90% | 10.18 |
| kv early | 99.02% | 98.44% | 99.04% | 12.13 |
| hidden last | 99.19% | 98.13% | 99.32% | 10.83 |
| fc last | 99.18% | 98.91% | 99.34% | 10.24 |
| kv last | 99.15% | 98.95% | 99.31% | 12.17 |

early 为请求开始的 400 step、总上下文预算 4096；last 在 4200 step 后截取
400 step、总预算 16384。每个 trace 约 16 MB，没有保存整个请求的巨大 trace。
两套统计实现的 overlap 一致：DMA 区间并集与推理 kernel 区间并集的交集 / DMA
时间。kernel 时间占比不是 SM 利用率，未重叠 DMA 也不能直接当成推理阻塞时间。
本轮 H2D 均完整落在记录的推理首尾范围内，不存在尾部裁剪导致的低估。

KV 采集成本包含 capture annotation 外额外 stack 打包的 GPU kernel，通过其
CPU aten::stack→CUDA correlation 关联，逐轮唯一匹配。没有漏掉这部分计算，
也没有把 GPU user annotation 与 CPU annotation 重复算成两倍轮次。

全部 trace 的 H2D stage 内没有 CUDA synchronize，stage/commit 在推理线程；
D2H event 等待位于 capture 线程，CPU train 位于训练线程。last 没有主推理
stream 的 4 字节 pageable sequence-length D2H。可说明推理连续性保持良好，
不能表述为数据搬运零成本或 100% 完全重叠。

## 质量、数值与末 token 差异

三组的 AIME Q1/Q2 均自然结束，答案 **277 / 62** 正确，输出分别为
3521 / 11108 token，全部与不训练基线逐 token 相同。两题合计 decode 分别为
**310.76 / 311.47 / 312.12 token/s**；完整请求分别为
**291.01 / 305.18 / 301.72 token/s**，后者受 finish 时机波动影响更大。
这只是两题回归，不能当作广泛准确率评估。

九次 Q15 中八次逐 token 相同。FC 第三次前 16097 个 token 相同，最后一个
输出 token（零基 index=16097、绝对位置=16383）从 `inner`（9179）变成
`invalid`（8318）。输出长度相同，只有这一个位置不同，且两种续写都处于截断句中。

另外用不带 speculative decoding 的 target，在同一前缀上分别以 0/16/64 个
suffix decode 重新测量：三次均选择 `inner`，前缀均一致，两个候选 logprob
差均为 **0.125**。该检查未复现当时实际 verification 的执行形状和历史，因此
**不能将差异直接判定为 BF16 误差，也未证明是 payload 错位**。这是尚未定位的
输出一致性限制，不隐藏该样本、不从吞吐统计中删除，也不宣称 FC 完全无损。

各格式在 W0 和非零安装后均通过真实 CPU/GPU 数值检查：context relative RMSE
最大约 0.00151，logits relative RMSE 最大约 0.01363，forward KL 最大约
0.00108，target/draft 参数存储隔离通过。KV 的 context RMSE=0 是同一份 GPU
KV 复制到 CPU 的结果，不是新的独立投影精度证明。

## GPU KV 版本审计

另跑两个 1024-token 请求，各保留 160 个真实 batch，序列化放在 finish 后，
该运行不计入性能。每个 batch 的 KV payload 宽度为 10240；prompt、确认行、
proposal source version 和 context projection version 分别保存。

两个请求各观察到 2 个安装边界：context_version 比 proposal_version 高 1，
其他轮次相同。CPU 更新开始版本减去新采集 context 版本的集合均为 `{0, 1}`。
证明实际安装边界上两个版本确实可能不同，不能用 proposal version 替代 context
version。CPU 完成更新后保留旧 KV，未来还会继续增加历史版本差；未重建全历史。
两个请求均完成 3 次 CPU 更新、2 次 GPU 安装，验证 epoch close 与 reset 能继续推进。

## 代码、验证与复现

实现与不变量见[阶段 23](stage23_context_payload_implementation.md)。复用原有
trainer、capture queue、pinned buffer、stream 和 weight handoff；无新增上传线程
选择开关。`--context-representation` 是本次消融的格式选择，后两种格式必须与
`--no-context-grad` 配合。默认 `hidden` 保留，三组均自动去掉不更新参数的上传。

168 项现有/扩展测试通过，修改源码的 pre-commit 与 mypy Python 3.12 通过。
没有提交或推送。首次真实模型启动因当前虚拟环境缺少 ninja 失败两次，已用
`uv pip install --python .venv/bin/python ninja` 补齐 1.13.2；失败不计入任何测量。

```bash
export PATH="$PWD/.venv/bin:$PATH"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE
export MALLOC_MMAP_THRESHOLD_=1073741824 MALLOC_TRIM_THRESHOLD_=-1

.venv/bin/python online_draft/scripts/smoke_async_dflash.py \
  --online --target-graphs --train-threads 16 --cpu-isolation \
  --no-context-grad --context-representation fc \
  --grouped-query-attention --selective-last-layer \
  --learning-rate 2e-4 --max-train-rounds 16 --loss-decay-gamma 7 \
  --aime-questions 1 --aime-offset 14 --context-tokens 16384 \
  --max-model-len 18432 --decode-timeline --queue-metrics \
  --output /tmp/dflash-fc-context.json
```

替换 `fc` 为 `hidden` 或 `kv` 即可复现另外两组。early 增加
`--profile-dir <目录> --profile-max-iterations 400` 并设 `--context-tokens 4096`；
last 保留 16K 并增加 `--profile-delay-iterations 4200 --profile-max-iterations 400`。

## 产物

- [完整结果与校验汇总](context_payload_results.json)
- [实验目录、命令、日志、源码哈希与分析脚本](../../results/experiments/dflash_context_payload_20261019/)
- [hidden early](../../results/experiments/dflash_context_payload_20261019/hidden-early/early.pt.trace.json.gz) / [hidden last](../../results/experiments/dflash_context_payload_20261019/hidden-last/last.pt.trace.json.gz)
- [FC early](../../results/experiments/dflash_context_payload_20261019/fc-early/early.pt.trace.json.gz) / [FC last](../../results/experiments/dflash_context_payload_20261019/fc-last/last.pt.trace.json.gz)
- [KV early](../../results/experiments/dflash_context_payload_20261019/kv-early/early.pt.trace.json.gz) / [KV last](../../results/experiments/dflash_context_payload_20261019/kv-last/last.pt.trace.json.gz)
