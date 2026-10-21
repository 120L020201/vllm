# 阶段 18：DFlash 在线训练的完整系统优化与实验

2026-10-19，工作树 `/home/lixiang/work/myvllm-eagle3-dflash`，分支
`eagle3&dflash`，base `3979838d2d`。未提交或推送。

## 结果与推荐配置

本轮完成 7 组真实权重 CPU 基准、13 种系统配置的筛选与复测、四题完整生成
对照，以及 3 个截取 400 轮的 CPU/GPU trace。正式系统数据共 29 个请求，
profile 请求单列、不纳入吞吐。推荐组合为：

```text
grouped_query_attention = True
selective_last_layer = True
context_grad = False
max_train_rounds = 16
learning_rate = 2e-4
loss_decay_gamma = 7
BF16 + fused AdamW，CPU 训练 16 核
```

GQA 重排省掉长历史 K/V 的四倍展开；最后一层只输出监督行，保留完整 K/V。
继续训练原有 draft 参数范围，target embedding/head 冻结。context 不求导的
实验中，FC/hidden norm 与旧在线对照一样没有梯度；其他层仍通过 query 学习。
位置衰减 KL、全历史增量 KV、混合版本近似、GPU 实际安装关闭窗口、请求间 W0
reset 和双 slot/graph 保持不变。没有截断历史，也没有把未训练 block 延后积压。

仅使用 GQA、N16、lr=2e-4 的两次 decode 为 222.45 token/s，完整组合为
222.25 token/s，相差仅 0.09%，视为同一性能水平。最后一层选行的独立 CPU
收益约 2%，没有证据证明它能单独提高 decode；保留显式开关复现这项消融。
完整组合的后续四题和 trace 均完成验证。通用默认参数保留，以下命令显式启用
本轮测得的配置，避免将小题集结论变成未经验证的通用调度策略。

## 自然增长到 16K：重复对照

Qwen3-8B + Qwen3-8B-DFlash-b16，单 RTX PRO 6000 Blackwell 96 GB，
Xeon w7-3445，PyTorch 2.13.0+cu130。单活动请求，K=15，target/draft FULL
CUDA Graph，FLASHINFER，关闭 prefix caching。AIME 2026 第 15 题，thinking，
greedy seed=17，prompt=286，自然输出 16098 token，实际总上下文 16384；
`max_model_len=18432` 为 speculative block 留余量，没有填充 prompt 或禁用 EOS。

推理固定 CPU 18、单线程，采集固定 CPU 19、单线程，训练固定 CPU 0–15。
所有计时实验串行运行，OMP_WAIT_POLICY=PASSIVE，使用阶段 11 的 glibc 参数。
旧在线对照为 context_grad=False、N16、lr=1e-4、原 K/V 展开、最后一层全输出。
在线组均开启相同的轻量队列测量，不逐轮打印队列信息。

每组两个独立进程。吞吐按总 token / 总秒数合并，接受长度按总 accepted/drafts
合并。纯 decode 排除首 token；完整请求包含 prefill、输出、finish/reset 等待，
不包含引擎启动和模型加载。CPU snapshot/H2D 不在 `train_seconds` 范围内。

| 方案 | decode token/s | 平均 decode s | 完整请求 token/s | 接受长度 | GPU 安装 | CPU 更新 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 原始 DFlash，online 关闭 | 173.42 | 92.819 | 173.32 | 2.5355 | 0 / 0 | 0 / 0 |
| 优化前在线 | 204.59 | 78.678 | 201.49 | 3.0633 | 68 / 68 | 69 / 69 |
| 本轮推荐组合 | **222.25** | **72.428** | **219.67** | **3.3169** | 69 / 69 | 70 / 70 |

- 对原始 DFlash：decode **+28.15%**，完整请求吞吐 **+26.75%**。
- 对旧在线：decode **+8.63%**，完整请求吞吐 **+9.02%**。
- 推荐组合两次 decode 分别 222.36 / 222.14；旧在线 205.49 / 203.70；
  原始 173.49 / 173.35 token/s。没有固定异步安装轮次来消除真实运行波动。
- 这六次输出逐 token 相同，但本题在 16K 上限截断、没有最终答案，不能据此
  宣称解题准确率。完整作答验证见后面的四题实验。

| 总上下文区间 | 原始 token/s | 旧在线 token/s | 推荐组合 token/s |
| --- | ---: | ---: | ---: |
| 约 0–4K | 239.17 | 233.57 | 236.12 |
| 约 4–8K | 197.93 | 206.94 | 222.81 |
| 约 8–12K | 153.35 | 189.38 | 213.48 |
| 约 12–16K | 138.93 | 195.54 | 218.75 |

收益主要来自接受长度增加和验证轮数减少。平均每轮从 14.619 ms 到 14.921 ms，
仍增加约 **2.07%**；不能声称开启在线训练后每个 GPU round 本身更快或零开销。

![CPU 更新成本与分段 decode 对照](system_optimization_summary.png)

左图蓝点为单次消融，旧在线与推荐组合为两次合并；右图为两次分段合并。
图的可缩放版本为 [SVG](system_optimization_summary.svg)。

## 为什么没有选择更新次数最多的方案

下面列第一次筛选的实测，不与重复均值混用。E=原 K/V 展开，G=GQA 重排，
S=最后一层只输出监督行。所有在线行均关闭 context 梯度、保留完整历史。

| 布局 | 每步 block | 学习率 | CPU 平均更新 ms | GPU 安装次数 | decode token/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| E | 16 | 1e-4 | 1011 | 68 | 205.49 |
| G | 16 | 1e-4 | 910 | 74 | 206.88 |
| G | 8 | 1e-4 | 790 | 84 | 204.57 |
| G | 4 | 1e-4 | 651 | 100 | 203.48 |
| G | 2 | 1e-4 | 619 | 108 | 196.63 |
| E | 4 | 1e-4 | 792 | 85 | 202.87 |
| G+S | 16 | 1e-4 | 908 | 74 | 207.04 |
| G+S | 16 | 2e-4 | 903 | 69 | 222.36 |
| G+S | 16 | 4e-4 | 924 | 69 | 218.85 |
| G | 16 | 2e-4 | 899 | 70 | 222.04 |
| G+S | 8 | 2e-4 | 763 | 81 | 220.03 |
| G | 16 | 3e-4 | 913 | 68 | 222.08 |

N4/N2 的训练步确实更快，但每次更新的监督减少。GQA、lr=1e-4 的这次请求中，
N16 实际训练 1185 个 block、3735 个 teacher 行；N4 为 401/1194，N2 为
217/634。它们累计看到的有效监督显著更少，接受长度也下降。
这是支持保留 N16 的观测，不能把更低接受率完全归因于单一因素。

提高学习率到 2e-4 比继续减少 block 更有效；3e-4 未进一步改善，4e-4 回落。
没有把当前题集上的最佳步长解释为所有输入的普遍最优值。
最终生成更快、请求更早结束，所以 CPU 虽加速，安装总数只从 68 增到 69；
应同时看每步耗时、每秒安装率、监督量和接受长度。

## CPU 耗时、队列与版本

| Q15 两次运行统计 | 旧在线 | 推荐组合 |
| --- | ---: | ---: |
| 全部更新平均 ms | 1013.82 | 903.97 |
| 最前 8 个普通更新平均 ms，排除 bootstrap | 716.78 | 762.19 |
| 最后 8 个更新平均 ms | 1481.94 | 1105.51 |
| 前 8 个窗口平均完整 GPU round | 60.63 | 63.44 |
| 后 8 个窗口平均完整 GPU round | 102.19 | 78.88 |
| train queue 事件峰值 | 106 | 79 |
| capture queue 事件峰值 | 13 | 8 |
| 请求前四分之一 round 的平均 FIFO 等待 ms | 372.50 | 385.73 |
| 请求后四分之一 round 的平均 FIFO 等待 ms | 714.92 | 533.54 |
| CPU 开始版本减样本来源版本的最大值 | 1 | 1 |

完整更新平均减少 **10.83%**，末段减少 **25.40%**，末段 FIFO 等待减少
**25.37%**。短上下文没有同等收益，推荐组合的 early CPU 时间还略高。

queue 深度在提交事件时记录，可能包含关闭窗口标记，不等于严格 GPU round
数量。等待从 CPU batch 准备完成到训练线程取出 round，不含 D2H，也不含
round 已进入 epoch 后等 GPU 安装关闭窗口的时间。early/last 按 round 数四等分。
这些是本次 16K 请求的观测，不能证明更长请求不会积压。

样本版本差为 1 **不代表历史 KV 只落后一版**：prompt 和更早历史仍保留旧版本
KV。优化继续接受用户选定的混合版本近似，没有重投影全部历史。

CPU 实现、数学不变量、完整权重微基准和测试详见
[阶段 17](stage17_cpu_gqa_integration.md)。常用 BF16、fused AdamW、打包算子、
仅监督行的 LM head、线程隔离等优化已经启用；本轮没有重复启用此前无收益的
compile loss/MLP，也没有改为 FP32 master weights 或冻结前几层。

## 补充四题：自然结束、答案与完整请求成本

AIME 2026 第 1–4 题，各配置一个进程、每题从 W0 开始，仍允许总上下文到
16K，保留正常 EOS。三组都自然结束，输出 token 数为
3521 / 11108 / 9883 / 11175，合计 35687。

| 方案 | 合计 decode token/s | 完整请求 token/s | 完整请求总 s | 实际 GPU 安装 |
| --- | ---: | ---: | ---: | --- |
| 原始 DFlash | 262.45 | 262.14 | 136.136 | 0 / 0 / 0 / 0 |
| 旧在线 | 266.95 | 257.21 | 138.746 | 11 / 37 / 40 / 45 |
| 推荐组合 | **282.21** | **272.84** | **130.798** | 11 / 37 / 38 / 43 |

推荐组合对原始的 decode 提高 **7.53%**，完整请求吞吐提高 **4.08%**；对旧
在线分别提高 **5.72% / 6.08%**。四题答案依次为 277、62、79、70，三组均
4/4 正确，输出逐 token 相同。该集合也用于过往实验，不能称作完全独立的
留出测试集或完整 AIME 准确率评测。

| 题号 | 原始 decode token/s | 旧在线 | 推荐组合 |
| --- | ---: | ---: | ---: |
| 1 | 331.54 | 319.31 | 322.45 |
| 2 | 278.21 | 281.75 | 296.75 |
| 3 | 254.57 | 257.24 | 270.89 |
| 4 | 239.77 | 249.37 | 268.51 |

第一题仍比原始慢 **2.74%**，没有接受率收益，finish 还增加约 0.99 秒。
优化不能保证每个短请求都加速。四题每组只运行一遍，稳定性证据弱于 Q15 的
独立重复；表格保留这一限制。

## 候选中的一次输出分叉及诊断

推荐组合的两次 Q15 和四题均与对应 baseline 逐 token 相同。另一个候选
`gqa-n16-lr2-2` 在第 15109 个输出 token 开始分叉，前 15108 个一致；两次均
输出 16098 token，按相同下标比较有 866 个 token 不同。没有隐藏或删除这次结果。

在所有 Q15 候选共同一致的前 15000 个输出 token 内，按最后一个完整输出 batch
计时：原始为 177.71 token/s，旧在线 206.04，仅 GQA+2e-4 为 223.25，完整
组合 223.59。后缀文字变化不是本轮观察到加速的必要条件。

另外完全关闭 draft 和在线训练，固定分叉前相同 token 前缀，用原始 Qwen3-8B
重放；分别将最后 0/16/64 token 放到 decode，其余作为 prefill。三次前缀都匹配，
分叉位置均选择候选的 token 1741（`such`），原始 DFlash 那次为 279（`the`）。
target-only 的两个 token 的 logprob 差为 0.125 / 0.125 / 0.25。

这说明候选 token 可以由完全未训练的 target 产生；结合启动检查确认 mutable
draft slot 与 target 参数存储不重叠，证据支持执行布局/数值路径敏感的解释。
但没有保存那次发生分叉时的完整 target logits，不能声称精确复现并证明了
其唯一原因。[vLLM 的 speculative decoding 文档](https://github.com/vllm-project/vllm/blob/main/docs/features/speculative_decoding/README.md)
也区分分布意义的正确性与浮点执行路径下的逐位一致性。

诊断数据是 `target-margin-probe.json`。最初诊断启动因 PATH 缺少 Ninja 失败，
补齐项目虚拟环境 PATH 后成功；失败日志单独保留，没有计入任何模型结果。

## 截取的 CPU/GPU trace

两个推荐组合请求分别截取 early 与 last 各 400 个 worker step，另截原始
DFlash 的 late 400 step。全部启用 CPU all_threads，关闭 stack/shapes/memory
等大型记录项，gzip 压缩。early delay=0，推荐 last delay=4379，原始 last
delay=5576，后两者按各自正式运行约 14.8K 总上下文选择。
推荐最后一段覆盖的 token 更多，约到 16.1K；原始约到 15.7K，不能视为完全
相同位置的 attention 延迟对照。

| trace | 压缩大小 MiB | H2D 与推理 kernel 重叠 | D2H 与推理 kernel 重叠 |
| --- | ---: | ---: | ---: |
| 推荐 early | 15.95 | 97.15% | 87.56% |
| 推荐 last | 15.64 | 94.42% | 90.11% |
| 原始 last | 12.94 | 无在线传输 | 无在线传输 |

两段推荐 trace 均有 6 次 H2D，总量 12.584 GB、GPU copy 约 225 ms；H2D 在
stream 25，capture D2H 在 stream 45，推理在 stream 7。上传 CPU 范围内
**没有 CUDA synchronize**；commit CPU 中位数约 0.037 / 0.041 ms。
D2H 的 400 次 event 等待位于后台采集范围。late 没有 `cudaHostAlloc`，early
有 1 次；不能将启动阶段和稳态分配混为一谈。

| late 固定 400 轮 | 原始 | 推荐组合 |
| --- | ---: | ---: |
| 推理 graph launch 数 | 800 | 800 |
| 主要两类 GEMM 累计 ms | 4808.76 | 4815.10 |
| attention kernel 累计 ms | 912.60 | 918.63 |
| 推理 kernel 并集 ms | 6051.22 | 6084.02 |
| 推理 kernel 首尾跨度 ms | 6175.55 | 6607.25 |

主要 GEMM 仅增加约 **0.13%**。这些数据支持传输在独立流上高度重叠，没有
增加推理线程等待权重上传的路径；它们不证明 DMA 完全无带宽竞争。
all-thread profiler 会影响 Python 提交、训练节奏和导出暂停，三个 profile
请求的吞吐没有进入正式性能表，不能用其空隙直接估计线上开销。

### Overlap 口径复核与推理 metadata 的隐式同步

上表的 overlap 是 **在线 DMA 时间与推理 stream 上 kernel 时间并集的交集，
除以在线 DMA 时间**。它排除了 kernel 之间的空隙，不能将 `1 - overlap`
解释为推理被传输阻塞的比例，也不能用它证明没有带宽竞争。

复核旧 EAGLE3 的 `aime_2026_ii_8_greedy_batched_anchor_trace_late`，
使用同一口径得到 H2D **99.64%**、D2H **99.57%**，确实高于 DFlash。
这是不同请求和 profile 配置的历史 trace，只用于解释 timeline，不能据此
给出同配置性能差异。EAGLE3 的 kernel busy/span 为 99.25%，DFlash last
为 92.08%；DFlash trace 有更多计算间隙，不能把差异全部归为显示口径。

| trace | H2D 未与 kernel 重叠 ms | 其中位于推理 kernel 首尾之外 ms | D2H 未与 kernel 重叠 ms |
| --- | ---: | ---: | ---: |
| DFlash early | 6.440 | 0 | 8.084 |
| DFlash last | 12.576 | 6.247 | 5.507 |
| 旧 EAGLE3 late | 3.531 | 0 | 0.278 |

DFlash last 的 6.247 ms 位于最后一个被记录的推理 kernel 之后。只裁剪到
推理 kernel 首尾范围，H2D overlap 为 **97.11%**；这只是修正 trace 边界，
不改变原始统计值，也不能冒充新的运行性能。其余 H2D 非重叠部分和全部
D2H 非重叠部分落在 kernel 间隙中。最大的若干间隙位于 context KV 写入后、
query graph 前，CPU 同时执行 metadata 处理和 graph 提交。

此外，在线传输没有让推理流等待 DMA，不代表整个 DFlash 推理没有同步。
`DFlashSpeculator.propose` 每轮在 context KV 写入后重建 draft attention
metadata。`flashinfer.py` 中 `all_uses_trtllm = causal and (...)` 对
non-causal DFlash 为 False，随后读取 `common_attn_metadata.seq_lens_cpu`。
该属性在 `attention/backend.py` 中执行 `self.seq_lens.to("cpu")`，
形成推理主线程上的隐式 GPU→CPU 同步。当前 dedicated XQA decode 分支
使用 GPU `seq_lens`，这个 CPU 回读是否可以按实际 backend 路径省略，是
后续优化点；本次复核没有修改该路径或测量移除后的收益。

原始 DFlash last 和在线 last 的各 400 轮 trace 均记录了 **400 次、每次
4 字节、stream 7 上的 pageable D2H**。这和后台采集的 stream 45 不同，
说明该依赖在原始 DFlash 中也存在，不能归咎于在线训练采集。对应 CPU
`cudaMemcpyAsync` 范围包含等待此前 GPU 工作完成的时间，不能把整个 API
耗时算成新增 GPU 空闲或移除后可节省的时间。

共享 transport 中的 `copy_done.synchronize()` 则运行在 capture worker；
H2D 安装通过 `event.query()` 检查完成，未完成时继续使用旧 slot。两者都
需要与上述推理 metadata 回读分别解释。后台等待不能直接算成推理阻塞。

复核数据位于实验目录的 `copy-overlap-audit.json`、
`copy-gap-cpu-attribution.json` 和 `inference-metadata-d2h-audit.json`。

对推荐 trace 只统计完整更新、排除 bootstrap 与截断边缘，各有 5 次：

| 每次完整更新平均 ms | early | last |
| --- | ---: | ---: |
| 整次训练 | 759.17 | 1157.68 |
| forward | 180.29 | 238.51 |
| backward | 329.02 | 679.74 |
| optimizer | 140.77 | 137.28 |
| 新 KV 追加 | 31.32 | 36.81 |
| CPU Flash attention forward，包含于 forward | 13.00 | 61.13 |
| CPU Flash attention backward，包含于 backward | 39.89 | 406.32 |
| K/V repeat_interleave 与 ExpandBackward0 | 0 | 0 |

重排确实消除了 K/V 展开和对应梯度归并；长历史 attention backward 仍是后续
主要瓶颈。真正跳过 detached 历史 dK/dV 需要专用 backward 内核，当前没有实现，
也没有把“计算完后丢弃梯度”冒充已省去计算。

## 复现与检查

```bash
export PATH="$PWD/.venv/bin:/home/lixiang/work/myvllm-rebuild/.venv/bin:$PATH"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE
export MALLOC_MMAP_THRESHOLD_=1073741824 MALLOC_TRIM_THRESHOLD_=-1

.venv/bin/python online_draft/scripts/smoke_async_dflash.py \
  --online --target-graphs --train-threads 16 --cpu-isolation \
  --no-context-grad --grouped-query-attention --selective-last-layer \
  --learning-rate 2e-4 --max-train-rounds 16 --loss-decay-gamma 7 \
  --aime-questions 1 --aime-offset 14 --context-tokens 16384 \
  --max-model-len 18432 --decode-timeline --queue-metrics \
  --output /tmp/dflash-optimized-16k.json
```

四题改为 `--aime-questions 4 --aime-offset 0`。原始对照移除 online 及其诊断
开关；旧在线省略两个新实现开关并使用 lr=1e-4，仍指定 no-context-grad。
每个请求 reset 到 W0，不需要手动重启权重；独立重复时重启进程。

完整回归 **151 passed，14 个原有 JIT warning**；pre-commit 和 mypy-3.12
通过。启动数学对照检查 W0、非零权重安装、context 投影和监督行 logits，
并验证 target 与 mutable draft slot 的存储隔离。原有 EAGLE3/runtime/slot
回归包含在测试集中。

详细数据：[system_optimization_results.json](system_optimization_results.json)。
原始 JSON、日志、命令、聚合/绘图与诊断脚本均在
[实验目录](../../results/experiments/dflash_system_optimization_20261019/)。
训练环境未安装绘图库，图在 `uv run --with matplotlib` 的独立环境生成。

可直接打开的 trace：

- [推荐 early](../../results/experiments/dflash_system_optimization_20261019/optimized-early/early.pt.trace.json.gz)
- [推荐 last](../../results/experiments/dflash_system_optimization_20261019/optimized-last/last.pt.trace.json.gz)
- [原始 last](../../results/experiments/dflash_system_optimization_20261019/baseline-last/baseline-last.pt.trace.json.gz)

所有实验已经结束，没有后台训练或 GPU 实验残留。
