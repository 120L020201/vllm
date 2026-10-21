# 阶段 20：Capture、数据传输与推理线程重叠

本文记录阶段 20 的实现与实验。后续[阶段 21](stage21_eagle_weight_handoff.md)
已按用户要求移除后台上传开关和专用回调，统一采用 EAGLE3 的主线程提交方案。
本文含该旧开关的命令仅用于历史记录，当前运行命令以阶段 21 为准。

## 结果

完成三项修改：移除 XQA decode 不需要的 CPU 长度回读，合并 capture metadata，
由训练线程提交权重上传。最终 early / last trace 中，推理 stream 的 kernel 时间
占比分别为 **99.06% / 99.32%**，H2D 重叠 **98.97% / 99.09%**，D2H 重叠
**97.67% / 97.83%**。这些是时间线区间指标，不是 SM 利用率或吞吐提升比例。

无 profiler 的 AIME Q15 自然生成到总上下文 16K，最终两次在线运行合计
**227.39 token/s**，优化前在线为 **221.79 token/s**，提高 **2.52%**。
平均每个 draft/verify round 从 **14.940 ms** 降到 **14.514 ms**。
两次最终输出均与原始 DFlash 逐 token 相同。

完整四题验证全部正确、输出 token 与基线相同。最终在线 decode 合计
**289.46 token/s**，同版本不训练基线 **264.04 token/s**，提高 **9.63%**；
包含 prefill、请求结束等待和恢复 W0 的完整吞吐提高 **6.19%**。

## 实现与时序

### 1. Decode metadata 不再回读 GPU 长度

见[阶段 19](stage19_metadata_sync.md)。FlashInfer 的 dedicated XQA decode
直接使用 GPU sequence lengths。删除 `all_uses_trtllm` 判断中额外的 `causal`
约束后，non-causal DFlash 不再进入不需要的 `seq_lens.to("cpu")`。
native prefill、DCP 和 cascade 的 CPU metadata 分支仍依据各自需求执行。

这解决的是每轮隐式同步：旧 last trace 有 400 次主线程 4 字节 pageable D2H，
新 last trace 为 **0**。没有改变 attention mask 或计算 kernel。
原始 DFlash 不训练时也受益。

### 2. Capture 完整 decode packet 从 9 次 DMA 合并到 4 次

`DFlashCaptureAdapter` 现在保存以下四块数据：

| 数据 | 内容与用途 |
| --- | --- |
| proposal metadata | 两行 int64：query input IDs、绝对 query positions；保留源权重版本 |
| target metadata | target positions、rejected count、sampled count |
| target hidden | 原始 target 特征，供连续历史 KV 追加 |
| teacher logits | 已确认前缀和首拒绝位置的完整 teacher 分布来源 |

删除 CPU 训练从未读取的 target token IDs 和 proposal sampled IDs 快照。
没有减少监督位置，没有对 teacher 分布做 top-k、精度压缩或采样。
bootstrap 的 prompt、已确认历史、query 对齐和源版本信息均保留。

teacher/hidden 的 owned clone 和 metadata 的 cat/stack 仍在生产者 stream
提交：原始推理 buffer 会被后续 round 复用，不能直接把对原始地址的读取延后到
后台。主线程只提交快照与 ready event；capture 线程等待 ready、在私有 stream
执行 D2H，再进行 CPU softmax、位置校验和 batch 准备。
复用 pinned staging 前，batch 保存自己的数据。

late trace 的 400 个完整 round 对应 **1600 次 capture DMA**；旧布局应为
3600 次。early 包含一次 bootstrap，实测 1598 次。稳态 last trace 没有
`cudaHostAlloc`；early 的 5 次初始化分配没有变成每轮分配。

### 3. H2D 提交移到训练线程，推理只尝试安装

`AsyncDraftPipeline` 增加可选 `on_snapshot` 回调。DFlash 开启
`background_weight_upload` 后，将它连接到 `installer.stage`：

1. CPU 更新完成，训练线程复制到已有 pinned snapshot。
2. 设置 snapshot pending，在 pipeline 锁外调用 `installer.stage`。
3. 训练线程向 H2D stream 提交 inactive slot 的权重上传及派生 buffer 更新。
4. 推理边界调用 `commit_if_ready`；锁忙或 event 未完成就继续旧权重推理。
5. 上传完成后切换 slot，并通过原有有序队列关闭 epoch，允许下一次训练更新。

`commit_if_ready` 改为非阻塞获取 installer 锁，避免推理等后台线程提交全部
parameter copy。复用现有训练线程、双 slot、snapshot 和 epoch 协议，没有新增
上传线程或缓冲队列。EAGLE3 默认仍走原来的 snapshot 消费路径。

inactive slot 的 last-use event 仍由 H2D stream 等待；派生 context 投影权重 buffer
必须完成后才能安装；CPU snapshot 在安装完成前不能被下次更新覆盖。
历史 context KV 保留混合权重版本近似，context grad 关闭，训练方案没有改变。
请求结束仍排空后台工作并恢复 W0，不能把这部分成本当作 decode 已经消除。

## 实验条件与计时

RTX PRO 6000 Blackwell 96GB；本地 Qwen3-8B 和 Qwen3-8B-DFlash-b16；单 GPU、
单活动请求、K=15；target/draft FULL CUDA Graph；FLASHINFER；关闭 prefix cache。
greedy、seed=17、thinking 开启、EOS 正常生效。

沿用阶段 18：CPU GQA、最后一层仅计算监督行、context grad 关闭、最多 16 个
训练 round、lr=2e-4、按位置指数衰减 KL gamma=7、BF16 fused AdamW；冻结
embedding/head 及已有配置。推理核 18，capture 核 19，训练核 0–15；各角色
线程数 1/1/16，allocator 设置见复现命令。GPU 实验串行。

decode 排除 prefill 和 finish；完整请求包含两者及 reset，不包含引擎初始化。
合计吞吐按总 token / 总时间计算，不取逐请求吞吐算术平均。
所有性能数据来自未启用 profiler 的独立运行。

Q15 prompt 为 286 token，输出 16098 token，总上下文 16384。本题达到长度上限，
没有最终 boxed 答案，JSON 的 `correct=false` 不能用于衡量本轮准确率。
四题验证使用 AIME 2026 前四题，均自然结束。

## 16K 筛选和最终消融

| 配置 / 原始文件名 | decode token/s | 平均 round ms | CPU 更新 / GPU 安装 |
| --- | ---: | ---: | ---: |
| 修改前原始，`before-baseline` | 173.51 | 14.612 | 0 / 0 |
| 修改前在线，`before-online` | 221.79 | 14.940 | 70 / 69 |
| metadata 后原始，`metadata-baseline` | 174.57 | 14.524 | 0 / 0 |
| 仅 metadata 后在线，`metadata-online` | 226.13 | 14.525 | 69 / 68 |
| 再合并 capture，`packed-online` | 227.84 | 14.504 | 68 / 67 |
| 再后台上传，`background-online-1` | 227.69 | 14.514 | 70 / 69 |
| 后台上传重复，`background-online-2` | 227.08 | 14.514 | 69 / 68 |

中间候选 `metadata-online` 在零基位置 4549 与基线输出分叉，其吞吐不能视作
完全相同生成路径下的实现收益。没有保存该位置的 target logits，不能确定分叉
的唯一原因。后续 packed 和两次最终运行均逐 token 一致。
后台上传与前台提交的 packed 方案吞吐基本持平；它的直接证据是线程归属和
不阻塞 commit，不能单独宣称一次确定的吞吐提升。

| 最终代码的消融 | decode token/s | 相对不采集基线 | CPU 更新 / GPU 安装 |
| --- | ---: | ---: | ---: |
| 不采集、不训练 | 174.57 | — | 0 / 0 |
| 仅采集，`packed-capture` | 174.45 | -0.06% | 0 / 0 |
| 持续训练、不安装，`packed-train-extended` | 173.78 | -0.45% | 88 / 0 |
| 完整在线，两次合计 | 227.39 | +30.26% | 139 / 137 |

“不安装”需要显式关闭 epoch，否则生产协议会一直等待首次安装。实验重放
`background-online-1` 的 69 个安装轮次；W0 接受率较低，运行 6349 个 decode
round，完整在线约 4871 个，因此又每 80 round 延伸边界至尾段，共 87 个边界，
连同 bootstrap 完成 88 次更新。它保留真实训练计算，但不执行 H2D/安装；
它不是与在线逐样本相同的训练轨迹。原先只重放 69 个边界的结果
`packed-train-no-install` 也保留，因后段未继续更新，不用作完整控制组结论。
两个 replay 文件分别保存，不修改旧实验对应的输入。

完整在线两次合计相对修改前在线提高 **2.52%**，相对修改前原始提高 **31.05%**。
平均接受长度由修改前在线 3.3135 变成 3.3005，本次提升没有依赖接受率提高。
完整请求吞吐从 220.36 到 223.85 token/s，提高约 1.58%；最终两次 finish
分别为 1.195 s 和 0.935 s。

## 四题完整验证

| AIME 题号 | 输出 token | 答案 | 原始 decode token/s | 在线 decode token/s | CPU 更新 / GPU 安装 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 3521 | 277 | 334.05 | 331.91 | 12 / 11 |
| 2 | 11108 | 62 | 279.85 | 303.81 | 36 / 35 |
| 3 | 9883 | 79 | 256.04 | 278.13 | 38 / 37 |
| 4 | 11175 | 70 | 241.24 | 275.36 | 43 / 42 |
| 合计 | 35687 | 全部正确 | **264.04** | **289.46** | 129 / 125 |

全部输出与本轮基线、阶段 18 的基线逐 token 相同。平均接受长度 3.7281 →
4.1005；完整请求吞吐 263.67 → 280.00 token/s。
第 1 题较短，decode 仍下降 0.64%，且 finish 约 1.16 s。本次优化没有证明所有
短请求都能获益。四题各运行一次，不外推为完整题集准确率或稳定泛化提升。

## Early / last trace

每份截取 400 个 worker step。early 从请求开始，last 延迟 4200 step；旧 last
延迟 4379，比较的是相近晚段，不是逐轮完全相同的 context。early 运行的总上下文
预算为 4096，截取窗口在触及上限前结束。trace 导出造成暂停，其请求吞吐不参与
上面的性能统计。

| 指标 | 旧 early | 新 early | 旧 last | 新 last |
| --- | ---: | ---: | ---: | ---: |
| 推理 stream kernel 时间占比 | 90.32% | **99.06%** | 92.08% | **99.32%** |
| H2D 与推理 kernel 重叠 | 97.15% | **98.97%** | 94.42% | **99.09%** |
| D2H 与推理 kernel 重叠 | 87.56% | **97.67%** | 90.11% | **97.83%** |
| 窗口内推理 kernel 间隙合计 ms | 584.14 | **51.49** | 523.23 | **41.85** |

统计使用推理 stream 上 kernel 区间的并集，包含该 stream 上的快照/准备 kernel，
不是只计算模型矩阵乘。重叠分母是 capture/上传范围所关联 DMA 的时间并集。
profiler 会放大 CPU 提交成本，不能把 trace 间隙减少量直接换算成无 profiler
请求加速。旧 last 有约 6.247 ms H2D 落在最后一个已记录推理 kernel 之后；
裁掉边界后其 H2D 重叠约 97.11%，所以 94.42% → 99.09% 不全是实现收益。

新 trace 中 H2D 在 stream 25，D2H 在 stream 45，推理在 stream 7：

| CPU 工作 | early tid | last tid |
| --- | ---: | ---: |
| 推理 / slot commit | 443049 | 444931 |
| CPU train / H2D stage | 443140 | 445038 |
| capture / D2H | 443141 | 445039 |

H2D stage CPU 范围中没有 synchronize；其中位耗时约 0.892 / 0.720 ms，
现在线程归属是 trainer。主线程 commit 中位约 0.020 / 0.030 ms。
D2H 仍有每个 packet 的 `cudaEventSynchronize`，但全部在 capture 线程；
它保护 CPU 读取，并不要求推理 stream 等 capture 完成。

新 last 的 metadata pageable D2H 为零；early 仅一次，发生在首次 capture
之前的初始化 prefill。capture 的 snapshot 工作仍需主线程提交，没有宣称
全部 capture 都搬离主线程。剩余 1%–2% 未重叠和资源竞争不能视作已经消除。

无 profiler 的 Q15 队列中，最终 capture 峰值 7/8，train 峰值 77/79；修改前
分别为 8/90。尾段 CPU 训练仍约 1.094 s，修改前约 1.099 s；队列尾段等待仍
约 0.52–0.53 s。本轮改善推理提交和重叠，没有改变随历史增长的 CPU attention。

原始 trace：

- [early：400 step，约 16 MB](../../results/experiments/dflash_overlap_20261019/early/early.pt.trace.json.gz)
- [last：400 step，约 16 MB](../../results/experiments/dflash_overlap_20261019/last/last.pt.trace.json.gz)
- [逐请求、trace、验证与源码哈希汇总](transfer_overlap_results.json)
- [全部实验、命令、日志与线程审计](../../results/experiments/dflash_overlap_20261019/)

## 验证与复现

runtime/model 回归 153 项，加上 packed capture 的三个 CUDA 参数用例，共
**156 passed**。扩展已有 capture 测试覆盖 CPU/CUDA、接受 0/1/3 个 token、
混合整数 dtype、对齐和输入复用；并发测试覆盖后台 stage、非阻塞 commit、
snapshot/epoch 顺序和请求 reset。
XQA causal / non-causal 数值与无 CPU 长度访问检查 **2 passed**。
pre-commit 和手动 mypy Python 3.12 均通过。

四题在线启动时另执行真实 CPU/GPU 数值检查：W0 logits relative RMSE
0.01173、cosine 0.999880、KL 0.000787；一次非零安装后的对应值为 0.01312、
0.999862、0.001015；target 存储隔离检查通过。诊断结束恢复 W0 后才开始计时。
所有已完成 online 请求的 CPU/GPU 版本均恢复到 0。

复现本机最终 16K 配置：

```bash
export PATH="$PWD/.venv/bin:$PATH"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE
export MALLOC_MMAP_THRESHOLD_=1073741824 MALLOC_TRIM_THRESHOLD_=-1

.venv/bin/python online_draft/scripts/smoke_async_dflash.py \
  --online --target-graphs --train-threads 16 --cpu-isolation \
  --no-context-grad --grouped-query-attention --selective-last-layer \
  --background-weight-upload \
  --learning-rate 2e-4 --max-train-rounds 16 --loss-decay-gamma 7 \
  --aime-questions 1 --aime-offset 14 --context-tokens 16384 \
  --max-model-len 18432 --decode-timeline --queue-metrics \
  --output /tmp/dflash-overlap-16k.json
```

实际 serving 设置 `additional_config.online_draft.background_weight_upload=True`；
当前该项默认 False，需要显式开启。metadata 与 capture 合并不需要新开关。
其余模型、冻结参数、CPU GQA、context grad 配置按上述实验显式设置。
将题目参数改为 `--aime-questions 4 --aime-offset 0` 可复现四题。

本轮最终方案做两次 16K 重复；筛选和消融各一次，四题各一次。小于 1% 的差异
只作本次观测，不视为统计显著收益。保留所有原始结果，包括中间分叉候选。

## 补充复核：重叠不等于推理零成本

本节针对后续问题重新分析已有 trace，没有重新运行 GPU 模型实验，也没有修改
运行时代码。分析脚本和逐项数据为实验目录的 `analyze_overlap_cost.py` 与
`overlap-cost-audit.json`。

### 开关含义

`--background-weight-upload` 仅改变提交 H2D 的 CPU 线程。未开启时，推理线程
在 `DraftWeightHandoff.advance()` 中取 snapshot，再调用 `installer.stage()`；
GPU copy 已经是 pinned memory + `non_blocking=True` + 私有 H2D stream。
开启后，训练线程直接调用 `installer.stage()`，推理线程不再提交这批 copy。
它不是 D2H 开关，也不是“同步 copy / 异步 copy”的切换。上文最终在线与两份
新 trace 均已经开启；`packed-online` 是前台提交的对照。

### 仍存在的直接 GPU 成本

last trace 通过 CPU annotation → CUDA correlation → GPU event 关联得到：

| 主推理 stream 中的额外 capture 工作 | 400 轮累计 ms |
| --- | ---: |
| teacher logits 的 GPU→GPU 快照 | 1.591 |
| target capture 的 GPU→GPU copy | 0.719 |
| proposal capture 的 GPU→GPU copy | 0.406 |
| target metadata / proposal 打包 kernel | 1.374 |
| 合计 | **4.090** |

约 **10.23 微秒/轮**。这些工作与同一个 stream 的原推理顺序执行，需要保护
会被复用的原始 buffer，不能因后台 D2H 的 overlap 很高就忽略。它与无 profiler
的“仅采集”平均 round 增加约 9.38 微秒处于同一量级，但两种测量不能直接逐项
相减归因。CPU 端仍需构造 packet、提交 event、入队和检查安装。

此外，H2D stream 更新 DFlash 派生投影 buffer 的 kernel，在 late 五次上传中
累计约 0.380 ms。这与完整上传参数不同，是模型专用的额外 GPU 工作。

### 传输期间推理 kernel 是否变慢

对 early/last 的推理 graph 按 `graph node id` 匹配，选出至少 80% 执行时间与
H2D 重叠的 kernel。每个样本用附近 500 ms 内、相同节点、完全不与 H2D 重叠的
最多 12 次执行的中位耗时作为参考，至少需要 6 次参考。

| 观察项 | early | last |
| --- | ---: | ---: |
| 与 H2D 重叠的 GEMM 相对附近参考累计耗时 | +2.40% | +2.34% |
| 与 H2D 重叠的 attention 相对附近参考累计耗时 | +1.54% | +1.88% |
| H2D DMA 时间占整个 trace 推理跨度 | 4.10% | 3.07% |

这支持“传输期间存在小幅资源竞争”的解释，**不构成显存带宽瓶颈的独立因果
证明**：GPU 时钟、context、profiling、派生 buffer 更新等仍可能影响局部耗时。
2.34% 也不是整个请求变慢 2.34%；上传只占 late 窗口约 3.07% 的时间。
目前没有显存控制器/缓存性能计数器，不能把影响精确分成带宽与缓存份额。

late H2D 每次约 **2.097 GB / 37.57 ms**，传输期间约 55.8 GB/s；每轮 D2H
约 **5.518 MB / 0.144 ms**。前者是间歇上传，后者每轮出现；按整个 late 窗口
平均分别约 1.72 GB/s、0.36 GB/s。PCIe copy 使用 GPU 显存和主机内存路径，
与推理 kernel/CPU 训练会共享部分资源；GPU 推理本身主要访问显存，不能简单
描述为“模型推理和 copy 都在抢 PCIe”。

H2D 未与推理 kernel 重叠的时间总计仅 1.705 ms，D2H 为 1.249 ms；这仍不是
被传输阻塞的时间，因为它们可以落在原本就存在的 kernel 间隙中。可并行 copy
和 kernel 的条件见 [NVIDIA CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#asynchronous-and-overlapping-transfers-with-computation)。

### 与已有 EAGLE3 trace 的区别

重新用相同 DMA/推理 kernel 交集口径分析原目录中的
`aime_2026_ii_8_greedy_trace_late_torch_trace`。该历史 trace 是 K=7、不同题目和
不同训练配置，不能直接比较两者吞吐；下表只比较实际捕获到的传输。

| 指标 | 历史 EAGLE3 late | 当前 DFlash last |
| --- | ---: | ---: |
| 每轮 teacher logits | 0.512 MB | 4.862 MB |
| 每轮全部 D2H | 0.873 MB | 5.518 MB |
| 每轮 D2H DMA 时间 | 0.0309 ms | 0.1440 ms |
| 每次权重上传 | 0.537 GB | 2.097 GB |
| 每次 H2D DMA 时间 | 9.99 ms | 37.57 ms |
| H2D overlap | 99.62% | 99.09% |
| D2H overlap | 99.86% | 97.83% |
| 推理 stream kernel 时间占比 | 99.32% | 99.32% |

另复核 `aime_2026_ii_8_greedy_batched_anchor_trace_late_torch_trace`：2000 个
capture packet、97 次上传，packet 和单次上传字节数与上述 EAGLE3 trace 相同；
H2D/D2H overlap 为 99.64%/99.57%，推理 stream kernel 时间占比为 99.25%。
两份历史 EAGLE3 trace 均高度重叠，但都不是严格 100%。

EAGLE3 teacher 为 8×32000×2 字节，DFlash 为 16×151936×2 字节，约 9.50 倍；
加上各自 hidden/embedding 等数据，完整 D2H packet 约为 6.32 倍。
本地模型分别为 1 层/3 组特征与 5 层/5 组特征，权重更新量、CPU 训练量和
派生 buffer 工作不同。DFlash full logits 是主要 D2H 数据，而不是整份历史 KV。

历史 EAGLE3 上传统计仅包含 9 个 tensor，没有 32000×4096 BF16 lm_head
应有的 262144000 字节 copy；不能把 0.537 GB 外推为当前所有 EAGLE3 配置。
若在该集合加入此 head，上传量应约 0.799 GB，仍小于本轮 DFlash 的 2.097 GB。
历史 EAGLE3 的 CPU 提交方式也不是本轮 DFlash 的训练线程回调方式。

因此，旧 DFlash 较大的空隙主要已定位到 metadata 同步和提交成本；当前剩余
成本包括主 stream 快照/打包、CPU 提交，以及可能的显存/缓存/主机内存竞争。
传输量更大是一个真实差异，但当前证据不支持把剩余全部开销归结为带宽饱和，
也不支持“EAGLE3 完全无影响、DFlash 明显中断推理”的判断。
