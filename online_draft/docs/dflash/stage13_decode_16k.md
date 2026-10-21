# 阶段 13：自然生成到 16K 的纯 decode 对照

2026-10-16，`eagle3&dflash`，base `3979838d2d`，未提交工作树。

## 结论与测量范围

在本地 AIME 2026 第 15 题、prompt 加自然输出达到 16,384 token 的实验中，
当前长请求在线配置的纯 decode 吞吐为 **201.97 token/s**，原始 DFlash 为
**173.45 token/s**，提高 **16.44%**。相同输出的 decode 时间由平均 92.804 秒
减少到 79.699 秒，减少 **14.12%**。12K–16K 段的吞吐提高 **39.76%**。

这是同一道题两次独立进程重复的结果，不是 AIME 全集平均收益，也没有证明
上下文只要达到 16K 就一定加速。阶段 12 前四题、最大输出 8192 的在线方案
仍略慢于 baseline；本题后半段出现了更明显的接受率收益。

这里的 16K 是用户确认的**自然增长上下文**，不是 16K 输入的 prefill。
首选第 3 题在总上下文 10,040 时正常结束且答案 79 正确，因此只保留为预实验；
随后选择第 15 题。正式实验没有关闭 EOS、拼接重复文本或修改原题提示。
第 15 题在 16K 上限处截断，仍处于思考过程，没有最终 boxed 答案，不能据此
计算完整解题准确率。全部正式运行的 16,098 个输出 token 逐 token 相同。

## 配置与公平性

- Target `/data/models/Qwen3-8B`，draft `/data/models/Qwen3-8B-DFlash-b16`。
- 单 RTX PRO 6000 Blackwell 96 GB；Xeon w7-3445；BF16，batch 1，K=15。
- Target/draft FULL CUDA Graph，FLASHINFER，prefix caching 关闭。
- 本地 `/data/benchmarks/aime2026.jsonl` 第 15 题，thinking，greedy，seed=17。
- Prompt 286 token，输出预算 `16384 - 286 = 16098`；`max_model_len=18432`
  为 speculative block 留余量，实际上下文确实只增长到 16384。
- 两组推理均为 CPU 18、单线程；在线采集使用 CPU 19、单线程，训练使用
  CPU 0–15、16 线程。通过 worker RPC 实测记录，避免 baseline 忽略绑核设置。
- 在线用最近 16 个完整 block、位置衰减 KL gamma=7、T=1、lr=1e-4、fused AdamW，
  BF16 参数及 moments，不开启 FP32 master weights。保留 embedding/head 及配置指定的冻结。
- 保留全部连续历史，旧 KV detach，新段按当前权重追加，继续使用混合版本 KV。
  每个请求从 W0 开始；没有把前一次实验的更新带入下一次实验。
- 两组均使用 PASSIVE 等待和阶段 11 的可选 glibc 参数。正式运行串行、不开 profiler。

“原始 DFlash”指同一工作树和 checkpoint，关闭在线训练；不加载 CPU draft、
不 capture、不安装权重。没有更换 target/draft 或量化方式。

## 纯 decode 的定义

新增计时使用 `RequestOutput.metrics` 中的 engine-core 时间戳：

```text
decode_seconds = last_token_ts - first_token_ts
decode_tokens = num_generation_tokens - 1 = 16097
decode_tokens_per_second = decode_tokens / decode_seconds
```

这与 vLLM 自身 `request_decode_time_seconds` 的口径一致，排除 prefill、首 token
和请求结束后的 CPU 训练/reset 等待，但保留真实 decode 的调度、采集、安装及
CPU/GPU 竞争成本；它不是仅累加 CUDA kernel 时间。

分段时间直接记录 frontend 收到的 engine-core 原始输出时间戳，没有插入 GPU
同步或逐轮 RPC。每段边界取不超过 4096/8192/12288/16384 的最后一个完整输出
batch，所以中间边界可能相差几个 token；原始 JSON 保存精确位置。四段 token
及时间总和分别与上述 decode 指标一致。

每个独立进程完成模型和 graph 初始化后运行一道题，没有另加请求预热。
日志中首轮 decode 的 JIT 告警仍计入首段，两组均如此；后面三段没有这一影响。
实际 prefill 为 43–46 ms；在线 finish 等待为 0.818/1.707 秒，均未混入 decode。

## 两次独立进程的正式结果

| 方案 | decode 秒 | decode token/s | 平均接受长度 | GPU 安装次数 |
| --- | ---: | ---: | ---: | ---: |
| 原始 DFlash，第 1 次 | 92.788 | 173.481 | 2.5355 | 0 |
| 原始 DFlash，第 2 次 | 92.821 | 173.420 | 2.5355 | 0 |
| 在线，第 1 次 | 79.856 | 201.576 | 3.0171 | 64 |
| 在线，第 2 次 | 79.543 | 202.369 | 3.0322 | 64 |

吞吐按两次总 token / 总秒数合并，不取逐请求吞吐的简单平均。
接受长度使用 `1 + 总 accepted / 总 drafts`；最后一轮可能产生被输出上限裁掉的
token，因此它与实际输出 token / 输出 batch 有极小差异。两次在线 CPU 都完成
65 次更新，最后一次只在 finish 阶段完成，没有用于推理。

| 总上下文区间 | 原始 token/s | 在线 token/s | 在线变化 | 原始每轮确认 token | 在线每轮确认 token |
| --- | ---: | ---: | ---: | ---: | ---: |
| 约 0–4K | 239.15 | 234.14 | -2.09% | 3.302 | 3.335 |
| 约 4–8K | 197.94 | 203.44 | +2.78% | 2.846 | 3.016 |
| 约 8–12K | 153.40 | 184.47 | +20.26% | 2.259 | 2.791 |
| 约 12–16K | 138.95 | 194.20 | +39.76% | 2.109 | 3.024 |

在线不是让每轮推理变快。合计每轮耗时从约 14.617 ms 增到 14.974 ms，增加
约 2.44%；平均接受长度从 2.5355 增到 3.0246，提高约 19.29%，减少了验证轮数。
最后一段的每轮耗时为 15.175/15.571 ms，但每轮确认 token 为 2.109/3.024。
这解释了它为何能抵消后台训练和 capture 的开销。

同时，原始方案从短段到长段的下降不能全部归因于 attention：同一过程中生成
内容也在变化。原始每轮耗时仅从约 13.808 增到 15.175 ms，而每轮确认 token
从 3.302 降到 2.109；接受率下降是本题吞吐下降的主要可观测因素。

## 长上下文 trace

正式计时结束后另跑原始/在线诊断，按各自参考运行中接近 12K 的轮次延迟启动，
采集 600 个 worker step，启用所有 CPU 线程。profile 运行不进入性能表格。

按输出时间线估计，原始 trace 覆盖总上下文约 12.29K–13.65K，在线约
12.32K–14.15K；轮数相同、确认 token 数不同，所以它们不是严格相同位置的
kernel 延迟消融。两个 profile 请求的最终 token 也与正式 baseline 一致。

| trace 指标 | 原始 DFlash | 在线 |
| --- | ---: | ---: |
| 推理 stream | 7 | 7 |
| 推理 kernel 首尾跨度 ms | 9167.38 | 9828.91 |
| 推理 kernel 区间并集 ms | 8982.66 | 9031.83 |
| kernel 忙碌比例 | 97.98% | 91.89% |
| 两类主 GEMM 合计 ms | 7211.86 | 7218.24 |
| attention `kernel_mha` 合计 ms | 1277.21 | 1289.10 |
| `cudaGraphLaunch` 次数 | 1200 | 1200 |

原始 GEMM 占推理 kernel 时间约 80.3%，attention 约 14.2%。GPU 已高度忙碌，
不能把本题低吞吐简单解释成 Python 喂不饱 GPU。在线两类主 GEMM 的总时间
增加约 0.09%，未看到 DMA 让它们大幅变慢。

在线 trace 的空隙明显大于正式运行测得的每轮额外成本；全线程 profiler 本身
会干扰采集、Python/CUDA 提交及异步训练节奏，不能把 91.89% 直接当成生产 GPU
利用率，或把两个 trace 相差的全部空隙归给在线逻辑。trace 导出也在请求中造成
明显暂停，因此没有使用 profile 请求的 decode 时间评价速度。

### 传输是否仍被掩藏

在线的 600 轮 trace 内发生 6 次权重上传：

| 传输 | 数据量 GB | copy 时间 ms | 与推理 kernel 的时间交集 | stream |
| --- | ---: | ---: | ---: | ---: |
| 权重 H2D | 12.584 | 225.238 | 97.542% | 25 |
| Capture D2H | 3.311 | 80.580 | 90.505% | 45 |

H2D CPU 提交中位数 0.907 ms，最大 4.178 ms；这些上传范围内没有 CUDA
synchronize。全 trace 的 `cudaHostAlloc` 次数为 0。commit 中位数 0.043 ms。

D2H 的 600 次 `cudaEventSynchronize` 全在后台采集线程，等待 producer 以及
copy 完成。D2H CPU 范围中位数 10.818 ms 包含这部分等待，不是复制本身花了
10 ms；GPU 实际 D2H 平均约 0.134 ms/轮。推理主线程仍有原始 vLLM 就存在的
输出同步，不能把所有同步调用都说成“在线传输阻塞”。两组 trace 结束时都出现
一次 profiler 的 device synchronize，也不属于常规推理路径。

这支持上传/下载仍在独立流上与推理重叠，没有新增推理线程等待权重上传的行为。
重叠率不是零带宽竞争或零主线程提交开销的证明；正式计时仍观察到每轮约 2.44%
额外成本，结论应保留这一点。

### CPU 长上下文成本

两次正式在线请求中，最前 8 次更新平均分别为 0.799/0.803 秒，最后 8 次为
1.544/1.558 秒。query 训练数量保持最多 16 个 block，历史长度仍会增加成本。
在线 trace 中各 CPU 阶段中位数如下；边缘事件可能跨越 profile 边界，不将不同
阶段的 count 或总时间直接拼成完整训练步。

| 阶段 | 中位数 ms |
| --- | ---: |
| forward | 363.8 |
| backward | 826.8 |
| fused AdamW | 150.0 |
| loss | 4.7 |
| 新增 context KV 投影与追加 | 42.8 |
| CPU snapshot | 56.8 |

trace 中 `online_draft.rebuild_cache` 对应新增段投影与追加，**没有重建旧历史 KV**。
训练线程 CPU Flash Attention backward 累计 2470.9 ms，forward 1006.2 ms；
`aten::mm` 累计 2152.6 ms，`repeat_interleave` 263.8 ms，`aten::cat` 280.5 ms。
算子统计是 inclusive 时间，含父子事件，不能把整张算子表相加当作耗时分解。

## 由证据支持的后续优化点

1. **优先减少 CPU 长历史 attention 的内存展开/复制。** 当前
   `DFlashAttention.forward` 对拼接后的 K/V 执行 `repeat_interleave`，
   从 8 个 KV head 展开到 32 个 query head；历史越长，复制与反向成本越高。
   同时，训练前拼接旧/新 KV、训练后追加缓存都涉及历史 tensor。可分别消融
   无显式展开的 GQA 路径及预分配历史追加，必须验证 CPU Flash Attention 路径、
   完整非因果 block、位置 mask 和新段梯度不变。不能靠截断历史来冒充等价优化。
2. **减少未监督 teacher 行的 D2H。** 当前每轮复制完整 16 行 logits，CPU 只对
   `min(num_sampled, K)` 行做 teacher 分布。长段每轮约确认 3 个 token，存在可减少
   的传输量。候选是后台先取有效行数，再只传连续有效行，或在 GPU 侧打包；需
   比较额外小传输/同步的代价，保留首拒绝位置和所有连续历史，不让推理线程等计数。
   现有 D2H 总 copy 时间仅占本 trace 跨度约 0.82%，所以收益不能按字节缩减比例外推。
3. **GPU 侧优先评估小 batch BF16 GEMM。** 当前主耗时是两个 CUTLASS WMMA GEMM
   kernel，合计约 12 ms/验证轮；可基于真实 M=16 及各层 shape 比较适用于本卡的
   GEMM 路径。仅融合少量准备 kernel 或削减 Python 调度的预期空间较小。需要
   单独 kernel 基准及端到端验证，不能由 kernel 名称就断言现用实现不合理。

本轮没有据此修改生产实现，也没有重新搜索学习率。继续使用阶段 12 已选的
配置，以免混淆“当前方案在 16K 的表现”和“新增优化带来的收益”。

## 可复现命令

在 `myvllm-eagle3-dflash` 工作树运行：

```bash
export PATH="$PWD/.venv/bin:/home/lixiang/work/myvllm-rebuild/.venv/bin:$PATH"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE
export MALLOC_MMAP_THRESHOLD_=1073741824 MALLOC_TRIM_THRESHOLD_=-1

.venv/bin/python online_draft/scripts/smoke_async_dflash.py \
  --target-graphs --train-threads 16 --cpu-isolation \
  --learning-rate 1e-4 --max-train-rounds 16 --loss-decay-gamma 7 \
  --aime-questions 1 --aime-offset 14 \
  --context-tokens 16384 --max-model-len 18432 --decode-timeline \
  --output /tmp/dflash-16k-baseline.json
```

在线对照增加 `--online` 并修改输出路径，每组重启进程重复两次。
诊断 trace 另加 `--profile-dir`、`--profile-delay-iterations` 和
`--profile-max-iterations 600`。本次 baseline delay=4407，online delay=3974，
它们来自参考运行到约 12K 的实际输出轮数，不代表通用固定轮数。

## 修改、验证和证据

本阶段只扩展测量脚本和报告，未修改生产训练、KV 或权重安装路径。
`smoke_async_dflash.py` 增加题号偏移、总上下文预算、纯 decode 及分段计时、
baseline CPU 设置对齐、延迟/限量 profile；`summarize_dflash_trace.py` 增加
推理 kernel、CPU 阶段和同步调用统计，并支持没有在线传输的 baseline trace。

原始 JSON/日志/trace 保存在 `/tmp/dflash-decode-16k/`。第 3 题预实验与未对齐
baseline 绑核的初测分别保留为 `question3-preflight` 和 `baseline-default-host`，
它们不进入正式对照均值。正式文件为 `baseline-{1,2}` 和 `online-{1,2}`。

[持久化结果](decode_16k_results.json) 包含逐请求统计、token SHA256、原始文件
SHA256、精确分段、逐更新记录、trace 汇总及整理报告时的源码 SHA256；不重复
保存大段输出文本。随机 request_id 只在原始 JSON 中保留。

Trace 文件分别为：

- `/tmp/dflash-decode-16k/trace-baseline/rank0.1792162326656467725.pt.trace.json.gz`
- `/tmp/dflash-decode-16k/trace-online/rank0.1792162480400373191.pt.trace.json.gz`

验证包括四次正式模型运行、两个 profile 请求的 token 一致性检查、正式运行的
decode 时间/token/分段加和及实际 CPU 设置断言。脚本和文档的 pre-commit 通过，
包括 Ruff、格式、typos、SPDX、Markdown 检查、mypy-3.10；另跑 mypy-3.12 通过。
日志为 `final-checks.log` 和 `mypy312.log`。
没有修改生产代码或新增训练算法，因此本阶段没有重复运行阶段 12 已通过的
131 项生产路径回归。没有提交或推送 Git。
