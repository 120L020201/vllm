# 阶段 30：长输出 AIME 的逐轮接受长度与吞吐

## 实验目的与结果

比较分支基准 commit 的原始 vLLM DFlash 与阶段 29 最终在线训练方案。
这里的原始对照关闭全部在线训练，不是阶段 29 中使用旧 CPU backward 的在线训练对照。

两次运行合并后，平均接受长度从 **2.9827** 变为 **4.7327**（+58.67%）；
decode 吞吐从 **196.16** 变为 **316.62 token/s**（+61.41%）。
平均 GPU 验证 round 的 engine-core 输出周期从 15.205 变为 14.947 ms（-1.70%）。

这条 32K 输出在约 20K 后进入重复循环，并在长度上限处截断，因此不能把上述
61.41% 当作一般长推理的收益。额外测量能自然答完的 AIME 第 2 题，输出 11,108
token，原始与在线分别为 278.15 / 307.23 token/s（+10.45%），两组答案均正确。

上述 round 周期包含完整 decode 调度与执行间隔，不是单个 GPU kernel 时间。
当前方案也包含此前的 FlashInfer metadata 优化；本实验比较完整系统，不单独归因到某一项优化。
两组 round 在上下文位置上的分布不同，不能凭整体平均周期判断后台资源竞争是否消失。

## 配置与复现

- 日期：2026-10-20；GPU：RTX PRO 6000 Blackwell 96 GB；CPU：Xeon w7-3445。
- Target `/data/models/Qwen3-8B`；draft `/data/models/Qwen3-8B-DFlash-b16`。
- AIME 2026 第 15 题，`/data/benchmarks/aime2026.jsonl` 第 15 行，thinking chat template。
- Prompt 286 token；总上下文上限 32768，输出上限 32482；模型最大长度 34816。
- BF16，batch=1，15 个 speculative token，temperature=0，seed=17，尊重 EOS。
- Target/draft CUDA Graph、FlashInfer、关闭 prefix cache，GPU memory utilization=0.65。
- 推理固定 CPU 18、1 线程；在线采集 CPU 19、训练 CPU 0–15、16 线程。
- 在线：最近 16 round 的位置衰减 KL（gamma=7）、LR=2e-4、fused AdamW、GQA、selective last layer、阶段 29 CPU backward。
- FC 输出采集；context KV 不求梯度；optimizer 后重算增量 context；冻结配置与混合版本历史保持不变。
- 每个进程先运行 256 token 的短请求预热；请求 finish/reset 完成后再测长题。
- 顺序：original1 → online1 → online2 → original2。运行串行，不开 PyTorch profiler。
- 每次在线请求从 W0 开始；原始组无 CPU draft 训练、capture 或权重安装。

原始源码由 `git archive 3979838d2d561d18f0e2082a8b51c6b23d881136 vllm` 导出，
放在 `/tmp/dflash-acceptance-20261020/original_source`。只链接同一环境的编译扩展和构建生成的第三方 Python 文件。
worker 返回的 `vllm_source` / `dflash_source` 已确认来自该导出目录；在线组来自当前工作树。
第一次原始源码启动因缺少构建生成的 FlashMLA Python 接口失败，补全后重新运行；失败发生在模型初始化，未产生测量数据。

复现本次四次请求：

```bash
.venv/bin/python /tmp/dflash-acceptance-20261020/run_comparison.py
.venv/bin/python /tmp/dflash-acceptance-20261020/analyze.py
```

每次的 `*-command.json` 保存完整命令、环境、基准 commit 与源码 SHA256；实验不修改推理或训练实现。

## 统计口径

在 frontend 包装已有 `engine_core.get_output()`，读取每轮 `scheduler_stats.spec_decoding_stats`，
只追加 Python 标量到内存列表；没有额外 GPU tensor 回传、逐轮 RPC、GPU 同步或逐轮文件写入。
单请求下验证 `num_drafts == 1`，因此每条记录对应一个 speculative verification round。

```text
逐轮接受长度 = accepted_draft_tokens + 1
平均接受长度 = 1 + sum(accepted_draft_tokens) / num_rounds
decode tokens/s = (输出 token 数 - 1) / (last_token_ts - first_token_ts)
合并两次：分别汇总 accepted/rounds 与 tokens/seconds，再求比值
```

首个 prefill 生成 token 不计入 decode；接受长度中的 +1 是 target correction/bonus token。
GPU rounds 只数 verification；capture 内部还包含 bootstrap bookkeeping，因此其 completed_rounds 不能直接替代这里的轮数。
原始 JSON 的 metrics 字段是含预热的累计指标；本报告使用清空后的逐轮计数，未把累计值混入请求统计。
CSV 同时保存 draft 接受数与实际 emitted 数。达到长度上限时，最后一轮已验证的输出可能被裁短，
因此官方接受长度可以比该轮实际 emitted 数大；吞吐始终使用实际输出数。
两种方案在相同 round 编号下可能位于不同输出位置，所以图按 context token 位置对齐，
CSV/hover 仍保留各自真实 round 编号。128-round 平滑线只是辅助，全部原始点都保留。

## 每次请求结果

| 运行 | GPU rounds | 平均接受长度 | Decode 秒 | Decode token/s | Round ms | CPU 更新 / GPU 安装 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| original1 | 10890 | 2.9827 | 165.506 | 196.25 | 15.198 | 0 / 0 |
| online1 | 6857 | 4.7372 | 102.494 | 316.91 | 14.947 | 115 / 114 |
| online2 | 6870 | 4.7282 | 102.680 | 316.33 | 14.946 | 115 / 114 |
| original2 | 10890 | 2.9827 | 165.662 | 196.07 | 15.212 | 0 / 0 |

## 按上下文位置比较

以下汇总两次重复。边界取不超过 4K 整数倍的最后完整输出 batch；精确边界与每次值保存在 `segments.csv`。

| Context 区间 | 原始接受长度 | 在线接受长度 | 原始 token/s | 在线 token/s | 吞吐变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0–4K | 3.3021 | 3.3325 | 239.30 | 241.44 | +0.90% |
| 4–8K | 2.8459 | 3.3497 | 198.02 | 233.20 | +17.77% |
| 8–12K | 2.2587 | 3.2251 | 153.46 | 219.10 | +42.77% |
| 12–16K | 2.1076 | 3.4520 | 138.93 | 227.38 | +63.67% |
| 16–20K | 1.9971 | 3.7126 | 128.18 | 238.16 | +85.80% |
| 20–24K | 4.8042 | 10.8177 | 301.28 | 678.63 | +125.25% |
| 24–28K | 5.4940 | 14.5435 | 335.20 | 886.71 | +164.53% |
| 28–32K | 4.5877 | 14.7343 | 271.94 | 873.18 | +221.09% |

## 输出一致性与边界

四次输出逐 token 比较：**完全一致**。

| 运行 | 输出 tokens | 结束原因 | 与 original1 相同的前缀 tokens |
| --- | ---: | --- | ---: |
| original1 | 32482 | length | 32482 |
| online1 | 32482 | length | 32482 |
| online2 | 32482 | length | 32482 |
| original2 | 32482 | length | 32482 |

该题到 32K 仍未自然结束，所有请求均标记为 length 截断；本实验不能给出完整答题正确性结论。
约 20K 后输出进入重复循环，尾部反复出现 `One 10, one 8, one 6, one 4, one 2: same.`。
按输出 token 切成 4096-token 段，在每段枚举连续 32-token 子串，使用 `1 - unique / total` 计算段内重复率。
输出 20480–24575 段的重复率为 92.35%，24576–28671 段为 99.41%；此前五段为 0.15%–6.79%。
因此后半段的大幅提升包含对高度重复内容的适应，不能视为一般数学推理的同等加速。
原始结果中的 correct=false 由缺少最终 boxed 答案产生，不把它解释成已经完成但答错。
逐轮 emitted 数之和已核对为输出 token 数减 1；每轮满足 0 ≤ accepted_draft_tokens ≤ drafted_tokens。
各个 4K 分段的轮数、token 数和时间与全请求统计一致。在线请求结束后 CPU/GPU 版本恢复为 0。
这是同一道题上的两次重复，不证明整个 AIME 集合的平均收益，也不证明 GPU 完全不受后台训练影响。

## 逐轮数据与图

- [可交互逐轮图：悬停查看轮号和接受数；图例可切换第二次重复](/tmp/dflash-acceptance-20261020/acceptance-interactive.html)
- [静态图 PNG](/tmp/dflash-acceptance-20261020/acceptance.png) / [PDF](/tmp/dflash-acceptance-20261020/acceptance.pdf)
- [全部逐轮 CSV](/tmp/dflash-acceptance-20261020/all-rounds.csv)
- [原始第 1 次 CSV](/tmp/dflash-acceptance-20261020/original1-rounds.csv) / [在线第 1 次 CSV](/tmp/dflash-acceptance-20261020/online1-rounds.csv)
- [原始第 2 次 CSV](/tmp/dflash-acceptance-20261020/original2-rounds.csv) / [在线第 2 次 CSV](/tmp/dflash-acceptance-20261020/online2-rounds.csv)
- [分段 CSV](/tmp/dflash-acceptance-20261020/segments.csv) / [统计汇总 JSON](/tmp/dflash-acceptance-20261020/summary.json)

原始请求 JSON 包含完整输出 token、文本、engine 时间线和逐轮 scheduler 计数；均在同一实验目录。

## 补充：自然答完的长推理 AIME 第 2 题

第 15 题的尾部重复会放大全程收益，因此额外选择此前能自然结束的第 2 题。
仅将 `--aime-offset` 改为 1；使用相同上限与其余配置，原始和在线各运行一次。
该补充是单次对照，不能据此估计运行方差。

| 方案 | 输出 tokens | GPU rounds | 平均接受长度 | Decode 秒 | Decode token/s | 答案 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 原始 DFlash | 11108 | 2800 | 3.9675 | 39.932 | 278.15 | 62 |
| 当前在线 | 11108 | 2541 | 4.3711 | 36.152 | 307.23 | 62 |

平均接受长度变化 +10.17%，decode 吞吐变化 +10.45%。
两者 finish_reason 均为 stop / stop，答案正确性为 True / True；
输出 token 完全一致：True。在线 CPU 更新 42 次、GPU 安装 41 次。
原始组 prefill 为 0.133 秒，其中出现一次输入准备 kernel 的 JIT；在线组为 0.018 秒。
两组 decode 计时都从首 token 后开始，以上 prefill 差异不计入表内吞吐。

同一道第 15 题只看前 16K（两次合并），原始/在线平均接受长度为 2.5351 / 3.3380，
decode 为 173.50 / 229.80 token/s；这是同一条 32K 请求的前缀统计，并非另一次独立 16K 请求。

- [第 2 题逐轮交互图](/tmp/dflash-acceptance-20261020/aime2/acceptance-interactive.html)
- [第 2 题全部逐轮 CSV](/tmp/dflash-acceptance-20261020/aime2/all-rounds.csv)
- [第 2 题原始逐轮 CSV](/tmp/dflash-acceptance-20261020/aime2/original1-rounds.csv) / [在线逐轮 CSV](/tmp/dflash-acceptance-20261020/aime2/online1-rounds.csv)
- [第 2 题统计 JSON](/tmp/dflash-acceptance-20261020/aime2/summary.json)

复现：`.venv/bin/python /tmp/dflash-acceptance-20261020/aime2/run_comparison.py`。
