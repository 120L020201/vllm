# 阶段 32：筛选新的超长、无持续循环 AIME 样本

## 最终选题

**AIME 2026 第 27 题**，本地数据集 offset=26，内容是由两类等腰三角形组成的四面体，
求其内接球与外接球的球心距离。此前主对照的第 2、15 题均被明确排除，本阶段没有运行它们。

原始 DFlash 自然生成 **30,459 token** 后结束（finish_reason=stop），
prompt 186 token，最终上下文 30645 token。
最长连续周期片段只有 18 token，没有连续相同段落；输出没有第 15 题那种持续重复循环。

**质量边界：这次模型答案为 143，标准答案为 223，答题错误。**
该样本满足长输出、自然结束和无持续循环的性能测试条件，不将它描述成答对的长推理。

## 筛选过程

本阶段实际完成了五道新的原始 DFlash 请求。筛选优先找输出至少 16384 token、
尊重 EOS、自然结束且无持续循环的题，并单独核对答案。没有修改 prompt 来要求额外输出，
没有更改 temperature 或添加 repetition penalty；候选选择不参考在线加速幅度。

| 题号 | 输出 tokens | 结束原因 | 最长连续周期片段 | 答案情况 | 处理 |
| --- | ---: | --- | ---: | --- | --- |
| aime2026-14 | 38785 | length | 28562 | 未给出最终整数答案 | 长循环，排除 |
| aime2026-12 | 12459 | stop | 3 | 正确 | 正常结束，但少于 16K |
| aime2026-28 | 38744 | length | 8337 | 13，标准 107 | 长循环，排除 |
| aime2026-27 | 30459 | stop | 18 | 143，标准 223 | 选作长输出性能样本 |
| aime2026-29 | 38697 | length | 43 | 12，标准 157 | 达到上限仍未结束 |

第 30 题在筛选程序自动推进后开始生成，因已找到符合用户长度与非循环要求的第 27 题，
主动停止剩余搜索；第 30 题未完成，也不计入筛选结果。停止原因保存在 `screen-stop.json`。
初筛脚本的 selected 字段额外要求答案正确，五道题均未同时满足该额外条件；
最终按用户要求将“长度和非循环”与“答案正确性”分开，选题记录在 selected.json，
screen-summary.json 同时保留 strict_correct_answer_gate 和 selected_for_comparison，避免混淆。

## 非循环检查的范围

- 枚举 1–256 token 的周期，至少匹配三个周期，记录最长连续周期片段。第 27 题仅 18 token。
- 按空行划分非空段落，最多连续相同段落为 1，即没有紧邻重复段落。
- 每 4096 个输出 token 枚举 32-token 子串，计算 `1 - unique / total`；各段为 0.17%–7.87%。
- 人工检查完整输出的开头、中段与结尾，最终进入解答正文并给出 boxed 答案、自然结束。

这些检查针对此前的持续退化循环，不保证推理没有回顾、改算、重复公式或语义上相近的段落。
本例是主动筛选出来的长请求，不能当作整个 AIME 数据集的无偏性能估计。

## 推理与在线配置

- Target `/data/models/Qwen3-8B`；draft `/data/models/Qwen3-8B-DFlash-b16`。
- 原始 vLLM 来自 commit 3979838d2d561d18f0e2082a8b51c6b23d881136 的独立源码导出，关闭在线训练。
- GPU RTX PRO 6000 Blackwell 96 GB；CPU Xeon w7-3445；单请求、BF16、15 个 draft token。
- Temperature=0，seed=17，thinking template，尊重 EOS，关闭 prefix caching。
- max_model_len=40960，总上下文上限 38912；选定请求自然结束，未触及上限。
- Target/draft CUDA Graph、FlashInfer；推理固定 CPU 18，采集 CPU 19，训练 CPU 0–15 / 16 线程。
- 当前阶段 29 backward；最近 16 round、位置衰减 KL gamma=7、LR=2e-4、fused AdamW、GQA、selective last layer。
- 冻结配置、FC capture、无梯度 context KV、optimizer 后增量重投影和混合版本历史均保持原方案。
- 模型初始化后预热短请求；筛选的各题顺序执行；在线正式请求从 W0 开始；不开 profiler。

原始数据复用本次筛选中第 27 题的真实请求，随后新跑一次在线对照，属于每组一次测量。
筛选期间的重复分析在请求结束后执行，不包含在其 decode 时间中。

## 逐轮接受长度与吞吐

从 scheduler 的原有 spec_decoding_stats 逐轮读取 accepted_draft_tokens，接受长度定义为
`accepted_draft_tokens + 1`。Decode token/s 为 `(输出 tokens - 1)/(last_token_ts - first_token_ts)`，
不含加载、prefill 与 finish/reset。每轮标量在 frontend 内存中记录，没有新 GPU 同步或逐轮 RPC。

| 方案 | 输出 tokens | 验证 rounds | 平均接受长度 | Decode 秒 | Decode token/s | CPU 更新 / GPU 安装 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 原始 DFlash | 30459 | 11466 | 2.6565 | 177.825 | 171.28 | 0 / 0 |
| 当前在线 | 30459 | 7007 | 4.3471 | 106.864 | 285.02 | 115 / 114 |

平均接受长度变化 +63.64%，decode 吞吐变化 +66.40%。
完整输出逐 token 一致：True；共同前缀 30459 token。
在线结束原因为 stop，答案为 143，标准答案为 223。

这里比较完整系统（包括此前 FlashInfer metadata 优化），不把差异单独归因于 CPU backward。
单次对照不能估计运行方差，答题正确性与吞吐结果分别报告。

| Context 区间 | 原始接受长度 | 在线接受长度 | 原始 token/s | 在线 token/s |
| --- | ---: | ---: | ---: | ---: |
| 约 0–4K | 4.1246 | 4.1966 | 299.04 | 304.64 |
| 约 4–8K | 3.6610 | 4.1002 | 254.83 | 285.81 |
| 约 8–12K | 3.3377 | 4.5077 | 226.46 | 306.26 |
| 约 12–16K | 2.8858 | 4.6402 | 190.00 | 305.78 |
| 约 16–20K | 2.5012 | 4.5935 | 160.38 | 294.70 |
| 约 20–24K | 2.3286 | 4.8231 | 145.72 | 302.10 |
| 约 24–28K | 1.7605 | 3.7095 | 107.40 | 226.21 |
| 约 28K–自然结束 | 1.9175 | 4.4898 | 114.42 | 267.99 |

分段边界取不超过 4K 整数倍的最后完整输出 batch；精确边界保存在 segments.csv。
两种方案的 round 编号不对应相同输出位置，因此图按 context token 位置对齐，悬停仍可查看真实 round 编号。

## 产物与复现

- [选定题目原文](/tmp/dflash-aime-search-20261020/selected-question.md) / [完整原始输出](/tmp/dflash-aime-search-20261020/selected-output.txt)
- [逐轮交互图](/tmp/dflash-aime-search-20261020/comparison/acceptance-interactive.html)
- [PNG](/tmp/dflash-aime-search-20261020/comparison/acceptance.png) / [PDF](/tmp/dflash-aime-search-20261020/comparison/acceptance.pdf)
- [全部逐轮 CSV](/tmp/dflash-aime-search-20261020/comparison/all-rounds.csv)
- [原始 CSV](/tmp/dflash-aime-search-20261020/comparison/original1-rounds.csv) / [在线 CSV](/tmp/dflash-aime-search-20261020/comparison/online1-rounds.csv)
- [筛选记录](/tmp/dflash-aime-search-20261020/screen-summary.json) / [选题信息](/tmp/dflash-aime-search-20261020/selected.json)
- [对照汇总](/tmp/dflash-aime-search-20261020/comparison/summary.json) / [验证记录](/tmp/dflash-aime-search-20261020/comparison/validation.json)

```bash
.venv/bin/python /tmp/dflash-aime-search-20261020/comparison/run_comparison.py
.venv/bin/python /tmp/dflash-aime-search-20261020/comparison/analyze.py
.venv/bin/python /tmp/dflash-aime-search-20261020/validate_comparison.py
```

第一条命令重跑选定新题的在线请求；原始请求来自 screen-results.json 的 record_index=3，
其完整参数保存在 screen-command.json 及 comparison/original1-command.json。
本阶段不修改推理或训练实现，只增加实验记录、报告及索引。

## 每 256 round 分组的接受长度曲线

复用上述两次请求的逐轮 CSV，分别按 round 1–256、257–512 等不重叠区间，
对 `accepted_draft_tokens + 1` 求算术平均，两条曲线绘制在同一坐标轴中。
原始方案共 11466 round，得到 44 个完整分组和末尾 202 round；
在线方案共 7007 round，得到 27 个完整分组和末尾 95 round。
最后不足 256 round 的分组保留，以空心菱形标记。

主图横轴沿用 context token 位置，每点取该组各 round 的 `context_start` 均值。
另提供以 round 中点为横轴的版本；两种方案相同 round 编号不代表相同文本位置。
这两种图仅改变横轴，分组和纵轴数值相同。分组轮数求和与原始记录一致，
按组内轮数加权后的均值与全请求平均接受长度一致。

- [同图对比 PNG（context 横轴）](/tmp/dflash-aime-search-20261020/comparison/acceptance-256-context.png)
- [交互图（context 横轴）](/tmp/dflash-aime-search-20261020/comparison/acceptance-256-context.html)
- [同图对比 PNG（round 横轴）](/tmp/dflash-aime-search-20261020/comparison/acceptance-256-rounds.png)
- [分组 CSV](/tmp/dflash-aime-search-20261020/comparison/acceptance-256-groups.csv)
- [绘图脚本](/tmp/dflash-aime-search-20261020/comparison/plot_256.py)

此次仅重绘已有数据，没有重新运行推理或训练。
