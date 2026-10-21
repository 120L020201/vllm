# 阶段 31：以无持续重复循环的长输出 AIME 作为主对照

## 题目与选择依据

主对照改用本地 AIME 2026 第 2 题：求十进制表示中不含零、各位数字和为 13 的正整数回文数数量。
Target 为 Qwen3-8B，draft 为 Qwen3-8B-DFlash-b16。四次请求均自然结束，输出 11,108 token，
答案 62 正确，原始与在线的完整输出逐 token 一致。Prompt 为 106 token，最终上下文 11,214 token。

这是从已有样本中选择能够自然答完的长题；没有修改题目、采样温度、重复惩罚或推理上限。
它用于替换阶段 30 中进入循环的第 15 题作为主要性能例子，不是完整 AIME 集合的无偏评测。

## 重复检查

检查输出中的连续周期片段：周期长度从 1 到 256 token，至少连续匹配 3 个周期；
另检查连续完全相同的非空段落，并统计每个 4096-token 输出段中重复 32-token 子串的比例。

| 指标 | 第 2 题 | 第 15 题（参照） |
| --- | ---: | ---: |
| 最长连续周期片段 | 26 token | 11,715 token |
| 最长片段周期 | 2 token | 24 token |
| 最多连续相同段落 | 1 | 488 |
| 结束原因 | stop，自然结束 | length，触及上限 |

第 2 题三个输出段的重复 32-token 子串比例分别为 0.861%、0.221%、0.208%。
其短周期片段属于正常数字排列；人工检查结尾为按长度分类计数并给出最终答案。
因此本样本没有发现第 15 题那种持续重复循环。这不是承诺文本中没有任何重复公式或短句，
也不是穷尽所有形式的语义重复检测。

## 运行配置与统计口径

- 沿用阶段 30 配置：greedy、seed=17、BF16、batch=1、15 个 draft token、target/draft CUDA Graph。
- 总上下文上限仍为 32768，max_model_len=34816，尊重 EOS；实际远未触及上限。
- 推理 CPU 18；采集 CPU 19；训练 CPU 0–15、16 线程；在线使用阶段 29 backward。
- 在线仍训练最近 16 round；LR=2e-4；位置衰减 KL gamma=7；冻结配置和增量 context 策略不变。
- 原始源码来自基准 commit 3979838d2d561d18f0e2082a8b51c6b23d881136，关闭在线训练。
- 每个进程先预热 256 token，请求 finish/reset 后再测正式长题；不开 profiler。
- 第一次 original1 / online1 复用阶段 30 的同配置真实记录；本阶段新跑 online2 → original2。
- 每次 online 请求重新从 W0 开始；汇总使用原始 → 在线 → 在线 → 原始两次对照。

接受长度为 scheduler 的 accepted_draft_tokens + 1。Decode token/s 为
`(输出 token 数 - 1) / (last_token_ts - first_token_ts)`，不计 prefill、加载或 finish/reset。
每轮从已有 engine-core scheduler stats 读取标量，无额外 GPU 同步；首个 prefill 输出不计入验证 round。
两次合并分别汇总 accepted/rounds 和 tokens/seconds 后求比值。

## 结果

| 方案，两次合并 | 平均接受长度 | Decode token/s |
| --- | ---: | ---: |
| 原始 DFlash | 3.9675 | 278.21 |
| 当前在线 | 4.3806 | 308.16 |

平均接受长度增加 10.41%；
decode 吞吐增加 10.76%。

| 运行 | 验证 rounds | 平均接受长度 | Decode 秒 | Decode token/s | CPU 更新 / GPU 安装 |
| --- | ---: | ---: | ---: | ---: | --- |
| original1 | 2800 | 3.9675 | 39.932 | 278.15 | 0 / 0 |
| online1 | 2541 | 4.3711 | 36.152 | 307.23 | 42 / 41 |
| online2 | 2530 | 4.3901 | 35.935 | 309.09 | 42 / 41 |
| original2 | 2800 | 3.9675 | 39.913 | 278.28 | 0 / 0 |

## 按上下文位置观察

边界取不超过 4K 整数倍的最后完整输出 batch；末段到自然 EOS。精确边界保存在分段 CSV。

| Context 区间 | 原始接受长度 | 在线接受长度 | 原始 token/s | 在线 token/s |
| --- | ---: | ---: | ---: | ---: |
| 约 0–4K | 4.3824 | 4.3872 | 318.02 | 318.69 |
| 约 4–8K | 4.1447 | 4.4779 | 288.48 | 312.04 |
| 约 8K–11.2K | 3.3548 | 4.2472 | 229.32 | 290.59 |

逐轮波动与生成内容有关；不能把分段差异仅解释成上下文长度的因果作用。
此处比较完整系统，包括现有 FlashInfer metadata 优化，不单独归因于某个 CPU 算子。

## 逐轮产物与复现

- [交互图：全部原始点，悬停查看 round；图例切换第二次重复](/tmp/dflash-nonrepeating-aime2-20261020/acceptance-interactive.html)
- [PNG](/tmp/dflash-nonrepeating-aime2-20261020/acceptance.png) / [PDF](/tmp/dflash-nonrepeating-aime2-20261020/acceptance.pdf)
- [全部逐轮 CSV](/tmp/dflash-nonrepeating-aime2-20261020/all-rounds.csv)
- [原始第 2 次 CSV](/tmp/dflash-nonrepeating-aime2-20261020/original2-rounds.csv) / [在线第 2 次 CSV](/tmp/dflash-nonrepeating-aime2-20261020/online2-rounds.csv)
- [重复检查](/tmp/dflash-nonrepeating-aime2-20261020/repetition-check.json)
- [汇总 JSON](/tmp/dflash-nonrepeating-aime2-20261020/summary.json) / [验证记录](/tmp/dflash-nonrepeating-aime2-20261020/validation.json)

实验目录保留全部逐轮接受数、输出 token、文本、时间线、日志、命令与源码哈希。
图按 context 位置对齐，因为两组相同 round 编号不一定对应同一输出位置。

```bash
.venv/bin/python /tmp/dflash-nonrepeating-aime2-20261020/run_comparison.py
.venv/bin/python /tmp/dflash-nonrepeating-aime2-20261020/analyze.py
.venv/bin/python /tmp/dflash-nonrepeating-aime2-20261020/check_repetition.py
```

第一条命令重跑本阶段新增的 online2 / original2；既有 original1 / online1 及其命令清单保留在同目录。
本阶段不改动训练或推理实现，只新增实验报告和索引。
