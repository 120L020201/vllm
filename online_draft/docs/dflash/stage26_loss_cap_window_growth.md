# 阶段 26：训练最近 16 个 round 能否阻止 CPU 窗口变大

## 问题与测量对象

2026-10-20，分支 `eagle3&dflash`。本阶段针对用户所说的积压：
**一次 CPU optimizer 更新所接收的完整窗口 round 数随着请求推进而增加。**
测量的是 `window_rounds`，不是 FIFO 长度，也不是参与 loss 的 `trained_rounds`。

比较整个窗口参与 loss，以及只训练最后 32、16、8 个完整 round。
每组两次独立进程；所有组均传 GPU FC 输出，context 不参与反传，旧 KV 保留，
optimizer 后使用新权重重新投影整个窗口的全部增量 context。

正式运行时代码和默认值未修改。复用现有 `--max-train-rounds`，其中 0 表示
整个窗口。临时入口仅通过 callable RPC 包装训练 recipe、snapshot 发布和安装，
记录 CPU 时间、context 位置及窗口大小；不增加 CUDA 同步，不运行 profiler。

## 固定配置

- Qwen3-8B + Qwen3-8B-DFlash-b16，BF16，K=15，单 GPU/单请求。
- target 与两套 draft slot 使用 CUDA Graph；关闭 prefix caching。
- CPU 训练 16 线程，核心 0–15；推理核心 18，capture 核心 19。
- 开启 grouped-query attention 和 selective last layer；fused AdamW。
- 学习率 2e-4，KL 位置衰减 gamma=7；关闭额外全参数梯度有限值扫描。
- 使用 `OMP_WAIT_POLICY=PASSIVE`、`MALLOC_MMAP_THRESHOLD_=1073741824`、
  `MALLOC_TRIM_THRESHOLD_=-1`。
- 本地 AIME 2026 第 15 题，thinking、greedy、seed=17，自然生成至 EOS 或
  prompt + output 达到 16384；`max_model_len=18432`。

本题用于长请求性能和调度验证。达到 16K 截断不等于完成解题，不用其 `correct`
字段评价数学能力或宣称准确率提升。

## 指标和口径

每次更新记录：完整窗口 round 数、训练 round 数、监督位置数、CPU 开始/结束时间、
更新前后持久 cache 长度、窗口首尾 anchor。另记录 snapshot 和 GPU 安装时刻。

bootstrap 单独一轮更新，不纳入窗口增长均值。按窗口最后确认位置分为 0–4K、
4–8K、8–12K、12–16K；跨区间的大窗口归属其结束位置，不能把全量组的一次巨大
更新当作均匀覆盖该区间的多个观测。

最后一次 CPU 更新即使没有安装到 GPU，也是真实执行的工作，因此保留并单独标记。
同时报告 CPU 更新数和 GPU 安装数。finish/reset 后两侧版本归零，context 连续性、
实际训练 round 数和安装关闭窗口位置均通过结果中的断言检查。
区间结果先在每次重复内部求均值，再对两次重复等权平均，避免更新更多的重复
获得更大权重。全量组最后三个区间每次仅各有一次更新，需结合完整更新序列看待。

decode 吞吐排除 prefill 和结束等待；完整请求吞吐包含 finish/reset。
所有组比较 token 序列，避免把不同生成路径的时间差误当作同轨迹加速。

## 数学解释

窗口由两次 GPU 权重安装划分。第 i 个窗口大小为 Ni，CPU 更新时间为 Ti，
snapshot、上传、安装及窗口交付的其他延迟为 Ui，GPU 每轮平均耗时为 gi，则：

```text
N(i+1) ≈ (Ti + Ui) / gi
```

整个窗口都参与训练时，query、MLP、head、loss 及 backward 工作量随 Ni 增加；
attention 还受历史长度 Li 影响。在局部近似中：

```text
Ti ≈ C(Li) + [a + b·Li + c]·Ni
```

限制最近 B 轮后，当 Ni >= B：

```text
Ti ≈ C(Li) + [a + b·Li]·B + c·Ni
```

上限截断的是 query 训练项随窗口大小的增长，仍保留全部增量 context 的 c·Ni 项。
同样没有消除历史 attention 和历史 KV 复制随 Li 的增长。所以上限可以显著削弱
“大窗口导致长更新、长更新导致下一窗口更大”的反馈，却不保证窗口保持常数。

16 是可调的训练预算，不是由 K=15 或 block_size=16 推导出的唯一值。
这里 B=16 表示 16 个完整 block，而每个 block 自身又包含 16 个 query 位置。

## 实验记录

原始输出、入口、命令与源码 hash 位于
`/tmp/dflash-window-ablation-20261020/`。未重建之前删除的仓库实验目录。
运行顺序为 16、全量、8、32，然后逆序重复；第一次 8 轮运行的依赖环境不一致，
排除并补跑，保留排除原因和原始输出供审计。

绘图库最初安装到了实验 venv，短暂将 numpy 2.3.5 遮盖为 2.5.4；发现后移除
这些新增依赖，恢复 numpy 2.3.5，并改用独立绘图 venv。受影响的第一遍 8 轮进程
在此期间启动，故不纳入结果。该次输出在 token 837 后分歧，原因未定位，不能
直接归因于 numpy。其他已运行进程在环境变化前已加载原依赖；后续进程在恢复后启动。

## 实测结果

8 次有效运行均生成 16098 个输出 token，加 286 个 prompt token，达到 16384 上限。
八份输出 token 序列完全一致；CPU/GPU 请求结束后均恢复 W0。全部正式源文件 hash
与实验开始时相同，实验没有改动生产路径。

**限制 loss 数据量显著抑制了窗口膨胀，但最后 16 轮并没有消除窗口随上下文增长。
16 也不是唯一有效的上限；8 和 32 都有明显作用。**

窗口大小的分区间均值，单位为每次 CPU 更新收到的 GPU round 数：

| 参与 loss 的 round | 0–4K | 4–8K | 8–12K | 12–16K | 首末区间增长 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 整个窗口 | 185.1 | 639.5 | 1321.5 | 3009.5 | 1525.7% |
| 最后 32 轮 | 81.7 | 92.0 | 102.5 | 115.0 | 40.8% |
| 最后 16 轮 | 62.0 | 66.9 | 72.3 | 83.2 | 34.3% |
| 最后 8 轮 | 55.4 | 56.9 | 62.1 | 66.4 | 19.9% |

CPU 更新时间的分区间均值，单位为秒，包含 query 训练与更新后增量 KV 重算，
不包含随后单独记录的 snapshot 复制：

| 参与 loss 的 round | 0–4K | 4–8K | 8–12K | 12–16K |
| --- | ---: | ---: | ---: | ---: |
| 整个窗口 | 4.438 | 18.903 | 44.644 | 131.362 |
| 最后 32 轮 | 1.026 | 1.202 | 1.380 | 1.633 |
| 最后 16 轮 | 0.729 | 0.834 | 0.940 | 1.121 |
| 最后 8 轮 | 0.632 | 0.688 | 0.787 | 0.872 |

全请求结果，两次独立运行分别列出更新/安装次数。吞吐为 token 数之和除以时间之和：

| 参与 loss 的 round | CPU 更新次数 | GPU 安装次数 | 最大窗口 | Decode token/s | 完整请求 token/s |
| --- | --- | --- | ---: | ---: | ---: |
| 整个窗口 | 8 / 8 | 7 / 7 | 3134 | 176.86 | 74.96 |
| 最后 32 轮 | 52 / 50 | 51 / 49 | 129 | 225.89 | 222.71 |
| 最后 16 轮 | 68 / 71 | 67 / 70 | 99 | 228.42 | 225.28 |
| 最后 8 轮 | 84 / 81 | 83 / 80 | 80 | 225.56 | 222.37 |

全量组两次最后窗口分别为 2885、3134 轮，CPU 更新耗时 128.05、134.68 秒，
均未安装到 GPU。限制最近 16 轮后，最后 4K 区间的平均窗口约 83.2 轮，
单次更新约 1.12 秒，但比前 4K 的 62.0 轮仍增长 34.3%。

最近 8 轮把最后 4K 的平均窗口进一步降低到 66.4 轮，增长约 19.9%；
CPU 更新更频繁，但本题 decode 吞吐略低于 16 轮组。不能把“更新更多”直接等同于
“整个系统更快”，也不据两次单题截断运行更改默认配置。

[逐次更新曲线与吞吐图](/tmp/dflash-window-ablation-20261020/window_growth.png)
同时提供 [PDF](/tmp/dflash-window-ablation-20261020/window_growth.pdf)。
图中实线/虚线为两次重复，bootstrap 已排除；全量对照使用对数纵轴，受限方案另有
线性纵轴放大图，避免几千轮的全量结果掩盖几十轮的剩余增长。

## 如何回答“16 轮是否必要”

- **是否需要限制本次参与 loss 的工作量：** 在本机、当前学习率和完整在线配置下，
  全窗口训练出现连续的大幅膨胀；8/16/32 上限都显著抑制了这一现象。
- **是否必须恰好 16：** 否。实验没有发现 16 是唯一有效值；它是当前速度与更新量
  的一个折中，本题上其 decode 吞吐略高于 8/32。
- **16 是否能让窗口不再变大：** 否。两次重复都仍增长；区间平均增长约 34%。
- **减少到 8 是否彻底解决：** 否。增长降到约 20%，没有归零。

限制 B 后不再让全部 Ni 个 round 进入 query forward/backward，显著削弱了
训练成本对窗口大小的反馈。但完整历史 attention、KV 复制和全窗口增量投影仍然
存在，所以固定 B 并不是窗口大小的硬上限。本实验验证了系统现象；未做算子分时
消融，不能把剩余增长的具体百分比全部归因于某一个算子。

## 复现和验证

以下只需替换训练上限为 0、32、16、8。临时入口与原始运行脚本的区别仅为测量包装：

```bash
export PATH="$PWD/.venv/bin:$PATH"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE
export MALLOC_MMAP_THRESHOLD_=1073741824 MALLOC_TRIM_THRESHOLD_=-1

.venv/bin/python /tmp/dflash-window-ablation-20261020/run_request.py \
  --online --target-graphs --train-threads 16 --cpu-isolation \
  --grouped-query-attention --selective-last-layer --learning-rate 2e-4 \
  --max-train-rounds 16 --loss-decay-gamma 7 --max-model-len 18432 \
  --aime-questions 1 --aime-offset 14 --context-tokens 16384 \
  --decode-timeline --output /tmp/dflash-window-check.json
```

- [逐次更新与分区间汇总](/tmp/dflash-window-ablation-20261020/summary.json)
- [两次重复的合计结果](/tmp/dflash-window-ablation-20261020/aggregate.json)
- [源码 hash](/tmp/dflash-window-ablation-20261020/source_sha256.json)
- [安装周期与 CPU 更新时间](/tmp/dflash-window-ablation-20261020/cycle_timing.json)

原始文件命名为 `b0/b32/b16/b8-r1/r2.json`，对应日志和 command manifest 同目录。
每个有效请求的安装边界、训练 round 选择、连续 cache、请求 reset 都通过分析脚本
断言；八份 token hash 相同。环境不一致的第一次 b8-r1 保留为 `excluded-environment-*`，
不参与统计。临时入口最初一次语法错误尚未启动模型，记录为 `startup-syntax-*`，
也不参与统计。

正式运行时代码无改动，未重复运行源码单元测试。新增报告和 README 通过 pre-commit，
`git diff --check` 通过；没有提交或推送。

## 全窗口训练的完整增长过程

以下从 bootstrap 后第一次更新开始，列出窗口 round 数：

```text
重复 1：59 → 123 → 214 → 339 → 625 → 1275 → 2885
重复 2：59 → 125 → 209 → 353 → 654 → 1368 → 3134
```

重复 1 的对应 CPU 更新耗时为：

```text
1.54 s → 2.79 s → 4.52 s → 8.62 s → 18.21 s → 42.70 s → 128.05 s
```

最后一次更新未安装到 GPU。生成结束后，finish 仍等待已经启动的训练完成，
因此不能只报告 decode 性能而隐藏这次 CPU 工作与结束等待。

## 结论的适用范围

本实验同时改变参与 loss 的数据量及由此产生的更新频率，属于完整在线系统的
策略消融。接受率、GPU round 总数和 GPU 安装版本轨迹可以随之变化；它并非
固定权重、固定窗口、固定输入的 CPU kernel 微基准。

最近 16 轮上限针对的是计算预算，并不能单独证明“必须只选最近数据”优于其他
采样方式，也不能证明 16 对所有请求和学习率最优。本轮没有比较均匀采样或随机
采样。完整非因果 block 的 query 仍全部保留，改变的是 block 数而不是删除
block 内未监督 query。

若目标严格要求窗口不随历史增长，固定 B=16 不构成保证。可以继续减少历史复制、
优化长 context attention，或按更新周期预算动态降低 B；当 B=1 时仍超过预算，
则必须处理不可缩减的历史、optimizer、snapshot 等成本，不能再靠减少 loss
样本保证目标。上述是后续方案，本阶段未修改算法或默认设置。
