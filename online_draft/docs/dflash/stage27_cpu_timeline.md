# 阶段 27：当前 CPU 分阶段耗时与 GPU 推测解码周期

2026-10-20，分支 `eagle3&dflash`，base `3979838d2d`，未提交工作树。
本阶段测量当前固定 FC 采集、detached context、optimizer 后重建增量 KV 的实现。
没有修改训练、推理或传输源码，也没有实施上一轮讨论中的优化候选。

## 主要结果

- 两次轻量计时运行的 GPU 平均推测解码周期分别为 **14.497 / 14.511 ms**。
  从前 4K 的约 13.77 ms 增至最后 4K 的约 15.15 ms。
- CPU 单次更新在前 4K 约 **737–747 ms**，最后 4K 约 **1025–1155 ms**，
  均不含随后约 52–57 ms 的 pinned snapshot 复制。
- early/late trace 中 attention backward 的算子 self time 从 **40.6 ms**
  增至 **374.0 ms**；同期 GEMM backward 为 **234.1 / 220.7 ms**。
  本次片段中的 backward 增长主要出现在长历史 attention。
- `zero_grad(set_to_none=True)` 也有明显耗时：轻量计时约 **56–64 ms/更新**，
  主要涉及释放梯度存储及相关内存管理，不能把它当作可忽略的清理步骤。

这里统计的是训练主线程经过各阶段的 wall time，包含阶段内工作线程完成计算所需
时间，不是 16 个训练线程的 CPU time 之和。Capture 线程与训练线程、GPU 并行，
其时间不能直接加到一次训练更新时间上。

## 配置与测量方式

沿用阶段 26：Qwen3-8B + Qwen3-8B-DFlash-b16，BF16，K=15，单请求；target 与
两套 draft slot 使用 CUDA Graph，FLASHINFER，关闭 prefix caching。训练最近
16 个完整 round，保留全部连续 context 增量。学习率 2e-4、KL gamma=7；开启
grouped-query attention、selective last layer、fused AdamW，不使用 master
weights，不扫描全量梯度有限值。训练核心 0–15，推理核心 18，capture 核心 19。

本地 AIME 2026 第 15 题，thinking、greedy、seed=17；prompt 286 token，长请求
自然生成至总上下文 16384，max_model_len=18432。保留阶段 26 的 OMP 和 allocator
环境设置。

串行执行四次请求：

| 运行 | 上下文上限 | 测量方式 | CPU 更新 / GPU 安装 | Decode token/s |
| --- | ---: | --- | ---: | ---: |
| timed | 16384 | 轻量 host 计时 | 69 / 68 | 228.47 |
| early | 4096 | 从起始截取 400 worker step | 见原始结果 | 不作为性能结果 |
| late | 16384 | 延迟 4200 step 后截取 400 step | 见原始结果 | 不作为性能结果 |
| timed2 | 16384 | 轻量计时，补充 zero_grad 范围 | 71 / 70 | 227.31 |

轻量计时通过临时 callable RPC 包装现有函数和 profile range，只读取 host 时钟。
未增加 CUDA event、同步、tensor 操作或数据选择逻辑。PyTorch profiler 开启
all-threads，关闭 shape、stack、memory 记录；只保留 early/late 小片段。
Profiler 的启动、采集和导出会改变请求调度，因此其整请求吞吐不进入性能比较。

第一次轻量计时发现约 50 多毫秒未归类时间，随后增加 zero_grad 计时并运行 timed2。
timed/early/late 使用初始测量入口，timed2 使用补充入口；各自源码 hash 写入
command manifest，初始入口保存在 `run_request_initial.py`。算法源码相同。

## CPU 一次更新的完整拆分

下表采用 **timed2** 的非 profiler 计时，单位 ms/CPU 更新，数值为区间均值。
按窗口最后确认位置分桶；bootstrap 单独一轮，不进入表格。四个区间分别有
18、18、18、16 次更新。最后一笔已执行但未安装的 CPU 更新保留。

| 步骤 | 0–4K | 4–8K | 8–12K | 12–16K |
| --- | ---: | ---: | ---: | ---: |
| 输入整理、mask 与 forward 其他准备 | 5.16 | 5.42 | 5.50 | 5.29 |
| 更新前新增 context 投影 | 18.55 | 17.85 | 16.91 | 17.53 |
| 更新前历史 KV 拼接 | 1.96 | 4.36 | 6.48 | 8.80 |
| 五层 draft forward | 112.46 | 123.57 | 140.25 | 151.13 |
| LM head forward | 18.19 | 17.73 | 16.29 | 15.61 |
| 位置权重与 KL loss | 8.13 | 7.92 | 7.05 | 6.58 |
| Backward | 349.24 | 434.65 | 519.81 | 599.70 |
| Fused AdamW optimizer | 139.57 | 139.14 | 137.26 | 137.04 |
| zero_grad，两次调用合计 | 61.33 | 63.79 | 58.11 | 55.53 |
| 更新后新增 context 投影 | 18.65 | 16.91 | 16.74 | 16.96 |
| 更新后历史 KV 拼接 | 2.03 | 4.51 | 6.57 | 8.77 |
| 剩余清理、特征拼接和版本记录等 | 1.56 | 1.92 | 1.64 | 1.74 |
| **训练 recipe 合计** | **736.82** | **837.77** | **932.60** | **1024.67** |
| 随后复制 pinned snapshot | 52.47 | 52.57 | 52.45 | 52.03 |
| **训练加 snapshot** | **789.29** | **890.34** | **985.05** | **1076.70** |

各项为互斥划分，合计存在显示舍入误差。输入准备通过 forward 总范围扣除已分别
计时的 context 投影、拼接、draft forward、head 得到；剩余项为 recipe 总时间
减去各已计时阶段。snapshot 单独计时，不属于 pipeline 的 train_seconds。

五层 draft forward 中，attention 路径合计从 **55.74 ms** 增至 **101.73 ms**；
MLP 合计为 **47.11 / 38.82 ms**。这些是 forward 的子项，不应再次加到总计。
Attention 路径包括 QKV 投影、norm/RoPE、context/query 拼接、SDPA 和输出投影。

第一次运行 timed 的训练 recipe 四个区间均值为 **747.36 / 819.74 / 958.10 /
1154.55 ms**。两次运行的窗口与监督行数不同：最后 4K 的平均窗口分别为
85.86 / 76.25 round，平均监督位置为 55.07 / 50.06。当前表不是固定输入微基准；
这两次差异不能解释成测量入口带来了优化收益。

## Backward 在 trace 内的算子拆分

只统计完整记录的非 bootstrap 更新：early 为版本 2–6，共 5 次，其 context end
为 485–1324；late 为版本 63–66，共 4 次，其 context end 为 14214–15112。
边界处截断的更新不参与。这里的 early/late 片段与上面的整个 0–4K、12–16K
分桶范围不同，不将两套数字混合相加。

| Backward 项目 | Early ms/更新 | Late ms/更新 |
| --- | ---: | ---: |
| CPU Flash Attention backward，self time | 40.63 | 374.02 |
| `aten::mm`，self time | 234.12 | 220.67 |
| 其余算子、autograd 调度及范围内剩余时间 | 55.59 | 62.15 |
| **Backward 总 wall time** | **330.33** | **656.84** |

按相同线程上的 CPU operator 嵌套关系计算 self time，避免将父子算子的 inclusive
时间重复相加。Flash Attention backward 的 inclusive 时间为 42.10 / 379.44 ms，
其中包含内部子操作，因此表中采用扣除子操作后的 self time。

这支持继续优化长 context attention backward。**374 ms 不是“全部可删除”的
历史 K/V 梯度时间**：算子仍包含必要的 dQ、softmax 反传和 query K/V 梯度。
本阶段没有对固定 context 的分段 backward 实施消融，不能宣称能省掉这 374 ms。

两份 trace 中 `Optimizer.zero_grad#AdamW.zero_grad` 的均值为 52.86 / 46.88 ms。
timed2 单独包装 zero_grad 后进一步确认其耗时；它是梯度引用清理和内存释放路径，
不是额外梯度有限值扫描。set_to_none 不会将整个梯度 tensor 填零，但仍可能产生
明显的释放成本。

## Capture、D2H 与上传

timed2 的 capture 线程每个 packet：

| CPU 范围 | 均值 ms | 中位数 ms | P95 ms |
| --- | ---: | ---: | ---: |
| D2H 提交及等待 ready/copy_done | 10.27 | 11.31 | 13.15 |
| CPU prepare：FC/元数据复制、teacher softmax 与归一化、入队 | 4.07 | 2.86 | 10.70 |

10.27 ms 包含等待 GPU 生产数据和拷贝完成，**不是 PCIe 搬运耗时，也不是 GPU
推理线程停顿**。trace 中实际 D2H DMA 为 early **0.143 ms/packet**、late
**0.126 ms/packet**。early 包含首次 prefill packet。

Capture prepare 的 trace 算子 self time，单位 ms/packet：

| 算子 | Early | Late |
| --- | ---: | ---: |
| `_softmax` | 0.605 | 0.598 |
| `sum` | 0.378 | 0.389 |
| `div_` | 0.626 | 0.679 |
| `copy_` | 2.355 | 1.814 |

`copy_` 汇总了 prepare 内不同数据复制及 dtype 转换，不等于 FC 单项耗时；sum
和 div 包括当前 softmax 后再次归一化的路径。以上来自 profiler，只用于定位，
不能直接替换非 profiler 的 capture 均值。

Pinned snapshot 复制约 52 ms，属于 CPU 训练线程。随后由推理线程提交 H2D，
early/late CPU 提交范围均值 **0.729 / 0.822 ms**，中位数均约 0.699 ms；实际
权重 DMA **34.54 / 34.58 ms/次**，commit CPU 范围 **0.062 / 0.053 ms/次**。
H2D 私有 stream 与推理并行，不能将其 34.6 ms 逐轮加到 GPU 推理周期上。

## GPU 一轮推测解码的时间

这里“一轮”指稳态的一次 target 验证、采样/接受处理和下一次 DFlash proposal，
不是一个输出 token，也不是一次 CPU optimizer 更新。

先使用不启用 PyTorch profiler 的 engine-core 输出 batch 时间戳计算相邻轮次
间隔。该请求每个 decode round 对应一个输出 batch；两个运行分别有 4860 / 4880
个 speculative round，均生成 16098 个 token。全程 decode 时间除以 round 数：

| 测量 | 第一次 ms/round | 第二次 ms/round |
| --- | ---: | ---: |
| 0–4K | 13.762 | 13.772 |
| 4–8K | 14.338 | 14.351 |
| 8–12K | 14.697 | 14.703 |
| 12–16K | 15.146 | 15.162 |
| **全请求平均** | **14.497** | **14.511** |

这是端到端稳态轮次周期，包含正常 host 调度和 kernel 间隙。GPU trace 中再用
连续两次 target graph 首 kernel 的时间差验证周期，并利用 cudaGraphLaunch
correlation 关联每次 graph 的所有 kernel：

| 同一 GPU 周期的拆分 | Early ms/round | Late ms/round |
| --- | ---: | ---: |
| Target verification graph，首 kernel 至末 kernel | 10.375 | 11.736 |
| Draft query graph，首 kernel 至末 kernel | 2.238 | 2.425 |
| Graph 外工作与间隙，按周期差额计算 | 1.137 | 1.103 |
| **完整周期** | **13.749** | **15.264** |

Early/late 分别使用 398/399 个完整相邻周期，每个周期恰有一个 target graph
和一个 draft graph。Target graph 每次 518 个 kernel，两个 draft slot 的 graph
每次均为 78 个 kernel。表中的 draft graph 不代表所有 draft 相关工作：context
准备、head、采样等 graph 外工作计入差额；差额同时含 kernel 间隙，不能全部称为
采样时间。

## 文件与复现

全部原始结果及临时测量脚本位于 `/tmp/dflash-cpu-timeline-20261020/`。

- [Early CPU/GPU PyTorch trace，约 16 MB](/tmp/dflash-cpu-timeline-20261020/early/rank0.1792468071442592911.pt.trace.json.gz)
- [Late CPU/GPU PyTorch trace，约 16 MB](/tmp/dflash-cpu-timeline-20261020/late/rank0.1792468209452278409.pt.trace.json.gz)
- [完整 16K 请求 CPU 阶段时间线，轻量计时](/tmp/dflash-cpu-timeline-20261020/cpu-wall-timeline.json.gz)
- [CPU 各阶段 CSV](/tmp/dflash-cpu-timeline-20261020/cpu-stages.csv)
- [逐次更新、分桶统计与 GPU 轮次时间](/tmp/dflash-cpu-timeline-20261020/timing_summary.json)
- [Early 算子与 GPU graph 统计](/tmp/dflash-cpu-timeline-20261020/early-trace-summary.json)
- [Late 算子与 GPU graph 统计](/tmp/dflash-cpu-timeline-20261020/late-trace-summary.json)
- [输出、版本复位、context 连续性及源码检查](/tmp/dflash-cpu-timeline-20261020/validation.json)

trace 可用 Perfetto 打开；轻量 CPU 时间线没有 GPU kernel 或算子事件，完整
CPU/GPU 关联请使用 early/late PyTorch trace。各运行参数和环境保存在对应的
`*-command.json`，运行日志和原始结果为同名前缀的 `.log`、`.json`。

复现 timed2：

```bash
export PATH="$PWD/.venv/bin:$PATH"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE
export MALLOC_MMAP_THRESHOLD_=1073741824 MALLOC_TRIM_THRESHOLD_=-1

.venv/bin/python /tmp/dflash-cpu-timeline-20261020/run_request.py \
  --online --target-graphs --train-threads 16 --cpu-isolation \
  --grouped-query-attention --selective-last-layer --learning-rate 2e-4 \
  --max-train-rounds 16 --loss-decay-gamma 7 --max-model-len 18432 \
  --aime-questions 1 --aime-offset 14 --context-tokens 16384 \
  --decode-timeline --output /tmp/dflash-timeline-recheck.json
```

early 使用 `--context-tokens 4096 --profile-dir <early目录>`，加
`--profile-max-iterations 400`；late 使用 16384 上下文，加
`--profile-dir <late目录> --profile-delay-iterations 4200 --profile-max-iterations 400`。
4200 只适用于当前题目和配置，应以实际输出轨迹确认采样位置。

四次输出均与阶段 26 基线逐 token 一致，early 比较对应前缀；context 连续、训练
轮数上限和 CPU/GPU 请求结束版本归零检查通过。长请求达到 16K 截断，不能用作
完成解题的准确率结论。记录的训练和推理源码 hash 在实验前后相同。
