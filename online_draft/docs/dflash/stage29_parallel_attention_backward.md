# 阶段 29：拆分 CPU attention backward，跳过历史 KV 梯度

2026-10-20，工作目录 `myvllm-eagle3-dflash`，分支 `eagle3&dflash`。
本阶段实现、验证并测量 CPU attention backward 优化。

最终方案保留原生 forward，将 backward 按 head × query tile 并行，并省掉历史
context 的 K/V 梯度。16K 单层 attention backward 为 92.85→24.60 ms；完整请求的
12–16K 区间，全部 backward 为 597.10→329.35 ms，完整训练为 1016.21→748.95 ms。
CPU 更新次数增加约 16%，GPU decode 基本持平。129 项测试与两道 AIME 回归通过。

## 实现选择

保留 PyTorch CPU Flash Attention forward，新增一个 CPU backward 算子。
算子复用 ATen 的线程池、向量运算和 BRGEMM，不自行实现底层矩阵乘法，也不修改
GPU attention。Python autograd 包装负责保存 forward 的 output/logsumexp，
区分历史 context 和 query K/V，并把必要梯度交回现有模型。

仅在 Python 中改变 attention 的 batch/head 形状，不能让原生 backward 跳过
历史 K/V 梯度。它还会增加 KV 复制或梯度归并。阶段 28 已测到这种重排的收益有限，
因此本阶段直接改变 backward 内部的任务划分和计算范围。

当前安装的 PyTorch CPU Flash backward 按 batch × KV head 分配主要任务，
不能通过调大线程数自动拆开一个 head 内的 query。依据是与安装版本相同 commit 的
[CPU FlashAttention 源码](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/cpu/FlashAttentionKernel.cpp)。

主要文件：

| 文件 | 职责 |
| --- | --- |
| `src/online_draft/models/csrc/attention_backward.cpp` | query 分块、KV packing、梯度计算与归并 |
| `src/online_draft/models/cpu_attention.py` | CPU extension 初始化、原生 forward、autograd 接线 |
| `src/online_draft/models/qwen3_dflash.py` | 将 context 与 query K/V 分开传入包装函数 |
| `tests/test_dflash_training.py` | forward、必要梯度、分块边界与冻结行为验证 |
| `pyproject.toml` | 打包 C++ 源码并声明 Ninja 依赖 |

没有增加 CLI 开关。训练模型初始化时加载/编译 CPU extension，在请求开始前完成；
decode 期间不会构建该 extension。EAGLE3、capture、H2D/D2H stream、权重安装、optimizer、
KV 更新策略和最后 16 round 的 loss 选择保持原有逻辑。

## backward 的任务划分

当前前四层的典型形状为：

```text
训练 16 个 block，每个 block 16 个位置
32 个 query head，8 个 KV head，head_dim = 128
GQA 重排后 Q = [8, 1024, 128]
K/V = [8, L + 256, 128]

原生 backward：8 个主要并行任务
新 backward：query tile = 128，8 × ceil(1024 / 128) = 64 个任务
key tile = 512
```

64 个任务由已有 16 个训练线程执行，不创建 64 个线程。最后一层沿用 selective
query，实际任务数取决于监督位置数量。

每个任务独占自己的 dQ 行，因此 dQ 无需原子加法。不同 query tile 对同一 query
K/V 的梯度写入各自的 FP32 partial buffer，所有任务结束后再归并并转换为参数
计算使用的 dtype。旧历史不需要梯度，因此不分配对应的 partial buffer。

BF16 路径把 K/V 打包成 BRGEMM 使用的布局。K/V 转置按 512 行分块完成，避免
对完整长序列做低效转置；softmax 重建、类型转换使用 ATen 向量操作。临时 attention
矩阵仅为 query tile × key tile，不物化整张长上下文概率矩阵。

Boolean mask 沿用 query dtype 存成 0/−inf；BF16 backward 只把当前 512 列的小段
转换为 FP32。最初包装将整张 mask 存成 FP32，16K 单层 forward 测到 18.40 ms，
改回 BF16 后为 16.01 ms，原生为 16.07 ms；backward 为 23.87→24.78 ms。
最终选择后者，减少长期保存的 mask 内存，也消除了这项 forward 额外开销。
显式输入的 FP32 additive mask 保留其精度。

## 为什么可以省掉历史 dK/dV

把完整 K/V 写成历史与当前 query 两部分：

```text
K = [Kc; Ku]，V = [Vc; Vu]
P = softmax(s Q Kᵀ + M)，O = P V，s = 1 / sqrt(d)
G = dLoss / dO
δi = sum_j(Gij Oij)
S = P ⊙ (G Vᵀ - δ)

dQ  = s S K
dKu = s S[:, context_end:]ᵀ Q
dVu =   P[:, context_end:]ᵀ G
```

`Kc/Vc` 已 detach，故模型参数的链式求导中不存在经过历史 context 的路径。
可以省掉 `dKc/dVc`，但仍必须用完整历史计算 P、G Vᵀ 和 dQ。新实现没有截断历史、
没有省略 softmax 分母，也没有删除非监督 query 的 K/V。

共享的 QKV 投影和 k_norm 仍由 query 路径更新；FC、hidden norm、embedding、
lm_head 继续冻结。optimizer 完成后，仍使用新权重重新投影本窗口全部确认位置的
FC 输出，然后追加到保持原值的历史 KV。

忽略 softmax 的逐元素计算，原生 Flash backward 需要重建 QKᵀ，加上四个梯度
矩阵乘法；新实现只有三个矩阵乘法覆盖完整历史，另外两个只覆盖 query 后缀。
若历史长度为 L、query K/V 长度为 U，主要 GEMM 工作量之比约为：

```text
新 / 旧 = [3(L + U) + 2U] / [5(L + U)]
L = 16384，U = 256 时，约为 0.606
```

这是约 39% 的 GEMM 工作量减少。额外加速来自并行度、BRGEMM 和向量化，不能把
全部 wall-time 收益仅归因于“多用了 8 个核心”。

实现使用 FP32 累加 dQ 和各 query tile 的梯度，归并后再转 BF16。数学梯度一致，
但 BF16 的累加顺序、舍入位置与原生 backward 不同，不承诺逐 bit 梯度一致。

## 实验环境与统计方法

- Target：`/data/models/Qwen3-8B`；draft：`/data/models/Qwen3-8B-DFlash-b16`。
- CPU：Xeon w7-3445，20 个物理核；训练固定核心 0–15、16 线程，推理 18、采集 19。
- GPU：RTX PRO 6000 Blackwell，96 GB。
- PyTorch：`2.13.0+cu130`，commit `cf30153c4c131c8164ee7798e5022d810682e2cb`。
- BF16、GQA、selective last layer、fused AdamW，learning rate `2e-4`，
  最后 16 round、位置衰减 KL gamma=7；target/draft 均启用已有 CUDA Graph。
- `OMP_WAIT_POLICY=PASSIVE`，沿用 malloc mmap/trim 配置；不同性能实验串行运行。

完整系统对照使用同一道 AIME 问题（offset=14），prompt 286 token，输出自然增长
至总上下文 16384。先按 baseline → 初版 → 初版 → baseline 完成四次对照，确认
backward 收益；修正 mask 表示后，再跑最终版两次。下面最终系统结果使用两次
baseline 与两次最终版，初版单独保留为消融。计时不含模型加载、编译和 prefill。
CPU 使用轻量 host span，普通吞吐实验不启用 PyTorch profiler；trace 另跑。

旧 CPU 包在修改前完整保存到 `/tmp/dflash-backward-20261020/baseline/`，通过
`PYTHONPATH` 选择；不在运行中切换源文件。每次运行保存命令、环境、源码 SHA256。

## 结果

### 固定形状 attention

每个配置预热 3 次、计时 9 次。两个独立进程分别正序、逆序运行，表中取两次
中位数的均值。包含 Python/autograd 接线、KV 拼接、mask 转换及梯度分配，不包含
QKV/o_proj、其他网络层或 optimizer。固定 16 个训练 block，改变历史长度。

| 历史长度 | 原生 forward ms | 新 forward ms | 原生 backward ms | 新 backward ms | backward 加速 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1K | 2.30 | 1.92 | 7.80 | 3.49 | 2.24× |
| 4K | 6.32 | 4.65 | 24.47 | 7.46 | 3.28× |
| 8K | 9.09 | 8.28 | 46.73 | 13.46 | 3.47× |
| 16K | 16.04 | 16.15 | 92.85 | 24.60 | 3.77× |

短序列 forward 差异包含包装和 autograd graph 的不同，不能归因于 forward kernel
被重写；实际仍调用同一个 ATen CPU Flash forward。

16K backward 连续 30 次，通过进程 CPU 时间和 `/proc/self/task/*/stat` 验证：

| 指标 | 原生 | 最终版 |
| --- | ---: | ---: |
| CPU time / wall time，两次均值 | 7.23 | 13.88 |
| 每次实验累计 CPU time >50 ms 的线程数 | 8 | 16 |

另对同一 packed/suffix 原型只改变 query tile，采用直接算子计时：tile=1024
产生 8 个任务，为 49.61 ms；tile=128 产生 64 个任务，为 21.91 ms。
该消融同时改变局部工作集和 GEMM 形状，不把全部差值解释为核心数变化。
它使用初版 FP32 mask、没有完整 Python 包装，不能与上表的 24.60 ms 混作同一口径。

### 完整系统与 trace

以下为两次运行各自均值再取平均，排除 bootstrap。不同运行的安装时刻、窗口和
监督位置数量会变化，因此完整系统收益与固定形状算子收益分别报告。

| 总上下文区间 | 原生全部 backward ms | 新全部 backward ms | 原生完整训练 ms | 新完整训练 ms | 原生窗口 round | 新窗口 round |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0–4K | 338.84 | 299.69 | 717.77 | 679.88 | 61.06 | 58.92 |
| 4–8K | 425.16 | 307.97 | 813.33 | 700.23 | 65.42 | 57.52 |
| 8–12K | 500.34 | 318.98 | 900.75 | 728.68 | 70.12 | 58.65 |
| 12–16K | 597.10 | 329.35 | 1016.21 | 748.95 | 75.60 | 58.33 |

完整训练包含输入准备、context 投影/拼接、网络 forward、KL、全部 backward、
optimizer、zero_grad 和 optimizer 后增量 context 重算；不包含随后独立发布的
snapshot。全部 backward 包含 MLP、投影、norm 和冻结 lm_head 的输入梯度，
不是单个 attention backward。

12–16K 区间，全部 backward 减少 **44.8%**，完整训练减少 **26.3%**。
Draft forward 为 151.55→151.23 ms；最终实现没有保留初版的长上下文 forward
额外开销。窗口从平均 75.60 round 缩短至 58.33 round，但 loss 仍固定最多 16 round。

| 整个请求 | 原生，两次 | 最终版，两次 |
| --- | --- | --- |
| Decode token/s | 227.91 / 229.10 | 229.74 / 230.07 |
| CPU 完成更新 | 73 / 71 | 83 / 84 |
| GPU 安装更新 | 72 / 70 | 82 / 83 |
| GPU round 平均时间，两次均值 | 14.505 ms | 14.516 ms |

CPU 更新次数均值增加约 **16.0%**。Decode 均值 228.50→229.90 token/s，约 +0.6%，
GPU round 时间相差约 0.01 ms。这里的结论是没有观察到 GPU 吞吐下降，不能把
约 0.6% 的变化当成有统计把握的推理加速，更不能声称 GPU 完全没有受到任何影响。

本次 16K 范围内，新方案的窗口均值约 57–59 round，明显缓解了此前的增长。
这是实际系统平衡点的变化，并没有把 epoch 的完整窗口硬截成 16 round。

初版 FP32 mask 的对应两次结果保留在 `optimized1/2.json`：decode 均值
229.76 token/s、CPU 更新 85/84 次、12–16K 完整训练 758.69 ms。最终版保留 BF16
mask，主要依据同形状 forward 的改进和更小的 mask 存储；不以小样本更新次数的
差异声称该细节必然提高整个请求的训练频率。

另外看最终版 12–16K 区间的训练时间分解：

| 步骤 | ms / update |
| --- | ---: |
| 输入和 mask 准备 | 5.19 |
| 更新前 context 投影 / 拼接 | 15.00 / 8.77 |
| Draft forward / lm_head / KL | 151.23 / 15.72 / 6.85 |
| 全部 backward | 329.35 |
| Optimizer / zero_grad | 137.05 / 55.00 |
| 更新后 context 投影 / 拼接 | 14.42 / 8.93 |
| 其他 | 1.43 |
| 完整训练 | 748.95 |
| 随后发布 snapshot | 51.87 |

Snapshot 加上训练约 800.82 ms；GPU 安装边界还包含 capture、handoff 和实际安装
时机，因此不能直接用 748.95 / 14.516 推导完整窗口 round 数。

### Early/last trace

最终 trace 仅截取 400 个 GPU iteration，分别从开头和第 4200 iteration 开始。
PyTorch profiler 停止/导出会明显扰动请求耗时，这两次运行不参与上述吞吐统计。

- [Early trace，约 16 MB](/tmp/dflash-backward-20261020/final_early/rank0.1792489491494605971.pt.trace.json.gz)
- [Last trace，约 16 MB](/tmp/dflash-backward-20261020/final_late/rank0.1792489627143719602.pt.trace.json.gz)

在 Perfetto/PyTorch trace viewer 中搜索 `online_draft.backward`、
`online_draft_cpu::attention_backward` 或 `_ContextAttentionBackward`。
OpenMP 内部的每个 query tile 不是独立的 PyTorch operator event；16 个线程的实际
利用率由前述 `/proc` 实验验证，不从单条 trainer lane 推断。

只选取完整非 bootstrap 更新，且剔除跨越采集边界的更新：early 为版本 2–6，
context end=482–1292；last 为版本 73–78，context end=14373–15454。

| Trace 指标 | Early | Last |
| --- | ---: | ---: |
| 五层新 attention backward 合计，inclusive | 22.68 ms | 98.89 ms |
| 全部 backward | 307.62 ms | 335.09 ms |
| Backward 内 `aten::mm` 合计，self | 231.57 ms | 183.43 ms |
| Target graph span | 10.376 ms | 11.756 ms |
| Draft graph span | 2.238 ms | 2.428 ms |
| 相邻 target 开始时间，即 GPU round | 13.750 ms | 15.298 ms |
| H2D copy 与推理 kernel 时间重叠 | 99.07% | 99.20% |
| D2H copy 与推理 kernel 时间重叠 | 93.34% | 93.86% |

这里的 overlap 分母是 copy 的 GPU 时间，分子是它与推理 stream kernel 时间并集
的交集；不是 CPU 提交范围，也不是 GPU 利用率。并未达到 100%。H2D 提交范围没有
Synchronize 调用；D2H 仍有 capture 线程原有的 `cudaEventSynchronize` 等待 host
buffer 就绪，不能把它误读为推理线程新增了同步。

阶段 27 使用相同基线源码的 trace，五层原生 attention backward 为
42.10 / 379.44 ms，GPU round 为 13.749 / 15.264 ms，H2D overlap 为
99.08% / 99.18%，D2H 为 92.62% / 93.21%。已逐文件核对其 CPU 源码 hash
与本次 baseline 相同。这只是辅助算子/时间线对照；窗口和监督位置不完全相同，
主要性能结论仍采用本阶段未开 profiler 的重复实验。

Last trace 中 attention 已缩短至约 99 ms，剩余 backward 的主要实测热点是
`aten::mm`（约 183 ms self）。Optimizer、zero_grad、snapshot 也已成为完整更新
周期中不可忽略的固定成本；继续仅优化 attention 不会按同样倍数缩短全部训练。

## 验证范围与限制

已新增跨 query/key tile 边界的 FP32、BF16 梯度测试，包含非连续 Q、空历史、
非对齐历史长度、selective query 数量、完全被 mask 的行，以及 boolean/FP32
additive mask。使用相同 CPU Flash
forward 布局时，输出逐 bit 相同；FP32 梯度与参考一致，BF16 梯度与 FP32 参考
按相对 L2 误差检查。

原有测试继续覆盖非因果 mask、GQA、packed/sequential 梯度、冻结参数、bootstrap、
历史 KV 不变和 optimizer 后新增 context 使用新权重。

另用实际 GQA 形状、BF16 输入和 FP32 SDPA 梯度作为参考，固定 seed=17：

| 历史长度 / 梯度 | 原生 BF16 相对 L2 误差 | 新 BF16 相对 L2 误差 |
| --- | ---: | ---: |
| 1K / dQ | 0.312% | 0.237% |
| 16K / dQ | 0.717% | 0.237% |
| 16K / query dK | 0.312% | 0.235% |
| 16K / query dV | 0.309% | 0.233% |

这两种长度的 BF16 forward 输出逐 bit 相同；16K 时新旧 dQ 的相对 L2 差异为
0.716%，query dK/dV 为 0.297% / 0.294%。在这个 fixture 中，FP32 累加后的
新梯度更接近 FP32 参考；不把这个局部结果扩大为模型质量更好的结论。

最终执行 `.venv/bin/python -m pytest online_draft/tests -q`：**129 passed**。
14 条 warning 均来自已有 EAGLE3 测试使用的 `torch.jit.script_method` 弃用提示。
本阶段 Python/C++/打包配置的 pre-commit 检查通过。
另外执行 `uv build online_draft --wheel --python .venv/bin/python --no-build-isolation`，
wheel 构建成功，并核对其包含 `models/csrc/attention_backward.cpp`。

完整模型还执行了启动阶段 CPU/GPU 数学对照与非零权重安装检查：W0 的 logits
relative RMSE=0.01147、forward KL=0.000771；非零安装后分别为 0.01306、0.000870，
均通过已有阈值。fused KV 权重副本与 CPU 新权重逐 bit 对齐，检查结束后恢复 W0。
此诊断只在请求前运行，其同步不计入 decode。

AIME 两道完整生成回归：新旧均回答 **277、62**，均正确，输出分别为 3521、11108
token，token 序列逐个一致；连续请求结束后 CPU/GPU version 均恢复为 0。
16K 性能题的输出序列同样一致，但该题被长度上限截断，不能计为答对。
这些是实现回归检查，不是足以证明在线训练质量提升的统计评测。

算子针对当前 CPU 在线训练路径实现：CPU、FP32/BF16、非因果 attention、无 dropout、
二维 mask，不支持高阶梯度。BF16 packed BRGEMM 和编译参数针对当前 AVX512/AMX
工作站验证，依赖当前 PyTorch 的内部 ATen 接口；没有做跨 CPU/跨 PyTorch 版本验收。
需要 C++ 编译器和 Ninja，首次构建在请求前完成，缓存命中后直接加载。

新实现仍需扫描完整历史，训练复杂度仍含 O(query_count × context_length)。它降低
CPU 周期随上下文增长的斜率，不构成“窗口从此不增长”的算法保证。更多更新还会产生
更多 H2D 权重上传；是否影响 GPU 必须看系统测量，而不能只凭 CPU 微基准判断。

## 原始产物

本阶段产物目录：`/tmp/dflash-backward-20261020/`。

- `run_one.py` / `run_request.py`：完整请求、host span 和 early/last trace。
- `baseline*-command.json` / `optimized*-command.json`：可复现命令、环境与源码 hash。
- `bench_final.py`：完整 attention forward/backward 的同形状对照和线程 CPU 时间。
- `bench_new.py`：原型 query tile 选择；探索代码只保存在实验目录。
- `bench_mask.py`：mask 表示的同形状消融。
- `bench_ablation.py`：8/64 个任务的直接算子对照。
- [固定形状计时汇总](/tmp/dflash-backward-20261020/micro_summary.json)
- [完整系统分阶段汇总](/tmp/dflash-backward-20261020/system_summary.json)
- [数值误差](/tmp/dflash-backward-20261020/numerics.json)
- [Early 算子汇总](/tmp/dflash-backward-20261020/final_early-trace-summary.json)
- [Last 算子汇总](/tmp/dflash-backward-20261020/final_late-trace-summary.json)
- [Early 传输汇总](/tmp/dflash-backward-20261020/final_early-overlap.json)
- [Last 传输汇总](/tmp/dflash-backward-20261020/final_late-overlap.json)
- `all-tests-final.log` / `precommit-final.log`：最终测试与检查输出。
