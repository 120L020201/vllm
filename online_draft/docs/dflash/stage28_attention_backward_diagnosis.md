# 阶段 28：为什么 attention backward 比 forward 增长明显

2026-10-20，`eagle3&dflash`，沿用阶段 27 的训练源码。只分析已有 trace，并执行
CPU attention 隔离微基准；没有修改模型、训练策略、推理路径或默认配置。

## 结论

当前主要是**长历史 attention 与 CPU backward 实现效率的叠加**。
固定 16 个训练 block，单独增加历史长度即可复现；不能概括为 CPU 无法训练大 batch。
Backward 与 forward 的 FLOPs 有关系，但 wall time 不要求维持固定倍数。

一个已核实的具体限制：当前 GQA 布局下，CPU Flash Attention backward 的主要
计算只分成 8 个并行任务；forward 可以进一步沿 query 分块，利用 16 个训练核心。
隔离实验实际观察到 forward 有 16 个活跃线程，backward 主要有 8 个。

## 先统一 trace 比较范围

阶段 27 的 draft forward 包含 attention、MLP、投影等，backward 还包含 head 的
输入梯度；不能直接用这两个总数推导 attention 的 forward/backward 比例。
本阶段对同一组完整非 bootstrap 更新提取 CPU SDPA 算子的 inclusive 时间：

| 指标 | Early，版本 2–6 | Late，版本 63–66 |
| --- | ---: | ---: |
| 窗口最后 anchor 的均值 | 912.4 | 14651.5 |
| 五层 attention forward 合计 | 12.98 ms | 58.85 ms |
| 五层 attention backward 合计 | 42.10 ms | 379.44 ms |
| Backward / forward | 3.24 | 6.45 |

每次更新对应 5 个 forward、5 个 backward 算子。这里采用 inclusive 时间，
与阶段 27 的 backward self time 40.63 / 374.02 ms 有小幅差别，不能混用。
Early/late 来自不同更新，窗口、监督位置和最后一层 query 数并非完全一致，
因此又做了下面的固定形状对照。

## 固定训练数据量，只改变历史长度

使用安装版本 PyTorch `2.13.0+cu130`，commit
`cf30153c4c131c8164ee7798e5022d810682e2cb`。
BF16、CPU 核心 0–15、16 线程，环境沿用阶段 27。单独调用 SDPA 和 autograd.grad，
不加载 target/draft、不运行 GPU、不执行 optimizer、capture 或 snapshot。

训练 16 个 block，每个 16 个位置，即 256 个原始 query；32 个 Q head、8 个 KV
head，head_dim=128。沿用当前 GQA 重排，SDPA 输入 Q 为 `[1,8,1024,128]`，
K/V 为 `[1,8,L+256,128]`。Mask 包含不同 block 的历史边界和 block 内非因果可见性。
这是前四层的 query 规模；没有将它当成整五层训练或最后一层 selective 的测试。

每组预热 3 次、计时 7 次，两个独立进程以正序、逆序运行；下表为两个进程各自
中位数的均值。Forward 包含 SDPA 的 boolean mask 转换；backward 包含 autograd
调用和返回梯度的分配。输入、mask 构建和输出对象释放在计时外。

| 历史长度 L | Forward ms | Backward ms | 比值 |
| --- | ---: | ---: | ---: |
| 1024 | 1.64 | 7.69 | 4.69 |
| 4096 | 5.39 | 24.25 | 4.50 |
| 8192 | 8.62 | 47.09 | 5.46 |
| 16384 | 15.35 | 92.39 | 6.02 |

没有增加训练 block 数，backward 仍显著增长。4K→8K→16K 的 backward 约为
24→47→92 ms，近似线性增长，没有证据表明它突然变成了关于历史长度的平方复杂度。
Forward 的常数成本、缓存与 kernel 执行效率不同，有限尺寸下的耗时比例并非常数。
没有采集硬件 cache-miss 或内存带宽计数器，不将缓存失效率作为已证明的唯一原因。

固定历史 16K，改变训练 block 数的辅助检查：

| 训练 block 数 | Forward ms | Backward ms |
| --- | ---: | ---: |
| 8 | 10.48 | 71.44 |
| 16 | 15.35 | 92.39 |
| 32 | 26.47 | 179.12 |

减少 query 有帮助，但不足以消除长历史梯度和数据读写成本。Kernel 分块大小还会
随 query 长度变化，不能期望 block 减半就精确减半耗时。

## 为什么 CPU 的前后向效率不对称

核对了与安装版本 commit 相同的
[PyTorch CPU FlashAttention 源码](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/cpu/FlashAttentionKernel.cpp)。
Forward 按 batch、head、query 分块并行；backward 的外层并行仅按 batch 和 KV
head 划分，query 和 key 分块在每个任务内部循环。Forward 另有 packed BRGEMM
分支，backward 采用 GEMM 路径，两者不能假设具有相同有效吞吐。

把当前输入形状代入：

```text
实际 batch = 1
重排后的 head = 8
重排后的 query length = 1024

forward：query block = 256，1 × 8 × 4 = 32 个外层任务
backward：1 × 8 = 8 个外层任务
```

反向需要把多个 query 对同一 K/V 的贡献累加。进一步拆分 query 并行必须处理
这些梯度归并，不能直接把 forward 的并行循环复制到 backward。

另用直接 ATen 前后向算子测试 16K 形状，连续各运行 30 次，记录进程 CPU 时间、
wall time 和 `/proc/self/task/*/stat` 的每线程 CPU tick：

| 指标 | Forward | Backward |
| --- | ---: | ---: |
| CPU time / wall time，约等于平均占用核心数 | 14.84 | 7.26 |
| 累计 CPU time 超过 50 ms 的线程数 | 16 | 8 |

另外 8 个 OpenMP 线程在 backward 中各消耗约 20–30 ms，主要计算集中于前面
8 个线程。这里说明的是 kernel 的并行任务分配，不是训练进程整体只有 8 个线程。
MLP、QKV/head GEMM、optimizer 等仍有各自的并行策略。

将配置从 8 线程提高到 16 线程，16K、16 block 的隔离结果为：

| 配置 | Forward ms | Backward ms |
| --- | ---: | ---: |
| 8 线程 | 26.45 | 87.23 |
| 16 线程 | 15.35 | 92.39 |

Forward 明显受益，backward 没有得到对应加速。两组均绑定 0–15，改变的是 PyTorch
线程数；不能据此直接将完整训练改为 8 线程，因为其他阶段可能变慢。

## FLOPs 倍数与耗时倍数的区别

令 Q 长度为 M、K/V 长度为 S，每个 head 的维度为 d。忽略 scale/mask 的表示：

```text
forward：SCORE = Q Kᵀ，P = softmax(SCORE)，O = P V
backward：dV = Pᵀ dO，dP = dO Vᵀ
          dSCORE = P ⊙ (dP - rowsum(P ⊙ dP))
          dQ = dSCORE K，dK = dSCOREᵀ Q
```

Forward 有两个主要矩阵乘法。Backward 有四个，还需在 Flash Attention 中重算
attention 概率，对应额外 QKᵀ；因此仅大矩阵乘法 FLOPs 就约为 forward 的 2.5 倍。
这是工作量，不是 wall time 保证。每秒完成的运算量还受并行度、实现路径、
内存读写、归并和数值转换影响。

当前 forward 的主要输出大小随 M 增长，backward 除 dQ 外还要生成长度为 S 的
dK/dV，并跨 query 块累加。即使 context 在 autograd 图中 detach，当前拼接后的
SDPA 仍计算整段 dK/dV，再由拼接节点丢弃历史部分。

固定 M 时，前后向的主要 attention 运算量都随 S 近似线性增长。但前后向的系数
和固定成本不同；随着历史变长，backward 中效率较低的长序列部分成为主导，总时间
的倍数自然会变化。这不违反链式求导或复杂度分析。

## 对优化方向的含义

- 增加 CPU 线程数不足以解决当前 8 个 backward 外层任务的限制。
- 要保留完整历史，可以研究按 query 或 key 分块并行的 backward，以及固定 context
  下只计算必要 dQ 和 query 后缀 dK/dV 的实现；必须保留完整 attention softmax。
- 单纯截掉历史或进一步减少训练 round 会改变训练条件，需要另做质量与系统对照。
- 本次也记录了把 KV 重复到 16/32 head、FP32 等辅助形状结果，位于原始 JSON。
  它们仅用于诊断 kernel 效率，没有实施参数更新一致性或完整系统性能验证，不据此
  修改 GQA、dtype 或默认配置。

## 产物和验证

- [两次微基准汇总](/tmp/dflash-attention-diagnosis-20261020/aggregate.json)
- [第一次原始计时](/tmp/dflash-attention-diagnosis-20261020/run1.json)
- [反序第二次原始计时](/tmp/dflash-attention-diagnosis-20261020/run2.json)
- [实际线程 CPU 时间](/tmp/dflash-attention-diagnosis-20261020/thread_utilization.json)
- [同一组 trace 更新的前后向比较](/tmp/dflash-attention-diagnosis-20261020/trace_comparison.json)
- [微基准脚本](/tmp/dflash-attention-diagnosis-20261020/bench_sdpa.py)
- [线程利用率脚本](/tmp/dflash-attention-diagnosis-20261020/bench_threads.py)
- [脚本 hash、环境与源码版本](/tmp/dflash-attention-diagnosis-20261020/manifest.json)

没有更改正式 Python 代码，没有新增测试或启动整模型 GPU 实验。当前结论是对
现有 trace、固定形状实验和对应版本源码的联合解释，不是新优化后的性能承诺。
