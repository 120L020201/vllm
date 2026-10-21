# 阶段 16：长上下文 CPU 训练加速候选与 GQA 微基准

2026-10-19，`eagle3&dflash`，base `3979838d2d`，未提交工作树。

## 结论与范围

优先验证两条路线：减少每步参与训练的完整 query block 数量，以及改变 CPU
GQA 的张量布局，减少长历史 K/V 的复制和梯度归并。前者改变每步训练样本量，
后者保持 attention 数学定义，但 BF16 梯度的舍入顺序改变。

本阶段只增加独立 CPU 微基准和报告，没有修改线上模型、训练策略或 GPU 链路。
GQA 布局在两次独立进程的 16000-token 历史实验中均减少 attention 前后向耗时，
但结果有明显跨进程波动，不能据此承诺整次训练或 decode 的加速比例。
尚未运行候选布局的真实 checkpoint 训练或 AIME 服务实验。

## 瓶颈证据

从阶段 15 的 early / last trace 只统计完整 CPU 更新，排除 bootstrap 和窗口
两端截断事件。early 为 version 2–7，共 6 次；last 为 version 66–68，共 3 次。

| 每次完整更新平均耗时，ms | early | last |
| --- | ---: | ---: |
| 整次训练 | 715.71 | 1477.52 |
| forward | 171.16 | 382.02 |
| backward | 306.48 | 845.45 |
| optimizer | 139.73 | 136.46 |
| 更新后追加 context KV | 34.22 | 38.83 |
| CPU Flash attention forward，包含在 forward 中 | 10.32 | 178.60 |
| CPU Flash attention backward，包含在 backward 中 | 35.54 | 436.18 |
| K/V repeat_interleave | 2.42 | 46.08 |
| ExpandBackward0 梯度归并 | 3.97 | 66.25 |

forward/backward 及其内部 attention 是包含关系，不可逐行相加。两段来自不同
请求中的不同历史长度，且 profiler 会干扰运行，因此用于归因，不是受控加速比。
新增约 762 ms 中，attention 内核增加约 569 ms；optimizer 基本不变。

## GQA 布局候选

当前 `DFlashAttention.forward` 将 8 个 KV head 用 `repeat_interleave(4)` 展开为
32 个 head，再调用 CPU Flash SDPA。即便 context detach，query K/V 仍需要
梯度，拼接后的完整 K/V 仍进入 attention backward。

同一个 KV head 对应的 4 个 query head 可以折叠到 query 维：

```text
原来：Q [32, 256, 128]，K/V [8, L, 128] → 展开到 [32, L, 128]
候选：Q [8, 1024, 128]，K/V 保持 [8, L, 128]
      mask [256, L] 沿 query 维重复 4 次
      attention 输出再还原为 [32, 256, 128]
```

这里 L 包含历史和当前训练 query 的 K/V。每一行仍对应原来的 query、KV head
和可见位置，softmax 仍沿 key 维进行；query 行之间不发生额外混合。
这是根据 [PyTorch SDPA 的逐行定义](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)
推导的等价重排，要求保持本模型的显式 mask、`is_causal=False` 和 dropout=0。
不能将该写法直接套入按重排后行号构造的 causal mask。

本机实测 PyTorch 是 `2.13.0+cu130`，以上链接是当前可访问的 2.14 API 文档。
实际后端由本机 profiler 确认：两组都调用
`aten::_scaled_dot_product_flash_attention_for_cpu` 及其 backward。
候选布局没有使用 `enable_gqa=True`，也没有退回显式 attention 概率矩阵的实现。

## 微基准设计与结果

脚本：[benchmark_dflash_cpu_gqa.py](../../../benchmarks/kernels/benchmark_dflash_cpu_gqa.py)。
CPU 0–15，16 线程，BF16，query=256、Q heads=32、KV heads=8、head dim=128。
完整 16-token 非因果 block，context 可见前缀随 block 改变。旧 context 不求导，
当前 query、K/V 求导。每种形状先预热两次，再交替执行两组各 8 次；表格取
各组总耗时的中位数。计时包含拼接、布局转换、attention 和 backward，排除
投影、MLP、LM head、loss、optimizer、capture 和参数安装。

| 历史长度 | 进程 | 展开 K/V，ms | 折叠 query，ms | 耗时减少 |
| --- | --- | ---: | ---: | ---: |
| 1024 | run1 | 9.71 | 9.93 | -2.30% |
| 1024 | run2 | 16.99 | 12.81 | 24.59% |
| 16000 | run1 | 128.62 | 106.26 | 17.38% |
| 16000 | run2 | 170.96 | 114.77 | 32.87% |

16000 历史时，run1 前向中位数为 30.29→15.48 ms，反向 98.16→90.85 ms。
profile 确认候选消除了 K/V `repeat_interleave` 及其 `ExpandBackward0`。
但所有 query 仍需读取完整历史，长历史 attention backward 仍是主要成本。

run1 是最初的临时脚本；持久化的 run2 将 FP32 检查从 4 Q / 2 KV heads 改为
8 Q / 2 KV heads，以覆盖本模型的 4:1 比例，因此之后生成的随机 BF16 数据也
不同。两个进程内部两组始终使用相同输入、mask 和上游梯度。不能将两进程
当成完全固定输入的重复；两组的绝对时间也存在明显波动。短历史收益未确定。

FP32 检查包含输出与 Q、新 K、新 V 的梯度，通过 `atol=1e-5, rtol=1e-4`。
两个 BF16 实验的输出和 dQ 与基线逐元素相同，新 K/V 梯度相对 RMS 差异约
0.33%，来自将跨 head 梯度归并移入 attention 内核后舍入顺序不同。
这些结果只覆盖本微基准输入，不能声称完整模型逐元素等价或学习效果已验证。

## 后续候选及验收

1. **减少完整训练 block：N=16→8→4。** 历史长度固定时，query 数从 256 降到
   128/64，减少五层 query forward/backward。所有轮次的确认历史仍连续消费、
   投影和追加；旧轮次不积攒成以后的训练任务。每个被选中的 block 保留完整
   16 个 query，不能因某些位置没有 KL 而删除非因果 block 内的输入。
   可以评估随长度调整 N，例如 4K/8K/16K 分别 16/8/4；这只是候选预算。
   固定 optimizer、snapshot 和 history 开销意味着总步耗时不会严格减半。
   阶段 10 的短请求 N8 虽增加安装次数，吞吐略低于 N16；当前关闭 context
   梯度的自然 16K 配置尚未进行 N8/N4 对照。
2. **将上述 GQA 布局接入候选路径。** 先在已有训练测试中验证多 block、chunk
   边界、位置 mask、梯度和两种 context_grad 设置，再用真实 checkpoint 比较
   1K/8K/16K 全训练步。只有模型训练和 decode 对照通过后才考虑默认切换。
3. **为 detached context 专门实现 backward。** dQ 仍必须读取全部历史，dK/dV
   只需当前 query 后缀。安装版 CPU Flash backward 的 schema 总是返回三者，
   没有前缀跳过或按输入选择梯度的参数。真正省去历史 dK/dV 计算需要更改
   内核；调用原 backward 后再切片没有这项计算收益。必须保留完整 softmax
   归一化，不能把 context/query 分别 softmax 后直接拼接。收益尚未实测。
4. **只训练最后 1–2 层。** 在 context 不求导、embedding 冻结的条件下，冻结
   前面完整层可省去那些层的 query backward，仍保留五层 forward。它改变
   训练参数范围，必须对比 KL、接受长度和 decode 吞吐，不能自动替代当前配置。
   仅加 LoRA 或冻结任意中间层不保证消除需要传播输入梯度的 attention backward。
5. **历史 KV 预分配和追加。** 可减少反复 `cat` 的旧历史复制，但目前追加
   context 的约 39 ms 明显小于 attention 成本，应在前几项后处理，并保证
   autograd 保存张量的生命周期和更新后新段的权重版本语义。

BF16、CPU Flash SDPA、打包 QKV/gate-up、仅有效监督行的 LM head、fused AdamW、
`set_to_none=True`、16 核隔离及 PASSIVE 已启用。阶段 11 的 compile loss/MLP
没有显著收益，FP32 master weights 和 foreach/普通 AdamW 更慢。没有证据支持
把更多 PyTorch 开关全部打开就能解决长历史成本。

下一轮真实服务验收应保持 AIME 第 15 题、自然增长至总 16K、相同初始权重和
线程/传输配置，比较 CPU step、每次更新覆盖轮数、实际安装次数、接受长度、
纯 decode token/s，以及 early/last 队列年龄；更快训练不自动等于更快 decode。

## 产物与复现

数据位于 [dflash_cpu_gqa_probe_20261019](../../results/experiments/dflash_cpu_gqa_probe_20261019/)：
`run1.json`、`run2.json` 含逐次时间、数值误差和算子统计；
`complete_step_trace_costs.json` 保存完整训练步的原始范围和聚合结果。

```bash
OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OMP_WAIT_POLICY=PASSIVE \
MALLOC_MMAP_THRESHOLD_=1073741824 MALLOC_TRIM_THRESHOLD_=-1 \
taskset -c 0-15 .venv/bin/python \
  benchmarks/kernels/benchmark_dflash_cpu_gqa.py \
  --threads 16 --repeat 8 --output /tmp/dflash-cpu-gqa.json
```

验证：两次 CPU 微基准完成，FP32 输出/梯度检查通过；脚本 Ruff check/format 通过。
没有修改运行时实现，因此本阶段没有重跑完整模型回归或 GPU 生成实验。
