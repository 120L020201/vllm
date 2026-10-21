# 阶段 15：关闭 context grad 的 early / last CPU GPU trace

2026-10-19，`eagle3&dflash`，base `3979838d2d`，未提交工作树。

## 交付文件

本轮使用真实 AIME 请求，同时记录 CPU 在线训练、GPU 推理、采集 D2H、权重
H2D 和安装事件。不是阶段 14 的 CPU 合成输入微基准。

| 窗口 | trace 文件 | gzip 大小 MiB | 展开大小 MiB | 估计覆盖总上下文 |
| --- | --- | ---: | ---: | --- |
| early | [early.pt.trace.json.gz](../../results/experiments/dflash_nograd_16k_early_last_20261019/early/early.pt.trace.json.gz) | 16.01 | 421.34 | 286–1580 |
| last | [last.pt.trace.json.gz](../../results/experiments/dflash_nograd_16k_early_last_20261019/last/last.pt.trace.json.gz) | 15.48 | 413.12 | 15070–16219 |

两份均包含 **400 个 `execute_context_*` worker 范围**，分别约 95.5 万和
92.1 万个 trace event。early 包含首次 prefill；last 全部是 decode。
GPU 推理 kernel 首尾跨度分别约 6.03 秒和 6.62 秒。
gzip 减少磁盘与传输体积，查看器仍需展开数据；展开后也远小于此前 EAGLE3
early trace 的约 4 GiB。

结果目录：

```text
online_draft/results/experiments/dflash_nograd_16k_early_last_20261019/
├── early/early.pt.trace.json.gz
├── early/profiler_out_0.txt
├── last/last.pt.trace.json.gz
├── last/profiler_out_0.txt
├── early.json / last.json
├── early.log / last.log
├── early-summary.json / last-summary.json
├── commands.json
└── README.md
```

`profiler_out_0.txt` 是 PyTorch 的算子汇总。`*-summary.json` 是从 trace 提取的
CPU 阶段、GPU kernel、copy 重叠和线程编号。`early.json/last.json` 保存完整请求
输出、时间线和在线更新记录。可追溯汇总、文件 SHA256、源码 SHA256 和命令也
保存在 [nograd_early_last_trace_results.json](nograd_early_last_trace_results.json)。

## 截取方式与配置

参考原工作树 `online_draft/scripts/profile_aime_2026_online.py` 的 EAGLE3 做法：
`torch_profiler_all_threads=True`、关闭 stack、忽略 frontend，使用
`delay_iterations` 与 `max_iterations` 控制 worker 采集区间。
EAGLE3 旧样本用 3000 步，本轮缩短为 400 步，并开启默认 gzip。
shape、memory、FLOPs 记录均保持关闭。无需修改 profiler 或推理/训练代码。

- `early`：delay=0，max=400，对应 worker step 1–400。
- `last`：delay=4815，max=400，对应 worker step 4815–5214。
- last 的起点依据阶段 14 `nograd-1` 无 profiler 时间线中达到约 15000 context
  的位置选择；上表的范围再由本次实际请求时间线估算，避免沿用旧请求的长度。
- last 指接近请求末尾的 decode 区间，不包括最后约 165 个 token 和 finish/reset。
- 两份来自两个串行、独立进程的完整请求，每次从 W0 开始。early 导出造成的
  暂停不会影响 last 请求之前的训练状态；它们不是同一次请求连续截出的两段。

模型仍为 `/data/models/Qwen3-8B` 和 `/data/models/Qwen3-8B-DFlash-b16`，本地
AIME 2026 第 15 题，thinking、greedy、seed=17。prompt 286，输出 16098，总上下文
16384，`max_model_len=18432`。单 GPU、K=15、target/draft FULL CUDA Graph、
FLASHINFER、关闭 prefix caching。

显式使用 `context_grad=False`、最近 16 个完整 query block、位置衰减 KL gamma=7、
lr=1e-4、fused AdamW、BF16、无 FP32 master weights。推理 CPU 18 单线程，采集
CPU 19 单线程，训练 CPU 0–15 共 16 线程。PASSIVE 等待和阶段 14 allocator 环境
保持不变。记录中的 `context_grad` 均核实为 False。

这两次是诊断运行。profiler 的采集、导出和停止会改变耗时与更新节奏，不能用
请求里的吞吐评价算法加速，也不能把 early/last 的差异全归因于上下文长度。

## 在查看器中定位

用 Perfetto 打开对应 `.pt.trace.json.gz`。先按线程/stream 定位，再搜索以下名称：

| 角色 | early CPU TID / GPU stream | last CPU TID / GPU stream | 搜索内容 |
| --- | --- | --- | --- |
| 推理主线程 | TID 399069 | TID 399322 | `execute_context_`、`cudaGraphLaunch` |
| CPU 训练 | TID 399145 | TID 399400 | `online_draft.cpu_train` |
| CPU 采集 | TID 399146 | TID 399401 | `online_draft.d2h`、`online_draft.prepare` |
| GPU 推理 | stream 7 | stream 7 | target/draft kernel |
| 权重上传 | stream 25 | stream 25 | `Memcpy HtoD` |
| 采集下载 | stream 45 | stream 45 | `Memcpy DtoH` |

训练线程可以展开 `online_draft.forward`、`loss`、`backward`、`optimizer`、
`rebuild_cache`、`cpu_snapshot`；其中 `rebuild_cache` 仍只是新增确认段的 KV
投影和追加，不是重算全部历史。推理主线程上的 `h2d_stage[version=...]`
标记上传提交，`h2d_commit[version=...]` 标记实际切换权重。

early 安装 version 1–7，记录到 version 1–8 的 CPU 训练范围。最后一个训练
范围跨越截取终点：有 forward，尚无对应完整 loss/backward。
last 从 version 65 的 optimizer/收尾开始，完整观察到 version 66–68 的训练，
末端还包含 version 69 的部分训练；GPU 安装为 version 65–68。
不要把边缘处不完整范围当成完整训练步，也不要将各阶段不同样本数的中位数
直接相加。全线程记录可以看到 CPU 算子范围，不是逐物理核的调度利用率采样。

## 本次 trace 的检查结果

| 指标 | early | last |
| --- | ---: | ---: |
| worker 范围数量 | 400 | 400 |
| CUDA Graph launch 次数 | 799 | 800 |
| 权重上传 / 安装次数 | 7 / 7 | 4 / 4 |
| 权重 H2D 字节数 GB | 14.681 | 8.389 |
| 权重 H2D GPU copy 时间 ms | 262.67 | 150.50 |
| H2D 与推理 kernel 的时间重叠 | 97.27% | 97.43% |
| Capture D2H 字节数 GB | 2.213 | 2.207 |
| Capture D2H GPU copy 时间 ms | 59.29 | 54.74 |
| D2H 与推理 kernel 的时间重叠 | 88.50% | 87.02% |
| `h2d_stage` CPU 范围中的同步调用 | 0 | 0 |
| 全 trace `cudaHostAlloc` 次数 | 2 | 0 |

copy 通过 CPU 标注内 CUDA correlation ID 关联，重叠率按时间区间并集及交集
计算。它能确认传输与推理重叠，不代表零带宽竞争或零 CPU 提交成本。
每份有 400 次 D2H 范围内同步，均在后台采集线程；其 CPU 范围时间包含等待，
不能当作 GPU copy 的耗时。early 两次 `cudaHostAlloc` 分别约 26.3 / 28.3 微秒，
位于 prefill 的 `aten::empty` 和普通 `aten::to/_to_copy` 路径，不在在线
H2D/D2H 标注范围内。不能把 early 说成全 trace 零 pinned allocation。

CPU 阶段的 trace 中位数，仅用于阅读本次片段：

| 阶段 | early ms | last ms |
| --- | ---: | ---: |
| forward | 162.4 | 383.5 |
| backward | 304.0 | 856.6 |
| optimizer | 139.8 | 136.9 |
| 新增 KV 投影/追加 | 34.0 | 42.5 |
| snapshot | 58.0 | 56.7 |

关闭 context 投影梯度后，长历史 attention 的反向仍在，可以在 last 的训练
线程中继续展开 `aten::_scaled_dot_product_flash_attention_for_cpu_backward`。
early 包含 bootstrap 和初始化，last 的训练输入、历史和窗口大小也不同；
正式梯度消融的性能结论仍使用阶段 14 的无 profiler 对照。

## 复现与验证

在当前 worktree 的 `.venv` 中，沿用阶段 14 的环境：

```bash
export PATH="$PWD/.venv/bin:/home/lixiang/work/myvllm-rebuild/.venv/bin:$PATH"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE
export MALLOC_MMAP_THRESHOLD_=1073741824 MALLOC_TRIM_THRESHOLD_=-1

.venv/bin/python online_draft/scripts/smoke_async_dflash.py \
  --online --no-context-grad --target-graphs \
  --train-threads 16 --cpu-isolation --learning-rate 1e-4 \
  --max-train-rounds 16 --loss-decay-gamma 7 \
  --aime-questions 1 --aime-offset 14 --context-tokens 16384 \
  --max-model-len 18432 --decode-timeline \
  --profile-dir /tmp/dflash-nograd-early \
  --profile-delay-iterations 0 --profile-max-iterations 400 \
  --output /tmp/dflash-nograd-early.json
```

last 将 delay 改为 4815，并使用独立的输出目录和 JSON。该 delay 仅适用于本题、
本机、该配置的当前轨迹；其他请求应根据自己的时间线选择。

两次实际模型运行都达到 16384 context，16098 个输出 token 与阶段 14 无
profiler `nograd-1` 逐 token 相同。两份 gzip 解压、JSON 解析及 CPU/GPU/copy
事件存在性检查通过，worker 范围数均为 400。输出依然在 16K 处截断，没有
完整数学准确率结论。本阶段仅运行现有脚本、增加结果与文档，没有修改训练
或推理代码，因此未重复运行无关单元测试。
