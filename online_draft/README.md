# 在线 draft 训练代码导航

这里实现 EAGLE3、DFlash 和 MTP 的请求内异步 CPU 在线训练（target 支持 Qwen3 与
Qwen3.5，MTP 仅 Qwen3.5）。运行时使用单 GPU、
单活动请求：GPU 持续推理，后台采集线程搬运数据，后台训练线程更新 CPU draft，
推理线程在安全位置安装已完成上传的新权重。每个请求结束后恢复初始权重 W0。

本文介绍代码结构、启动配置与脚本入口（见[运行](#运行)）。EAGLE3 与 DFlash 的
差异见 [EAGLE3 / DFlash 对比](docs/dflash/change_summary_eagle3_comparison.md)。

## 目录

```text
online_draft/
├── src/online_draft/
│   ├── models/                       # CPU 可微模型及 checkpoint 转换
│   │   ├── layers.py                 # 共用算子：RMSNorm（含 Gemma）、partial RoPE、MLP、attention
│   │   ├── qwen3_eagle3.py           # EAGLE3（Qwen3 与 Qwen3.5 draft）
│   │   ├── qwen3_dflash.py           # DFlash，逐层因果与滑动窗口
│   │   ├── qwen3_5_mtp.py            # Qwen3.5 MTP（门控注意力）
│   │   ├── cpu_attention.py           # CPU attention / autograd 接入
│   │   └── csrc/attention_backward.cpp
│   ├── runtime/                      # 采集、搬运、调度与权重交接
│   │   ├── contracts.py              # CapturePacket、PreparedRound、TrainRounds
│   │   ├── capture/
│   │   │   ├── base.py               # 共用 packet、adapter 状态机、历史构造器
│   │   │   ├── eagle3.py             # EAGLE3 staging 布局、采集字段、batch 组装
│   │   │   └── dflash.py             # DFlash 同上
│   │   ├── transport.py              # pinned staging、capture FIFO、D2H、event、finish
│   │   ├── async_pipeline.py         # 训练 FIFO、窗口、snapshot、reset
│   │   ├── weights.py                # CPU snapshot、H2D installer
│   │   └── cpu_worker.py             # 各 CPU 角色的线程数及 affinity
│   └── training/
│       ├── trainer.py                # 公共 optimizer、step、W0、version
│       ├── eagle3/                   # 自回归 draft 的训练路径，EAGLE3 与 MTP 共用
│       │   ├── draft_model.py        # 训练所需的模型接口 AutoregressiveDraftModel
│       │   ├── batch.py              # 单轮 batch 定义
│       │   ├── inputs.py             # 位置对齐、query 输入与 attention mask
│       │   ├── distill.py            # teacher 分布、监督权重与 KL
│       │   ├── cache.py              # detached context、确认历史 KV 追加
│       │   └── window.py             # 多轮窗口及训练入口
│       └── dflash/
│           ├── batch.py              # 单轮 query block 定义
│           ├── distill.py            # 位置衰减权重与 KL
│           └── window.py             # context KV、完整 block 前向、窗口训练入口
├── scripts/                          # 实机 smoke、数学检查、trace 解析
├── tests/                            # CPU 模型、训练及运行时行为测试
└── docs/                             # 当前说明与历史实验报告

vllm/v1/worker/
├── gpu_worker.py                     # 在线组件的四处生命周期调用点
└── gpu/
    ├── online_draft.py               # create_online_draft / OnlineDraftRuntime
    └── spec_decode/
        ├── weight_slots.py           # 共用 GPU slot 分配和参数绑定
        ├── online.py                 # 共用 speculator 在线接入与 capture/安装顺序
        ├── online_state.py           # 共用安装、窗口关闭和 request reset
        ├── eagle/
        │   └── speculator.py         # EAGLE3：采集 aux hidden
        ├── mtp/
        │   └── speculator.py         # MTP：采集 target 最终 hidden
        ├── autoregressive/
        │   ├── online.py             # EAGLE3 与 MTP 共用的在线采集钩子
        │   └── speculator.py         # EAGLE 等自回归推测解码共用路径
        └── dflash/
            ├── online.py             # DFlash 派生权重 buffer slot
            ├── speculator.py         # DFlash 推理与采集字段
            └── cudagraph.py          # DFlash query graph
```

目录树省略了 `__init__.py` 和无关的推理文件。`runtime/__init__.py` 只提供
`profile_range`，不承担运行时装配。两种方法的训练代码按同样的职责分组：batch 定义、loss、窗口训练入口；
DFlash 的 context cache 较简单，和窗口训练放在一起。

## 按职责找代码

| 要处理的问题 | 代码位置 | 边界 |
| --- | --- | --- |
| 哪个时刻采集，记录什么 | 两种 `speculator.py` 与 `runtime/capture/` | 推理侧选 hook；adapter 保存 proposal、teacher 和版本，生成 packet |
| GPU 数据如何成为一轮训练样本 | `runtime/capture/` 的 packet 与 batch 构造器 | `base.py` 负责连续历史、prompt 累积和 clone；模型文件只组装各自的 batch 与监督位置 |
| D2H 怎么执行，何时可以读 CPU 数据 | `runtime/transport.py` | 按 FIFO 等待 ready event、在 D2H stream 复制、等待完成后调用构造器 |
| 何时训练、哪些轮属于同一窗口 | `runtime/async_pipeline.py` | bootstrap、epoch close、训练队列及快照发布；不解释模型专用 batch |
| 一次更新计算什么 | `training/eagle3/`、`training/dflash/` | query 选择、mask、loss、context KV 构建与更新后的历史追加 |
| 如何做 backward / optimizer / reset | `training/trainer.py` | 持有 CPU 模型、optimizer、W0、可训练参数名和 CPU version |
| CPU 权重如何上传并生效 | `runtime/weights.py` | snapshot 数据与版本；installer stage、完成检查、active slot/version、reset |
| GPU 参数存储如何分配和绑定 | `spec_decode/weight_slots.py` | 只为选中的 draft 自有参数分配双 slot，保持 target 共享参数 |
| DFlash 派生参数如何保持一致 | `dflash/online.py` | 按 slot 维护 fused KV、norm buffer，并在 commit 前准备完成 |
| 谁组装上述对象 | `gpu/online_draft.py` | 选择模型和训练函数，连接回调，提供 worker 生命周期入口 |

`DFlashCaptureAdapter` 的 Adapter 指它把 vLLM 推理回调中的数据转换为采集协议。
它本身不训练模型。`capture/` 属于 `runtime/`，因为它依赖请求状态、GPU 完成事件、
异步队列及跨线程数据生命周期；其中的 CPU batch 构造器仍保留模型专用语义。

权重交接集中在 `runtime/weights.py`，主要对象是 `DraftWeightSnapshot` 和
`DraftWeightInstaller`。推理侧通过 `pipeline.poll_snapshot` 回调领取快照，再由
`installer.advance(snapshot)` 完成交接。GPU 模型参数绑定放在 vLLM 一侧，训练包
不依赖具体推理模型。

## 线程与数据所有权

| 执行位置 | 持有的状态 | 发出的数据 |
| --- | --- | --- |
| 推理线程 / GPU | adapter 的 proposal、teacher；在线状态；活动 slot；GPU KV | GPU packet + ready event，epoch close、finish 标记 |
| capture 线程 | `CaptureQueue`、D2H stream、CPU batch 构造器的连续历史 | 带 request/step/source version 的 `PreparedRound` |
| training 线程 | CPU 模型、optimizer、当前窗口、CPU 持久 KV | 已更新版本的 CPU snapshot |
| GPU H2D stream | installer 提交的复制与派生 buffer 准备 | 完成 event，由推理线程查询 |

H2D stream 不是额外的 Python 工作线程。Capture 等待 D2H 完成发生在采集线程；
推理侧安装只查询完成事件，不等待 CPU 训练或尚未完成的 H2D。
请求 finish/reset 则允许等待后台工作和上传完成。

两种模型的 D2H 都写入启动时预分配、复用的 pinned staging（`PinnedStaging`），
因此 capture 队列每次只搬运一个 packet；dtype 不一致或超出容量直接报错，不做
隐式转换或截断。Batch 构造器必须在下一次搬运覆盖 staging 之前，把提交给训练
线程或跨轮保留的数据转成独立持有的内容。CPU snapshot 也复用一份 pinned
存储，但下一次训练由上次安装成功后的 epoch close 触发，因此不会在 H2D 仍读取
该快照时覆盖它。

## 一条完整调用链

```mermaid
flowchart LR
    S[模型 speculator] --> A[CaptureAdapter]
    A --> Q[CaptureQueue / D2H]
    Q --> B[模型 CPU batch 构造器]
    B -->|PreparedRound| P[AsyncDraftPipeline]
    P --> T[模型窗口训练函数 + DraftTrainer]
    T -->|新 cache 与训练结果| P
    P -->|poll_snapshot| O[DraftOnlineState]
    O --> W[DraftWeightInstaller / H2D]
    W -->|安装成功后绑定 slot| S
```

装配入口是 [`create_online_draft`](../vllm/v1/worker/gpu/online_draft.py)。
它按方法注入绑定了 `max_train_rounds` 的 `train_eagle3_rounds`（EAGLE3、MTP）或
`train_dflash_window`（DFlash），所以 pipeline 不需要按模型分支。训练函数接收 trainer、
完整 round 的 batch 元组、旧 cache 和 profiling 开关，返回新 cache 与训练结果。

两种模型共用同一个步内顺序，由 `spec_decode/online.py` 的
`_capture_target_and_install` 实现：

```text
capture target packet → poll/stage/commit/bind → 安装成功则 close_epoch
  → 依赖权重的 draft 计算 → capture_proposal
EAGLE3: 上述顺序位于 FC 之前（FC 可训练，需使用新权重）
DFlash: 上述顺序位于 FC 之后（FC 冻结，采集其输出）、context KV / query 之前
```

两者只在采集哪些张量、顺序位于 FC 前还是后上不同。Packet 必须先于对应的 epoch
标记进入 FIFO；非采集步（dummy / profile）不安装权重。第一条完整 round 携带 prompt bootstrap，单独训练一次；后续
只在新权重真正安装时关闭窗口。样本中的 `source_weight_version` 记录 proposal
生成时的 GPU 版本，CPU 训练版本可以领先于它。

[`gpu_worker.py`](../vllm/v1/worker/gpu_worker.py) 保留四处生命周期接入：

1. 模型加载后创建 `OnlineDraftRuntime`，完成装配，capture 此时尚未启用。
2. Warmup 完成后调用 `enable_capture()`，先应用 inference CPU 配置，再挂上 adapter。
3. 每次执行前传播后台错误，并处理已结束请求。Finish 经过 capture FIFO 和训练
   FIFO，等待已开始工作，丢弃未闭合窗口及不完整尾部，再恢复 CPU/GPU W0。
4. Shutdown 先关闭 capture，再关闭 pipeline，最后释放 model runner。

关闭在线训练时不创建上述 runtime、CPU 模型、工作线程或 capture buffer。

## 依赖与方法差异

独立 `online_draft` 包不导入 vLLM。vLLM 的装配层和推理状态对象依赖训练包；
公共 transport 只依赖 packet 协议，公共 pipeline 只依赖训练回调、trainer 和
通用 round 元数据。模型的 batch/KV 细节留在模型自己的 capture 与 training 中。

`training` 和 `runtime` 不是完全单向的目录依赖：训练函数复用
`runtime.profile_range`，pipeline 调用公共 trainer；模块层面没有运行时导入环。
EAGLE3 cache 对 window 类型的引用在 `TYPE_CHECKING` 下，不会触发运行时循环。
三种 CPU 模型的 RMSNorm、RoPE、MLP 与 attention 基础算子集中在 `models/layers.py`。

推理共用 slot，但 DFlash 仍需要派生 buffer 和每个 slot 一套 query graph。
EAGLE3 与 MTP 在线时同样为每个 slot 捕获一套 prefill/decode FULL graph，prompt 步
与不支持 fused decode 的后端走 eager（见[报告五](docs/reports/05_eagle3_online_overhead_root_cause.md)）。
各方法有各自的 loss、query 布局与 context 构建方式。

仓库根目录的 [`methods/`](../methods/README.md) 是另一组 EAGLE3 实验策略入口，
包含 TTS、随机采样和 OSpec 等策略，复用部分训练组件。它不由这里的
`create_online_draft` 装配，与这里的在线训练路径相互独立。

## 运行

### 环境变量与 vLLM 配置

在线训练由环境变量开启，方法由 `speculative_config.method` 选择：

| 环境变量 | 作用 |
| --- | --- |
| `VLLM_USE_V2_MODEL_RUNNER=1` | 必需，在线训练只接在 V2 model runner 上 |
| `VLLM_ONLINE_DRAFT_TRAIN=1` | 开启在线训练；为 0 时不创建 CPU 模型、线程和 capture buffer |
| `VLLM_ONLINE_DRAFT_PROFILE=1` | 可选，记录在线训练的 profile 时间线 |
| `OMP_NUM_THREADS` / `MKL_NUM_THREADS` | 未配置 `cpu_threads` 时 CPU 训练使用的线程数，需在启动命令中设置；两者都没有时 vLLM 在 warmup 后把 Torch 线程设为 1，CPU 训练只有单线程 |
| `VLLM_ENABLE_V1_MULTIPROCESSING` | 保持默认，不要设为 0：engine 与 `LLM` 前端同进程时，前端的 `torch.set_num_threads` 会让训练线程只剩单线程 |

启动时 `_validate_online_draft_config` 检查以下条件，不满足时报出全部未满足项：

| 配置 | 要求 |
| --- | --- |
| `speculative_config.method` | `eagle3`、`dflash` 或 `mtp` |
| target `model_type` | EAGLE3 / DFlash：`qwen3` 或 `qwen3_5`；MTP：`qwen3_5` |
| `tensor_parallel_size` / `pipeline_parallel_size` | 1 |
| `max_num_seqs` | 1 |
| `enable_prefix_caching` | `False` |
| draft 输入 | 纯文本；Qwen3.5 等多模态 checkpoint 需设置 `limit_mm_per_prompt={"image": 0, "video": 0}` |

GPU attention 后端：Qwen3.5 EAGLE3 的 draft head_dim 为 160，FlashInfer 不支持，
且 V2 runner 中 EAGLE 的 draft prefill 复用 target 的 attention metadata，因此
target 与 draft 都需用 `FLASH_ATTN`。Qwen3-8B 可用 `FLASHINFER`。

CPU attention backward 使用独立 C++ extension，需要 C++ 编译器和 Ninja；首次构建
在 CPU 模型初始化时完成，不进入 decode。

### `additional_config["online_draft"]`

各方法只接受自己读取的键，出现其他键直接报错。

| 键 | 适用方法 | 默认 | 说明 |
| --- | --- | --- | --- |
| `learning_rate` | 全部 | `1e-5` | AdamW 学习率，须为正 |
| `fused_optimizer` | 全部 | `True` | fused AdamW |
| `master_weights` | 全部 | `False` | FP32 master weights |
| `frozen_parameter_prefixes` | 全部 | `[]` | 按名字前缀冻结参数，例如 `["lm_head"]` |
| `check_gradients` | 全部 | `False` | 每步检查梯度是否有限 |
| `log_updates` | 全部 | `False` | 记录每次 GPU 权重安装 |
| `log_queue_sizes` | 全部 | `False` | 记录 capture / 训练队列长度 |
| `cpu_threads` | 全部 | `{}` | `inference` / `capture` / `training` 各自的线程数 |
| `cpu_affinity` | 全部 | `{}` | 同上三个角色绑定的 CPU 列表 |
| `attention_backend` | EAGLE3、MTP | `"eager"` | CPU attention：`"eager"` 或 `"flash_attention"` |
| `max_train_rounds` | 全部 | `0` | 每次训练只用两次安装之间最近 N 个 round；0 表示全部 |
| `grouped_query_attention` | DFlash | `False` | CPU attention 按 GQA 重排，不展开历史 K/V |
| `selective_last_layer` | DFlash | `False` | 最后一层只计算监督行 |

MTP 与 target 共享 lm_head，不训练 lm_head。EAGLE3 draft 的 lm_head 与 target
完全相同时（例如 Qwen3.6-27B PRISM `full`），vLLM 会共享它，需设置
`frozen_parameter_prefixes: ["lm_head"]`。

### 示例

CPU 训练的线程数有两种配法，二选一；两者都不配时 CPU 训练只有单线程。

#### 写法一：环境变量指定线程数，不绑核

把下面的代码保存为 `launch.py`，在启动命令中设置线程数：

```bash
OMP_NUM_THREADS=20 MKL_NUM_THREADS=20 .venv/bin/python launch.py
```

```python
import os

os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
os.environ["VLLM_ONLINE_DRAFT_TRAIN"] = "1"

from vllm import LLM

llm = LLM(
    model="/data/models/Qwen3.5-4B-BF16/Qwen3.5-4B",
    limit_mm_per_prompt={"image": 0, "video": 0},
    max_num_seqs=1,
    enable_prefix_caching=False,
    attention_config={"backend": "FLASH_ATTN"},
    speculative_config={
        "method": "eagle3",
        "model": "/data/models/Qwen3.5-4B-BF16/Qwen3.5-4B-Eagle3",
        "num_speculative_tokens": 7,
        "attention_backend": "FLASH_ATTN",
    },
    additional_config={
        "online_draft": {
            "learning_rate": 1e-5,
            "attention_backend": "flash_attention",
            "log_updates": True,
        }
    },
)
```

CPU 训练使用 `OMP_NUM_THREADS` 个 OpenMP 线程。推理、capture 与训练线程都不绑核，
由操作系统调度，任何机器都可以直接使用。

#### 写法二：`cpu_threads` 与 `cpu_affinity` 按机器拓扑绑核

在写法一的 `online_draft` 中加入：

```python
"cpu_threads": {"inference": 1, "capture": 1, "training": 16},
"cpu_affinity": {
    "inference": [18],
    "capture": [19],
    "training": list(range(16)),
},
```

推理、capture 与训练线程各自设置线程数与绑核，不依赖 `OMP_NUM_THREADS`。上面是本工作站
（20 个物理核，CPU n 与 n+20 是同一物理核的两个超线程）的取值：训练 16 线程用 CPU 0–15，
推理与 capture 各占一个物理核，它们的超线程 38、39 空闲，训练线程不会与推理线程共享物理核。
Qwen3.5 与 Qwen3.6 的 AIME 实验都使用这一布局。换机器需按
`/sys/devices/system/cpu/cpu*/topology/thread_siblings_list` 重新选择。

#### 改用 MTP 或 DFlash

MTP：`method` 改为 `"mtp"`，去掉 `speculative_config.model`，`num_speculative_tokens`
常用 3，`online_draft` 不变。

DFlash：换成 DFlash draft，`num_speculative_tokens` 常用 15；`online_draft` 去掉 EAGLE3 的
`attention_backend`，加入 DFlash 的键：

```python
speculative_config={
    "method": "dflash",
    "model": "/data/models/Qwen3.5-4B-BF16/Qwen3.5-4B-DFlash",
    "num_speculative_tokens": 15,
    "attention_backend": "FLASH_ATTN",
},
additional_config={
    "online_draft": {
        "learning_rate": 1e-5,
        "max_train_rounds": 16,
        "grouped_query_attention": True,
        "selective_last_layer": True,
        "log_updates": True,
    }
},
```

#### `max_train_rounds`

每次更新拿到两次权重安装之间的全部 round。`max_train_rounds=N` 时只对其中最近 N 个 round
计算 loss 与 backward；更早的 round 不产生 loss，但它们确认的 target 特征照常进入训练的
上下文（DFlash 的 context KV，EAGLE3 与 MTP 的 confirmed spine 与持久 KV），之后的 round
能看到。默认 0，训练全部 round；三种方法都接受这个键，写在 `online_draft` 中。

N 越大单次训练越久，两次安装之间积累的 round 越多，安装越稀疏。DFlash 建议配 16：
Qwen3-8B 的扫描（`results/experiments/2026-10-28_dflash_train_rounds_8b_9b`，FlashInfer，
lr 1e-4，前 8 题）中，0 的窗口最大到 2,450 个 round，接受长度 3.23，低于 frozen 的 3.55；
16 为 3.76。

DFlash 的实验用写法二绑核，并在启动命令中设置
`OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE`。

### 实机 smoke

[`smoke_async_dflash.py`](scripts/smoke_async_dflash.py) 自动设置上述环境变量，
用 `--target` 和 `--method` 选择模型组合：

| `--target` | 可用 `--method` |
| --- | --- |
| `qwen3-8b`（默认） | `dflash`、`eagle3` |
| `qwen3.5-4b` | `dflash`、`eagle3`、`mtp` |

```bash
# Qwen3.5-4B 三种方法的 CPU/GPU 对拍与在线 smoke
OMP_NUM_THREADS=20 MKL_NUM_THREADS=20 .venv/bin/python \
  online_draft/scripts/smoke_async_dflash.py --online --check-math \
  --target qwen3.5-4b --method eagle3 --speculative-tokens 7 \
  --attention-backend FLASH_ATTN --max-tokens 256 --requests 1 \
  --output /tmp/online-eagle3.json

# Qwen3-8B DFlash，target 与两套 draft slot 使用 CUDA graph，并按本机绑核
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE .venv/bin/python \
  online_draft/scripts/smoke_async_dflash.py \
  --online --target-graphs --train-threads 16 --cpu-isolation \
  --grouped-query-attention --selective-last-layer \
  --max-tokens 1024 --requests 2 --output /tmp/dflash-online.json
```

MTP 与 DFlash 常用的 `--speculative-tokens` 分别为 3 和 15。去掉 `--online` 运行
冻结 draft 的推理。

| 参数 | 作用 |
| --- | --- |
| `--attention-backend` | target 与 draft 的 GPU attention 后端，默认 `FLASHINFER` |
| `--learning-rate` | 默认 `1e-5`，长请求可用 `2e-4` |
| `--train-threads N` | 训练线程数，推理与采集各 1 线程 |
| `--cpu-isolation` | 本工作站（20 物理核）专用：推理绑 18、采集 19、训练 0..N-1 |
| `--target-graphs` | target 与 draft slot 开启 CUDA graph |
| `--check-math` | 请求前做 CPU/GPU 前向、非零 slot 安装与 reset 对拍；会同步 GPU，不属于推理路径 |
| `--max-train-rounds`、`--grouped-query-attention`、`--selective-last-layer`、`--master-weights` | 对应 `online_draft` 配置项；`--max-train-rounds` 默认 16，`0` 训练全部 round |
| `--queue-metrics` | 仅在线 DFlash：记录队列长度和驻留时间 |
| `--aime-questions`、`--aime-offset`、`--context-tokens`、`--max-model-len` | 使用本地 AIME 数据自然生成至 EOS 或上下文上限 |
| `--gsm8k-questions` | 复用 GSM8K 回归工具 |
| `--decode-timeline` | 记录 decode 分段指标 |
| `--profile-dir`、`--profile-delay-iterations`、`--profile-max-iterations` | 截取 PyTorch trace；profile 请求的吞吐不作为性能结果 |

`--cpu-isolation` 只对应本工作站；换机器需按实际 topology 配置 `cpu_threads`
与 `cpu_affinity`。脚本使用可信 callable RPC，只在本地测试进程中开启 pickle
序列化。

## 脚本与验证入口

- [`smoke_async_dflash.py`](scripts/smoke_async_dflash.py)：EAGLE3、DFlash、MTP 的
  baseline 与 online、请求指标及 profile，用法见[实机 smoke](#实机-smoke)。
- [`smoke_async_eagle3.py`](scripts/smoke_async_eagle3.py) 与
  [`smoke_async_eagle3_two_requests.py`](scripts/smoke_async_eagle3_two_requests.py)：
  EAGLE3 单请求与两请求实机 smoke；在线运行需外部设置 `VLLM_USE_V2_MODEL_RUNNER=1`
  和 `VLLM_ONLINE_DRAFT_TRAIN=1`。
- [`check_dflash_math.py`](scripts/check_dflash_math.py)、
  [`check_eagle3_math.py`](scripts/check_eagle3_math.py) 与
  [`check_mtp_math.py`](scripts/check_mtp_math.py)：`smoke_async_dflash.py --online
  --check-math` 按方法调用的 CPU/GPU 实机对拍，覆盖完整前向、安装后的权重与 reset。
- [`summarize_dflash_trace.py`](scripts/summarize_dflash_trace.py)：解析 profiler trace。
- [`benchmark_aime_2026.py`](scripts/benchmark_aime_2026.py)：AIME 公共 runner，支持
  target-only / EAGLE3 的 eager / graph 基线；模型与数据路径集中在文件顶部。
- [`benchmark_aime_2026_online.py`](scripts/benchmark_aime_2026_online.py) 与
  [`benchmark_aime_2026_speculative_baseline.py`](scripts/benchmark_aime_2026_speculative_baseline.py)：
  EAGLE3 在线训练和冻结 draft 的对照入口，可用 `--problem-limit` 限制题数。
- [`profile_aime_2026_online.py`](scripts/profile_aime_2026_online.py)：
  EAGLE3 指定 AIME 题目的 profile，可用 `--no-profile` 运行普通请求。它设置了
  `VLLM_ENABLE_V1_MULTIPROCESSING=0`，CPU 训练只有单线程，trace 中的训练耗时偏大；
  需要多线程训练的 trace 时，engine 用默认的独立进程运行。

脚本中的模型、数据路径及 CPU 绑核配置对应本地工作站。
回归入口如下，在仓库根目录及已配置好的虚拟环境中执行：

```bash
source .venv/bin/activate
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m pytest \
  online_draft/tests methods/tests \
  tests/v1/spec_decode/test_eagle_weight_slots.py \
  tests/v1/spec_decode/test_online_draft_parity.py \
  tests/v1/worker/test_gpu_worker_online_draft.py \
  tests/v1/worker/test_gpu_autoregressive_speculator.py \
  tests/v1/spec_decode/test_dflash_prepare_inputs.py \
  tests/v1/spec_decode/test_dflash_causality.py \
  tests/v1/spec_decode/test_eagle_draft_attn_metadata.py -q
```

该回归验证模型计算、数据对齐、线程队列、安装/reset 及推理接入；不等同于真实
请求的模型评估或性能复测。`docs/` 下的阶段报告记录写作时的实现和实验数据，其中的
模块名、脚本与产物路径可能已变化，代码导航以本文为准。

EAGLE3 的真实 AIME、权重安装和两请求 reset 检查见
[EAGLE3 实机验证报告](docs/experiments/eagle3_refactor_validation.md)。
