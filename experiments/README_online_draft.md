# Online draft CPU training

`online_draft+eagle3`、`online_draft+dflash` 和 `online_draft+mtp` 共用
`online_draft.runtime.AsyncDraftPipeline`：GPU 侧异步捕获 verification 信号，
CPU worker 训练独立副本，再异步安装权重。它们不使用 TTS 的 CUDA side-stream
trainer。

MTP 没有独立 draft checkpoint，权重必须内置在 target 中。在线 MTP 只支持含
`mtp_num_hidden_layers` 的 Qwen3.5；含 `num_nextn_predict_layers` 的 Qwen3
（如 `Qwen3-8B-MTP`）只能走 `tts+mtp`（`methods/tts_common/vllm_ext`）。标准
`Qwen/Qwen3-8B` 不含 MTP 权重，只能运行 EAGLE3 或 DFlash。

```bash
bash experiments/prepare.sh \
  --model-size 8b --draft-method mtp
bash experiments/prepare.sh \
  --model-size qwen3.5-4b --draft-method mtp
```

## 默认配方

loss 形式是方法定义的一部分；DFlash 只有温度可配置。CPU 训练开关的默认值见下文
“CPU 训练优化”。

| 方法 | loss | 证据（旧 32 GB GPU） |
| --- | --- | --- |
| EAGLE3 | 每轮深度 d 权重 `0.9**d` 的 forward KL，除以监督行数 | Qwen3.5-4B 30 题三次同批配对 +0.46 / +0.63 / +0.64 AL（每次 28/30 题更高）；Qwen3-8B 30 题 frozen 2.284 → 2.657 |
| MTP（Qwen3.5、Gemma4） | 与 EAGLE3 相同 | Qwen3.5-4B 仍低于冻结：30 题三次同批 −0.07 到 −0.16 AL；全参数 loss 中最好。Gemma4 未测 |
| DFlash | forward KL，权重 `exp(-d/7)`，除以权重和；`DFLASH_MAX_TRAIN_ROUNDS=16` | 默认 T=1。`ONLINE_DRAFT_DFLASH_LOSS_TEMPERATURE=0.6` 把 teacher 与 draft 都锐化到 T=0.6（teacher 不截断）：Qwen3.5-4B 30 题三次同批 +0.10 / −0.25 / −0.16 AL |

EAGLE3 的证据只覆盖 Qwen3.5-4B 与 Qwen3-8B target；Llama3.1、Gemma4 target 与
Gemma4 MTP 换用此 loss 后尚未重测。

## CPU 训练优化

`experiments/run.sh` 的开关和默认值（0 关闭，1 开启）：

```bash
ONLINE_DRAFT_MASTER_WEIGHTS=1           # FP32 master weights
ONLINE_DRAFT_CONTINUOUS_TRAINING=0      # 队列排空即训练，安装槽空出时发布最新权重
ONLINE_DRAFT_LR_REFERENCE_ROWS=0        # N>0 时每步 lr 乘 min(1, 监督行数 / N)；0 关闭
ONLINE_DRAFT_DFLASH_LOSS_TEMPERATURE=0  # DFlash KL 温度；0 保持 T=1
ONLINE_DRAFT_LR=1e-5                    # 学习率
MAX_TRAIN_ROUNDS=                       # 每次更新只训练最近 N 个 round；空时 DFlash 16，其余 0
ONLINE_DRAFT_GROUPED_QUERY_ATTENTION=0  # 仅 DFlash：CPU attention 按 GQA 重排
ONLINE_DRAFT_SELECTIVE_LAST_LAYER=0     # 仅 DFlash：最后一层只算监督行
ONLINE_DRAFT_CPU_THREADS=               # JSON，例如 '{"inference":1,"capture":1,"training":16}'
ONLINE_DRAFT_UPDATE_METRICS=0           # 1 时每个请求结束后把每次 CPU 更新的明细追加到 UPDATE_METRICS_FILE
ONLINE_DRAFT_CPU_AFFINITY=              # JSON，例如 '{"inference":[18],"capture":[19],"training":[0,1,2,3]}'
```

`DFLASH_MAX_TRAIN_ROUNDS` 是 DFlash 的旧名，`MAX_TRAIN_ROUNDS` 未设置时生效。`ONLINE_DRAFT_CPU_*` 都不设时不绑核，线程数由 `OMP_NUM_THREADS`（默认 20）决定。

`UPDATE_METRICS_FILE` 默认是 `$RUN_DIR/updates.jsonl`，每个请求一行：`request_id`、`cpu_version` 和 `updates` 列表。列表每项对应一次 CPU 训练，含 `rounds`、`trained_rounds`、`target_count`、`train_seconds`、`loss`、`start_step`/`end_step` 和 `source_versions`。warmup 请求不写入。

以下两项是 trainer 的固定行为，没有开关：

- 无 bias 的可训练 linear 把权重梯度写入常驻缓冲，缓冲在第一次 backward 时分配。
- `master_weights` 开启且可训练参数都是连续的 CPU BF16 张量时，用释放 GIL 的单遍
  原生 AdamW；原生 kernel 编译失败时退回 FP32 副本 + torch AdamW。

BF16 原地 AdamW 在 lr=1e-5 时约 82% 的元素更新被舍入丢弃，RMSNorm 权重从不变化；
FP32 master weights 是 AL 收益的主要来源（Qwen3-8B 5 题 AL@2000：2.377 → 2.564，
去掉它降到 2.202）。其余几项主要增加更新次数（约 1.5 倍）和 decode 速度，单独对
AL 的影响在重复噪声内。

原生 AdamW 一遍读 BF16 梯度、写回 FP32 master、两个矩和 BF16 参数。243M 参数、
20 线程（Xeon w7-3445）单步 70 ms，FP32 副本 + fused AdamW 为 120 ms。原生 kernel
首次使用时以 `-march=native` 编译（需要 g++ 与 ninja），失败时日志出现
`native master AdamW unavailable, using FP32 master copies`。

## 启动示例

```bash
MODEL=./methods/artifacts/models/Qwen3-8B \
SPEC_MODEL=./methods/artifacts/models/Qwen3-8B_eagle3 \
bash experiments/run.sh online_draft+eagle3

MODEL=./methods/artifacts/models/Qwen3-8B \
SPEC_MODEL=./methods/artifacts/models/Qwen3-8B_dflash \
bash experiments/run.sh online_draft+dflash

# MTP 权重内置在 target 中，不要设置 SPEC_MODEL；多模态 Qwen3.5 需 LANGUAGE_MODEL_ONLY=1。
MODEL=./methods/artifacts/models/Qwen3.5-4B LANGUAGE_MODEL_ONLY=1 \
bash experiments/run.sh online_draft+mtp
```

`online_draft` 是 `online_draft+eagle3` 的兼容简写。三种 online 方法都要求
`RESUME=0`，并在每个请求结束时恢复初始 draft 权重。Qwen3.5-4B EAGLE3 的 draft
head_dim 为 160，FlashInfer 不支持，需 `GPU_ATTENTION_BACKEND=FLASH_ATTN`。
