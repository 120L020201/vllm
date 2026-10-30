# Online draft CPU training

`online_draft+eagle3`、`online_draft+dflash` 和 `online_draft+mtp` 共用
`online_draft.runtime.AsyncDraftPipeline`：GPU 侧异步捕获 verification 信号，
CPU worker 训练独立副本，再异步安装权重。它们不使用 TTS 的 CUDA side-stream
trainer。

MTP 没有独立 draft checkpoint，权重必须内置在 target 中。CPU recipe 支持含
`num_nextn_predict_layers` 的 Qwen3（如 `Qwen3-8B-MTP`），以及含
`mtp_num_hidden_layers` 的 Qwen3.5。标准 `Qwen/Qwen3-8B` 不含 MTP 权重，
只能运行 EAGLE3 或 DFlash。

```bash
bash experiments/prepare.sh \
  --model-size 8b --draft-method mtp
bash experiments/prepare.sh \
  --model-size qwen3.5-4b --draft-method mtp
```

## Loss 预设

只保留一个入口变量：

```bash
ONLINE_DRAFT_LOSS=uniform  # 默认，KL+Uni，所有位置权重为 1
ONLINE_DRAFT_LOSS=position # 每轮深度权重 0.9**depth
ONLINE_DRAFT_LOSS=entropy  # exp(-2 * normalized_teacher_entropy) * KL
```

三种损失都按监督行数归一化，并适用于 EAGLE3、DFlash 和 MTP 的共享 CPU
pipeline。旧的 coverage、rank、rejection、sampling、LoRA、replay 等实验变量已
移除；传入未知 `additional_config["online_draft"]` 字段会 fail closed。

DFlash 另有 sampler-aligned 配方：teacher 按实际 `temperature -> top-k -> top-p`
截断，student 保持全词表归一化，从而惩罚只能被 rejection sampling 拒绝的尾部
概率。`objective=overlap` 直接优化 expected acceptance。也可用
`ONLINE_DRAFT_DFLASH_LOSS_REFERENCE_WEIGHT` 对冻结初始 draft 做 trust-region
KL；`..._REFERENCE_DECAY_TOKENS` 可按长度释放，但多 seed 下稳定性不如 dense
temperature KL，建议只在明确调参后使用。

单条 `aime2026[1]` 的 5-seed 平均 AL：冻结 `4.577`，历史配方 `4.438`，
`T=0.6` 全词表 KL `4.562`，overlap 加 anchor `0.5` 为 `4.557`。短单请求
没有足够的新信息，在线训练不应期待超过冻结；若必须更新，使用下面的 dense
temperature KL 以减小漂移：

```bash
ONLINE_DRAFT_LOSS=position \
ONLINE_DRAFT_DFLASH_LOSS_GAMMA=7 \
ONLINE_DRAFT_DFLASH_LOSS_NORMALIZATION=weights \
ONLINE_DRAFT_DFLASH_LOSS_OBJECTIVE=kl \
ONLINE_DRAFT_DFLASH_LOSS_TEMPERATURE=0.6 \
ONLINE_DRAFT_DFLASH_LOSS_TOP_K=0 \
  bash experiments/run.sh online_draft+dflash
```

多请求场景可保留历史配方或 overlap + anchor；固定 5 题上它们与历史配方接近，
且都优于冻结。

## CPU 训练优化

以下三个开关可以独立启用，默认关闭：

```bash
ONLINE_DRAFT_MASTER_WEIGHTS=1       # FP32 master weights + native AdamW
ONLINE_DRAFT_PERSISTENT_GRADS=1     # 复用 linear weight 梯度缓冲
ONLINE_DRAFT_CONTINUOUS_TRAINING=1  # 队列排空即训练，不等待下次权重安装
```

`ONLINE_DRAFT_MASTER_WEIGHTS` 保持 BF16 参数供 forward/backward 使用，同时在
FP32 master weights/moments 上更新并回写 BF16；原生 C++ kernel 释放 GIL，
构建失败时自动回退到 ATen。`ONLINE_DRAFT_PERSISTENT_GRADS` 只替换无 bias 的
`nn.Linear`，将权重梯度直接写入持续复用的缓冲区。`ONLINE_DRAFT_CONTINUOUS_TRAINING`
允许 CPU 在已有权重快照等待 H2D 安装时继续训练，epoch 关闭时只发布最新版本。

如需按监督行数缩放学习率，可额外设置 `ONLINE_DRAFT_LR_REFERENCE_ROWS`（正数）；
不设置时保持配置的完整学习率。

需要复现某次运行的更新节奏时，用 `ONLINE_DRAFT_TRAIN_SCHEDULE` 固定训练边界：
值是累计已验证 round 数（严格递增），到达边界立即训练并发布快照，最后一项之后
该请求停止训练。例如 `63,111,151,197,241,287,330,376,419,459,501,543,587,634`
复现 10-29 22:18 那次第 1 题的 14 次更新。

## 启动示例

```bash
MODEL=./methods/artifacts/models/Qwen3-8B \
SPEC_MODEL=./methods/artifacts/models/Qwen3-8B_eagle3 \
ONLINE_DRAFT_LOSS=uniform \
ONLINE_DRAFT_MASTER_WEIGHTS=1 \
ONLINE_DRAFT_PERSISTENT_GRADS=1 \
ONLINE_DRAFT_CONTINUOUS_TRAINING=1 \
bash experiments/run.sh online_draft+eagle3

MODEL=./methods/artifacts/models/Qwen3-8B \
SPEC_MODEL=./methods/artifacts/models/Qwen3-8B_dflash \
ONLINE_DRAFT_LOSS=position \
bash experiments/run.sh online_draft+dflash

# MODEL 必须是含内置 mtp.* 权重的 Qwen3/Qwen3.5 checkpoint；不要设置 SPEC_MODEL。
MODEL=./methods/artifacts/models/Qwen3-8B-MTP \
ONLINE_DRAFT_LOSS=entropy \
bash experiments/run.sh online_draft+mtp
```

`online_draft` 是 `online_draft+eagle3` 的兼容简写。三种 online 方法都要求
`RESUME=0`，并在每个请求结束时恢复初始 draft 权重。
