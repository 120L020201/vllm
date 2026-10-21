# DFlash 改动总览与 EAGLE3 在线训练策略对比

整理日期：2026-10-19。代码范围：`/home/lixiang/work/myvllm-eagle3-dflash`，
分支 `eagle3&dflash`，基线 `3979838d2d`。下文比较的是**本仓库当前接入的在线
训练路径**，不是两个方法的所有论文训练方案。数值实验来自 2026-10-16 的报告，
本次只核对代码与整理文档，没有重新跑 GPU 实验，也没有提交 Git。

当前 DFlash 复用了 EAGLE3 的请求生命周期、异步调度、trainer 和权重安装机制。
主要分歧在模型输入、监督加权、query 选择、context 梯度及 GPU graph 接入。
“复用运行时”不等于“训练策略完全相同”；一些共享运行时优化也已作用于 EAGLE3。

## 全部改动按职责归档

相对上述基线，已有跟踪文件修改 15 个，其中运行代码 9 个、测试 6 个；另新增
运行文件 5 个、测试文件 1 个、实验脚本 6 个，以及阶段报告和结果 JSON。
普通 `git diff --stat` 不包含这些尚未跟踪的新文件，不能只按它显示的行数估计范围。

### 新增 DFlash 模型与训练方法

| 文件 | 最终职责 |
| --- | --- |
| [`models/qwen3_dflash.py`](../../src/online_draft/models/qwen3_dflash.py) | CPU 可微 DFlash；从已加载的 GPU draft 复制真实参数；打包 QKV、gate/up；独立 context 投影；非因果完整 block；CPU Flash SDPA；attention 按完整 block 分组 |
| [`training/dflash/window.py`](../../src/online_draft/training/dflash/window.py) | DFlash batch/cache；窗口打包与位置 mask；最近 N 个完整 query block；位置衰减 KL；新增 context 梯度；更新后连续 KV 追加；记录缓存段版本 |
| [`runtime/capture/dflash.py`](../../src/online_draft/runtime/capture/dflash.py) | prompt bootstrap、完整 proposal、FC 输出、teacher logits 和 verification 计数的捕获及对齐；预分配 pinned D2H staging；CPU batch 构造器持有连续历史并选择有效监督行 |
| [`dflash/online.py`](../../../vllm/v1/worker/gpu/spec_decode/dflash/online.py) | DFlash 参数 slot 及 fused KV/norm 派生 buffer 一致性；安装、窗口关闭、请求结束及 reset |
| [`dflash/speculator.py`](../../../vllm/v1/worker/gpu/spec_decode/dflash/speculator.py) | 在 target 特征进入 FC 前 capture；接入 teacher/proposal 回调；每个参数 slot 一套 query graph；按活动 slot 执行；在线关闭时沿用原推理路径 |

### 共用组件的改动

| 文件 | 改动及 EAGLE3 影响 |
| --- | --- |
| [`runtime/async_pipeline.py`](../../src/online_draft/runtime/async_pipeline.py) | 两种模型共用 `AsyncDraftPipeline`，由 worker 注入各自的训练函数，EAGLE3 window 包装留在训练层；共享 bootstrap/安装划分窗口/reset；复用一份 pinned snapshot；记录 CPU/source version、完整/训练轮数、目标数和耗时 |
| [`runtime/contracts.py`](../../src/online_draft/runtime/contracts.py) | 通用 `PreparedRound` 与 `CapturePacket` 协议；独立于模型专用采集实现 |
| [`runtime/capture/eagle3.py`](../../src/online_draft/runtime/capture/eagle3.py) | EAGLE3 packet、GPU adapter 和专用 CPU batch 构造器；持有 confirmed 历史并处理位置对齐 |
| [`runtime/transport.py`](../../src/online_draft/runtime/transport.py) | 通用 FIFO 和 D2H 搬运；失败后继续处理清理信号，避免 finish 等待悬空，错误仍传播给推理主路径 |
| [`runtime/weights.py`](../../src/online_draft/runtime/weights.py) | 共用 `DraftWeightSnapshot` 和 `DraftWeightInstaller`；推理侧领取 snapshot 后调用 `advance(snapshot)`，先 stage 再非阻塞检查 commit；管理活动 slot/version、旧 slot 最后使用事件、派生 buffer 准备和 reset；移除独立 handoff 对象，绑定和 epoch close 仍由各模型负责 |
| [`eagle/weight_slots.py`](../../../vllm/v1/worker/gpu/spec_decode/eagle/weight_slots.py) | 提取 `DraftWeightSlots` 并保留 EAGLE3 别名；仍只给选中的 draft 自有参数分配两份 GPU 存储，不改 target 共享参数 |
| [`runtime/cpu_worker.py`](../../src/online_draft/runtime/cpu_worker.py) | 新增 inference/capture/training 三种角色的线程局部 OpenMP/MKL 配置及 affinity；EAGLE3 同样可以配置 |
| [`training/trainer.py`](../../src/online_draft/training/trainer.py) | 新增可选 FP32 master weights 和配套梯度/reset；默认关闭，原 AdamW 行为保留 |
| [`gpu_worker.py`](../../../vllm/v1/worker/gpu_worker.py) | 按方法加载 CPU 模型及 recipe；DFlash 配置检查与默认 N=16、gamma=7、fused=True；保留 EAGLE3 loss、全量窗口及 fused=False 默认；统一角色配置和生命周期 |

共享 snapshot 只复用一份 pinned buffer的前提是：下一次训练由上一次 H2D 真正
commit 后的 epoch close 触发，CPU 不会在上传仍读取快照时覆盖该 buffer。
这是现有时序保证，不能在以后改成无约束连续 optimizer update 时直接照搬。

### 测试、实验与报告

新增 [`test_dflash_training.py`](../../tests/test_dflash_training.py)，覆盖完整 block
attention、bootstrap、新 context 梯度、旧 KV 不变、窗口打包等价性、位置衰减及
减少 query 后历史仍完整。扩展六个现有测试文件：`test_async_pipeline.py`、
`test_capture_transport.py`、`test_draft_weight_installer.py`、`test_trainer.py`、
`tests/v1/spec_decode/test_eagle_weight_slots.py`、
`tests/v1/worker/test_gpu_worker_online_draft.py`。主要验证线程隔离、位置对齐、
FIFO/reset、pinned buffer、slot 连续覆盖、双 graph/派生 buffer 及 FP32 小更新。

六个新增脚本均位于 `online_draft/scripts/`：

| 脚本 | 用途 |
| --- | --- |
| `smoke_async_dflash.py` | baseline/采集/训练不安装/完整在线消融；AIME/GSM8K；纯 decode 和上下文分段；指定区间 profile |
| `check_dflash_math.py` | 真实 CPU/GPU logits、安装后的派生参数及 reset 数值检查 |
| `benchmark_dflash_cpu.py` | CPU 算子阶段、线程数、optimizer、compile、FP32 master weights 微基准 |
| `audit_dflash_updates.py` | 真实 capture 回放、权重有效变化与同窗 loss 审计 |
| `summarize_dflash_ablation.py` | 消融结果汇总，区分生成与完整请求时间 |
| `summarize_dflash_trace.py` | copy/kernel 重叠、stream、同步、CPU 阶段和算子统计 |

阶段 0–3 完成接入与正确性，4–8 完成采集/训练/安装消融、双 graph、CPU 配置和
长窗口优化，9–12 完成更新审计、位置衰减 KL、减量训练与长请求验证，13 完成
16K 纯 decode 对照和 trace。索引及六份结果 JSON 见 [README](README.md)。

## 两者保留的共同策略

1. 单 GPU、单活动请求内，GPU 推理与 CPU 训练异步运行；target 不参与在线更新。
2. 每个请求从 W0 开始，第一条完整 verification round 携带 prompt bootstrap，
   单独做一次更新；无跨请求累计学习或 replay buffer。
3. 后续只有 GPU **真正安装**新权重时才关闭窗口。每个已关闭窗口只做一次 optimizer
   update，没有多 epoch 重放。GPU 不等待 CPU 训练完成。
4. CPU 使用自己的当前参数训练旧 GPU 版本采集的样本；保留 request/step/source
   version。CPU optimizer 版本与 GPU 已安装版本分开统计，不假定它们相等。
5. 当前线上目标都监督“已接受前缀 + 首拒绝位置”的 teacher 分布，温度 1，
   teacher detach；全接受时 bonus 不单独产生 loss。EAGLE3 库里另有
   `PROPOSAL_CANVAS`，但 worker 实际使用默认 `CONFIRMED_PATH`。
6. 旧历史 KV 固定且不回传梯度；新段用处理时的参数构建，更新后不刷新全部历史。
   CPU/GPU 都接受各自历史里的混合版本近似，CPU KV 并非从 GPU KV 直接下载。
7. 保持所有已提交完整 round 的 FIFO 顺序和连续历史；不施加训练背压。finish
   有序处理已提交事件、等待已开始工作、丢弃未闭合窗口/不完整尾部，恢复 CPU/GPU
   W0、optimizer、历史和快照。错误向上传播，不能默默退回旧版本继续服务。
8. 共用双 GPU 参数 slot、独立 H2D stream、完成事件查询和后台 D2H；正式 decode
   与 finish/reset 等待分别计时。传输重叠不意味着零资源竞争或零提交成本。

## 训练策略的主要区别

| 维度 | 当前 EAGLE3 | 当前 DFlash |
| --- | --- | --- |
| 本地 draft 结构 | 1 层，3 组 target 特征，32K draft 词表 | 5 层，5 组 target 特征，151936 全词表 |
| proposal | 自回归展开，后续输入含旧 draft 的 recurrent hidden | 1 个 anchor + K 个 mask，一次并行预测；b16 对应 K=15 |
| 捕获输入 | token embedding、recurrent hidden、target 原始辅助特征、teacher、确认位置 | token IDs/positions、完整 query block、FC 前 target 原始特征、teacher、确认/拒绝计数 |
| teacher 词表 | GPU 先取映射后的 draft 词表 logits，CPU softmax 为条件分布 | 完整 target 词表 logits；CPU 转分布并选有效行 |
| loss 权重 | 普通位置 1，首拒绝位置 2 | 每 block 内第 k 个预测位置 `exp(-(k-1)/7)`，不再额外首拒绝 ×2 |
| 窗口选择 | 已关闭窗口的全部完整 round 参与训练 | worker 默认只选最近 16 个完整 query block，0 表示全量 |
| query 数量 | `CONFIRMED_PATH` 可只保留有监督的因果 query 行 | 所选 block 的 anchor 和拒绝后 mask 也必须保留在 forward 中；只在 loss/head 选择监督行 |
| 跨轮隔离 | 每轮看其 anchor 之前的确认上下文及本轮因果 query 前缀 | 每轮看 anchor 之前的确认 context 及本轮完整非因果 block，不能看其他 block 的 query |
| 临时 context 梯度 | 训练前以 `no_grad` 构建确认上下文 KV；它整体作为常量 | 默认 bootstrap prompt 和本窗口新增、对所选 query 可见的 context 投影参与梯度；旧持久 KV 为常量。可用 `context_grad=False` 关闭新增投影梯度，见[阶段 14 消融](stage14_context_gradient_ablation.md) |
| query 输入梯度 | 每轮 anchor 的 FC 重新计算并可训练；后续 recurrent hidden 是固定采集值，不跨 GPU rollout 反传 | 当前 CPU 权重计算整个 query block，没有采集并固定的 recurrent hidden 链 |
| 永久 KV 追加 | 更新后用新权重对新增确认路径做 no-grad draft forward | 更新后用新权重对全部新增确认特征做 no-grad context 投影 |
| 可训练参数 | draft 自有 FC/decoder/norm/独立 lm_head，叠加冻结前缀；输入 embedding 为捕获常量 | draft 自有 FC/decoder/norm，叠加冻结前缀；来自 target 的 embedding 和 lm_head 冻结 |
| KV 版本记录 | 持久 KV 为 tensor tuple；round/update 有版本信息 | 另有 `DFlashCache.segments`，记录 CPU KV 段终点及构建版本 |

“只训练最近 16 个 block”并不意味着只留 16 个 block 的 context，更不意味着
“每 16 轮就更新一次”。例如安装边界间采集了 90 轮：EAGLE3 训练这 90 轮的
有效 query；DFlash 默认只对最后 16 轮的完整 block 求 loss，但仍处理前 74 轮
的连续 context，并在更新后将 90 轮全部确认历史追加到持久 KV。它减少的是
query/MLP/head/loss 计算，不能消除长历史 attention 的成本。

两种 loss 都按整个训练窗口的实际监督权重归一化：

```text
L = sum_over_supervised_positions(w * KL(teacher || draft)) / sum(w)

EAGLE3: w = 2 at the first rejected prediction, otherwise 1
DFlash: w = exp(-(k - 1) / gamma), gamma = 7, k restarts in each block
```

DFlash 沿用位置衰减形式，并结合本项目 teacher KL 和在线监督位置；不能把整个
在线 recipe 称为对官方离线训练方案的原样复刻。早期阶段 1/6 记录的首拒绝 ×2
及全量 loss，已被用户后来授权的阶段 10 策略取代；EAGLE3 原损失保留。

源码依据：[`eagle3/inputs.py`](../../src/online_draft/training/eagle3/inputs.py)、
[`eagle3/cache.py`](../../src/online_draft/training/eagle3/cache.py)、
[`eagle3/distill.py`](../../src/online_draft/training/eagle3/distill.py)、
[`dflash/window.py`](../../src/online_draft/training/dflash/window.py)。

## KV 落后与权重落后的区别

两者都存在三种不同的旧状态，不能统称为“版本不一致”：

- **采样版本旧**：GPU 用 Wg 产生 proposal 和训练输入，CPU 用当前 Wc 计算 loss。
  不要求 Wc=Wg，不为等待新数据暂停 GPU，也没有 importance weighting 等补偿。
- **历史 KV 旧**：CPU/GPU 历史中各段由当时的权重构建；只追加新段，不保证整段
  cache 等价于用当前权重重跑全历史。DFlash 的段版本元数据用于追踪，不用于自动刷新。
- **GPU 安装滞后**：CPU 完成一次更新后，还要生成 snapshot、H2D 和派生 buffer，
  event ready 后 GPU 才切换；CPU 已完成不代表 GPU 已使用。finish 时可能有一个
  已训练但未用于生成的版本。

DFlash 还有必须严格消除的一类错误：**当前参数与其派生权重副本不同版本**。
因此同一 slot 的 `_fused_kv_weight`、`_k_norm_weights` 在 H2D 流上准备完成后才
允许 commit，绑定时更新 `_hidden_norm_weight` 别名，reset 也同步恢复。
这一问题不同于双方主动接受的混合版本历史 KV，不能用“允许旧 KV”来放宽。

## 工程优化哪些共用，哪些只在 DFlash

| 项目 | EAGLE3 | DFlash |
| --- | --- | --- |
| pinned 权重 snapshot 复用 | 共用新 pipeline，已生效 | 已生效 |
| 独立 H2D / slot 使用事件保护 | 共用 installer，已生效 | 已生效，并准备专有派生 buffer |
| 按角色设置线程数/affinity | 支持，未配置时不改变原行为 | 支持，实验使用训练 16 核、推理/采集各 1 核 |
| D2H staging 预分配复用 | 原 packet 路径仍逐 tensor 分配 CPU pinned destination，allocator 可能复用底层内存 | 专有 `DFlashCaptureBuffers` 启动预分配，由串行采集线程复用 |
| online draft CUDA Graph | `explicit_input_embeds=True` 导致 draft eager | 双 slot 各有一套 query FULL graph，当前活动 slot 选择对应 graph |
| target CUDA Graph | 可独立开启 | 可独立开启 |
| 默认 fused AdamW | False，可显式开启 | worker 默认 True |
| FP32 master weights | 共享 trainer 提供，可选，默认 False | 同左；本次消融更慢，没有选为最优 |
| 最近 N block / 衰减 KL | 未接入 | 已接入 |

EAGLE3 的 eager 依据是 `EagleSpeculator.explicit_input_embeds` 及
`AutoRegressiveSpeculator` 的 `need_eager=is_profile or self.explicit_input_embeds`。
不能因为 target 开了 FULL graph，或两条链路都有双权重 slot，就认为 EAGLE3
在线 draft 也已获得 DFlash 的双 graph 优化。

默认与实验配置也要区分：两者生产默认学习率仍为 **1e-5**、weight decay 0、
`check_gradients=True`、`master_weights=False`。DFlash 长请求候选显式使用
**1e-4**、N=16、fused AdamW、`check_gradients=False`、16 核训练及独立绑核；
loss 有限值检查保留。可选 glibc 参数和 `OMP_WAIT_POLICY=PASSIVE` 通过实验启动
环境设置，不是 runtime 自动修改的全局默认。

loss/MLP `torch.compile`、FP32 master weights、foreach/普通 AdamW 都做过对照，
没有把所有开关一概启用。当前采用的组合见阶段 11/12；“都支持”不代表“都更快”。

## 实验已经说明了什么

- [阶段 5](stage5_eagle_cpu_diagnosis.md) 的短请求 EAGLE3 对照中，online 也有开销：
  baseline 约 139.2、online 113.3 token/s。它与 DFlash 的 K 值、模型结构不同，
  不能拿它直接判定两种方法谁更快。
- [阶段 10](stage10_frequent_updates.md) 的 DFlash 全量/最近 16 block 对照中，
  四题 GPU 安装次数从 15 增到 49，完整请求吞吐从 182.47 增到 272.39 token/s。
  主要收益是缩短训练与结束等待，生成吞吐仍低于无训练 baseline。
- [阶段 12](stage12_long_request_validation.md) 的前四题、8192 最大输出实验中，
  高学习率使接受长度提高约 2.1%–2.2%，生成吞吐仍比原始 DFlash 低约 0.7%–1.0%。
- [阶段 13](stage13_decode_16k.md) 的第 15 题自然生成到总上下文 16K，两次重复
  得到纯 decode 173.45→201.97 token/s，增加 16.44%；最后 4K 段增加 39.76%。
  每次 GPU 安装 64 次，输出一致；仍是单题截断结果，不是完整准确率或普遍加速证明。

历史 EAGLE3 较容易及时完成更新，与其 1 层、较小 head、可裁掉无监督 query，
以及 detached context 的计算方式相符。DFlash 的全量窗口原来容易出现“训练越慢，
窗口越大，下一次更慢”的反馈；最近 16 block 主要缓解了这个问题。不过它仍有
5 层完整 block 和可微新增 context，长历史成本并未消失。

此前阶段 12 的完整生产回归为 131 passed；阶段 13 的脚本检查和实际 GPU 对照
均已记录。本次仅复核这些证据，不把历史通过记录写成本日重新测试的结果。
后续 CPU GQA/历史缓存复制、有效 teacher 行传输、GPU 小 batch GEMM 是阶段 13
提出的待验证方向，目前均不能算作已经实现的改动。
