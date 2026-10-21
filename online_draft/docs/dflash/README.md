# DFlash 请求内在线训练

当前实现固定传输 **GPU FC 输出**，CPU 使用当前权重构建 detached context KV；
每次 optimizer 更新后，重新投影本窗口新增 context 并追加到历史 KV。
历史 KV 不重算，继续接受混合权重版本。实现复用 EAGLE3 的 trainer、FIFO、
pinned snapshot、weight handoff 和双 slot。

最新技术说明见[阶段 25：固定 FC 采集与增量 KV 重算](stage25_fc_incremental_context.md)。
训练轮数限制与窗口增长的验证见
[阶段 26：全窗口与最后 32/16/8 轮消融](stage26_loss_cap_window_growth.md)。
当前 CPU 分阶段耗时、GPU 轮次时间与新 trace 见
[阶段 27：CPU timeline 与 GPU 推测解码周期](stage27_cpu_timeline.md)。
Attention 前后向耗时比例与 CPU 并行限制见
[阶段 28：attention backward 增长原因](stage28_attention_backward_diagnosis.md)。
CPU backward 的 query 分块、历史 KV 梯度裁剪与完整系统实验见
[阶段 29：并行 attention backward](stage29_parallel_attention_backward.md)。
原始 DFlash 与当前方案的长输出 AIME 逐轮接受长度及 decode 吞吐见
[阶段 30：长输出接受长度对比](stage30_long_aime_acceptance.md)。
无持续重复循环、自然答完的长输出主对照见
[阶段 31：AIME 第 2 题重复验证](stage31_nonrepeating_long_aime.md)。
排除第 2、15 题后，新题筛选及自然输出 30K token 的对照见
[阶段 32：新的超长 AIME 样本](stage32_new_long_aime_selection.md)。

## 数据与梯度

- GPU capture 位于 `combine_hidden_states` 之后、GPU context KV 构建之前。
  Qwen3-8B 每个位置仅传 4096 维 BF16 FC 输出；teacher logits 和位置元数据另行传输。
- CPU batch 使用 `prompt_fc` 和 `confirmed_fc`；拒绝位置以后的 context 行不入历史。
  loss 只选择已确认预测及首拒绝位置，完整非因果 query block 仍参与 forward。
- FC、hidden norm、target embedding 和 lm_head 冻结，不进入 optimizer 或 H2D snapshot。
  用户已有的 `frozen_parameter_prefixes` 配置继续叠加生效。
- `project_context` 固定在 no-grad 下执行；五层 attention 的共享 K/V 和 k_norm
  仍通过 query forward/backward 更新。
- 一次 CPU 更新先用更新前权重构建本窗口所需的临时 context，再执行 forward、
  位置衰减 KL、backward 和 optimizer。随后用更新后权重重算本窗口全部确认位置的
  context KV；bootstrap 同时包含 prompt。已有历史 KV 值与版本保持不变。
- 默认最近 16 个完整 block 参与 loss，所有确认位置仍连续进入历史；
  `max_train_rounds=0` 表示整个窗口参与 loss。KL 温度为 1，位置权重为
  `exp(-offset / 7)`，每个 block 从第一个预测位置重新计数，按权重和归一化。

无 `context_representation`、`context_grad` 配置或对应 CLI 开关。
DFlash 仅保留这条 FC 路径，EAGLE3 的训练策略保持原样。

## 异步执行与版本

Capture 代码按模型集中在 `runtime/capture/`：

- [`capture/dflash.py`](../../src/online_draft/runtime/capture/dflash.py) 包含
  DFlash packet、pinned staging、GPU adapter 和 CPU batch 构造器。
- [`capture/eagle3.py`](../../src/online_draft/runtime/capture/eagle3.py) 包含
  EAGLE3 packet、GPU adapter 和 CPU batch 构造器。
- [`contracts.py`](../../src/online_draft/runtime/contracts.py) 定义通用
  `CapturePacket` 协议与 `PreparedRound`，不依赖模型专用 batch。
- [`transport.py`](../../src/online_draft/runtime/transport.py) 管理通用 FIFO、
  D2H stream、CUDA event、finish barrier 和错误传播。

Adapter 持有推理线程的 proposal/teacher 状态；batch 构造器只在 capture 线程
访问 prompt 与连续历史。Packet 的 `to_cpu()` 显式描述各模型的复制布局，由
transport 等待生产事件、发起复制并等待完成后调用 batch 构造器。提交给训练
线程的 batch 不引用下一次 D2H 会覆盖的 staging 内容。

推理线程提交 FC 快照与 ready event，capture 线程在独立 D2H stream 搬运。
训练线程发布 pinned snapshot；推理线程沿用 EAGLE3 的 handoff，在私有 stream
提交 H2D，并通过 event query 安装完成的权重，不等待 CPU 训练。
GPU 安装不会重算其历史 KV；后续新增段使用已安装的新权重。

每个请求从 W0 开始，第一个完整 round 连同 prompt 单独更新。此后只在 GPU 真正
安装新版本时关闭 epoch；本轮 packet 先入队，再入 epoch 标记。每个 proposal 保留
`source_weight_version`，CPU cache segments 记录该段更新后的 CPU 投影版本。
finish 有序处理已提交任务，清理尾部并将 CPU、GPU、optimizer 与 cache 重置。

## 运行

支持单 GPU、单活动请求、Qwen3-8B、DFlash-b16、非因果 full attention；
关闭 prefix caching。模型默认从 `/data/models` 加载。
CPU attention backward 使用独立 C++ extension，复用 PyTorch forward 和 BRGEMM。
需要 C++ 编译器、Ninja；首次构建在 CPU 模型初始化时完成，不进入 decode。
当前 BF16 算子在本机 AVX512/AMX CPU 和 PyTorch 2.13 上验证，详见阶段 29。

```bash
export PATH="$PWD/.venv/bin:$PATH"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE

.venv/bin/python online_draft/scripts/smoke_async_dflash.py \
  --online --target-graphs --train-threads 16 --cpu-isolation \
  --grouped-query-attention --selective-last-layer \
  --max-tokens 1024 --requests 2 --output /tmp/dflash-online.json
```

本机有 20 个物理核，示例隔离推理至 18、采集至 19、训练至 0–15。
换机器需按实际 topology 配置 `additional_config.online_draft.cpu_threads`
和 `cpu_affinity`。脚本的 `--cpu-isolation` 仅对应此工作站。

`--target-graphs` 开启 target 与两套 draft slot 的 CUDA Graph。
`--check-math` 在请求前验证 CPU/GPU 数学及一次非零 slot 更新，之后恢复 W0；
该诊断会同步 GPU，不属于推理路径。`--queue-metrics` 记录队列长度和驻留时间。
`--profile-dir` 配合 `--profile-delay-iterations`、`--profile-max-iterations`
可截取 early/last PyTorch trace；`summarize_dflash_trace.py` 用于解析时间线。
profile 请求的吞吐不作为普通性能结果。

`--aime-questions 2 --context-tokens 16384 --max-model-len 18432`
使用本地 AIME 数据，自然生成至 EOS 或总上下文上限；`--decode-timeline` 记录分段指标。
长请求可沿用 `--learning-rate 2e-4`；默认仍为 `1e-5`。
`--gsm8k-questions` 复用现有 GSM8K 回归工具。去掉 `--online` 可验证原始推理。
保留 fused AdamW、可选 FP32 master weights、GQA 与 selective last layer 配置。

实际应用需设置 `VLLM_USE_V2_MODEL_RUNNER=1`、`VLLM_ONLINE_DRAFT_TRAIN=1`，
并使用 `speculative_config.method="dflash"`。关闭 online 时不创建 CPU 模型、
训练线程或 capture buffer。运行脚本使用可信 callable RPC，因此只在其本地测试
进程中开启 pickle 序列化。

## 验证

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m pytest \
  online_draft/tests \
  tests/v1/spec_decode/test_eagle_weight_slots.py \
  tests/v1/worker/test_gpu_worker_online_draft.py \
  tests/v1/spec_decode/test_dflash_prepare_inputs.py \
  tests/v1/spec_decode/test_dflash_causality.py -q
```

## 历史技术报告

以下报告保留当时的设计、数据和结论。阶段 25 已按要求删除全部 DFlash 实验目录、
原始 trace、JSON 汇总、生成图和专用消融脚本；历史报告中的产物路径与消融命令
不再可用，当前接口以本文和阶段 25 为准。历史性能数字不代表本次清理后重新测量。

- [改动总览与 EAGLE3 在线训练策略对比](change_summary_eagle3_comparison.md)
- [阶段 0：设计与边界](stage0_design.md)
- [阶段 1：CPU 模型与训练](stage1_cpu_training.md)
- [阶段 2：Capture 与安装](stage2_capture_and_install.md)
- [阶段 3：模型验证、性能与限制](stage3_validation.md)
- [阶段 4：采集、CPU 训练和权重安装消融](stage4_ablation.md)
- [阶段 5：EAGLE3 对照与 CPU 瓶颈](stage5_eagle_cpu_diagnosis.md)
- [阶段 6：双 slot graph 与 CPU 调优](stage6_optimization.md)
- [阶段 7：真实长请求与 attention 分组](stage7_long_windows.md)
- [阶段 8：最终实测与验证](stage8_final_validation.md)
- [阶段 9：真实样本更新审计](stage9_update_audit.md)
- [阶段 10：位置衰减 KL 与增加更新次数](stage10_frequent_updates.md)
- [阶段 11：CPU PyTorch 优化](stage11_cpu_optimizations.md)
- [阶段 12：长请求效果与频繁传输验证](stage12_long_request_validation.md)
- [阶段 13：自然生成到 16K 的纯 decode 与 trace](stage13_decode_16k.md)
- [阶段 14：context KV 梯度消融](stage14_context_gradient_ablation.md)
- [阶段 15：关闭 context grad 的 early / last CPU GPU trace](stage15_nograd_early_last_trace.md)
- [阶段 16：CPU 训练加速候选与 GQA 微基准](stage16_cpu_training_speed_options.md)
- [阶段 17：GQA 重排接入与完整 CPU 训练基准](stage17_cpu_gqa_integration.md)
- [阶段 18：完整系统优化、四题验证与 early/last trace](stage18_system_optimization.md)
- [阶段 19：移除 XQA decode 的冗余 CPU 长度回读](stage19_metadata_sync.md)
- [阶段 20：Capture、数据传输与推理线程重叠](stage20_transfer_overlap.md)
- [阶段 21：统一使用 EAGLE3 权重交接方案](stage21_eagle_weight_handoff.md)
- [阶段 22：当前方案性能与 early/last overlap](stage22_eagle_handoff_overlap.md)
- [阶段 23：context 采集格式与版本实现](stage23_context_payload_implementation.md)
- [阶段 24：hidden、FC 输出、KV 完整系统消融](stage24_context_payload_results.md)
