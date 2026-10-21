# 阶段 6：双 slot CUDA Graph 与 CPU 资源调优

2026-10-16，基于 `eagle3&dflash` 工作区继续优化，base commit 为 `3979838d2d`。
本报告记录代码与同配置四模式实验。真实题目、传输时间线和最终回归另见阶段 7。
原始 JSON/日志在 `/tmp/dflash-tuning/`、`/tmp/dflash-tuning-*.log`；可持久审查的
逐请求统计在 [tuning_results.json](tuning_results.json)。

## 保留的算法与生命周期

每请求恢复 W0，bootstrap 单独更新，后续 epoch 仍只在 GPU 真正安装时关闭；FIFO
不丢 round，不取消已经启动的训练。训练全部 draft 自有且未被配置冻结的参数，
target embedding/head 冻结。监督已确认前缀和首拒绝位置，forward KL、温度 1、
首拒绝权重 2。完整 anchor + 15 mask 保留非因果 attention。历史 KV detach，
新增段用当前权重投影，继续接受混合版本历史。没有用改变这些策略来缩短耗时。

## 双 slot graph

`DFlashSpeculator` 为两个固定参数 slot 分别建立 `DFlashCudaGraphManager`，启动时
绑定各 slot 捕获一次。请求内按 installer 的 `active_slot` 选择 graph；不重新
capture、不把权重复制回固定的第三份参数、不因等待上传而阻塞推理。

GPU context 投影仍在 query graph 之外，使用当前绑定参数及同 slot 的 fused KV/
norm buffer。H2D stream 同时准备参数与派生 buffer，完成 event.query 返回 true
才切换。旧 slot 的最后使用事件保护下一次覆盖，推理 stream 不等待 H2D。
reset 在结束边界恢复活动 slot 的 W0 和派生 buffer；graph 持有的地址不变。

新增 CUDA 回归连续安装 3 个明显非零版本，并在两个 slot 上执行 reset，分别核对
graph/eager 输出和 W0 输出。真实请求额外记录每个 slot 的 graph 数；3072-token
实验中均为 `[1, 1]`，实际跨越 W1、W2。关闭 online 时仍只有原来的一个 manager。

## CPU 并行度与亲和性

新增小型 `CPUWorkerConfig`，分别在 inference、capture、training 所属线程设置。
使用当前 PyTorch OpenMP/MKL 的线程局部 API，避免在后台调用全局
`torch.set_num_threads()` 改动推理的并行配置。线程数显式配置时要求该运行库支持；
不支持就初始化报错。CPU ID 校验当前允许的 affinity。未提供配置时不改变原行为。

生产配置通过 `additional_config.online_draft` 传入，核心号不写死在 runtime：

```json
{
  "cpu_threads": {"inference": 1, "capture": 1, "training": 8},
  "cpu_affinity": {
    "inference": [18],
    "capture": [19],
    "training": [0, 1, 2, 3, 4, 5, 6, 7]
  }
}
```

上述映射只适用于本机：0–19 为不同物理核，20–39 为它们的 SMT siblings。
16 核训练时将 training 改为 16、核列表改为 0–15。始终设置
`OMP_WAIT_POLICY=PASSIVE`。smoke 的 `--cpu-isolation` 是这台机器的实验快捷配置。
单元测试验证子线程的线程数和 affinity 确实生效，且不改变调用线程设置。

## CPU attention 分块

线性投影、MLP 和 optimizer 仍按整个窗口批量执行，只有 attention 按 8 个完整
query block 分组。每组只取可能可见的历史 context 与组内 query K/V，并保留每轮
独立的 attention mask。因此不会引入跨轮 query attention，也不会删掉拒绝后
未监督位置。梯度测试比较分组大小 1、2、整窗口与逐轮参考，覆盖投影参数梯度。

真实 checkpoint、相同合成输入、8 线程、192 轮，交替两次测量：整窗口 attention
8.246/8.151 秒，分组 7.967/7.954 秒，均值改善约 2.9%。BF16 求和顺序不同使 loss
相差约 0.000098；不能声称逐 bit 等价。该 CPU benchmark 仅用于计算成本，不是
真实 teacher 的学习质量结果。

## 被移除的候选

optimizer 状态预分配复用曾通过 fresh AdamW 的逐参数/动量/reset 数值对照，但
16 核短请求初筛的结束等待从约 2.57 秒变为 3.33 秒。提前完成 bootstrap 会改变
安装时刻和后续窗口，从而启动额外训练。没有获得稳定的完整请求收益，因此最终
删除该选项、缓冲区与专用测试，继续沿用现有 AdamW 状态清理方式。

仅用 CPU 隔离、保留 eager 的 16 核方案，生成约 3.33 秒，未解决主要干扰。
双 graph 即使配默认 4 线程，生成也可达约 2.46 秒，但结束等待约 6.36 秒。因此
graph 是主要生成优化，CPU 配置主要改善训练速度及请求结束等待。

## 1024-token 最终四模式对照

Qwen3-8B/DFlash-b16，BF16、FlashInfer、K=15、greedy seed=17，单请求，无 prefix
cache；target 和 draft 均 FULL graph。8 核隔离训练，使用阶段 4 相同 prompt。
9 组交错顺序，每组四模式，第一组预热排除，保留 8 组配对数据。baseline 保留
空闲 CPU 模型与双 slot，只禁用 capture；它是同进程对照。

| 模式 | 生成 ms | 标准差 ms | token/s | 结束额外等待 s |
| --- | ---: | ---: | ---: | ---: |
| baseline | 2408.64 | 0.59 | 425.14 | 0.001 |
| capture | 2432.36 | 1.83 | 420.99 | 0.283 |
| train，不安装 | 2460.00 | 2.25 | 416.26 | 1.173 |
| online | 2461.30 | 7.05 | 416.04 | 1.309 |

采集增量 23.72 ms，CPU 训练增量 27.63 ms，安装及新权重使用的净差值 1.30 ms；
最后一项配对范围 −14.84 到 +14.22 ms，不能当作孤立 DMA 成本。总体多 52.65 ms，
生成吞吐较本次 graph baseline 低 **2.14%**，较阶段 4 online 的 329.7 token/s
提升约 **26.2%**。额外生成时间从阶段 4 的 544.9 ms 降低约 90.3%，两阶段
baseline 执行配置不同，明确包含启用 graph 的收益。

加入测得结束等待后的吞吐约 271.6 token/s，阶段 4 为 130.7 token/s。
这是显著改善，但仍远低于无训练 baseline；不可只报 416 token/s 隐去结束成本。
旧版统计的结束等待从 metrics 查询后开始，不包括很短的 metrics RPC；新版另加
`end_to_end_s`，覆盖 generate 开始到 reset 确认完成的完整区间。

36 次输出的 1024 个 token 完全相同。online 的 CPU 完成 W2，GPU 在生成中安装
W1；每次都恢复 W0。train 回放同一安装轮次，全部窗口边界一致、对应 loss 差值为
0。平均 draft rounds 为 baseline 175、online 174.875，尚无可靠接受率提升证据。

## 3072-token 压力对照

每种线程配置 4 组、16 请求，排除首组，均使用同一固定 prompt 并忽略 EOS。

| 训练核数 | baseline 生成 s | online 生成 s | online 等待 s | GPU 更新 |
| --- | ---: | ---: | ---: | --- |
| 8 | 4.4080 | 4.5082 | 8.2791 | W1、W2 |
| 16 | 4.4085 | 4.5407 | 1.7998 | W1、W2 |

16 核略多占用生成时间，但生成加结束等待的吞吐约 484.5 token/s，8 核约 240.2。
8 核第三个窗口为 172–202 轮，耗时 7.38–10.31 秒；16 核为 104–124 轮，耗时
3.12–3.73 秒。这是实际安装决定窗口长度的反馈效应，不能只看每秒 CPU 更新数。

所有 32 次输出逐 token 一致，train/online 窗口边界一致；W1 安装后 draft 路径
变化，使后续 teacher/上下文有差别，对应 loss 最大差分别 0.00281/0.00826，不再
要求它们为 0。不能据此把生成差值全部归因于纯传输。

短请求优先 8 核、长请求优先 16 核是当前实测建议，不在 runtime 中猜测请求时长
并自动切换。固定长度测试在 EOS 后仍生成，因此后续真实 AIME 测量是必要补充。

## 复现

```bash
export PATH="$PWD/.venv/bin:/home/lixiang/work/myvllm-rebuild/.venv/bin:$PATH"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE

.venv/bin/python online_draft/scripts/smoke_async_dflash.py \
  --target-graphs --train-threads 8 --cpu-isolation \
  --interleaved-blocks 8 --max-tokens 1024 --check-math \
  --output /tmp/dflash-tuning/final-short8.json

.venv/bin/python online_draft/scripts/smoke_async_dflash.py \
  --target-graphs --train-threads 16 --cpu-isolation \
  --interleaved-blocks 3 --max-tokens 3072 \
  --output /tmp/dflash-tuning/long16.json
```

真实权重 CPU/GPU 数值诊断通过：W0 logits cosine 0.999848、KL 0.000793；人为
1.01 倍非零更新并安装后 cosine 0.999839、KL 0.000907，再恢复 W0。CPU/runtime
针对性测试 33 项通过；最终全集、传输时间线和模型评估汇总在下一阶段。
