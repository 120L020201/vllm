# 阶段 14：context KV 梯度消融

2026-10-19，`eagle3&dflash`，base `3979838d2d`，未提交工作树。

## 结论

**将新增 context KV 作为常量可行，本题在线效果接近，但没有证据表明它能大幅
提高 decode 吞吐。** 关闭 context 梯度后，受控 CPU 热训练步中位数减少
**5.91%**，真实请求的平均训练步减少 **4.19%**，GPU 安装从每次 64 次增加到
66–67 次；两次合计 decode 吞吐仅提高 **0.48%**，小于保留梯度组的重复波动。

| 方案，两次合计 | decode token/s | 平均 decode 秒 | 平均接受长度 | GPU 安装次数 | CPU 平均训练步 ms |
| --- | ---: | ---: | ---: | --- | ---: |
| 原始 DFlash，online 关闭 | 173.418 | 92.822 | 2.5355 | 0 / 0 | — |
| 在线，context 参与梯度 | 202.642 | 79.436 | 3.0346 | 64 / 64 | 1095.4 |
| 在线，context 不参与梯度 | 203.614 | 79.057 | 3.0427 | 67 / 66 | 1049.5 |

吞吐按总 token / 总时间合并，接受长度按 `1 + 总 accepted / 总 drafts` 合并，
CPU 训练时间按所有实际更新求均值。关闭梯度组比原始 DFlash 快 **17.41%**，
但这一收益主要来自整个在线方案，不能全部归功于 context detach。

六次运行的 16098 个输出 token 完全一致，均在 16K 处截断、没有最终 boxed
答案。这支持本次生成路径没有变化，不能证明数学准确率或其他题目上的泛化。
保留 `context_grad=True` 默认值，并提供 False 作为节省 CPU 的候选策略；
没有把单题、两次重复的 0.48% 差异当成更优默认值的依据。

## 实验问题与实现

比较 DFlash 新增 context KV 投影参与反传与整体作为常量两种策略。
它们都保留全部历史，不截断 context，不重建旧历史 KV。旧持久 KV 在两组中
始终 detach；变化仅发生在 CPU 训练 forward 中，当前窗口的新 context 投影。

`forward_dflash_window` / `train_dflash_window` 新增 `context_grad=True`。
worker 从 `additional_config.online_draft.context_grad` 传入；两个实验脚本用
`--no-context-grad` 选择 False，并把实际配置写入 JSON。生产默认仍为 True。

False 在 `project_context` 和旧/新 context 拼接外使用 `no_grad` 语义；
query block 的五层 forward、KL、backward 和 optimizer 保持可训练。默认 True
仍遵从外层梯度开关，不会在调用方的 `no_grad` 中强制开启 autograd。
更新后的持久 KV 追加原本就在 `no_grad` 下，两组完全相同。

梯度差异可以写成：

```text
保留 context 梯度：dL/dθ = ∂L/∂θ|context + ∂L/∂C_new · ∂C_new/∂θ
关闭 context 梯度：dL/dθ = ∂L/∂θ|context
```

因此关闭后 `fc` 和 `hidden_norm` 没有其他 query 路径，梯度为 None，实际不更新；
各层共享 K/V 投影和 `k_norm` 仍通过 query 路径学习。没有另改冻结前缀配置，
AdamW 自动跳过无梯度参数；snapshot/slot 仍包含原有全部参数，所以传输字节数
没有随着本消融缩减。这是有意保持传输条件相同。

这与 EAGLE3 的 detached spine 在“将确认历史作为条件”这一点一致，但 EAGLE3
的 anchor 仍经过可训练 FC，因此不能认为两种模型的 FC 梯度行为相同。

## 验证设计

在已有 `test_dflash_training.py` 扩展参数化测试，避免另建模型或测试框架。
训练 recipe 的契约是：返回对齐的监督 logits/teacher，并在更新后返回连续、
无梯度、旧前缀不变的持久 KV。最便宜的验证是 tiny model 单元测试：

- 相同权重、相同完整 block 输入，关闭梯度前后的 logits 与 teacher 相同。
- False 时 prompt/新增特征、FC、hidden norm 无梯度；共享 K/V 和 k_norm
  仍有非零梯度；True 时新增 context 路径有非零梯度。
- 两种模式都只监督已确认前缀和首拒绝位置，跳过 anchor/bonus；bootstrap
  仅在 True 时更新 FC，embedding/head 保持冻结。
- 两种模式都保留未选中 query 的确认历史，旧 KV 逐元素不变，新段按更新后的
  当前权重生成，长度与版本段一致。

相关完整测试集 **136 passed，14 个已有 PyTorch JIT 弃用告警**：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OMP_WAIT_POLICY=PASSIVE \
  .venv/bin/python -m pytest online_draft/tests \
  tests/v1/spec_decode/test_eagle_weight_slots.py \
  tests/v1/worker/test_gpu_worker_online_draft.py \
  tests/v1/spec_decode/test_dflash_prepare_inputs.py \
  tests/v1/spec_decode/test_dflash_causality.py -q
```

## 公平性与计时

复用阶段 13 的 Qwen3-8B / DFlash-b16、单 RTX PRO 6000、16 核 CPU 训练配置。
保留最近 16 个完整 query block、gamma=7 的位置衰减 KL、T=1、lr=1e-4、
fused AdamW、BF16 参数/moments、不使用 FP32 master weights；每个请求从 W0
开始。推理绑 CPU 18，采集绑 CPU 19，训练绑 CPU 0–15。

正式 GPU 实验为 AIME 2026 第 15 题，prompt 286、输出上限 16098，自然增长到
总上下文 16384。两组 online 和原始 DFlash 各两个独立进程，串行交错顺序为
baseline、grad、nograd、nograd、grad、baseline。正式计时不开 profiler。

纯 decode 使用 engine-core 的 last-token 减 first-token 时间戳，分子为
16097 token，排除 prefill 和 finish/reset 等待。原始 DFlash 同一工作树关闭
online；不加载 CPU 模型、不采集或安装权重。仍允许 EOS，不能强迫问题生成到
16K，也不能将截断请求用于完整数学准确率结论。

CPU 微基准另行串行运行：真实 checkpoint、12288 token prompt、96 个连续
round、每轮 3 个确认 token，只训练最后 16 个完整 block。每组重复同一个窗口
5 次，参数逐次更新；首步单列，后 4 步报告热步统计。每组两个独立进程，
补测时反向运行 nograd、grad，共比较每组 8 个热步。输入由同一个 seed=17
生成，teacher 为均匀分布，仅用于比较计算成本，不代表真实训练质量。
两组 bootstrap 都从 W0 开始，但各自按该策略更新后才进入窗口计时，所以后续
权重可以不同；形状、输入及初始 checkpoint 相同。

CPU profile 另用独立进程运行，每组两个窗口，profile 数据仅用来解释开销，
不混入无 profiler 的性能结果。CPU 微基准、GPU 正式请求、CPU profile 之间
不并发，避免实验相互争用核心和内存带宽。

真实在线窗口仍以实际 GPU 安装为界，因此两组的窗口长度、具体训练样本及更新
时机可以不同；固定的是选择策略和超参数，没有强行固定安装时刻。受控 CPU
微基准用于隔离计算成本，真实请求用于衡量整个异步闭环的结果。不把不同真实
窗口上的 KL 数值直接作为两种方法学习质量的比较。

## 逐次结果与长上下文分段

| 正式运行 | decode 秒 | token/s | 平均接受长度 | CPU 更新 / GPU 安装 |
| --- | ---: | ---: | ---: | --- |
| baseline-1 | 92.788 | 173.481 | 2.5355 | 0 / 0 |
| grad-1 | 78.775 | 204.341 | 3.0570 | 65 / 64 |
| nograd-1 | 78.978 | 203.816 | 3.0456 | 68 / 67 |
| nograd-2 | 79.135 | 203.412 | 3.0398 | 67 / 66 |
| grad-2 | 80.096 | 200.972 | 3.0125 | 65 / 64 |
| baseline-2 | 92.856 | 173.355 | 2.5355 | 0 / 0 |

第一对关闭梯度略慢，第二对略快，方向不一致。两次 grad 相差约 1.68%，
所以不能据此声称 0.48% 的平均提升具有统计显著性。
每组最后一次 CPU 更新在 finish 时完成，没有安装给推理；不能把 CPU 次数
当成实际生效的 GPU 次数。两次 grad 总训练 query block 均为 1025，nograd
分别为 1073、1057，以上数量包含 finish 的最后一个窗口。

| 总上下文区间 | 原始 token/s | context 有梯度 | context 无梯度 |
| --- | ---: | ---: | ---: |
| 约 0–4K | 239.150 | 233.787 | 233.986 |
| 约 4–8K | 197.910 | 203.563 | 205.428 |
| 约 8–12K | 153.357 | 185.111 | 188.121 |
| 约 12–16K | 138.921 | 196.054 | 194.461 |

分段边界使用不超过该长度的完整输出 batch，与阶段 13 相同。关闭梯度在最后
一段没有优势；这也不支持“更长上下文一定更适合 detach”的结论。
整段平均每轮 decode 为 14.974 / 14.942 ms，两种在线方案十分接近。

| 真实请求训练统计 | grad-1 / grad-2 | nograd-1 / nograd-2 |
| --- | --- | --- |
| 前 8 次更新均值 ms | 769.2 / 795.8 | 749.6 / 738.7 |
| 后 8 次更新均值 ms | 1614.2 / 1571.7 | 1490.2 / 1563.6 |
| 全部更新均值 ms | 1085.5 / 1105.3 | 1036.4 / 1062.8 |
| finish RPC 等待秒 | 0.996 / 1.057 | 1.390 / 1.685 |

`train_seconds` 包含 recipe 的 forward、loss、backward、optimizer、更新后 KV
追加，排除 snapshot 和传输。finish 等待受请求结束时训练进度影响，关闭梯度
并没有在这两次运行中缩短它；本轮关注的纯 decode 不包含这段等待。

## CPU 微基准与 profile：省在哪里

| 无 profiler，两个独立进程 | context 有梯度 | context 无梯度 |
| --- | ---: | ---: |
| 8 个热步中位数秒 | 1.0143 | 0.9544 |
| 8 个热步平均秒 | 1.0192 | 0.9630 |
| 12K prompt bootstrap 平均秒 | 2.6625 | 1.8840 |
| 进程峰值 RSS GiB，两个进程均值 | 16.321 | 15.133 |

热步中位数减少 5.91%，平均耗时减少 5.51%。峰值 RSS 约少 1.19 GiB，是整个
CPU 进程的峰值，不能当成 GPU 显存或纯 autograd 激活差额。12K bootstrap
收益较大，因为首次需要投影整个 12K prompt；实际 AIME prompt 只有 286 token，
不能将微基准 bootstrap 的 29.24% 耗时下降外推到本题每次在线更新。

独立 CPU profile，每组两个窗口的每步均值：

| 阶段 | context 有梯度 ms | context 无梯度 ms |
| --- | ---: | ---: |
| forward | 260.1 | 262.9 |
| KL loss | 5.0 | 5.0 |
| backward | 642.5 | 589.0 |
| fused AdamW | 146.6 | 135.0 |
| 更新后新增 KV 投影/追加 | 27.7 | 28.3 |
| 其中：CPU Flash Attention backward | 320.9 | 321.7 |

最后一行包含在 backward 内，不能重复相加。profile 会影响执行时间，因此
上表用于拆解原因，正式速度使用前面的无 profiler 数据。

backward 减少约 53.5 ms，optimizer 减少约 11.6 ms。两窗口 `aten::mm` 次数
从 170 降到 148；FC 和 hidden norm 无梯度后也无需 AdamW 更新。
但每组都仍调用 **10 次 CPU Flash Attention backward**，累计耗时分别为
641.7 / 643.4 ms，基本不变；`repeat_interleave` 都为 20 次，累计约 40.5 / 40.7 ms。

代码中的原因是 `DFlashAttention.forward` 仍将 context K/V 与可训练 query
K/V 拼接，再交给 SDPA。整个 K/V 输入仍要求梯度，现有 CPU Flash backward
仍执行这条长历史 attention 的反向，最后才在 autograd 拼接边界丢弃 context
部分的梯度；query 的梯度也仍需要历史 K/V。关闭投影反传不等于绕过 attention
backward，更不等于只对最后一个 block 做局部 attention。

旧历史早已 detach，这次只省掉新增 context 的投影反传，五层 query、长历史
attention 和大部分参数的 optimizer 仍保留。这解释了为何成本只下降几个百分点。
若继续优化该瓶颈，需要单独处理长历史 attention 的反向和 K/V 展开成本，
不能用当前 `no_grad` 开关声称已解决。

本轮未改 capture、snapshot、H2D/D2H stream 或双 slot，上传内容仍相同；也未
重新测量 GPU copy 重叠率，因此不将阶段 13 的 trace 重叠数字当成本轮新测量。
较多安装仍会增加总上传次数，正式 decode 已包含其实际影响。

## 交付与检查

- recipe、worker、CPU benchmark、在线 benchmark 接入一个布尔开关，默认保留
  原行为；复用现有训练、安装、缓存和模型组件，没有新增训练框架。
- 现有训练测试扩展到两种梯度设置，相关完整测试集 136 项通过。
- 五个改动 Python 文件及本轮文档的 pre-commit 检查通过；mypy 3.12 单独检查通过。
- [逐次数据、配置与源码 SHA256](context_gradient_ablation_results.json) 保留
  每次 CPU/GPU 更新、训练耗时、分段性能及输出 token hash。完整输出和 trace
  位于下述原始目录。历史阶段 13 的结果未改写。

## 复现

```bash
export PATH="$PWD/.venv/bin:/home/lixiang/work/myvllm-rebuild/.venv/bin:$PATH"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE
export MALLOC_MMAP_THRESHOLD_=1073741824 MALLOC_TRIM_THRESHOLD_=-1

.venv/bin/python online_draft/scripts/smoke_async_dflash.py \
  --online --target-graphs --train-threads 16 --cpu-isolation \
  --learning-rate 1e-4 --max-train-rounds 16 \
  --aime-questions 1 --aime-offset 14 --context-tokens 16384 \
  --max-model-len 18432 --decode-timeline --output /tmp/grad.json

taskset -c 0-15 .venv/bin/python online_draft/scripts/benchmark_dflash_cpu.py \
  --threads 16 --rounds 96 --prompt-tokens 12288 --confirmed-tokens 3 \
  --max-train-rounds 16 --repeat 5 --output /tmp/cpu-grad.json
```

两条命令分别加 `--no-context-grad` 即为关闭 context 梯度的组；GPU 命令移除
`--online` 为原始 DFlash。CPU profile 将 repeat 改成 2 并加 `--profile`。
原始 JSON、日志和 trace 位于 `/tmp/dflash-context-grad-ablation-20261019/`，
完整运行顺序和参数保存在其中的 `commands.json`。
