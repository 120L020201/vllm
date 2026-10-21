# 阶段 4：在线路径的同配置消融

## 问题与控制变量

2026-10-16，在 `eagle3&dflash` worktree 上测量离线、仅采集、采集加 CPU 训练但不
安装、完整在线四种模式，拆分前次热请求额外约 411 ms 的来源。模型、prompt、seed、
BF16、FlashInfer、K=15、1024 输出 token、单请求、关闭 prefix cache 均保持阶段 3
的 eager 配置。target 与 draft 都关闭 CUDA Graph，所有模式设置
`OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE`。吞吐实验不启用
profiler；不同进程串行运行，不让 GPU 实验相互竞争。

| 模式 | GPU capture / D2H / CPU 样本准备 | CPU 训练与 snapshot | 请求内上传及安装 |
| --- | --- | --- | --- |
| baseline | 关闭 | 关闭 | 关闭 |
| capture | 开启 | 关闭 | 关闭 |
| train | 开启 | 开启 | 关闭，GPU 始终 W0 |
| online | 开启 | 开启 | 正常异步安装 |

仅采集包含真实 logits/hidden clone、event、队列、D2H、teacher softmax 和 batch
构建，不用假数据替代。为保持相同内存布局和在线初始化路径，后三种模式都创建
CPU model、pinned buffer、双 GPU slot 和空闲后台线程；capture 不向 trainer 提交
round，生成完仍执行原有 reset。因此 capture 的结束时间包含原有恢复 W0 的成本，
不是单独 D2H drain 的成本。

## 保持训练窗口一致

生产路径只有在新权重实际安装后才关闭 epoch。直接关闭安装会令 CPU 只做 bootstrap，
不能和完整在线比较。本实验先记录 online 每个请求的安装轮次，再让 train 在相同
completed-round 边界把 close-epoch 事件放入同一个 FIFO。trainer 线程在 bootstrap
完成后消费并丢弃 snapshot，然后正常执行关闭窗口和下一次更新。没有让推理线程
等待 CPU ready，也没有用固定轮数改动生产策略。

两种训练模式都保留 snapshot copy；train 仅省略请求内 H2D、派生 GPU buffer 更新、
commit 和 bind。逐请求检查两边 start/end step、窗口轮数、CPU version 与 loss；
检查所有模式输出 token 完全相同，capture/train 的 GPU 始终 W0，完整在线实际
安装了 W1，每个请求结束均恢复 CPU/GPU W0。安装后的 draft 接受轮数仍可能变化，
所以 online − train 是整个安装路径与权重使用的净差值，不是孤立 DMA 时间。

消融开关只位于 `smoke_async_dflash.py` 的测试进程 callable RPC 中，不向生产
runtime 添加模式或修改训练数学。`summarize_dflash_ablation.py` 从原始 JSON
校验以上条件，按累计指标差分计算单请求验证轮数，排除每个进程首个请求后汇总。

## 运行方法

在本 worktree 的虚拟环境中运行；`a` 为一个独立测量组，重复组使用不同输出名。

```bash
export PATH="$PWD/.venv/bin:/home/lixiang/work/myvllm-rebuild/.venv/bin:$PATH"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE

.venv/bin/python online_draft/scripts/smoke_async_dflash.py \
  --ablation online --max-tokens 1024 --requests 6 \
  --output /tmp/dflash-ablation-20261016/online-a.json

.venv/bin/python online_draft/scripts/smoke_async_dflash.py \
  --max-tokens 1024 --requests 6 \
  --output /tmp/dflash-ablation-20261016/baseline-a.json

.venv/bin/python online_draft/scripts/smoke_async_dflash.py \
  --ablation capture --max-tokens 1024 --requests 6 \
  --output /tmp/dflash-ablation-20261016/capture-a.json

.venv/bin/python online_draft/scripts/smoke_async_dflash.py \
  --ablation train --max-tokens 1024 --requests 6 \
  --replay-epochs /tmp/dflash-ablation-20261016/online-a.json \
  --output /tmp/dflash-ablation-20261016/train-a.json

.venv/bin/python online_draft/scripts/summarize_dflash_ablation.py \
  /tmp/dflash-ablation-20261016 --blocks a \
  --output /tmp/dflash-ablation-20261016/summary-a.json
```

## 分进程重复实验与测量漂移

先串行完成 A/B 两组。A 每模式 6 次请求，B 每模式 8 次；每个进程首个请求排除，
每模式保留 12 次。A 顺序为 online、baseline、capture、train；B 顺序为 online、
train、capture、baseline。所有请求均从 W0 开始，先等待前一请求 finish 后再继续。

| 模式 | A 组平均生成时间 ms | B 组平均生成时间 ms | 合并平均 ms | 合并标准差 ms |
| --- | ---: | ---: | ---: | ---: |
| baseline | 2645.4 | 2835.9 | 2756.5 | 303.0 |
| capture | 2682.0 | 2654.3 | 2665.8 | 38.1 |
| train | 3149.6 | 3149.2 | 3149.4 | 121.1 |
| online | 3152.6 | 3178.0 | 3167.4 | 142.6 |

后台训练的增量分别为 467.5 / 495.0 ms，方向一致。上传与安装的净差值分别为
3.0 / 28.8 ms，远小于请求波动。baseline 尤其不稳定，单次范围为 2505–3530 ms，
直接合并相减会得出 capture 为 −90.7 ms 的不合理归因。不能把该负值解释为采集
优化了推理，也不能只选较快的 baseline 请求。因此保留全部数据，增加同进程交错
实验控制进程初始化差异和时间漂移。

## 同进程交错实验

只加载一次模型，使用 seed=17 打乱每组四种模式的顺序；为取得真实训练窗口，
约束同组 online 先于 train。共 9 组、36 次请求，第一组四种模式均作预热排除，
留下 8 组、32 次配对测量。每组包含完整的四种模式，模式切换只在 finish 后进行，
开始前断言 CPU/GPU version 都是 0。

该实验的 baseline 关闭 speculator capture hook，GPU 始终 W0。在线 CPU model、
buffer、空闲线程和失败检查仍驻留，且 online draft 的 eager 选项保持开启；因此它
是用于归因的同进程对照，不能冒充完全关闭 online 功能的独立进程测量。target 和
draft 本来就都为 eager，切换不改变图执行方式。baseline 请求不执行在线 reset；
其他模式按原有流程 reset。上一个请求的 finish 等待已计在上一个请求中。

```bash
.venv/bin/python online_draft/scripts/smoke_async_dflash.py \
  --interleaved-blocks 8 --max-tokens 1024 \
  --output /tmp/dflash-ablation-20261016/interleaved.json

.venv/bin/python online_draft/scripts/summarize_dflash_ablation.py \
  /tmp/dflash-ablation-20261016 \
  --interleaved /tmp/dflash-ablation-20261016/interleaved.json \
  --output /tmp/dflash-ablation-20261016/summary-interleaved.json
```

| 模式 | 平均生成时间 ms | 标准差 ms | token/s | 结束额外等待 s |
| --- | ---: | ---: | ---: | ---: |
| baseline | 2561.4 | 33.2 | 399.8 | 0.0012 |
| capture | 2695.7 | 30.0 | 379.9 | 0.2955 |
| train | 3143.3 | 147.3 | 325.8 | 4.4459 |
| online | 3106.3 | 163.2 | 329.7 | 4.7264 |

吞吐使用 `1024 / 平均生成时间`，不把每个请求 token/s 的平均当作整体吞吐。

| 增量 | 均值 ms | 8 组配对差值范围 ms | 可支持的解释 |
| --- | ---: | ---: | --- |
| capture − baseline | +134.3 | +112.7 到 +172.3 | 每组均增加，采集路径有稳定成本 |
| train − capture | +447.6 | +234.5 到 +594.1 | 每组均增加，后台训练是主要增量 |
| online − train | −37.0 | −444.0 到 +335.7 | 方向不稳定，未检出稳定安装增量 |

同进程总增量为 **544.9 ms**，对应 token/s 下降 **17.5%**。这不是把先前单次的
411 ms 精确拆成固定常数；原测量样本少，异步训练的窗口长度和开始时机也会变化。
分进程的两组恰好合并为约 411 ms 总差值，但其 baseline 漂移使分项估计失真。

较可信的尺度是：**采集约 0.13 秒，加入 CPU 训练与 snapshot 再增加约
0.45–0.50 秒；请求内 H2D、派生 buffer、安装及新权重使用的净增量尚不能从噪声中
分离出来。** 不能将 −37 ms 解释为上传加速，或宣称 H2D 零开销。非安装模式的
验证轮数固定为 175；online 平均 174.5，稍有算法路径变化。将每次耗时按验证轮数
归一化后，train 为 17.962 ms/轮，online 为 17.800 ms/轮，仍不能给出稳定安装惩罚。

## 机制判断与下一步

采集差值包含 GPU clone、event/入队、独立 stream D2H，以及 CPU softmax 和 batch
构建，不能将 134 ms 全称为 PCIe 时间。训练差值包含 CPU forward/backward、
optimizer、新增 KV 投影和 snapshot copy，反映这些后台工作对推理调度的整体
干扰。它没有继续区分 CPU 核调度、GIL、内存带宽与频率因素；不能仅凭本实验指定
其中一个为根因。

两种独立实验都显示训练带来的增量约半秒，而 H2D/安装净差值远不稳定。因此下一步
优化优先级应为：先测 CPU 训练资源隔离对生成耗时的影响，再减少 capture 主路径
的 clone/发射和小包成本。不能只继续压缩 H2D copy 时间就预期消除全部吞吐损失。

结束 barrier 仍是单独的大成本。同进程 online 加上结束等待后的吞吐约
**130.7 token/s**。窗口训练已经开始就必须完成，再恢复 W0；本次没有取消更新或
改变 finish 语义。train 模式同样存在约 4.45 秒等待，说明主要结束成本并不依赖
GPU 安装。capture 的约 0.30 秒还包括 CPU 模型恢复和 GPU W0 reset，不能误作
采集队列单独排空时间。

## 验证与交付

两组独立进程 56 次请求，加上同进程 36 次，共 **92 次真实模型请求**；排除各模式
预热后，分进程计时 48 次，同进程计时 32 次。所有输出的 1024 个 token 逐个一致，
checksum 与阶段 3 相同。所有 capture 请求无 optimizer update；所有 train 请求
GPU 始终 W0；所有 online 请求实际安装 W1；训练模式 CPU 均完成 W1/W2，并在
finish 后恢复 CPU/GPU W0。

每个 train/online 请求的训练窗口完全匹配，所有对应 loss 的最大绝对差为 **0**。
同进程热请求第二窗口轮数为 `[97, 90, 92, 99, 88, 102, 96, 102]`，两模式相同。
完整在线仅安装 W1，W2 在结束阶段完成；本次仍没有学习收益的证据。

本阶段只修改 smoke 脚本并增加汇总脚本和报告，未修改生产训练/推理代码。所有数据
契约由汇总脚本直接校验；脚本及文档的 pre-commit hooks 通过。没有增加只镜像
instrumentation 实现的单元测试，也没有把阶段 3 的测试重复运行冒充新结果。

完整逐请求统计和两种实验设计的均值、中位数、标准差、版本与 loss 保存在
[ablation_results.json](ablation_results.json)。原始含 token IDs 的 JSON 位于
`/tmp/dflash-ablation-20261016/`，日志位于 `/tmp/dflash-ablation-*.log`。
