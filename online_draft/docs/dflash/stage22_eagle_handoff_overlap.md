# 阶段 22：统一 EAGLE3 交接后的性能与 overlap 实测

## 结论与范围

本阶段直接测量阶段 21 的当前代码：训练线程发布 snapshot，推理线程提交私有
stream 上的异步 H2D，并在轮次边界 query/commit。没有恢复后台上传开关，
没有修改推理或训练源码。

当前两次无 profiler 的 16K 请求合计 **225.87 token/s**，平均每轮 **14.509 ms**。
新 last trace 的 H2D/D2H overlap 为 **99.21%/98.03%**，推理 stream kernel
时间占比为 **99.32%**，与已有 EAGLE3 late trace 的 99.25%–99.32% 接近。
机制一致，推理连续性接近；D2H overlap 仍稍低，不能表述为完全相同或零开销。

## 实验条件

同一 RTX PRO 6000 Blackwell，Qwen3-8B + DFlash-b16，K=15，单 GPU、单请求，
FULL target/draft graph、FLASHINFER、无 prefix cache。沿用阶段 21 的 CPU
GQA、最后一层监督行计算、context grad 关闭、16 训练 round、lr=2e-4、KL
gamma=7、CPU affinity 和 allocator 设置。未使用任何上传线程选择配置。

新增三个串行 GPU 运行：early trace、无 profiler 的 Q15 重复、last trace。
early 从请求开始截取 400 worker step，总上下文预算 4096；last 延迟 4200 step
后截取 400 step，总上下文预算 16384。last 不是请求结束后的训练收尾 trace。
early/last 的 graph launch 数分别为 799/800，early 包含非 graph 的首次 prefill。

profile 的整个请求 decode 指标分别约 77.02/154.53 token/s，受采集与导出暂停
影响，**均不进入性能表**。性能表使用上一阶段和本阶段各一次不带 profiler 的
运行；源码哈希已核对一致，合计吞吐采用总 token / 总时间。

## 无 profiler 的性能

Q15 prompt=286，输出 16098 token，总上下文达到 16384，属于长度截断性能实验，
没有最终 boxed 答案，不能作为准确率评测。

| 配置 | decode token/s | 平均 round ms | 平均接受长度 |
| --- | ---: | ---: | ---: |
| 阶段 19 修改前原始 DFlash，不训练 | 173.51 | 14.612 | 2.5355 |
| metadata 优化后的原始 DFlash，不训练 | 174.57 | 14.524 | 2.5355 |
| 阶段 20 后台提交，两次合计 | 227.39 | 14.514 | 3.3005 |
| 当前 EAGLE3 交接，第 1 次 | 225.68 | 14.506 | 3.2744 |
| 当前 EAGLE3 交接，第 2 次 | 226.06 | 14.512 | 3.2810 |
| 当前两次合计 | **225.87** | **14.509** | **3.2777** |

当前相对修改前原始 DFlash 提高 **30.18%**，相对后台提交组合约低 **0.67%**。
平均 round 没有变慢，接受长度略低；不能把 token/s 差异直接解释为主线程提交
H2D 的耗时。两次重复不足以证明亚百分比差异具有统计显著性。

两次均完成 69 次 CPU 更新、68 次 GPU 安装；finish 分别为 1.272/0.760 秒。
包含 prefill、finish 和 reset 的完整请求合计 **222.52 token/s**，不包含引擎初始化。
两次输出均与原始基线逐 token 相同，结束后 CPU/GPU 版本恢复到 0。

同一当前代码在阶段 21 完成的两道 AIME 仍作为质量回归：答案 277/62 正确，
输出 3521/11108 token，均与基线相同。两题 decode 合计 291.22 → **311.26**
token/s，约提高 6.88%；完整请求 290.59 → **293.86**，约提高 1.12%。
不能用阶段 20 四题后台方案的 289.46 token/s 冒充当前方案的四题实测。

## 新 trace 的 overlap

| 指标 | 当前 DFlash early | 当前 DFlash last | 历史 EAGLE3 greedy late | 历史 EAGLE3 batched late |
| --- | ---: | ---: | ---: | ---: |
| H2D 与推理 kernel 重叠 | 99.09% | **99.21%** | 99.62% | 99.64% |
| D2H 与推理 kernel 重叠 | 96.57% | **98.03%** | 99.86% | 99.57% |
| 推理 stream kernel 时间占比 | 98.69% | **99.32%** | 99.32% | 99.25% |

EAGLE3 数值来自上一阶段已按同一口径复核的历史 trace，本轮没有重新跑 EAGLE3。
它使用另一道 AIME、K=7 和不同上传参数集合；表格仅比较实际时间线重叠，
不能据此比较两个方法的绝对吞吐或训练质量。相同的 CPU 提交方案也不会保证
不同大小、不同频率的 copy 获得完全相同的百分比。

重叠定义为在线 DMA 区间并集与推理 stream kernel 区间并集的交集 / DMA 时间。
推理 kernel 时间占比为其时间并集 / 首尾跨度，包含该 stream 的准备 kernel，
不是 SM 利用率。early/last 的 H2D 均完整落在已记录推理首尾范围内，本轮没有
旧 trace 中的上传越过尾边界问题，裁剪边界前后的比例相同。

| 当前 DFlash 400 step | early | last |
| --- | ---: | ---: |
| 推理 kernel 首尾跨度 ms | 5524.856 | 6104.935 |
| 推理 kernel 时间并集 ms | 5452.583 | 6063.709 |
| kernel 间隙合计 ms | 72.273 | 41.227 |
| H2D 次数 / 总 GB | 6 / 12.584 | 4 / 8.389 |
| H2D DMA 总 ms | 226.121 | 150.838 |
| D2H 总 GB / DMA 总 ms | 2.213 / 59.220 | 2.207 / 57.405 |
| H2D 未与 kernel 重叠 ms | 2.052 | 1.185 |
| D2H 未与 kernel 重叠 ms | 2.032 | 1.129 |

未重叠 DMA 时间不是推理被阻塞时间，它可以落在原本就存在的 kernel 间隙。
不能将 1-overlap 直接换算成请求性能损失。

## 线程、同步与剩余成本

线程审计直接关联实际 annotation 所在 tid：

| 工作 | early tid | last tid |
| --- | ---: | ---: |
| 推理、H2D stage、commit | 452958 | 454229 |
| CPU train | 453053 | 454338 |
| capture D2H | 453054 | 454339 |

H2D 在 stream 25、D2H 在 stream 45、推理在 stream 7。两份 trace 的 H2D stage
范围均无 CUDA synchronize；其 CPU 中位耗时约 **0.697/0.695 ms**，由推理线程
承担提交成本，但主要被 GPU 已排队的推理工作覆盖。

D2H 每份仍有 400 次 event synchronize，全部位于 capture 线程，用于保护 CPU
读取。late 没有主推理 stream 的 4 字节 pageable sequence-length D2H；early
仅有首次 capture 前的一次 prefill 读取。稳态 last 没有 pinned allocation。

主推理 stream 上的 capture copy/打包仍约 **10.08/10.19 微秒每轮**。
按相同 graph 节点与附近无 H2D 的执行对照，last 在 H2D 期间的 GEMM 累计耗时
高约 2.29%、attention 高约 1.69%。这是局部观察，不是整个请求慢了该比例，也
不能排除时钟、缓存或其他变化。当前结果仍不支持“完全没有资源竞争”的结论。

历史 EAGLE3 每轮 D2H 约 0.873 MB，DFlash 为 5.518 MB，约 6.32 倍。
其中 teacher 分别为 8×32000×2 与 16×151936×2 字节；DFlash 在不减少监督
数据的前提下需要搬运更多数据。单次上传在该历史 EAGLE3 配置为 0.537 GB，
当前 DFlash 为 2.097 GB。数据量和执行节奏不同，是 overlap 比例不必相同的
背景条件，但尚不能据此将剩余差异全部归因于带宽。

## 交付与复现

- [当前 early trace，400 step，16.2 MB](../../results/experiments/dflash_eagle_overlap_20261019/early/early.pt.trace.json.gz)
- [当前 last trace，400 step，15.9 MB](../../results/experiments/dflash_eagle_overlap_20261019/last/last.pt.trace.json.gz)
- [性能、trace、线程审计和命令汇总](eagle_overlap_results.json)
- [原始实验目录](../../results/experiments/dflash_eagle_overlap_20261019/)

启动参数与[阶段 21](stage21_eagle_weight_handoff.md#当前运行命令)相同。
early 增加 `--profile-dir <目录> --profile-max-iterations 400`，并将总上下文
预算设为 4096；last 在 16K 命令上增加 `--profile-dir <目录>`、
`--profile-delay-iterations 4200 --profile-max-iterations 400`。
测吞吐时不启用 profiler，不需要上传线程开关。

两套独立统计实现的 overlap 结果一致；线程归属、无上传同步、输出 token、
request reset 和源码哈希检查通过。本阶段没有修改 runtime，未重复阶段 21
已经通过的 154 项测试。新的诊断脚本通过 Ruff，报告和结果经过 pre-commit 检查。
