# 阶段 11：CPU PyTorch 优化实测

2026-10-16，Xeon w7-3445、20 物理核/40 线程、AMX BF16，PyTorch 2.13 + CUDA 13。
此阶段仅运行 CPU 实验，GPU 请求实验暂停，避免混入资源竞争。
结果在 [frequent_update_results.json](frequent_update_results.json) 的 `cpu` 字段；
原始日志、profile/trace 在 `/tmp/dflash-frequent-updates/cpu-*`。

## 已启用与对照

当前 BF16 Linear 使用本机 PyTorch/oneDNN/MKL 路径，attention 使用 CPU Flash SDPA；
QKV 和 gate/up 已打包，LM head 只处理有效监督行。AdamW 使用 fused 实现，
`zero_grad(set_to_none=True)`；无 profiler/anomaly detection 的运行作为吞吐基准。
训练使用 16 核并与推理/采集隔离，后台线程只改 OpenMP/MKL 局部配置；
`OMP_WAIT_POLICY=PASSIVE` 减少空转，避免全局 `torch.set_num_threads` 干扰推理。

[PyTorch 调优指南](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
支持按算子融合、梯度存储、CPU 线程/亲和性和分配器分别优化。这里分别比较
fused/foreach/普通 AdamW、8/16 线程、loss/MLP 的 `torch.compile` 和主权重精度。
fused 与 foreach 是互斥实现；CUDA TF32/cuDNN 或卷积 channels-last 不适用于
这个 CPU Transformer 训练链路，不能把无关开关打开后算作有效优化。

## 实验契约

加载完整真实 checkpoint，冻结 target embedding/head。2048-token 合成 prompt，
bootstrap 1 轮，随后 96 轮连续 history，每轮 3 个确认 token，只选最近 16 个完整
query block。位置衰减 KL，gamma=7。固定输入/cache 重复更新四次，第一遍单列，
后三遍的中位数为热时间；用于计算成本，不评价学习能力。

所有变体相同 seed、参数和输入。初次 fused/foreach/loop 有 profile 的结果用于归因；
表格使用另外补跑的无 profiler 对照，避免把 profiler 开销误算成编译收益。
CPU 独立脚本使用 torch.set_num_threads；线上仍是按角色的局部线程与亲和性配置。

| 方案 | bootstrap s | 热窗口中位数 s | 进程峰值 RSS GiB |
| --- | ---: | ---: | ---: |
| fused AdamW，16 线程 | 1.390 | 0.808 | 14.09 |
| foreach AdamW，16 线程 | 1.786 | 1.182 | 15.99 |
| 普通 AdamW，16 线程 | 1.889 | 1.326 | 14.48 |
| fused，8 线程 | 1.832 | 1.214 | 14.04 |
| fused + compile loss | 5.789 | 0.800 | 14.17 |
| fused + compile MLP | 5.391 | 0.830 | 14.15 |
| fused + FP32 主权重 | 2.080 | 1.317 | 25.82 |
| fused + glibc 参数 | 1.476 | 0.637 | 14.75 |

loss/MLP 使用 `torch.compile(dynamic=True, fullgraph=True)`；首次调用的 bootstrap
包含当前缓存状态下编译/加载开销，不声称是清空全部编译缓存后的冷编译时间。
loss 热时间仅快约 1%，MLP 反而略慢，且融合改变部分 BF16 舍入；不加入默认线上路径。
若将来启用编译，必须启动阶段覆盖 shape=1 和多行动态形状，不能把首次编译放进请求。

## Profile 与分配器

带 profile 的四次 fused 窗口累计约 3.49 秒：矩阵乘占 self CPU 39.8%，
fused AdamW 17.1%，zero_grad 9.7%，CPU Flash Attention backward 7.5%。
forward / loss / backward / 新 KV 投影平均约 157 / 13 / 441 / 22 ms。
loss 自身占比很小，说明仅编译 KL 无法解决主要成本。

`set_to_none=True` 已省去逐元素清零，但大梯度释放/再分配仍耗时。采用
`MALLOC_MMAP_THRESHOLD_=1073741824 MALLOC_TRIM_THRESHOLD_=-1`，让本次实验中
小于 1 GiB 的普通 CPU 分配更多复用 glibc heap，并避免频繁 trim。热窗口约
0.808→0.637 秒（-21.2%），本微基准峰值 RSS 增约 0.66 GiB。
代价是释放的普通 CPU 内存更倾向保留在进程内，实际服务 RSS 要按请求长度检查。
两个环境变量应在启动进程前设置；不是 GPU allocator、pinned buffer 或显存开关。

真实四题的 16-block 对照中，安装次数 49→56，但完整请求吞吐 272.39→271.62，
基本持平。更快的更新也会改变安装时刻、窗口和 proposal，不能假定所有请求都
单调更快；分配器参数作为可选调优，继续在长请求检验，不作为隐式进程全局修改。

FP32 主权重增加约 11.7 GiB 峰值 RSS，热窗口变为约 1.63 倍。它用于保留精度
的对照，当前不是吞吐默认配置。线上默认仍使用 BF16 + fused AdamW，N=16。

## 复现

```bash
OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OMP_WAIT_POLICY=PASSIVE \
.venv/bin/python online_draft/scripts/benchmark_dflash_cpu.py \
  --threads 16 --rounds 96 --prompt-tokens 2048 --confirmed-tokens 3 \
  --max-train-rounds 16 --repeat 4 --output /tmp/cpu-fused.json
```

分别加 `--optimizer foreach/loop`、`--compile-loss`、`--compile-mlp`、
`--master-weights`，或换 `--threads 8`。分配器变体在同命令前加上述两个环境变量。
使用 `--profile` 时另存诊断结果，不能与无 profiler 时间直接比较。

当前源码完整回归 131 passed，14 条原有 JIT 弃用 warning，pre-commit 通过。
长请求与频繁安装的传输验证见阶段 12。
