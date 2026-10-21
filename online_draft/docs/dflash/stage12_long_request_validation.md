# 阶段 12：频繁更新的长请求效果与传输验证

2026-10-16，`eagle3&dflash`，base `3979838d2d`，未提交工作树。
阶段 10/11 已验证减少 query block、fused AdamW 和 CPU 分配策略；本阶段确认
8192-token 请求中是否真正安装更多更新、是否改善接受长度，以及传输是否阻塞推理。

## 配置与选择

Qwen3-8B / Qwen3-8B-DFlash-b16，AIME 2026 前四题，thinking、greedy seed=17，
最大输出 8192、上下文 12288。单 GPU/单活动请求，target/draft FULL graph，
训练核 0–15、推理核 18、采集核 19，OMP/MKL 局部线程 16/1/1，PASSIVE 等待。
所有线上候选均用 gamma=7 的位置衰减 KL、最近 16 个完整 block、BF16 参数和
moments、fused AdamW、冻结 target embedding/head。额外冻结前缀配置仍生效。
在线候选均在启动前设置两个阶段 11 的 glibc 参数。各实验串行，无 profiler；
最后另跑传输 profile，它的时间不计入吞吐。

分别运行 baseline、lr=1e-5、lr=1e-4，并重新启动独立进程重复 lr=1e-4。
没有为了重复实验固定异步安装的轮次，允许真实 CPU/GPU 时间差改变训练窗口。

当前通用默认保留 N=16、lr=1e-5、FP32 主权重关闭；DFlash 默认启用 fused。
本组长请求的实测较佳配置是 N=16、lr=1e-4、16 核、可选 glibc 参数。
短请求上高学习率未显示相同优势，使用显式学习率选项，不按长度添加未经验证的自动切换。

## 长请求实测

每组输出 token 数为 3521/8192/8192/8192，合计 28,097。全部候选、两次高学习率
运行均与本阶段 baseline 逐 token 一致。第一题正常结束且答案 277 正确，其余
达到上限；不能把这四题解释成完整 AIME 准确率评测。

| 方案 | 四题实际 GPU 安装 | 合计生成 s | 合计完整请求 s | 生成 token/s | 完整请求 token/s | 合计接受长度 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 0/0/0/0 | 101.677 | 101.695 | 276.34 | 276.29 | 3.8929 |
| lr=1e-5 | 10/26/33/33 | 105.171 | 109.370 | 267.16 | 256.90 | 3.8821 |
| lr=1e-4 | 11/25/32/32 | 102.403 | 106.588 | 274.38 | 263.60 | 3.9789 |
| lr=1e-4 重复 | 10/25/32/32 | 102.701 | 105.843 | 273.58 | 265.46 | 3.9733 |

合计接受长度使用四题总 accepted / 总 drafts，非逐题平均。较大步长的两次运行
相对 baseline 提高约 2.21% / 2.06%；生成时间增加约 0.71% / 1.01%。
这组请求上出现了重复的接受率收益，但仍未抵消所有 capture/训练/安装及 finish 成本，
完整请求吞吐仍低于不训练约 4.6% / 3.9%。不能宣传在线训练已净加速 baseline。

| 题号 | baseline 接受长度 | lr=1e-5 | lr=1e-4 | lr=1e-4 重复 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 4.5537 | 4.5361 | 4.5244 | 4.5186 |
| 2 | 4.2243 | 4.2070 | 4.2930 | 4.2907 |
| 3 | 3.6333 | 3.6252 | 3.7309 | 3.7190 |
| 4 | 3.6404 | 3.6340 | 3.7591 | 3.7573 |

收益集中于后三个长请求，首题略退化。只在相同四题重复，未使用独立 held-out
题集确认泛化，不把它视为普遍最佳学习率。更新次数只是必要的时序条件之一；
1e-5 已能安装几十次但接受率仍未提高，说明每步有效幅度与训练信号同样重要。

## 与先前全量方案比较

[阶段 8](stage8_final_validation.md) 的相同四题、8192-token 设置中，GPU 只安装
3/4/5/5 次，生成吞吐 269.17、完整请求吞吐 201.38 token/s，结束等待合计约 35 秒。
当前 lr=1e-4 两轮安装合计 100/99 次，约为原 17 次的 5.9 倍；每题结束等待
约 0.4–1.5 秒，合计约 4.17/3.13 秒，完整请求吞吐提高约 31%–32%。
这是损失、采样、分配器和学习率组合的历史对照，不能把全部收益归因于一个开关。
严格同配置的 N=0/8/16/32 采样消融见阶段 10。

当前 lr=1e-4 首轮 CPU 完成 104 次更新、GPU 安装 100 次；通常每题最后一个
已开始的更新只在 finish 阶段完成，没有用于生成。CPU 平均每步约 0.874 秒，
最长约 1.144 秒。完整窗口的 query 计算被上限控制，不再随窗口扩张到几十秒；
所有确认特征仍用于连续历史追加，未重建旧 KV，也未改变用户选择的混合版本近似。

## 频繁安装下的 H2D/D2H

另跑第 1 题、最大 4096 token、lr=1e-5、同 N=16/CPU/allocator 配置，实际安装
11 次、CPU 完成 12 次。启用 profiler 后生成约 11.92 秒，该值不参与性能表格。

| 传输 | 数据量 GB | GPU copy 时间 ms | 与推理 kernel 时间交集 | stream |
| --- | ---: | ---: | ---: | ---: |
| 在线权重 H2D | 23.070 | 413.812 | 96.856% | 25 |
| capture D2H | 4.295 | 109.687 | 86.397% | 45 |

推理 kernel 在 stream 7。11 次 H2D CPU 提交中位数约 0.800 ms、最大 1.080 ms，
这些范围内没有 CUDA synchronize；trace 全局没有 cudaHostAlloc。
D2H 的 778 次 cudaEventSynchronize 发生在后台采集范围，等待 producer event；
不是推理线程在等待 D2H。独立流、预分配 pinned buffer、event.query 安装和
旧 slot 最后使用事件均保留。

重叠比例不是零带宽竞争证明，未与 kernel 相交的复制时间也不等于推理被阻塞。
结果支持频繁安装仍不引入推理路径的显式上传等待，而非承诺所有 DMA 成本完全为零。

原始 trace：`/tmp/dflash-frequent-updates/trace/rank0.1792160110479236988.pt.trace.json.gz`。
统计由 `summarize_dflash_trace.py` 关联 CPU 标注、CUDA correlation ID 和 stream，
先合并区间再计算交集，保存在 `trace-summary.json` 及持久化结果文件。

## 运行与验证

本机长请求候选：

```bash
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE
export MALLOC_MMAP_THRESHOLD_=1073741824 MALLOC_TRIM_THRESHOLD_=-1
.venv/bin/python online_draft/scripts/smoke_async_dflash.py \
  --online --target-graphs --train-threads 16 --cpu-isolation \
  --max-train-rounds 16 --learning-rate 1e-4 \
  --aime-questions 4 --max-tokens 8192 --max-model-len 12288 \
  --output /tmp/dflash-long-online.json
```

生产配置对应 `additional_config.online_draft` 中 `max_train_rounds=16`、
`loss_decay_gamma=7.0`、`learning_rate=1e-4`、`fused_optimizer=True`、
`master_weights=False`，其余 CPU 配置沿 README，性能实验显式关闭全参数梯度扫描。
glibc 参数在进程启动前配置，代价和可选性质见阶段 11。

完整回归命令与 README 相同：131 passed、14 条原有 JIT warning，日志
`/tmp/dflash-frequent-updates/tests.log`。pre-commit 与 mypy-3.10/3.12 通过。
更新审计脚本也用第一份真实 capture 重放两次 N=16、lr=1e-4、FP32 主权重更新；
同窗 KL 8.009→3.638、0.928→0.603，验证新 loss/采样/主权重路径能够计算有效更新。
这只是诊断回放，不是 FP32 线上性能优于 BF16 的证据，结果保存在持久化 JSON。
所有原始请求日志/JSON 在同目录；
[frequent_update_results.json](frequent_update_results.json) 保留去掉文本后的统计、
配置、token hash、逐更新数据、CPU 消融和传输摘要。没有提交或推送 Git。
持久化文件省略随机 request_id，仍保留请求序号、题号、step 与权重版本；
原始 JSON 保留完整字段。
