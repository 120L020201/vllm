# 阶段 21：统一使用 EAGLE3 权重交接方案

## 目的与最终路径

按用户要求，DFlash 直接使用 EAGLE3 已有的权重交接方案，不再提供上传线程选择。
CPU 训练、capture D2H 和 GPU 推理仍异步进行，权重 H2D 由推理线程提交到私有
CUDA stream。

```text
训练线程：optimizer → pinned snapshot → 发布 ready snapshot
推理线程：poll_snapshot → stage 到 inactive slot → event.query
GPU：     推理 stream 持续执行；H2D stream 异步上传
推理线程：上传完成 → commit / bind → close epoch
```

推理线程提交 copy 会有 CPU API 成本，但不会主动等待整批 H2D 完成。
上传尚未完成时 `commit_if_ready()` 返回 None，继续使用当前权重。
实际安装仍是后续训练 epoch 的边界，没有改成固定 N 轮更新。

## 代码修改

| 文件 | 本阶段修改 |
| --- | --- |
| `runtime/async_pipeline.py` | 删除 `on_snapshot` 参数、字段和回调分支；训练完成统一发布 `_ready_snapshot` |
| `vllm/v1/worker/gpu_worker.py` | 删除 DFlash 的 `background_weight_upload` 配置读取及 callback 注入 |
| `runtime/draft_weight_installer.py` | 删除专供并发 stage/commit 使用的 try-lock 分支，恢复同线程提交路径使用的锁作用域；event 仍只 query |
| `scripts/smoke_async_dflash.py` | 删除 CLI 开关、additional config、结果字段及消融中的 callback 切换 |
| 两个现有 runtime 测试文件 | 删除后台 callback、并发 staging 专用的两个测试和相关 import |
| 文档 | README 改为当前命令；阶段 20 标注为历史方案并保留其原始实验记录 |

直接复用现有 `DraftWeightHandoff.advance()`，没有新建 DFlash 专用 handoff，
没有添加替代开关、兼容分支或新上传线程。当前运行源码、测试与公共脚本中已无
`background_weight_upload`、`--background-weight-upload`、`on_snapshot` 或
`snapshot_callback` 引用。历史实验命令和数据仍保留原样，以免改写已有证据。

## 保留的不变量

- DFlash 的四块 capture 数据、pinned staging 和后台 D2H 继续使用。
- XQA decode 不再读取多余 CPU sequence lengths 的优化继续使用。
- 权重上传到 inactive slot；覆盖之前由上传 stream 等待该 slot 的 last-use event。
- DFlash 的 fused KV/norm 派生投影 buffer 随对应 slot 一起准备，完成后才能安装。
- snapshot 在 GPU 安装之前保持稳定；安装通过有序队列关闭 epoch，才允许下次更新。
- request finish 排空必要工作并恢复 CPU/GPU W0；训练方案、冻结配置和混合历史 KV
  约定保持不变。

CPU 训练与 GPU 推理之间仍只有完成事件查询，没有新增等待 CPU 更新或 H2D
完成的推理 barrier。请求结束的 finish/reset 等待仍存在，单独计时。

## 验证

现有 runtime、CPU 模型、capture、slot、worker 和 DFlash 输入/因果性测试共
**154 passed**，其中包括未完成上传只 query、不等待的测试，以及 snapshot 发布、
epoch 边界、派生 buffer 准备和 reset 测试。比阶段 20 少的两项是随功能一起删除的
后台上传专用测试。

修改文件的 pre-commit 与手动 mypy Python 3.12 均通过。还与阶段 20 开始前
保存的源码逐字节比较：共用 pipeline、handoff 和 installer 三个文件完全相同，
本次确实移除了后台上传引入的分支。

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OMP_WAIT_POLICY=PASSIVE \
  taskset -c 16,17 .venv/bin/python -m pytest \
  online_draft/tests \
  tests/v1/spec_decode/test_eagle_weight_slots.py \
  tests/v1/worker/test_gpu_worker_online_draft.py \
  tests/v1/spec_decode/test_dflash_prepare_inputs.py \
  tests/v1/spec_decode/test_dflash_causality.py -q
```

## 真实模型运行

性能运行不启用 profiler，GPU 实验串行。Q15 运行前执行 `--check-math`：
target 参数存储隔离通过；W0 logits relative RMSE=0.01173、cosine=0.999880、
KL=0.000787；一次非零安装后的对应值为 0.01312、0.999862、0.001015。
数值检查结束恢复 W0，再开始正式请求计时。

| 请求 | 输出 token | decode token/s | 平均 round ms | CPU 更新 / GPU 安装 | finish 秒 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q15，总上下文到 16K | 16098 | 225.68 | 14.506 | 69 / 68 | 1.272 |
| AIME Q1 | 3521 | 334.31 | 13.710 | 12 / 11 | 1.270 |
| AIME Q2 | 11108 | 304.61 | 14.221 | 37 / 36 | 1.430 |

三个请求均与阶段 20 不训练基线逐 token 相同，finish 后 CPU/GPU version
均为 0。Q1/Q2 在同一引擎中顺序执行，验证跨请求恢复；两题自然结束，答案
277、62 均正确。Q15 达到长度上限，没有最终 boxed 答案，只用于长请求性能验证。

Q15 本轮平均接受长度为 3.2744；阶段 20 后台方案两次合计为 3.3005。
本轮平均 round 14.506 ms，与阶段 20 前台提交候选的 14.504 ms、后台方案的
14.514 ms 接近。不能把本次 225.68 与此前 227.39 token/s 的差异全部归因于
上传提交线程：异步更新节奏和接受率也会变化。本阶段每个配置只运行一次，
用于真实安装、输出和 reset 回归，没有宣称新的稳定性能提升。

[逐请求结果、命令与源码哈希](eagle_weight_handoff_results.json)。

## 当前运行命令

本机 Qwen3-8B、DFlash-b16、K=15、单 GPU 单请求、FULL target/draft graph，
沿用 context grad 关闭、CPU GQA、最后一层监督行计算、16 训练 round、lr=2e-4、
KL gamma=7 的实验配置：

```bash
export PATH="$PWD/.venv/bin:$PATH"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE
export MALLOC_MMAP_THRESHOLD_=1073741824 MALLOC_TRIM_THRESHOLD_=-1

.venv/bin/python online_draft/scripts/smoke_async_dflash.py \
  --online --target-graphs --train-threads 16 --cpu-isolation \
  --no-context-grad --grouped-query-attention --selective-last-layer \
  --learning-rate 2e-4 --max-train-rounds 16 --loss-decay-gamma 7 \
  --aime-questions 1 --aime-offset 14 --context-tokens 16384 \
  --max-model-len 18432 --decode-timeline --queue-metrics \
  --output /tmp/dflash-eagle-handoff-16k.json
```

服务端无需新增配置项。旧命令中的 `--background-weight-upload` 应删除；
`additional_config.online_draft` 中也不再配置 `background_weight_upload`。

本阶段原始命令、日志、结果和源码哈希保存在
[实验目录](../../results/experiments/dflash_eagle_handoff_20261019/)。阶段 20 的
99% overlap 数值来自当时的后台提交 trace，不能当作本阶段重新测量的数据。
后续[阶段 22](stage22_eagle_handoff_overlap.md)已补采当前方案的 early/last trace，
并完成第二次无 profiler 的 16K 性能测量。
