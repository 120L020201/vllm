# 阶段 25：固定 FC 采集与增量 context 重算

## 目标与最终语义

本阶段按用户要求删除实验分支和实验产物，只保留传输 FC 输出的在线训练实现。
工作目录为 `/home/lixiang/work/myvllm-eagle3-dflash`，分支为 `eagle3&dflash`，
基于 `3979838d2d561d18f0e2082a8b51c6b23d881136` 的未提交工作区继续修改。

GPU 执行 `combine_hidden_states` 得到 FC 输出，经异步 D2H 交给 CPU。
CPU 不再执行 FC，也不直接使用 GPU 生成的 KV；CPU 根据自己的当前权重投影
context KV。context 始终不参与反传，optimizer 更新后用新权重重新投影本窗口
全部新增确认位置，再追加到原来的历史 KV。历史 KV 不重新投影。

这里的“增量”覆盖整个训练窗口的连续确认历史，不仅是最后一轮，也不受最近
16 个 loss block 的选择限制。第一个窗口还包括全部 prompt bootstrap。

## 数据流与版本

设进入窗口时 CPU 权重为 Wc，历史 KV 长度为 H：

1. capture 线程过滤 rejected context 行，按绝对位置连续组装 `prompt_fc`、
   `confirmed_fc`、完整 query block 和 teacher 分布。
2. CPU 用 Wc 投影本窗口 query 所需的新 context，临时拼接到旧 KV，执行 query
   forward。块间 attention mask 隔离未来 context 和其他 block 的 query。
3. 选择最近 `max_train_rounds` 个完整 block，在已确认预测与首拒绝位置计算
   位置衰减 forward KL，执行 backward 与 optimizer，得到 Wc+1。
4. 使用 Wc+1 对本窗口全部新增确认 context 重新执行 hidden norm、每层 K/V
   投影、K norm 与绝对位置 RoPE，追加到旧 KV。该步骤保持 no-grad。
5. 新段记录 `(新历史末尾, Wc+1 版本)`；旧段的值和版本不变。训练线程随后发布
   pinned 权重 snapshot，推理线程按原 EAGLE3 handoff 异步上传并安装。

临时 context 的重投影服务于当前 loss，更新后的重投影服务于后续窗口。
最后一轮刚确认的位置不会进入本轮 query 的条件，但会在 optimizer 之后进入
持久 cache。CPU 与 GPU 的安装进度仍可不同，旧历史仍是混合版本近似。

GPU 安装新版本不会重算其历史 context KV，只影响之后新增位置的 KV。
FC 和 hidden norm 固定后，不再需要采集 GPU KV 投影版本；proposal 的
`source_weight_version` 与 CPU cache 的分段版本仍保留，各自表达不同的来源。

## 梯度与参数集合

CPU 模型构造时冻结 FC、hidden norm、target embedding 与 lm_head。
`project_context(fc_output, positions)` 固定使用 `torch.no_grad()`，只接受 FC
输出，不再提供 raw hidden 或 packed KV 接口。既有冻结前缀配置继续叠加生效。

五层 query forward 仍使用完整 attention/MLP，K/V 投影和 k_norm 通过 query
路径更新。冻结 FC 和 hidden norm 也将它们排除在 optimizer、snapshot、H2D
和双 slot 的可变参数集合之外。CPU 保留 FC 模块以匹配加载后的模型参数结构，
线上 CPU 训练不执行它。

损失、最近窗口选择、fused AdamW、可选 FP32 master weights、GQA、最后一层
选择输出的优化以及 request reset 语义保持现有实现。

## Capture 与搬运

capture 位置前移至 FC 输出生成后、GPU context KV 构建前。此时 FC 已可供搬运，
无需等待 GPU 后面的 context 投影 kernel 完成；ready event 覆盖 FC 快照与位置
元数据。D2H 仍由 capture 线程在独立 stream 上执行，pinned staging 和 CPU batch
的所有权关系保持原样。

本轮 packet 先提交到 FIFO，再提交安装触发的 epoch close，保持原有窗口边界。
H2D 沿用训练线程发布 snapshot、推理线程提交上传和 event query 安装的方式。
本阶段没有新增同步调用或上传线程开关。它仍有 GPU 快照复制、CPU 提交及资源竞争
成本，不能将异步执行等同于推理完全无开销；本阶段未重跑 overlap trace。

Qwen3-8B 的 FC 输出为每位置 4096 维 BF16，即 8 KiB；此前 raw target 特征为
20480 维，即 40 KiB。teacher logits 的传输量不变。当前每次权重 snapshot 为
1,929,472,512 字节。

## 删除范围

- 移除 `context_representation`、`context_grad` 配置及 CLI 开关。
- 移除 raw hidden/直接 KV 分支、KV 打包解包、GPU context 版本字段及 GPU 模型
  为直接 KV 实验增加的返回值。
- 移除仅采集、训练不安装、安装轮次回放、交错消融和训练样本离线审计入口。
- 删除 `audit_dflash_updates.py`、`benchmark_dflash_cpu.py`、
  `summarize_dflash_ablation.py` 与 `benchmark_dflash_cpu_gqa.py`。
- 删除 `online_draft/results` 下全部 7 个实验目录，共 364 个文件，
  1,481,952,353 字节；同时删除文档目录中的 13 份实验 JSON 汇总和 2 张生成图。
- 保留阶段技术报告、基本生成/评测脚本、CPU/GPU 数学校验和 trace 解析工具。
  README 明确说明历史报告中的实验产物路径与消融命令已经失效。

新验证输出写入 `/tmp`，不重新建立仓库实验目录。本阶段没有提交或推送代码。

## 验证

复用原有测试文件，移除已删除策略的参数组合，并加强增量 cache 的行为验证：

- 更新后的 suffix 等于使用更新后模型重新投影的 KV，且与更新前投影不同。
- 旧 cache 前缀逐元素完全不变，版本分段连续且对应 CPU 更新版本。
- context 输入、FC 和 hidden norm 无梯度，共享 K/V 和 k_norm 有 query 梯度。
- loss 只训练最近 block 时，完整确认历史仍保留；packed 与逐块执行的 loss 和
  参数梯度一致。
- 分块 prompt、teacher 行、拒绝位置、pinned staging 复用与 GPU 双 slot/reset
  行为保持正确。

实际执行：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OMP_WAIT_POLICY=PASSIVE \
taskset -c 16,17 .venv/bin/python -m pytest \
  online_draft/tests \
  tests/v1/spec_decode/test_eagle_weight_slots.py \
  tests/v1/worker/test_gpu_worker_online_draft.py \
  tests/v1/spec_decode/test_dflash_prepare_inputs.py \
  tests/v1/spec_decode/test_dflash_causality.py -q
```

结果：143 passed，14 条现有 Torch JIT 弃用提示。测试数量下降是删除了 raw/KV
格式及 context grad 的消融组合。首次运行发现双 slot 测试仍传 raw hidden，按新
FC 输入契约修正后，上述完整测试集通过。

真实模型验证使用 Qwen3-8B + Qwen3-8B-DFlash-b16、BF16、K=15、target/draft
CUDA Graph、16 个 CPU 训练线程、最近 16 个 block、学习率 2e-4，执行启动前
W0/非零安装数学检查和两个连续 AIME 请求，自然生成至 EOS 或总上下文 16K：

```bash
PATH="$PWD/.venv/bin:$PATH" \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE \
MALLOC_MMAP_THRESHOLD_=1073741824 MALLOC_TRIM_THRESHOLD_=-1 \
.venv/bin/python online_draft/scripts/smoke_async_dflash.py \
  --online --target-graphs --train-threads 16 --cpu-isolation \
  --grouped-query-attention --selective-last-layer --learning-rate 2e-4 \
  --max-train-rounds 16 --loss-decay-gamma 7 --max-model-len 18432 \
  --aime-questions 2 --context-tokens 16384 --check-math \
  --output /tmp/dflash-fc-validation.json
```

两个请求均正常 EOS 结束且答案正确，CPU/GPU 在每次请求后均恢复版本 0，
capture round 计数归零，两套 slot 各保留一个 CUDA Graph。

| AIME 题目 | 答案 | 输出 tokens | CPU 更新 | GPU 安装 | 纯 decode token/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2026 第 1 题 | 277 | 3521 | 13 | 12 | 327.76 |
| 2026 第 2 题 | 62 | 11108 | 37 | 36 | 305.51 |

这是一轮功能回归，非重复性能消融；不据此宣称吞吐提升。实际请求分别在达到
16K 上限前自然结束，本次未运行长请求截断位置的 token 一致性对照。
阶段 24 的单次 FC 16K 截断末 token 差异尚未定位，本阶段也未声称解决该差异。

启动数学检查通过，target 与可变 draft slot 的存储独立：

| 检查 | W0 | 非零权重安装后 |
| --- | ---: | ---: |
| context 相对 RMSE | 0.001256 | 0.001199 |
| logits 相对 RMSE | 0.011475 | 0.013062 |
| logits 余弦相似度 | 0.999884 | 0.999859 |
| forward KL | 0.000771 | 0.000870 |

涉及文件的 pre-commit hooks、mypy-3.12 和 `git diff --check` 均通过。
验证日志为 `/tmp/dflash-fc-tests.log`、`/tmp/dflash-fc-validation.log`、
`/tmp/dflash-fc-precommit.log`、`/tmp/dflash-fc-mypy.log`；删除清单位于
`/tmp/dflash-fc-cleanup-manifest.json`。这些是本次变更的临时验证记录，不属于已删除
的仓库实验目录。
