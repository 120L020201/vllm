# 阶段 8：最终实测、传输与回归

2026-10-16，工作树 `/home/lixiang/work/myvllm-eagle3-dflash`，分支 `eagle3&dflash`，
base commit `3979838d2d`，改动尚未提交。最终选择是双 slot FULL CUDA Graph、
按角色设置 CPU 线程与 affinity、CPU attention 每组 64 轮、学习率 1e-5。
这是本次候选和工作负载中验证过的版本，不声称所有模型、长度和机器的全局最优。

最终完整数据、窗口/版本、数值检查、代码 SHA-256 均保存于
[tuning_results.json](tuning_results.json)。该文件去掉无分析价值的随机 request
ID，保留请求顺序与输出 hash；含完整 token/text 的原始 JSON 在
`/tmp/dflash-tuning/`，执行日志为 `/tmp/dflash-tuning-*.log`。

## 最终代码

- DFlash 两个固定参数 slot 分别捕获 query graph，安装时直接选择对应 graph。
  context 投影参数和派生 fused KV/norm buffer 同版本切换；旧 slot 最后使用事件
  保护覆盖，reset 可在任意活动 slot 恢复 W0。
- CPU inference/capture/training 分别配置线程数及亲和性，配置在线程自身生效，
  避免训练线程调整全局 PyTorch 线程池。未显式配置时保持原行为。
- CPU attention 以 64 个完整 block 分组，避免整窗口跨轮 query 计算及过小分组
  对 context KV 的重复复制。线性层和 MLP 仍批量执行，一个完整窗口一次更新。
- 沿用独立 D2H/H2D stream、启动时 pinned buffer 预分配和原有有序 FIFO。
  没有更改冻结参数、KL、完整非因果 query、混合历史 KV、真实安装分界或 finish。

optimizer 状态复用、GQA query 重排、18 核训练没有获得稳定收益，均未作为最终
实现保留。两种较大学习率 3e-5/1e-4 也未显示稳定优势，保留原有 1e-5。

## 1024 token：四模式交错消融

最终版本使用 8 核训练、inference/capture 各 1 核；FULL target/draft graph。
模型、prompt、greedy seed、K=15 等与阶段 6 相同。36 次真实请求，首组四模式预热，
每模式留下 8 次；不启用 profiler。baseline 保留空闲 CPU 模型/双 slot，关闭
capture，是同进程的成本对照。

| 模式 | 平均生成 ms | 生成 token/s | 结束额外等待 s | 完整请求 token/s |
| --- | ---: | ---: | ---: | ---: |
| baseline | 2408.37 | 425.18 | 0.0009 | 424.54 |
| 仅采集 | 2430.15 | 421.37 | 0.2819 | 377.16 |
| 加训练，不安装 | 2459.68 | 416.31 | 1.2806 | 273.54 |
| 完整在线 | 2466.43 | 415.18 | 1.4474 | 261.42 |

生成期间的增量依次是 **21.78 / 29.53 / 6.74 ms**，合计 **58.06 ms**。最后
一项包括安装、新权重使用以及验证路径变化，不是独立 DMA 延迟。在线生成吞吐
比本次 graph baseline 低 **2.35%**；阶段 4 eager 对照中约低 17.5%。

相对于阶段 4 online，生成吞吐从约 329.7 提升到 415.2 token/s，约 **+26%**；
完整请求吞吐从约 130.7 提升到 261.4 token/s，约 **2 倍**。这是启用 graph、CPU
资源调优及分组的整体效果，不能把不同阶段相减解读成单个改动的纯因果贡献。

本次完整请求指标从 generate 开始计到确认 finish/reset 完成，包含 metrics/RPC
间隔；与仅计 generate 的吞吐分开列出。结束等待仍是实际成本。

36 次输出逐 token 一致；online CPU 完成 W2、GPU 安装 W1，每次恢复 W0。
train 回放相同安装轮次，窗口边界一致、对应 loss 差值为 0。

## 3072 token：较长窗口

16 核训练，其余配置相同。16 次请求，首组预热后，每模式 3 次。

| 模式 | 生成 s | 生成 token/s | 结束额外等待 s | 完整请求 token/s |
| --- | ---: | ---: | ---: | ---: |
| baseline | 4.4094 | 696.69 | 0.0006 | 696.13 |
| 仅采集 | 4.4594 | 688.88 | 0.2746 | 648.54 |
| 加训练，不安装 | 4.5484 | 675.40 | 1.2169 | 532.49 |
| 完整在线 | 4.5494 | 675.26 | 1.4647 | 510.58 |

生成吞吐较 baseline 低约 **3.08%**。所有输出一致，online GPU 安装 W1/W2，CPU
完成 W3。窗口边界回放一致；安装后 proposal/上下文变化使对应 loss 最大差为
0.00947，不能再要求 train 与 online 的训练输入或 loss 完全相同。

该固定长度任务忽略 EOS，长输出包含模型正常回答后的延续；它用于成本回归，
不代表真实推理任务的接受率或质量。真实任务结果见下面的 AIME。

## AIME 长推理与学习率

本地 AIME 2026 前四题，thinking 开启、greedy seed=17，最多 8192 输出 token、
上下文 12288，16 核训练。这里 baseline 为完全关闭 online 的独立进程。

| 题号 | token | baseline 生成 s | 最终 online 生成 s | online 结束等待 s | GPU 版本 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 3521 | 10.664 | 10.957 | 5.561 | 3 |
| 2 | 8192 | 27.361 | 28.109 | 2.378 | 4 |
| 3 | 8192 | 31.852 | 32.692 | 11.573 | 5 |
| 4 | 8192 | 31.781 | 32.624 | 15.619 | 5 |

四题输出分别与 baseline 完全一致。第一题正常结束且答案 277 正确；后三题达到
8192 token 上限，未得到最终正确答案。不能把这组结果包装成完整 AIME 准确率。
它验证了真实长 prompt、多次 slot 重用和更新后的生成，没有证明学习改善。

按总 token/总时间统计，baseline/online 生成吞吐约 **276.39/269.17**，online
完整请求吞吐约 **201.38 token/s**。最终分组相对于阶段 7 初筛的 8 轮分组，使
四题合计结束等待从约 102 秒降至约 35 秒，第三题从约 71.6 秒降至约 11.6 秒。
第四题由于更早安装而启动额外窗口，结束等待增加；不能保证每题都单调加速。

额外用前两题比较 3e-5、1e-4，输出均保持相同。第二题平均接受长度从 baseline
4.224 分别变为 4.222/4.240，第一题则从 4.554 变为 4.530/4.536；方向不一致，
不足以选更大学习率。最终保持 1e-5，尚无可靠的接受率或准确率提升证据。

## GSM8K 回归

test split 前 16 题，zero-shot/chat、thinking 关闭、greedy，最多 1024 token。
baseline/online 均 **15/16**，所有输出逐 token 一致。online 有 4 题实际安装 W1，
其余在首次更新安装前结束；因此比前期全未安装的 8 题检查多了一点更新后覆盖，
仍然是小样本回归。长请求按 16 核配置运行这些短题，结束成本会很明显，不能把
生成吞吐直接当成连续处理大量短请求的总吞吐。

## H2D/D2H 时间线

独立运行最终 FULL graph + 8 核配置的两个 1024-token 请求，启动时所有模型和
在线 pinned staging 已建立，然后开启 PyTorch profiler。以下合并两个请求，
profiler 数据不参与前述吞吐统计。

| 操作 | 数据量 GB | GPU copy 时间 ms | 与推理 kernel 的时间交集 | stream |
| --- | ---: | ---: | ---: | ---: |
| 在线权重 H2D | 4.1945 | 75.064 | 97.13% | 25 |
| 样本 D2H | 1.9413 | 39.131 | 94.29% | 45 |

推理 kernel 位于 stream 7。两次 H2D CPU 提交分别 0.621/0.582 ms，其范围内没有
CUDA synchronize 调用。D2H 的 `cudaEventSynchronize` 在 capture 后台线程，推理
线程没有等它。上传完成与 slot 最后使用仍通过原有 event 协调。

trace 全局出现 1 次 30.629 微秒 `cudaHostAlloc`，位于普通 `aten::to/_to_copy`
调用内，不在在线传输标注范围。不能把最终 trace 写成全局零 pinned allocation；
在线预分配已经去除了前期数百毫秒的动态 pinned 分配成本。

交集比例不是零带宽竞争证明，未重叠的时间也不等于推理被阻塞。证据支持大部分
传输与推理重叠，以及推理不显式等待上传，仍有约 2%–3% 的生成净成本。

原始 trace：
`/tmp/dflash-tuning/final-trace/rank0.1792146164224618641.pt.trace.json.gz`。
新增 `summarize_dflash_trace.py` 按 CPU 标注范围关联 CUDA correlation ID，区分
在线复制与 reset/其他复制，合并区间后计算与推理 stream kernel 的交集。

## 数值和代码检查

最终真实 checkpoint 的 W0 与 1.01 倍非零权重安装检查均通过，CPU/GPU logits
cosine 约 0.99985/0.99984，forward KL 约 0.00079/0.00091；之后恢复 W0。
单元测试额外覆盖连续三次非零安装后的双 graph slot 重用及两个 slot 上的 reset。

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OMP_WAIT_POLICY=PASSIVE \
.venv/bin/python -m pytest online_draft/tests \
  tests/v1/spec_decode/test_eagle_weight_slots.py \
  tests/v1/worker/test_gpu_worker_online_draft.py \
  tests/v1/spec_decode/test_dflash_prepare_inputs.py \
  tests/v1/spec_decode/test_dflash_causality.py -q
```

结果 **128 passed**，14 条原有 PyTorch JIT 弃用 warning，耗时 10.22 秒。
变更文件的 pre-commit hooks、ruff、SPDX、mypy-3.10/3.12 与 `git diff --check`
均通过。详细日志为 `/tmp/dflash-tuning-final-tests.log`、
`/tmp/dflash-tuning-precommit-final.log`、`/tmp/dflash-tuning-mypy.log`。

## 使用与边界

本工作站优先启用 target/draft FULL graph；短请求可用 8 个训练核，较长推理用
16 个，inference/capture 各保留一个独立物理核，设置 `OMP_WAIT_POLICY=PASSIVE`。
启动方法及生产 `additional_config` 见 [README](README.md)。默认冻结和历史 KV
策略均保持用户已经确认的设置。

已经把主要的生成干扰降下来，但单请求 CPU 训练仍会落后 GPU，已关闭的大窗口
训练与 W0 reset 仍然造成结束等待。没有通过丢弃完整 round、取消更新或改变 epoch
边界来隐藏它。未在本阶段验证完整 32K 请求和全量质量数据；现有结果支持当前
工作负载的系统优化与正确性回归，尚不支持在线训练已经获得净学习收益的结论。
