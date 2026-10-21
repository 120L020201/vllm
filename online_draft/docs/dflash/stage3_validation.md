# 阶段 3：真实模型验证、性能与交付边界

## 环境与复现

日期 2026-10-15；worktree `/home/lixiang/work/myvllm-eagle3-dflash`；分支
`eagle3&dflash`，基线 commit `3979838d2d`。代码未提交。

硬件为 Xeon w7-3445（20 核、40 线程）和 RTX PRO 6000 Blackwell 96 GB。
PyTorch 2.13.0+cu130；target `/data/models/Qwen3-8B`，draft
`/data/models/Qwen3-8B-DFlash-b16`。使用 BF16、FlashInfer、K=15、max_model_len=4096、
单请求、关闭 prefix caching，学习率 1e-5、fused AdamW、冻结共享 embedding/head。
测试进程关闭逐参数 finite-gradient 扫描，有限 loss 检查仍执行；产品默认配置未改。

通过 uv 创建本 worktree 的 `.venv`，重用原 worktree 已安装的运行依赖和匹配的
编译扩展，导入路径确认优先使用当前源码。启动中缺失的本地生成模块/`ninja` 已由
现有环境补齐。Nsight Systems 启动受本机 cuDNN symbol 加载问题阻断，最终使用
PyTorch profiler 的 all-threads/CUDA trace，未将失败的 Nsight 数据作为依据。

## 实现后的训练优化

最初逐 round forward 的 107 轮窗口耗时 91.4 秒。最终 recipe 将窗口完整 query
拼接为一次 forward，仍只执行一次 optimizer step。每个 query 的 attention mask
满足两个条件：target context 的绝对位置小于该 round 的 anchor；query KV 只属于
同一个 round。自己的完整非因果 block 可双向读取，其他 round 的 query 与未来
target H 都不可见。loss 仍只选择 accepted 前缀和首拒绝位置。

小模型以逐 round 可微计算为独立参照，对比 packed loss 和每个可训练参数的 Adam
一阶矩（对应本次 gradient），结果一致。已有旧 KV 保留、新段重投影、完整 masked
后缀参与 forward 的测试继续通过。没有将窗口改为固定大小，也没有丢弃活动请求中
的完整 round。约 110 轮真实窗口现在耗时 7.4 秒，结束等待约 6.7 秒。

## 数值与正确性

`--check-math` 使用真实加载的 vLLM GPU 线性层、RMSNorm、RoPE、MLP、残差及
context fused projection，与 CPU 模型比较。为隔离 cache 调度，诊断中暂时把 GPU
attention backend 换成带固定 context 的 SDPA，完成后恢复；这不是完整 FlashInfer
算子等价测试。正常生成 smoke 则使用实际 FlashInfer 与 paged KV。

诊断还把所有 mutable 参数乘 1.01，经真实 H2D/commit/bind 安装非零版本，逐值
检查 fused KV buffer，并再次比较 logits，最后恢复 CPU/GPU W0。该诊断只在请求
开始前同步，不进入解码路径。

| 指标 | W0 | 非零更新安装后 |
| --- | ---: | ---: |
| context KV 最大相对 RMSE | 0.001435 | 0.001612 |
| logits 相对 RMSE | 0.011955 | 0.012943 |
| logits cosine | 0.999848 | 0.999839 |
| forward KL | 0.000793 | 0.000907 |

两次 1024-token 数学解释请求都与 offline DFlash 的 token IDs 逐个相同，均得到
`5050`。每次 CPU 完成 bootstrap W1 和一个后续窗口 W2；GPU 在请求内安装 W1，
结束后 CPU/GPU version 都回到 0。第二请求重新从 W0 开始，没有沿用前一请求的
optimizer、context 或待安装 snapshot。输出 checksum、各轮版本和 loss 见
[results.json](results.json)。这证明本次配置下的正确性，不能证明长期质量提升。

另外复用 `tests/evals/gsm8k/gsm8k_eval.py` 的数据加载和答案提取，对 test split
前 8 题做 zero-shot/chat、thinking 关闭、greedy、最多 1024 token 的成对评估。
online/offline 均为 **8/8**，8 题生成 token 全部相同。各题只有 62–369 个输出 token，
都在 GPU 安装 W1 前结束，因此该小样本只验证 capture/训练/reset 路径没有破坏
短请求；不能证明训练后权重在 GSM8K 上的准确率，也不能视为全量 benchmark。
安装后的正确性证据来自上面的长请求与非零权重数值比较。

## 传输证据

初版首次 snapshot 的 `cudaHostAlloc` 约 635 ms，另外 capture 线程的动态 pinned
分配累计约 157 ms。最终把 snapshot 和 DFlash D2H staging 都提前分配，最终 trace
中 **cudaHostAlloc 次数为 0**。保留 GPU clone 的所有权；CPU batch 在 staging 复用前
取得独立数据，不依赖 pinned buffer 的后续内容。

最终 target FULL CUDA Graph + online draft eager 的 1024-token trace：

| 操作 | 数据量 | GPU copy 累计时间 | 与推理 kernel 时间重叠 |
| --- | ---: | ---: | ---: |
| W1 H2D，独立 stream 25 | 2.097 GB | 37.486 ms | 83.64% |
| 样本 D2H，独立 stream 41 | 0.973 GB | 19.388 ms | 81.75% |

CPU stage 只发射复制与派生 buffer 更新，耗时 0.573 ms；commit query/切换耗时
0.050 ms。CPU snapshot copy 两次分别约 93.3/86.4 ms，位于训练线程。
上传完成 event 未就绪时继续使用旧版本；没有新增推理 stream 等待上传的依赖。
旧 slot 的最后使用 event 由传输 stream 等待。

重叠比例按 GPU memcpy 区间与推理 stream kernel 区间的交集累计计算；未重叠段
不等于 GPU 因复制而阻塞，但这些数据也不支持“100% 掩藏”或“零带宽竞争”。原始
trace 保存在 `/tmp/dflash-torch-packed/`，精简统计保存在 results.json。

## 同配置吞吐与结束成本

以下成对测量都使用 eager target/draft、4 个 CPU 线程、`OMP_WAIT_POLICY=PASSIVE`、
相同 prompt、seed=17、temperature=0、ignore_eos、1024 输出 token。计时从 generate
开始到输出返回，另行测量随后 finish/reset barrier；初始化和 profiler 不计入吞吐。

| 模式 | 第一次 token/s | 第二次 token/s | 请求结束额外等待 |
| --- | ---: | ---: | ---: |
| offline DFlash | 397.52 | 399.88 | 约毫秒级 |
| online，顺序窗口 | 333.10 | 329.56 | 约 90 秒 |
| online，最终 packed 窗口 | 301.10 | 344.59 | 6.66 / 6.70 秒 |

第一次请求包含 shape/JIT 和分配器冷启动影响；热请求的最终吞吐仍比 baseline 低
约 13.8%。这是整体在线路径的差异，不能全归因于 DMA，也不能因使用独立 stream
就忽略。CPU 训练、capture clone、Python 发射和资源竞争仍可能贡献开销；当前数据
没有把这些因素完全拆开。未证明“在线训练提升总体吞吐”或“对推理完全没有影响”。

finish/reset 保持既有语义：已经开始的窗口完成后再重置，未关闭的尾窗口丢弃。
所以结束等待会影响下一请求的启动。没有为缩短等待而偷偷取消当前训练或改变
epoch 边界。packed mask 随窗口增长，超长请求的内存与延迟仍需后续单独量化。

## 检查与可交付内容

主要回归命令见 [README](README.md)，结果 123 passed；额外使用本地 checkpoint
路径运行 `test_dflash_lookahead.py`，3 passed，共 **126 项**。最初该 suite 的远程
checkpoint 查询阻塞，终止后通过 pytest collection 插件替换成本地模型路径重跑，
未改动原测试文件。14 个 PyTorch `jit.script_method` deprecation warnings 为现有依赖。

改动文件的 pre-commit hooks、ruff、SPDX 检查及 mypy-3.12 通过。real-model
smoke、双请求 reset、W0/非零更新数值检查、target graph 与实际传输时间线均已运行。
运行方法和所有可调项见 README；不需要引入新服务或第二个训练进程。
