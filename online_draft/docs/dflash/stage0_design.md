# 阶段 0：接入设计与验证边界

日期：2026-10-15。工作目录 `/home/lixiang/work/myvllm-eagle3-dflash`，分支
`eagle3&dflash`，起点 `3979838d2d`。用户授权完成 DFlash 在线训练并逐阶段写报告。

## 已确认算法

Target 为 `/data/models/Qwen3-8B`，draft 为
`/data/models/Qwen3-8B-DFlash-b16`。训练使用真实 verification 的 teacher 分布，
只监督 accepted 前缀与首个 rejected 位置。保留完整 masked query block。
CPU/GPU 保留旧 context KV，新确认段使用各自当前参数投影，接受混合版本。
每请求 W0 bootstrap；后续窗口由 GPU 实际安装划分；完整 round 不丢弃。

## 实现分阶段

1. 可微 CPU 模型、窗口 loss、混合 context cache，验证位置、梯度与新旧 cache。
2. 复用 FIFO、snapshot、双 slot；增加 DFlash packet 和 capture；验证传输生命周期。
3. V2 worker/speculator 接入、真实模型 smoke/reset/评估、传输与吞吐观测。

CPU 模型采用与 GPU 相同的 packed 参数名，启动时从已加载 GPU 权重复制，避免
另写一套 checkpoint 名称转换。冻结共享 embedding/head 的 CPU 副本，训练 draft
自有参数。既有冻结配置继续可用。用户随后确认旧 KV detach、prompt/窗口新增
context 投影参与梯度，KL 温度 1，首拒绝权重沿用 EAGLE3。

## 异步契约

D2H 在 capture 线程独立 CUDA stream 执行，GPU 主线程仅 clone 输入并入队。
H2D 在独立 stream 写 inactive slot；主线程只 query event，未完成继续旧版本。
为 DFlash 派生 KV 权重预建两个 buffer 集合，在传输 stream 更新 inactive 集合，
commit 时只换引用。旧 slot 再使用前建立其最后读取与 H2D 的 stream 依赖。

请求 finish/reset 是允许等待的生命周期 barrier，不能把它计作无阻塞 decode。
以上消除显式传输等待，不保证复制引擎、带宽、CPU 调度没有吞吐影响；必须测量。

## 测试设计（写测试前）

模型模块负责给定 context KV 和完整 masked block 的可微 logits；I/O 是 CPU
tensor 与标准参数；防止错误 RoPE/QK norm、未来泄漏和截断 query；最低成本是小模型
CPU 数值测试，并与独立公式/本地 HF 数学参照比较。

训练模块负责窗口一次更新及 confirmed-only 增量 cache；输入连续 round 与旧
detached KV，输出新 cache/version；防止首拒绝错位、旧历史重写、fc 无梯度；小模型
单元测试覆盖，生命周期复用现有 async suite。

传输模块负责 FIFO 与 buffer 所有权；输入 GPU packet/event，输出稳定 CPU 数据；
防止未就绪读取、旧 slot 覆写、finish 死锁；扩展现有 transport/installer 测试，
必要时 CUDA 测试以 event gate 确定性验证，而不是靠 wall-clock sleep 猜测。

## 环境

使用 uv 创建本 worktree `.venv` 并安装 lint 依赖与 hooks。重用原工作区已安装的
PyTorch/测试依赖和匹配的编译扩展，Python 路径优先指向本 worktree；核对实际导入
路径。此复用是本机验证环境安排，不改变项目的安装依赖声明。
