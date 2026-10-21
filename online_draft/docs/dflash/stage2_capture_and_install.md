# 阶段 2：Capture、异步传输与权重安装

## 数据闭环

`DFlashCaptureAdapter` 在 target logits 被 sampler 修改前保存 teacher，在
`combine_hidden_states` 之前保存原始多层特征。每个 packet 关联上一轮 proposal
的完整 anchor/mask 输入、绝对位置、draft token IDs、GPU source version，以及
当前 verification 的 target positions/token IDs、num_sampled、num_rejected。
GPU 上只 clone、record event、入队；取标量、裁切有效历史、softmax 和 teacher
行对齐全部在 capture 线程完成。

分块 prefill 逐段累积真实 H；仅最后 prefill 的 proposal 可以启动 bootstrap。
验证 a 个 accepted token 时，新增历史为 anchor + a 个 accepted，监督为
min(a+1,K) 行；全接受时不为 bonus 增加不存在的 draft loss。首拒绝后的 H
不会进入历史。连续位置检查用于识别不支持的 prefix-cache/缺段行为。

## 复用与所有权

EAGLE3 原有 FIFO、epoch scheduler、snapshot、handoff、双 slot 和 trainer 被复用。
通用组件改用 Draft/Capture 命名，保留 EAGLE3 名称别名和 recipe facade，使原有
调用点继续工作。DFlash 独有代码仅处理 packet、模型、训练 recipe 和派生 buffer。

D2H worker 等待 producer event 后，独立 stream 将 packet 复制到 pinned CPU
buffer；等待 copy event 的线程是 capture worker。GPU clone 的引用一直持有到
复制完成。CPU 训练读取已完成的普通 tensor，不读取仍在 DMA 的数据。

阶段 3 时间线发现动态 pinned allocation 干扰后，最终改为启动时预分配：DFlash
使用一组按最大 target chunk/query/vocab 配置的 pinned staging，capture worker
串行复用；交给训练的 confirmed H、位置和 IDs 拥有独立普通 CPU storage，teacher
softmax 同样产生新 tensor。下一轮 D2H 不会覆盖已提交 batch，已有测试覆盖此所有权。

H2D 使用独立 stream 写 inactive slot，snapshot 持有 pinned CPU 源，直到
completion event 就绪才释放。主线程 `commit_if_ready` 只 query，不等待。CPU
每次只有一个待安装 snapshot，安装消息按 FIFO 划定下一训练窗口。

最终 snapshot 同样复用启动时分配的一组 pinned tensor。下一次写入只能发生在
上一份完成 H2D 并 commit 后到达的 close-epoch 事件中，finish/reset 顺序也保证
跨请求复用前旧传输结束。这避免在 decode 期间反复注册约 2.1 GB 的主机内存。

## 派生参数与旧 slot

DFlash 的 `_fused_kv_weight` 和 `_k_norm_weights` 有各自双份预分配 buffer。
H2D stream 在参数复制之后重建 inactive 派生 buffer，随后才 record 完成事件。
commit 同时绑定参数、派生 buffer、`_hidden_norm_weight` alias，且发生在当轮
context 的 fc 投影之前。这样新 context 不会使用新 fc 配旧 KV projection。

旧 slot 退出活动状态时，在推理 stream 记录最后使用事件。它下次作为 inactive
slot 被覆盖前，H2D stream 等待这个事件；推理 stream 不等待传输 stream。
reset 同样恢复 W0 并重建派生 buffer；历史 KV 则保持用户选择的增量混合版本。

## 生命周期与验证

finish 是允许等待的请求边界：capture barrier → CPU 已提交工作完成并 reset →
GPU reset。未关闭窗口与未验证 proposal 丢弃，已进入训练的窗口完成后再 reset。
capture 后台失败保留原异常并继续处理 finish 标记，防止错误发生时 barrier 永久
等待；后续推理边界主动抛出错误，不降级继续旧权重。

扩展 transport/installer suite，覆盖 teacher clone、分块 prompt、三种接受边界、
派生 buffer 在 commit/reset 前更新和未完成 event 只 query。原有 CUDA pinned
D2H 测试也在本机 GPU 上实际运行。

执行 `OMP_NUM_THREADS=2 .venv/bin/python -m pytest online_draft/tests
tests/v1/spec_decode/test_eagle_weight_slots.py
tests/v1/worker/test_gpu_worker_online_draft.py -q`，本阶段结果 **105 passed**。
真实 Qwen3-8B 已完成两次 512-token 请求，日志两次从 version=1 开始安装；请求
后 CPU/GPU version 均回到 0。性能、数值和 target graph 验证在阶段 3 记录。

这里实现的是传输不在 decode 主路径显式等待。clone kernel、内存带宽、CPU
调度与 Python 发射成本仍需用实际时间线和成对 baseline 测量，不能据此声称零开销。
