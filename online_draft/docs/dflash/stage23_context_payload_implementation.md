# 阶段 23：context 采集格式消融实现

## 目的与实验边界

比较关闭 context grad 后传输原始多层 hidden、FC 输出、GPU context KV 的成本。
三组共用相同的 CPU 训练、安装边界、KL、query forward 和权重上传集合。
本阶段实现候选路径并验证契约；性能和质量结论见后续阶段 24 实测报告。

工作区为 `myvllm-eagle3-dflash`，分支 `eagle3&dflash`，基线提交
`3979838d2d561d18f0e2082a8b51c6b23d881136`。原有未提交改动保留，未创建提交。
实验前源码保存在 `results/experiments/dflash_context_payload_20261019/before/`。

## 三种 payload

当前 checkpoint 的 hidden=4096、5 组 target 特征、5 层 draft、8 KV heads、
head_dim=128，使用 BF16。

| 格式 | 每 token 布局 | 字节 | CPU context 操作 |
| --- | --- | ---: | --- |
| hidden | 5×4096 | 40960 | FC、hidden_norm、每层 K/V、K norm、RoPE |
| fc | 4096 | 8192 | hidden_norm、每层 K/V、K norm、RoPE |
| kv | 5×2×8×128 | 20480 | 解包、追加缓存 |

KV 从 GPU 当轮正常投影的临时张量取得，K 已完成归一化和 RoPE。
没有重新计算 GPU context，没有读取整个 paged cache，也没有传输全历史。
target 位置、拒绝数和采样数沿用既有 metadata；只在 capture worker 上去掉拒绝行。
prompt chunk、已确认历史和首拒绝位置 teacher 对齐保持原契约。

## 梯度与版本

`context_grad=False` 时明确冻结 `model.fc` 和 `model.hidden_norm`，在创建 trainer、
snapshot、GPU slots 之前移出可变参数集合。三组均减少 167780352 字节的上传，
每次 snapshot 从 2097252864 降到 1929472512 字节；没有冻结共享 K/V 和 k_norm。
用户已有的 `frozen_parameter_prefixes` 仍继续应用。

hidden 和 fc 两组沿用 CPU 当前版本投影：forward 前构建窗口 context，更新后
对新增历史投影并追加；旧历史不重建。fc 将已冻结的 FC 计算移到 GPU 已有结果上，
CPU/GPU BF16 数值仍需实测检查，不宣称逐位一致。

kv 组保留 GPU 已生成的 K/V，包括 CPU optimizer 更新之后。
每个 prompt chunk 和新增已确认段记录 `(exclusive_end, GPU projection_version)`，
不将它们误记为 CPU 更新版本。proposal 的 source version 单独保留。
这组包含新窗口 context 的 GPU 版本滞后，是算法近似的消融，不能仅当作无损压缩。

安装仍在 FC/KV 投影之前执行。为拿到真实 KV，三组的 capture 统一放在 GPU
context 投影之后，epoch close marker 放在该 target packet 提交之后。
因此安装位置和归属窗口不变，后台消费顺序仍为本轮数据 → close epoch。
相比旧 hook，capture 提交时刻稍后；当前三组对照具有相同位置。

## 线程与生命周期

复用已有 capture queue、pinned staging、D2H stream 和 EAGLE3 weight handoff。
没有增加训练/上传线程或同步等待。GPU payload clone 与 ready event 保证张量
所有权；CPU batch 在 staging 下次复用前复制已确认数据。
KV 打包自身有 GPU kernel 成本，纳入实测，不假设其开销为零。

基准脚本增加 `--context-representation hidden|fc|kv` 作为这次消融的选择参数。
启动边界拒绝在 context grad 开启时使用压缩格式，避免静默丢失 FC/context 梯度。
没有新增数据上传方式开关。

## 验证设计与已完成检查

沿用现有测试文件。transport 的职责是保留连续历史、teacher 对齐及 buffer 所有权；
扩展原有 prompt-chunk 测试，覆盖三种格式、CPU/CUDA 和拒绝边界，并检查投影版本
与 proposal version 分离。模型测试验证同版本 payload 的 logits、共享权重更新和
更新后的缓存版本，使用 tiny CPU 模型而非额外端到端测试。

现有在线系统、slot、worker、DFlash 输入和因果性测试共 **168 passed**：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OMP_WAIT_POLICY=PASSIVE \
  taskset -c 16,17 .venv/bin/python -m pytest online_draft/tests \
  tests/v1/spec_decode/test_eagle_weight_slots.py \
  tests/v1/worker/test_gpu_worker_online_draft.py \
  tests/v1/spec_decode/test_dflash_prepare_inputs.py \
  tests/v1/spec_decode/test_dflash_causality.py -q
```

真实 checkpoint 数值校验、16K 两轮重复、两题自然结束质量回归和截断 trace
由阶段 24 完成。代码与实验记录由 AI 协助实现，尚未提交或发布 PR。
