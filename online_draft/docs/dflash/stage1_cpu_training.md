# 阶段 1：CPU DFlash 模型与在线训练窗口

## 实现

`models/qwen3_dflash.py` 实现 Qwen3 DFlash 的 context KV 注入与完整非因果
masked block。参数名与 vLLM 的 packed QKV/gate-up 布局一致，复用已有 RMSNorm
和 RoPE。启动时从已加载的 GPU draft 复制普通 CPU tensor，并复制其共享的
target embedding/head，因此无需加载另一份完整 CPU target。

`training/dflash_window.py` 接收连续 round 和旧 detached context KV，窗口一次
forward-KL/AdamW 更新；复用 `window_forward_kl_loss` 和 `DraftTrainer`。
监督从 query 行 1 开始，接受 a 个 token 时取 min(a+1,K) 个 teacher 目标。
T=1，首拒绝权重 2，其余 1，按全窗口目标权重归一化，沿用 EAGLE3。

用户确认训练所有 draft 自有参数、冻结 embedding/head，保留已有冻结配置。
bootstrap prompt 与窗口新增 context 可微投影；旧段不重建、不传播梯度。
更新后用新权重投影新增 confirmed H 并追加，记录位置段与 CPU version。

## 位置与数值契约

anchor A 的可见 context 为 positions < A，query 是 [A..A+K]。
target 在 A+i-1 的 teacher 对齐 query A+i。首拒绝后 query 仍全部保留，
只关闭 loss；否则双向 attention 的前面输出会变化。

最新 correction/bonus 未经 target forward，没有真实 H，不能放入 context。
训练时只使用当时可见的新段。旧 cache 保留混合权重版本，是已选近似。
如果后续窗口只有一个 round，且其 context 全部已缓存，该轮不会产生 fc 分支
梯度；不能为了得到梯度而隐式重算旧历史。

## 验证

命令：

```bash
OMP_NUM_THREADS=2 .venv/bin/python -m pytest \
  online_draft/tests/test_dflash_training.py \
  online_draft/tests/test_trainer.py -q
```

结果：13 passed。包含独立显式 attention 数学参照、a=0/部分/全接受、bootstrap
fc 参数更新、冻结 embedding/head 不变、窗口仅一次版本增长、旧 KV 逐值保留、
新段与更新后直接投影相同、截断 mask 后缀会改变输出。

共享异步调度/传输的初步重用另经既有 10 项测试通过；完整 GPU 接入仍在后续
阶段验证。本阶段尚未声称真实 Qwen3-8B CPU/GPU logits 已通过数值比较。
