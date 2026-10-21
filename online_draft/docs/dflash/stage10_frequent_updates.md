# 阶段 10：位置衰减 KL、减少训练 block 与更新频率

2026-10-16，Qwen3-8B / Qwen3-8B-DFlash-b16，`eagle3&dflash`。
用户本轮明确授权减少训练数据并采用 DFlash 位置衰减，覆盖此前“每个完整 round
均参与 loss”和“首拒绝位置 ×2”的约定。完整历史、实际安装分界、请求内 W0 reset
及异步传输仍保留。

## 损失与采样

[DFlash 论文 §4.2、附录 A.1](https://arxiv.org/html/2602.06036v1#S4.SS2)
对 block 中第 k 个预测位置使用 `w(k)=exp(-(k-1)/gamma)`；b16 模型 gamma=7。
论文原始目标为交叉熵。本实现按用户要求，把相同位置衰减用于 teacher→draft
forward KL，温度 1；只监督已接受前缀及首拒绝位置，anchor、bonus 和拒绝后位置
不参与 loss。每个 block 从 k=1 重新计数，取消 EAGLE3 的拒绝位置额外 ×2。

具体归一化为 `sum(w * KL) / sum(w)`，分母只包含实际监督位置的权重；这是本项目
KL 的加权均值约定，不能声称论文给出了完全相同的 KL 归一化实现。
EAGLE3 继续使用原来的损失。

新增 `max_train_rounds`：0 表示全量；正数 N 表示只训练窗口中最近 N 个完整 block。
每个所选 block 保留全部 1+15 query，保证非因果 attention 的数值语义。
旧 cache detach；窗口所有新增原始特征仍投影并参与所选 query 的上下文梯度；
更新后，将窗口所有已确认特征投影为新 KV 并追加，包括未选中 block 的确认段。
因此减少的是 query/MLP/LM-head/loss 计算，历史位置连续性和 KV 版本策略不变。

首轮 bootstrap 仍单独更新，之后每次实际 GPU 安装关闭一个窗口并触发一次更新。
没有把 epoch 改成固定 N 轮，没有减少 capture 的数据完整性。统计新增
`trained_rounds` 和 `target_count`，与完整窗口 `rounds` 分开，防止误把省略的
loss block 说成训练过。默认上限选 16，支持显式 0/8/32 复现消融。

## 同配置真实请求消融

AIME 2026 前四题、thinking、greedy seed=17、最多输出 4096、上下文 8192，
单 GPU/单活动请求；target/draft FULL graph，训练 16 物理核，推理/采集各留一核。
每行是独立进程，同四题顺序，无 profiler，无并行后台训练实验。
累计 15,809 输出 token，第一题自然结束且答案正确，其余三题截断；不作为完整 AIME 准确率。
全部方案输出逐 token 等于 baseline。

| 方案 | 四题 GPU 安装次数 | 合计生成 s | 合计额外收尾 s | 生成 token/s | 完整请求 token/s |
| --- | --- | ---: | ---: | ---: | ---: |
| 不训练 | 0/0/0/0 | 53.816 | 0.014 | 293.76 | 293.68 |
| 衰减 KL，全量 | 3/4/4/4 | 55.478 | 31.162 | 284.96 | 182.47 |
| 最近 8 block | 10/13/14/17 | 55.595 | 3.033 | 284.36 | 269.65 |
| 最近 16 block | 9/11/14/15 | 55.605 | 2.433 | 284.31 | 272.39 |
| 最近 32 block | 8/9/11/12 | 55.527 | 4.774 | 284.71 | 262.17 |
| 16 + FP32 主权重 | 6/8/9/10 | 55.423 | 6.798 | 285.24 | 254.08 |
| 16 + lr=1e-4 | 10/12/15/16 | 55.317 | 4.519 | 285.79 | 264.20 |
| 16 + FP32 主权重 + lr=1e-4 | 6/7/9/10 | 54.986 | 5.020 | 287.51 | 263.46 |
| 16 + CPU 分配器参数 | 10/13/16/17 | 55.672 | 2.531 | 283.97 | 271.62 |

未标学习率时均为 1e-5；未标主权重时，参数及 AdamW moments 为 BF16。
额外收尾是 `end_to_end_s - elapsed_s`，包含 finish/reset 和少量 metrics/RPC 开销。
安装次数是 GPU 实际版本，不是 CPU optimizer 步数；CPU 通常还多完成一个未安装版本。

16 block 相比全量的安装总数 15→49，约 3.27 倍；完整请求吞吐 +49.3%，
主要来自尾部等待消失。生成吞吐与全量几乎相同，仍比不训练低约 3.2%。
8 block 虽有更多安装，在本组请求中没有更好的完整请求吞吐。
分配器参数为 `MALLOC_MMAP_THRESHOLD_=1073741824 MALLOC_TRIM_THRESHOLD_=-1`；
它让 CPU 更新更快，但这一组的完整请求总时间与普通 16 block 接近，不能声称
CPU 微基准的百分比能等比例变成服务吞吐收益。

## 接受率、精度和学习率

| 方案 | 第 1 题接受长度 | 第 2 题 | 第 3 题 | 第 4 题 |
| --- | ---: | ---: | ---: | ---: |
| baseline | 4.5537 | 4.3959 | 3.7987 | 3.6934 |
| 全量衰减 KL | 4.5419 | 4.3912 | 3.7847 | 3.6868 |
| 16 block | 4.5537 | 4.3912 | 3.7987 | 3.7001 |
| 16 + FP32 主权重 | 4.5478 | 4.3771 | 3.7917 | 3.6703 |
| 16 + lr=1e-4 | 4.5013 | 4.3912 | 3.8415 | 3.7441 |
| 16 + FP32 主权重 + lr=1e-4 | 4.4955 | 4.3912 | 3.8343 | 3.7236 |

接受长度由每题增量 `1 + accepted_tokens / num_drafts` 计算，指标原始计数是累计值，
必须先减前一题。1e-4 在后两题有改善，但首题下降。小样本且每题仅一轮，
不足以确认稳定学习收益或选定更大学习率。暂留 1e-5，后续长请求继续对照。

新增可选 `TrainerConfig.master_weights`：BF16 做 forward/backward，FP32 主参数
及一、二阶矩做 AdamW，更新后回写 BF16 推理参数；reset 同时恢复主参数和模型，
清空 moments/gradient。冻结参数前缀及 target embedding/head 不变。
此选项确实保留小更新余量，但本机单步更慢、实际安装更少，未改善本组接受率，
因此默认关闭。阶段 9 发现的 BF16 舍入不是“修好后必然提高接受率”的证明。

## 复现与验证

原始 JSON/日志在 `/tmp/dflash-frequent-updates/`；持久化的请求配置、所有更新、
接受长度和输出 hash 见 [frequent_update_results.json](frequent_update_results.json)。

```bash
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE
.venv/bin/python online_draft/scripts/smoke_async_dflash.py \
  --online --target-graphs --train-threads 16 --cpu-isolation \
  --aime-questions 4 --max-tokens 4096 --max-model-len 8192 \
  --max-train-rounds 16 --output /tmp/decay-last16.json
```

换成 `--max-train-rounds 0/8/32` 对比样本量，加 `--master-weights` 对比精度，
加 `--learning-rate 1e-4` 对比学习率；移除 `--online` 得 baseline。
本轮新增测试覆盖位置衰减重置、KL 数值、删 query 后上下文梯度、完整 KV 追加，
以及小于 BF16 间隔的 FP32 累积和跨请求 reset；复用已有 trainer/DFlash 测试文件。
针对性测试 25 passed；CPU 优化见阶段 11，最终长请求和频繁传输验证见阶段 12。
