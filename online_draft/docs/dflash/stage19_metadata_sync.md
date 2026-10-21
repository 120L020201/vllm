# 阶段 19：移除 XQA decode 的冗余 CPU 长度回读

## 问题与修改

阶段 18 的 trace 复核发现，DFlash 每轮写入 context KV 后、提交 query graph
之前，FlashInfer metadata builder 会回读 GPU `seq_lens`。单请求只有 4 字节，
但 `seq_lens.to("cpu")` 要等待主 stream 先前提交的工作完成。这与后台
capture D2H 和权重 H2D 是不同的路径，原始 DFlash 同样存在。

原因在 `FlashInferMetadataBuilder.build`：旧判断把 `causal` 作为
`all_uses_trtllm` 的必要条件。因此 non-causal DFlash 即使使用 dedicated XQA，
也会进入 CPU 长度读取分支。实际 XQA decode 使用 GPU block table 和
GPU sequence lengths，不使用这个 CPU 副本。

修改只去掉这个额外的 `causal` 限制，依据实际的 prefill/decode backend 判断
是否需要 CPU metadata。non-causal native prefill 仍然不能走 TRTLLM prefill，
其 CPU planning 路径保留；DCP 和 cascade 的独立条件也保留。
没有用 CPU 长度上界替换真实 GPU 长度，没有改变 attention mask 或计算 kernel。

## 同配置完整请求对照

RTX PRO 6000 Blackwell，Qwen3-8B + DFlash-b16，单请求，K=15，target/draft
FULL graphs，FLASHINFER，AIME Q15 自然生成至 prompt+output=16384。
输出 16098 token；EOS 正常生效，本题达到长度上限，没有最终 boxed 答案。

在线沿用阶段 18 推荐：CPU GQA、最后一层监督行计算、context grad 关闭，
最多 16 个训练 round，lr=2e-4，KL gamma=7，16 个训练核。性能运行不启用
profiler。CPU affinity、allocator 设置和环境与阶段 18 一致。

| 配置 | decode token/s | generate 秒 | finish barrier 秒 | CPU 更新 / GPU 安装 |
| --- | ---: | ---: | ---: | ---: |
| 修改前原始 DFlash | 173.51 | 92.831 | 0.002 | 0 / 0 |
| 修改后原始 DFlash | 174.57 | 92.274 | 0.010 | 0 / 0 |
| 修改前在线 | 221.79 | 72.632 | 0.415 | 70 / 69 |
| 仅 metadata 修改后的在线 | 226.13 | 71.244 | 0.772 | 69 / 68 |

每项先运行一次作为筛选，不能把 0.61% 的原始 decode 差异视作已证明的稳定
收益。两个原始请求输出逐 token 相同；在线 metadata 候选在零基位置 4549
开始与原始输出分叉，因此完整请求的 1.96% 吞吐差异同时包含训练节奏、接受率
和生成路径变化。后续阶段保留分叉数据并使用不安装权重的采集对照区分开销。

## 验证

扩展现有 `test_flashinfer_xqa_decode_correctness`，覆盖 causal / non-causal，
在 builder 调用期间让 `seq_lens_cpu` 属性访问直接报错，同时保留与参考
attention 的数值对照：**2 passed**。

默认测试模型的远程配置下载等待超过一分钟，第一次运行被中止；未计作通过。
本机通过实验目录的 `run_attention_tests_local.py` 将该测试的 `model` 参数
设为 `/data/models/Qwen3-8B`，使用已有本地 checkpoint 配置。测试逻辑和断言
没有替换。测试 ID 保留 collection 时的默认模型名，实际参数由该脚本指定。

共享 pipeline / installer / handoff 的新增并发回归和原有测试 **16 passed**；
这些用于下一阶段后台上传，不计为本阶段吞吐收益。

数据和命令位于
[实验目录](../../results/experiments/dflash_overlap_20261019/)，本阶段对应
`before-baseline`、`before-online`、`metadata-baseline`、`metadata-online`。
完整 GPU 时间线和与 capture/H2D 改动组合后的验证在下一阶段报告。
