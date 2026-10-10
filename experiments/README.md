# 在线 Draft 实验脚本

本目录提供一套完整入口，用于下载数据和模型、启动四种方法、顺序运行完整
benchmark，并输出逐题性能统计。

## 1. 环境准备

该分支使用 `setuptools-scm` 从 Git 生成版本号。安装前必须确保
`v0.28.0` tag 可见，否则无 tag 的 clone 会生成错误的 `0.1.dev...` 版本。
不要手工修改生成的 `vllm/_version.py`。

```bash
git clone --branch online-draft-rebuild \
  https://github.com/120L020201/vllm.git vllm-osd
cd vllm-osd
git describe --tags --long
```

输出应以 `v0.28.0-` 开头。旧 clone 若缺少 tag，先从当前 fork 获取；仅当
fork 中也没有该 tag 时，才从上游仓库获取：

```bash
git fetch origin tag v0.28.0
# 仅作为备用：
git fetch https://github.com/vllm-project/vllm.git \
  refs/tags/v0.28.0:refs/tags/v0.28.0
```

所有 Python 命令都应通过项目的 `uv` 虚拟环境运行。这里保留经过验证的
CUDA 13.0 依赖组合，不要使用 `--torch-backend=auto`，以免新驱动自动选择
不兼容的 cu132 wheel：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12
source .venv/bin/activate

VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=cu130
uv pip install --extra-index-url https://flashinfer.ai/whl/ \
  flashinfer-python==0.6.16.post3 \
  flashinfer-cubin==0.6.16.post3
uv pip install -e online_draft
uv pip install -r methods/requirements.txt

uv pip install -r requirements/lint.txt
pre-commit install
```

启动服务前检查依赖与版本：

```bash
uv pip check
.venv/bin/python -c \
  'import importlib.metadata as m; print(m.version("vllm"))'
.venv/bin/python -c \
  'import importlib.metadata as m; print(m.version("flashinfer-python"), m.version("flashinfer-cubin"))'
vllm --version
```

vLLM 版本应类似
`0.28.1.devN+g<commit>[.dYYYYMMDD].precompiled`，两个 FlashInfer 包的版本必须
一致。切换提交后应重新运行 editable-install 命令，使生成的版本和当前 HEAD
保持一致。

GPQA Diamond 是 Hugging Face 受控数据集。第一次下载前，需要在网页接受访问
条款并登录：

```bash
.venv/bin/hf auth login
```

## 2. 一键下载数据和模型

默认下载六个完整数据集、Qwen3-4B/8B target 模型，以及对应 EAGLE3 draft：

```bash
nohup env \
  HF_ENDPOINT=https://hf-mirror.com \
  HF_HUB_DISABLE_XET=1 \
  PYTHON_BIN="$PWD/.venv/bin/python" \
  DATA_DIR=/srv/Datasets \
  MODEL_DIR=/srv/Models \
  DATASETS="aime2026 gpqa_diamond mmlu-pro-computer_science livecodebench-lite LongBench-v2 LongWriter-6k" \
  MODEL_SIZES="4b 8b" \
  MAX_WORKERS=8 \
  bash experiments/prepare.sh \
  > prepare_assets.log 2>&1 < /dev/null &
```

查看下载进度：

```bash
tail -f prepare_assets.log
```

脚本启动后会立即把所选数据集、模型规格、目标目录和下载阶段写入日志，并使用
Python 无缓冲输出。如果日志仍为 0 字节且 `ps` 中没有 `prepare.sh` 或
`prepare_datasets.py`/`prepare_models.py`，说明命令没有真正启动；可先执行：

```bash
DRY_RUN=1 bash experiments/prepare.sh
```

确认路径后再运行后台命令。

已有完整文件会被校验并跳过，因此同一命令可以安全地再次运行。脚本不会静默
覆盖行数错误的数据文件或不完整的模型目录。

### 选择性下载

只下载部分数据集和 4B 模型：

```bash
DATASETS="aime2026 gpqa_diamond LongBench-v2" \
MODEL_SIZES="4b" \
HF_ENDPOINT=https://hf-mirror.com \
bash experiments/prepare.sh
```

只下载数据集：

```bash
MODEL_SIZES="" bash experiments/prepare.sh
```

只下载模型：

```bash
DATASETS="" MODEL_SIZES="4b 8b" bash experiments/prepare.sh
```

仅查看即将执行的命令：

```bash
DRY_RUN=1 DATASETS="aime2026" MODEL_SIZES="4b" bash experiments/prepare.sh
```

## 3. 完整数据集

准备完成后，`/srv/Datasets` 应包含：

| Benchmark | 文件 | 完整条数 |
| --- | --- | ---: |
| AIME 2026 | `aime2026.jsonl` | 30 |
| GPQA Diamond | `gpqa_diamond.jsonl` | 198 |
| MMLU-Pro Computer Science | `mmlu-pro-computer_science.jsonl` | 410 |
| LiveCodeBench Lite release v6 | `livecodebench-lite.jsonl` | 1055 |
| LongBench v2 | `LongBench-v2.jsonl` | 503 |
| LongWriter-6k | `LongWriter-6k.jsonl` | 6000 |

检查实际条数：

```bash
wc -l \
  /srv/Datasets/aime2026.jsonl \
  /srv/Datasets/gpqa_diamond.jsonl \
  /srv/Datasets/mmlu-pro-computer_science.jsonl \
  /srv/Datasets/livecodebench-lite.jsonl \
  /srv/Datasets/LongBench-v2.jsonl \
  /srv/Datasets/LongWriter-6k.jsonl
```

数据准备器固定官方 revision，并且只有在完整条数校验通过后才原子发布最终
JSONL 文件。它直接下载已知的 JSONL/CSV/Parquet/JSON 文件，不会调用
Hugging Face `datasets` 去探测不存在的远程 `*.py` loader；未完成的原始文件
保存在 `/srv/Datasets/.downloads`，重跑时可继续复用。

## 4. 模型目录

准备完成后，`/srv/Models` 应包含：

```text
/srv/Models/Qwen3-4B
/srv/Models/Qwen3-4B_eagle3
/srv/Models/Qwen3-8B
/srv/Models/Qwen3-8B_eagle3
```

模型准备器会检查 `config.json`、hidden size、权重索引和所有权重分片。
未完成的模型保存在 `.Qwen3-4B.download` 或 `.Qwen3-8B.download` staging
目录；网络中断后重新执行相同命令，会从 Hugging Face 的 incomplete 文件继续
下载。不要同时启动多个针对同一模型目录的准备进程。

## 5. Benchmark 通用配置

推荐的 benchmark 列表为：

```bash
BENCHMARKS="aime2026 gpqa_diamond mmlu-pro-computer_science livecodebench-lite LongBench-v2"
```

在数据集名称后加 `[N]` 可从该数据集中确定性抽取 `N` 题；没有后缀的数据集
仍然全量运行。例如下面只从 AIME 抽取 15 题，其余三个数据集全量运行：

```bash
SEED=7
BENCHMARKS="aime2026[15] gpqa_diamond livecodebench-lite mmlu-pro-computer_science"
```

相同 `SEED` 会选择相同题目，也会作为 OpenAI 请求的生成 seed，并传给在线
方法的随机策略。更换 seed 或抽样数量时，断点恢复会自动丢弃配置不匹配的旧
结果。LongWriter-6k 有 6000 条长文本生成指令，通常应按需使用
`LongWriter-6k[N]`，而不是直接全量运行。

默认配置：

- `MAX_OUTPUT_TOKENS=32768`
- `MAX_MODEL_LEN=49152`
- `MAX_PROMPT_TOKENS=16384`
- `TEMPERATURE=0.6`
- `TOP_P=0.95`
- `TOP_K=20`
- `PRESENCE_PENALTY=1.5`
- `ENABLE_THINKING=1`
- `SEED=0`
- 单请求顺序执行

所有方法都通过 `/v1/chat/completions` 发送相同的 user message，由服务端应用
模型自带的 Qwen3 chat template。默认启用 thinking；设置 `ENABLE_THINKING=0`
可关闭。`PRESENCE_PENALTY=1.5` 遵循 Qwen3 对严重重复生成的建议。

Qwen3-4B/8B 原生声明的上下文长度是 40960。默认 49152 context 会显式启用
vLLM long-context override，额外区间使用 RoPE 外推，应通过实际评测确认质量。

显存规则：

- Qwen3-4B：硬上限 24 GiB，给 vLLM 的预算为 22 GiB。
- Qwen3-8B：硬上限 32 GiB，给 vLLM 的预算为 30 GiB。
- 两种配置都预留 2 GiB 给 CUDA 和运行时额外开销。

`MONITOR=1` 时会把周期性的 `nvidia-smi` 采样写到
`results/<method>.jsonl.gpu.csv`。

## 6. 运行 Base

```bash
nohup env \
  MONITOR=1 \
  RESUME=0 \
  MAX_MODEL_LEN=49152 \
  MAX_OUTPUT_TOKENS=32768 \
  GPU_MEMORY_HEADROOM_GIB=2 \
  GPU_MEMORY_UTILIZATION=1 \
  ENABLE_THINKING=1 \
  SEED=7 \
  PYTHON_BIN="$PWD/.venv/bin/python" \
  DATA_DIR=./methods/artifacts/datasets \
  BENCHMARKS="aime2026[15] gpqa_diamond livecodebench-lite mmlu-pro-computer_science" \
  MODEL=./methods/artifacts/models/Qwen3-8B \
  TEMPERATURE=0.6 TOP_P=0.95 TOP_K=20 \
  OUTPUT_FILE=results/qwen3_8b_base.jsonl \
  SERVER_LOG=results/qwen3_8b_base.server.log \
  bash experiments/run.sh base \
  >> qwen3_8b_base.log 2>&1 < /dev/null &
```

`MAX_MODEL_LEN` 必须大于 `MAX_OUTPUT_TOKENS`，并为输入 prompt 留出空间。因此
`MAX_MODEL_LEN=49152` 不能与 `MAX_OUTPUT_TOKENS=50000` 同时使用；如确实需要
最多生成 50000 tokens，应进一步提高模型上下文长度并验证 RoPE 外推质量。

## 7. 运行冻结 EAGLE3

```bash
nohup env \
  MONITOR=1 NSYS=0 \
  MAX_OUTPUT_TOKENS=32768 \
  PYTHON_BIN="$PWD/.venv/bin/python" \
  DATA_DIR=/srv/Datasets \
  BENCHMARKS="aime2026 gpqa_diamond mmlu-pro-computer_science livecodebench-lite LongBench-v2" \
  MODEL=/srv/Models/Qwen3-8B \
  SPEC_MODEL=/srv/Models/Qwen3-8B_eagle3 \
  SPEC_TOKENS=7 \
  TEMPERATURE=0.6 TOP_P=0.95 TOP_K=20 D_TEMPERATURE=0 \
  RUN_LOG=qwen3_8b_eagle3.log \
  bash experiments/run.sh eagle3 \
  > /dev/null 2>&1 < /dev/null &
```

## 8. 运行 TTS

```bash
nohup env \
  MONITOR=1 NSYS=0 \
  MAX_OUTPUT_TOKENS=32768 \
  PYTHON_BIN="$PWD/.venv/bin/python" \
  DATA_DIR=/srv/Datasets \
  BENCHMARKS="aime2026 gpqa_diamond mmlu-pro-computer_science livecodebench-lite LongBench-v2" \
  MODEL=/srv/Models/Qwen3-8B \
  SPEC_MODEL=/srv/Models/Qwen3-8B_eagle3 \
  SPEC_TOKENS=7 \
  TTS_UPDATE_STRIDE=1 UPDATE_DELAY=0 \
  TTS_RESET_PER_REQUEST=1 TTS_LEARNING_RATE=2e-5 \
  TEMPERATURE=0.6 TOP_P=0.95 TOP_K=20 D_TEMPERATURE=0 \
  RUN_LOG=qwen3_8b_tts.log \
  bash experiments/run.sh tts \
  > /dev/null 2>&1 < /dev/null &
```

## 9. 运行 OSpec

```bash
nohup env \
  MONITOR=1 NSYS=0 \
  MAX_OUTPUT_TOKENS=32768 \
  PYTHON_BIN="$PWD/.venv/bin/python" \
  DATA_DIR=/srv/Datasets \
  BENCHMARKS="aime2026 gpqa_diamond mmlu-pro-computer_science livecodebench-lite LongBench-v2" \
  BASE_MODEL_PATH=/srv/Models/Qwen3-8B \
  EA_MODEL_PATH_1=/srv/Models/Qwen3-8B_eagle3 \
  EA_MODEL_PATH_2=/srv/Models/Qwen3-8B_eagle3 \
  EA_MODEL_PATH_3=/srv/Models/Qwen3-8B_eagle3 \
  SPEC_TOKENS=7 \
  OSPEC_TRAIN_MAX_LEN=2048 BATCH_SIZE=2 \
  UPDATE_DELAY=0 CHUNK_SIZE=8 OSPEC_SKIP_LAST_CHUNK_TRAIN=1 \
  TEMPERATURE=0.6 TOP_P=0.95 TOP_K=20 D_TEMPERATURE=0 \
  RUN_LOG=qwen3_8b_ospec.log \
  bash experiments/run.sh ospec \
  > /dev/null 2>&1 < /dev/null &
```

当前 OSpec 在 CPU 侧复制一个 draft 为三个 learner，因此
`EA_MODEL_PATH_2/3` 仅为命令兼容字段。在线训练严格单请求有序；
`BATCH_SIZE=2` 不会把服务并发提高到 2。

## 10. 逐题日志和结果

每道题完成后，外层日志会输出：

```text
aime2026-10: AL=2.825, updates=1105, tokens=24987, time=720.8s, tokens/s=34.67, prefill=410 tokens/0.112s/3660.71 tokens/s, decode=24986 tokens/287.400s/86.94 tokens/s
```

指标含义：

- `AL`：`1 + accepted draft tokens / speculative rounds`；Base 为 `1.000`。
- `updates`：该请求实际完成的 CPU optimizer step 数；Base/EAGLE3 为 0。
- `tokens`：API 返回的 completion token 数。
- `time`：请求端到端时间，包括等待该请求的 CPU 更新完成。
- `tokens/s`：`tokens / time`。
- `prefill`：该题的 prompt token 数、vLLM 服务端 prefill 时间和 prefill
  tokens/s；prefill 时间包含首个输出 token。
- `decode`：该题首 token 之后的输出 token 数、vLLM 服务端 decode 时间和
  decode tokens/s。

详细生成结果写入 `results/<method>.jsonl`；vLLM 服务日志写入
`results/<method>.server.log`。每个数据集结束后还会输出严格按 decoding 轮数
加权的 `dataset_AL = 1 + Σaccepted_tokens / Σdrafts`。逐题原始计数和阶段耗时
写入 JSONL，相邻的 `results/<method>.summary.json` 保存数据集级汇总。Base
没有投机解码，因此其 `dataset_AL` 为 `N/A`。

## 11. 断点继续

默认 `RESUME=1`。重新执行相同命令和 `OUTPUT_FILE` 时：

- Base 和 EAGLE3 保留每道已成功题目的结果，只运行失败或尚未完成的题目。
- TTS 和 OSpec 只保留完整数据集；未完成数据集的旧结果会被清除，并从该
  数据集第 1 题重新运行。
- 损坏的末行、失败记录、请求配置不匹配或已变化 prompt 对应的旧记录不会被
  视为已完成；重复的成功记录会合并为一条。
- 结果文件会先原子整理再追加；run log、server log 和 GPU 监控 CSV 会继续
  追加。

设置 `RUN_LOG` 后，`RESUME=0` 会覆盖 run log、结果、server log 和 GPU CSV；
`RESUME=1` 会按对应恢复策略追加。外层 shell 应重定向到 `/dev/null`，避免与
脚本的日志管理重复。更换模型、采样参数或数据后应改用新的 `OUTPUT_FILE`，
或者设置 `RESUME=0` 从头运行。

```bash
RESUME=0 bash experiments/run.sh base
```

TTS/OSpec 续跑时会从原始 checkpoint 启动新服务，并用未完成数据集从第 1 题
重新建立该数据集内的在线状态；已经完整完成的数据集不会重放。

## 12. 已知兼容限制

- `D_TEMPERATURE` 会被接受，但当前 EAGLE3 draft 使用 greedy sampling。
- `UPDATE_DELAY` 当前不延迟更新。
- `OSPEC_TRAIN_MAX_LEN` 当前不会截断训练数据。
- `NSYS` 会被接受，但当前脚本没有自动包装 Nsight Systems。
- 在线方法在 8B draft 上会产生明显 CPU 训练延迟；可增大
  `TTS_UPDATE_STRIDE` 来降低更新频率。
