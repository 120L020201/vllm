# 在线 Draft 实验脚本

本目录提供一套完整入口，用于下载数据和模型、启动五种方法、顺序运行完整
benchmark，并输出逐题性能统计。

入口为 `base`、`eagle3`、`online_draft`、`tts`、`ospec`。
`eagle3` 只做冻结 draft 推理；`online_draft` 使用内置异步 CPU draft 训练；
TTS/OSpec 使用 GPU draft 训练。只有 Base 和 EAGLE3 支持断点恢复。

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

在数据集名称后加 `(N)` 可从该数据集中确定性抽取 `N` 题；加 `[N]` 则只运行
数据集中的第 `N` 题（从 1 开始计数）。没有后缀的数据集仍然全量运行。由于
圆括号对 shell 有特殊含义，`BENCHMARKS` 应保持用引号包裹。例如下面从 AIME
抽取 15 题，其余三个数据集全量运行：

```bash
SEED=7
BENCHMARKS="aime2026(15) gpqa_diamond livecodebench-lite mmlu-pro-computer_science"
```

只运行 AIME 2026 的第 10 题：

```bash
BENCHMARKS="aime2026[10]"
```

相同 `SEED` 会选择相同题目，也会作为 OpenAI 请求的生成 seed，并传给在线
方法的随机策略。更换 seed 或抽样数量时，断点恢复会自动丢弃配置不匹配的旧
结果。LongWriter-6k 有 6000 条长文本生成指令，通常应按需使用
`LongWriter-6k(N)`，而不是直接全量运行。

默认配置：

- `MAX_MODEL_LEN=32768`
- `MAX_OUTPUT_TOKENS=0`：不设固定上限，每题使用
  `MAX_MODEL_LEN - 完整 chat prompt token 数`
- `MAX_PROMPT_TOKENS=0`：保留完整 prompt，不截断
- `ENABLE_CHUNKED_PREFILL=0`
- `MAX_NUM_BATCHED_TOKENS=32768`
- `TEMPERATURE=0.6`
- `TOP_P=0.95`
- `TOP_K=20`
- `MIN_P=0.0`
- `PRESENCE_PENALTY=0.0`
- `FREQUENCY_PENALTY=0.0`
- `REPETITION_PENALTY=1.0`
- `STOP_TOKEN_IDS=151645,151643`
- `GPU_MEMORY_UTILIZATION=0.9`
- `GPU_ATTENTION_BACKEND=FLASHINFER`：显式指定 target 和 draft 的 GPU 推理后端
- `FLASHINFER_AUTOTUNE=0`：关闭启动时的 FlashInfer 内核调优，与 artifact 一致
- `WARMUP_TOKENS=16`
- `ENABLE_THINKING=1`
- `TRUST_REMOTE_CODE=1`
- `SEED=0`
- 单请求顺序执行

`GPU_ATTENTION_BACKEND` 会同时传给 target 的 `--attention-backend` 和 draft 的
`speculative_config.attention_backend`；Base 只设置 target。不依赖 vLLM 自动
选择后端；旧入口未显式指定时，本机实际选择的是 `FLASH_ATTN`，并非 FlashInfer。
`CPU_TRAINING_ATTENTION_BACKEND` 只影响可选的 CPU 训练，不控制 GPU 推理。
日志中的 `Using FlashInfer for top-p & top-k sampling` 也只说明采样器，不能用于
确认 Attention 后端。可设置 `FLASHINFER_AUTOTUNE=1` 启用内核调优，但长上下文
prefill 的调优可能增加启动显存峰值；此开关不影响 `WARMUP_TOKENS` 的请求预热。

所有方法都通过 `/v1/chat/completions` 发送相同的 user message，由服务端应用
模型自带的 Qwen3 chat template。默认启用 thinking；设置 `ENABLE_THINKING=0`
可关闭。正式计时前会用 `Warm up.` 发送一次最多生成 `WARMUP_TOKENS` 的请求；
该请求不写入结果，也不会进入在线 GPU 训练。服务端和每个生成请求都使用同一
`SEED`。每道正式请求先调用服务端 `/tokenize`，得到应用 chat template 后的完整
prompt 长度，再计算生成上限。将 `MAX_OUTPUT_TOKENS` 设为正数可额外限制输出；
将 `MAX_PROMPT_TOKENS` 设为正数才会启用左截断。

TTS/OSpec 的 draft 训练在 GPU 上使用 PyTorch eager；Base 的 target，以及
EAGLE3/TTS/OSpec 的 target、draft proposal 和 verification 都显式使用
`FULL_AND_PIECEWISE` CUDA Graph 模式。`online_draft` 仍走内置 CPU 在线训练
路径：当前底层会强制 GPU draft 的 prefill/decode 使用 eager，不能仅凭启动
参数认定 draft 使用 CUDA Graph；入口改名不会改变这一限制。
Chunked Prefill 默认关闭，prefix caching
也关闭。需要分块时显式设置，例如：

```bash
ENABLE_CHUNKED_PREFILL=1 MAX_NUM_BATCHED_TOKENS=512 \
  bash experiments/run.sh tts
```

默认 32768 context 位于 Qwen3-4B/8B 声明的 40960 token 原生范围内。8B OSpec
需要在同一 GPU 常驻三个 learner、optimizer state 和 serving 模型；RTX 5090
32GB、利用率 0.8 的实测中，32768 context 可能因 KV cache 空间不足而启动失败。
这种情况下应显式降低 `MAX_MODEL_LEN`，而不是让脚本静默改变实验配置。

显存规则：

- 默认不按模型大小设置硬上限，vLLM 使用 `GPU_MEMORY_UTILIZATION=0.9`。
- 仅当显式设置 `MAX_GPU_MEMORY_GIB` 时才启用硬上限。
- 启用硬上限后，`GPU_MEMORY_HEADROOM_GIB=2` 会为 CUDA 和运行时预留 2 GiB。

`MONITOR=0` 默认关闭资源监控。设为 `1` 后，每隔
`MONITOR_INTERVAL=5` 秒记录整机 CPU 利用率、整机已用/总内存、vLLM server
进程树的 CPU/RSS，以及每张 GPU 的显存和利用率。整机 CPU 是相邻采样间隔内
的利用率；
进程树 CPU 是 `ps` 给出的启动至今平均值，以单个逻辑核为 100%，多核工作时可能
超过 100%；RSS 是进程树各进程 RSS 之和，包含共享页的重复计数。
默认 5 秒间隔下，读取 `/proc`、执行一次 `ps` 和一次 `nvidia-smi` 的额外负载
通常很低；不建议为了更细粒度记录把间隔降到 1 秒以下。监控文件位于本次运行
目录下，文件名固定为 `monitor.csv`。

## 6. 运行 Base

```bash
nohup env \
  MONITOR=1 \
  RESUME=0 \
  MAX_MODEL_LEN=32768 \
  MAX_OUTPUT_TOKENS=0 \
  GPU_MEMORY_HEADROOM_GIB=2 \
  GPU_MEMORY_UTILIZATION=0.8 \
  ENABLE_THINKING=1 \
  SEED=7 \
  PYTHON_BIN="$PWD/.venv/bin/python" \
  DATA_DIR=./methods/artifacts/datasets \
  BENCHMARKS="aime2026(15) gpqa_diamond livecodebench-lite mmlu-pro-computer_science" \
  MODEL=./methods/artifacts/models/Qwen3-8B \
  TEMPERATURE=0.6 TOP_P=0.95 TOP_K=20 \
  bash experiments/run.sh base \
  >> qwen3_8b_base.log 2>&1 < /dev/null &
```

`MAX_OUTPUT_TOKENS=0` 表示使用完整上下文剩余空间；例如完整 chat prompt 为
768 token 时，最大生成长度为 `32768-768=32000`。设置
`MAX_OUTPUT_TOKENS=16384` 后则取 `min(16384, 32768-prompt_tokens)`。

## 7. 运行冻结 EAGLE3

```bash
nohup env \
  MONITOR=1 \
  MAX_OUTPUT_TOKENS=0 \
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
  MONITOR=1 \
  MAX_OUTPUT_TOKENS=0 \
  PYTHON_BIN="$PWD/.venv/bin/python" \
  DATA_DIR=/srv/Datasets \
  BENCHMARKS="aime2026 gpqa_diamond mmlu-pro-computer_science livecodebench-lite LongBench-v2" \
  MODEL=/srv/Models/Qwen3-8B \
  SPEC_MODEL=/srv/Models/Qwen3-8B_eagle3 \
  SPEC_TOKENS=7 \
  TTS_UPDATE_STRIDE=1 UPDATE_DELAY=0 \
  TTS_RESET_PER_REQUEST=1 TTS_LEARNING_RATE=1e-5 \
  TEMPERATURE=0.6 TOP_P=0.95 TOP_K=20 D_TEMPERATURE=0 \
  RUN_LOG=qwen3_8b_tts.log \
  bash experiments/run.sh tts \
  > /dev/null 2>&1 < /dev/null &
```

## 9. 运行 OSpec

```bash
nohup env \
  MONITOR=1 \
  MAX_OUTPUT_TOKENS=0 \
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

当前 OSpec 在 GPU 侧常驻三个 draft learner；`EA_MODEL_PATH_2/3` 仍仅为命令
兼容字段。在线训练严格单请求有序；
`BATCH_SIZE=2` 不会把服务并发提高到 2。

## 9.1 运行 online_draft（CPU 训练）

TTS/OSpec 默认继续使用本仓库的同步 GPU eager trainer。相关默认值为：

- `TTS_LEARNING_RATE=1e-5`
- `FUSED_OPTIMIZER=1`
- `LOG_UPDATES=0`

`bash experiments/run.sh online_draft` 启用 vLLM 内置的异步 CPU online-draft
trainer。脚本自动设置 `VLLM_ONLINE_DRAFT_TRAIN=1`，其他入口自动设置为 `0`。
不再使用 `eagle3 + VLLM_ONLINE_DRAFT_TRAIN=1` 的组合；与入口冲突的环境变量
会报错，请移除旧开关并改用新入口。CPU trainer 的默认配置为：

- `CHECK_GRADIENTS=1`
- `FROZEN_PARAMETER_PREFIXES=`（空，即不冻结）
- `FUSED_OPTIMIZER=1`
- `LOG_UPDATES=0`
- `CPU_TRAINING_ATTENTION_BACKEND=flash_attention`
- `VLLM_ONLINE_DRAFT_PROFILE=0`

多个冻结前缀使用逗号分隔。示例：

```bash
MODEL=./methods/artifacts/models/Qwen3-8B \
SPEC_MODEL=./methods/artifacts/models/Qwen3-8B_eagle3 \
DATA_DIR=./methods/artifacts/datasets \
BENCHMARKS="aime2026[14]" \
VLLM_ONLINE_DRAFT_PROFILE=0 \
CHECK_GRADIENTS=0 \
FROZEN_PARAMETER_PREFIXES=model.embed_tokens,lm_head \
FUSED_OPTIMIZER=1 LOG_UPDATES=1 \
  bash experiments/run.sh online_draft
```

`CHECK_GRADIENTS` 和 `FROZEN_PARAMETER_PREFIXES` 只配置可选的 CPU trainer；不会
关闭或冻结 TTS/OSpec 的 GPU trainer。`VLLM_ONLINE_DRAFT_PROFILE=1` 只允许用于
`online_draft`。`online_draft`、TTS、OSpec 均默认 `RESUME=0`，不支持续跑；
显式设置 `RESUME=1` 会在启动服务和改写输出前报错。

## 10. 逐题日志和结果

每道题完成后，外层日志会输出：

```text
aime2026-10: AL=2.825, updates=1105, tokens=24987, time=720.8s, tokens/s=34.67, prefill=410 tokens/0.112s/3660.71 tokens/s, decode=24986 tokens/287.400s/86.94 tokens/s
```

指标含义：

- `AL`：`1 + accepted draft tokens / speculative rounds`；Base 为 `1.000`。
- `updates`：该请求实际完成的 GPU optimizer step 数；Base/EAGLE3 为 0。
- `tokens`：API 返回的 completion token 数。
- `time`：请求端到端时间，包括该请求同步 GPU 更新的时间。
- `tokens/s`：`tokens / time`。
- `prefill`：该题的 prompt token 数、vLLM 服务端 prefill 时间和 prefill
  tokens/s；prefill 时间包含首个输出 token。
- `decode`：该题首 token 之后的输出 token 数、vLLM 服务端 decode 时间和
  decode tokens/s。

`run.sh` 会先切换到仓库根目录，并在启动时生成一次格式为
`YYYYMMDD_HHMMSS` 的 `RUN_TIMESTAMP`。默认运行目录为：

```text
results/<method>-<datasets>-<model>-<timestamp>/
```

`<datasets>` 是 `BENCHMARKS` 中各选择项用连字符连接后的名称，`<model>` 是目标
模型目录的 basename。例如：

```text
results/tts-aime2026(10)-gpqa_diamond-Qwen3-8B-20261019_210043/
```

目录内的文件名不再重复时间戳：

- `results.jsonl`：逐题结果
- `results.summary.json`：数据集级汇总
- `server.log`：vLLM server stdout/stderr
- `stats.json`：在线训练 worker 状态
- `monitor.csv`：`MONITOR=1` 时的资源采样

在当前仓库中，默认运行目录都位于 `/home/yuhengyu/vllm/results/`。可以显式设置
`RUN_DIR`，也可以分别覆盖 `OUTPUT_FILE`、`SERVER_LOG`、`STATS_FILE` 和
`MONITOR_FILE`。`RUN_LOG` 默认留空，所以 benchmark 进度、脚本提示和错误仍输出
到启动它的终端。每个数据集结束后还会输出严格按 decoding 轮数加权的
`dataset_AL = 1 + Σaccepted_tokens / Σdrafts`。Base 没有投机解码，因此其
`dataset_AL` 为 `N/A`。

## 11. 断点继续

只有 Base 和 EAGLE3 支持恢复，默认 `RESUME=1`。这两个入口重新执行相同命令
和 `OUTPUT_FILE` 时：

- Base 和 EAGLE3 保留每道已成功题目的结果，只运行失败或尚未完成的题目。
- 损坏的末行、失败记录、请求配置不匹配或已变化 prompt 对应的旧记录不会被
  视为已完成；重复的成功记录会合并为一条。
- 结果文件会先原子整理再追加；run log、server log 和资源监控 CSV 会继续
  追加。

设置 `RUN_LOG` 后，`RESUME=0` 会覆盖 run log、结果、server log 和监控 CSV；
`RESUME=1` 仅供 Base/EAGLE3 按题恢复并追加。外层 shell 应重定向到 `/dev/null`，避免与
脚本的日志管理重复。更换模型、采样参数或数据后应改用新的 `OUTPUT_FILE`，
或者设置 `RESUME=0` 从头运行。

默认时间戳会让每次启动使用新目录，因此要恢复一次旧运行，应复用它原来的
`RUN_DIR`；如果 method、datasets 和 model 均不变，也可以复用原时间戳，例如
`RUN_TIMESTAMP=20261019_143000 RESUME=1 ...`。

```bash
RESUME=0 bash experiments/run.sh base
```

`online_draft`、TTS、OSpec 默认且只允许 `RESUME=0`，每次都从原始 checkpoint
和所选题目的起点重新运行，不跳过已有结果。因为当前没有保存/恢复完整训练
状态，不能用已有结果替代训练过程。默认时间戳会创建新目录；显式复用
`RUN_DIR` 或 `OUTPUT_FILE` 时会覆盖对应旧输出，请自行保留需要的结果。

## 12. 已知兼容限制

- `D_TEMPERATURE` 会被接受，但当前 EAGLE3 draft 使用 greedy sampling。
- `UPDATE_DELAY` 当前不延迟更新。
- `OSPEC_TRAIN_MAX_LEN` 当前不会截断训练数据。
- 在线方法在 8B draft 上会产生明显 GPU 训练延迟；可增大
  `TTS_UPDATE_STRIDE` 来降低更新频率。
