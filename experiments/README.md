# 在线 Draft 实验脚本

本目录提供一套完整入口，用于下载数据和模型、启动各类方法、顺序运行完整
benchmark，并输出逐题性能统计。

入口为 `base`、`eagle3`、`dflash`、`mtp`、三种 `online_draft+...` 和三种
`tts+...`；旧的 `tts` 仍作为 `tts+eagle3` 的简写。三种 `online_draft` 共用
vLLM 内置异步 CPU trainer；TTS 使用 GPU draft 训练副本和 CUDA side stream。
Base、EAGLE3、DFlash 和冻结 MTP 支持断点恢复。三种精简 CPU loss 见
[Online draft CPU training](README_online_draft.md)。

新 CPU 训练方法及历史 loss 的复测结果见
[CPU 训练报告](online_draft_cpu_training_report.html)和
[Loss 复测报告](online_draft_sota_loss_study.md)。

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

## 2. 按需下载数据和模型

默认不下载任何模型或数据集。直接运行 `bash experiments/prepare.sh` 会提示
未选择下载项并退出，不启动 Python 下载器，也不创建下载目录。只有显式设置
`DATASETS`、`MODEL_SIZES` 或传入 `--dataset`、`--model-size` 时才下载对应项；
不再需要用 `MODEL_SIZES=""` 或 `DATASETS=""` 关闭另一类下载。已导出的环境
变量也算显式选择。

例如，显式下载五个完整数据集、Qwen3-4B/8B target 模型及对应 EAGLE3 draft
(EAGLE3 是 draft 下载的默认选择)：

```bash
nohup env \
  HF_ENDPOINT=https://hf-mirror.com \
  HF_HUB_DISABLE_XET=1 \
  PYTHON_BIN="$PWD/.venv/bin/python" \
  DATA_DIR=/srv/Datasets \
  MODEL_DIR=/srv/Models \
  DATASETS="aime2026 gpqa_diamond mmlu-pro-computer_science livecodebench-lite LongBench-v2" \
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
DRY_RUN=1 bash experiments/prepare.sh --dataset aime2026 --model-size 4b
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

只下载选定数据集（未设置 `MODEL_SIZES` 时不会下载模型）：

```bash
DATASETS="aime2026 gpqa_diamond" bash experiments/prepare.sh
```

只下载选定模型及对应 EAGLE3 draft（未设置 `DATASETS` 时不会下载数据集）：

```bash
MODEL_SIZES="4b 8b" bash experiments/prepare.sh
```

下载 DFlash draft，或同时下载两种 draft：

```bash
MODEL_DIR=./methods/artifacts/models \
MODEL_SIZES="8b" DRAFT_METHODS="dflash" \
HF_ENDPOINT=https://hf-mirror.com \
bash experiments/prepare.sh

MODEL_SIZES="8b" DRAFT_METHODS="eagle3 dflash" \
bash experiments/prepare.sh
```

下载器会把 8B DFlash checkpoint 保存为
`Qwen3-8B_dflash`；上例目录可直接作为 `SPEC_MODEL` 传给 `run.sh`。
单个 draft 也可用 `DRAFT_METHOD=dflash` 选择。未设置 draft 选择时仍默认
下载 EAGLE3，保持原有命令行为。

`--draft-method mtp` 不下载独立 draft：MTP 权重必须内置在 target checkpoint。
标准 `Qwen/Qwen3-8B` 不含 MTP 层；`--model-size 8b --draft-method mtp`
会准备带内置 MTP 的完整 `Qwen3-8B-MTP` target，并提示把 `MODEL` 设为该目录。
也可以使用含 `mtp_num_hidden_layers` 的 Qwen3.5 target：

```bash
bash experiments/prepare.sh --model-size 8b --draft-method mtp
bash experiments/prepare.sh --model-size qwen3.5-4b --draft-method mtp
```

也可以使用命令行选项，每个选项可重复。指定某类命令行选项时，会覆盖该类
环境变量的选择，不影响另一类：

```bash
bash experiments/prepare.sh --dataset aime2026 --dataset gpqa_diamond
bash experiments/prepare.sh --model-size 8b
bash experiments/prepare.sh --model-size 8b --draft-method dflash
bash experiments/prepare.sh --model-size 8b --draft-method eagle3 --draft-method dflash
bash experiments/prepare.sh --model-size 8b --draft-method mtp
bash experiments/prepare.sh --dataset aime2026 --model-size 8b
```

仅查看即将执行的命令：

```bash
DRY_RUN=1 DATASETS="aime2026" MODEL_SIZES="4b" bash experiments/prepare.sh
```

直接运行底层 Python 下载器也默认不下载，必须显式选择，可重复传入选项：

```bash
.venv/bin/python experiments/prepare_datasets.py --dataset aime2026
.venv/bin/python experiments/prepare_models.py --model-size 8b
.venv/bin/python experiments/prepare_models.py --model-size 8b --draft-method dflash
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

检查实际条数：

```bash
wc -l \
  /srv/Datasets/aime2026.jsonl \
  /srv/Datasets/gpqa_diamond.jsonl \
  /srv/Datasets/mmlu-pro-computer_science.jsonl \
  /srv/Datasets/livecodebench-lite.jsonl \
  /srv/Datasets/LongBench-v2.jsonl
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

## 4.1 各方法使用方式

所有入口都从仓库根目录执行，并要求先完成环境安装和所需模型、数据下载。下表
给出 `run.sh` 的方法名及额外输入；省略 `SPEC_MODEL` 的 MTP 入口使用 target
checkpoint 内嵌的 `mtp.*` 权重。

| 方法 | `run.sh` 参数 | draft/额外输入 | 默认 `SPEC_TOKENS` | 在线训练 | 默认 `RESUME` |
| --- | --- | --- | ---: | --- | ---: |
| Base | `base` | 无 | N/A | 否 | 1 |
| 冻结 EAGLE3 | `eagle3` | `SPEC_MODEL=<EAGLE3 draft>` | 7 | 否 | 1 |
| 冻结 DFlash | `dflash` | `SPEC_MODEL=<DFlash draft>` | 15 | 否 | 1 |
| 冻结 MTP | `mtp` | `MODEL=<含内嵌 MTP 的 target>` | 3 | 否 | 1 |
| Online EAGLE3 | `online_draft` 或 `online_draft+eagle3` | `SPEC_MODEL=<EAGLE3 draft>` | 7 | CPU 异步 | 0 |
| Online DFlash | `online_draft+dflash` | `SPEC_MODEL=<DFlash draft>` | 15 | CPU 异步 | 0 |
| Online MTP | `online_draft+mtp` | `MODEL=<含内嵌 MTP 的 target>` | 3 | CPU 异步 | 0 |
| TTS + EAGLE3 | `tts+eagle3`（`tts` 为简写） | `SPEC_MODEL=<EAGLE3 draft>` | 7 | GPU side stream | 0 |
| TTS + DFlash | `tts+dflash` | `SPEC_MODEL=<DFlash draft>` | 15 | GPU side stream | 0 |
| TTS + MTP | `tts+mtp` | `MODEL=<含内嵌 MTP 的 target>` | 3 | GPU side stream | 0 |
| OSpec | `ospec` | `BASE_MODEL_PATH`、`EA_MODEL_PATH_1` | 7 | GPU 异步 ensemble | 0 |

下面的示例复用相同的数据集、target 和上下文配置；按实验需要修改
`BENCHMARKS`、`SEED`、`MAX_MODEL_LEN` 等公共变量即可。

```bash
export PYTHON_BIN="$PWD/.venv/bin/python"
export DATA_DIR=/srv/Datasets
export BENCHMARKS="aime2026"
export MODEL=/srv/Models/Qwen3-8B
export MAX_MODEL_LEN=32768
export MAX_NUM_BATCHED_TOKENS=32768
```

冻结 EAGLE3：

```bash
SPEC_MODEL=/srv/Models/Qwen3-8B_eagle3 \
  bash experiments/run.sh eagle3
```

冻结 DFlash：

```bash
SPEC_MODEL=/srv/Models/Qwen3-8B_dflash \
  bash experiments/run.sh dflash
```

冻结 MTP。标准 `Qwen/Qwen3-8B` 不含 MTP 权重，必须换成包含 `mtp.*` 的
checkpoint；不要设置 `SPEC_MODEL`：

```bash
MODEL=/srv/Models/Qwen3-8B-MTP \
  bash experiments/run.sh mtp
```

三种 TTS 训练入口：

```bash
SPEC_MODEL=/srv/Models/Qwen3-8B_eagle3 \
TTS_UPDATE_STRIDE=1 TTS_RESET_PER_REQUEST=1 TTS_LEARNING_RATE=1e-5 \
  bash experiments/run.sh tts+eagle3

SPEC_MODEL=/srv/Models/Qwen3-8B_dflash \
ENABLE_CHUNKED_PREFILL=0 MAX_NUM_BATCHED_TOKENS="$MAX_MODEL_LEN" \
TTS_UPDATE_STRIDE=1 TTS_RESET_PER_REQUEST=1 TTS_LEARNING_RATE=1e-5 \
  bash experiments/run.sh tts+dflash

MODEL=/srv/Models/Qwen3-8B-MTP \
TTS_UPDATE_STRIDE=1 TTS_RESET_PER_REQUEST=1 TTS_LEARNING_RATE=1e-5 \
  bash experiments/run.sh tts+mtp
```

DFlash TTS 必须关闭 chunked prefill，并保证
`MAX_NUM_BATCHED_TOKENS >= MAX_MODEL_LEN`。Qwen3.5 多模态 target 上运行文本
benchmark 或任一 MTP 入口时，增加 `LANGUAGE_MODEL_ONLY=1`。三种
`online_draft+...` 和 `ospec` 的完整示例见下文相应章节。设置 `DRY_RUN=1`
可以只打印最终 server 和 benchmark 命令而不启动服务。

## 5. Benchmark 通用配置

`run.sh` 默认使用 `HOST=127.0.0.1 PORT=8000`。启动前会尝试绑定该地址；
端口已被占用时立即报错，不启动新服务、不复用或关闭旧服务，也不清空本次
配置指向的已有日志、结果和统计文件。可以显式设置 `PORT=8001` 换一个端口。
启动后，只有本次启动的进程仍存活、它或其子进程拥有该地址的监听 socket，
并且 `/health` 返回 200，才开始 benchmark。健康请求最多等待 2 秒，
总就绪等待最多 300 秒；无法确认监听归属时不会放行。`DRY_RUN=1` 不探测端口。
`HOST=0.0.0.0` / `HOST=::` 仍用于服务绑定，但客户端分别连接本机
`127.0.0.1` / `::1`。

推荐的 benchmark 列表为：

```bash
BENCHMARKS="aime2026 gpqa_diamond mmlu-pro-computer_science livecodebench-lite LongBench-v2"
```

在数据集名称后加 `(N)` 可从该数据集中确定性抽取 `N` 题；加 `[N]` 或
`[N,M,...]` 则只运行数据集中的指定题目（从 1 开始计数）。没有后缀的数据集
仍然全量运行。由于
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

只运行 AIME 2026 的第 3、8、13、18、23 题：

```bash
BENCHMARKS="aime2026[3,8,13,18,23]"
```

相同 `SEED` 会选择相同题目，也会作为 OpenAI 请求的生成 seed，并传给在线
方法的随机策略。更换 seed 或抽样数量时，断点恢复会自动丢弃配置不匹配的旧
结果。

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
`FULL_AND_PIECEWISE` CUDA Graph 模式。`online_draft+eagle3` 仍走内置 CPU 在线训练
路径：当前底层会强制 GPU EAGLE3 draft 的 prefill/decode 使用 eager，不能仅凭启动
参数认定 draft 使用 CUDA Graph。`online_draft+dflash` 由 vLLM 内置 DFlash capture
adapter 接入同一个异步 CPU pipeline。
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

`monitor.csv` 记录的是资源采样，不包含逐轮上下文位置或接收长度。`run.sh`
启动 TTS 时默认在 `server.log` 写入 `SPEC_DECODE_STEP` 逐轮记录；可设置
`VLLM_SPEC_DECODE_STEP_METRICS=0` 关闭。用下面的脚本聚合 AIME2026 的逐轮数据：

```bash
.venv/bin/python experiments/plot_tts_context_acceptance.py \
  results/<tts-experiment-dir> --window-tokens 512
```

脚本会生成原始逐轮数据、窗口统计 CSV 和 `tts_figure1b.png`。默认横轴是总上下文
长度（prompt + 已生成 token），窗口内均值用黑线表示，最小值至最大值用灰色阴影表示；
`--x-axis generated` 可改为不含 prompt 的已生成 token 数。旧实验若没有
`SPEC_DECODE_STEP` 记录，就无法从 `monitor.csv` 或逐题汇总恢复逐轮曲线，需要用
开启该记录的版本重新运行 TTS。

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
  TEMPERATURE=0.6 TOP_P=0.95 TOP_K=20 \
  RUN_LOG=qwen3_8b_eagle3.log \
  bash experiments/run.sh eagle3 \
  > /dev/null 2>&1 < /dev/null &
```

## 8. 运行 TTS

TTS drafter 可直接通过 `run.sh` 方法名选择：`tts+eagle3`、`tts+dflash` 或
`tts+mtp`。三个入口都接受同一套 `BENCHMARKS`，所以可用于此文档列出的任意数据集，也可
组合多个数据集或选择题目子集；例如 `BENCHMARKS="aime2026(15) gpqa_diamond"`。
EAGLE-3 默认使用 7 个 draft tokens，DFlash b16 默认使用 15 个，MTP 默认使用 3 个。
TTS 运行会自动记录每轮已生成 token 数、draft 数和接收数，以便绘制接收长度随
生成长度变化的曲线；详情见资源监控一节。

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
  TTS_UPDATE_STRIDE=1 \
  TTS_RESET_PER_REQUEST=1 TTS_LEARNING_RATE=1e-5 \
  TEMPERATURE=0.6 TOP_P=0.95 TOP_K=20 \
  RUN_LOG=qwen3_8b_tts.log \
  bash experiments/run.sh tts+eagle3 \
  > /dev/null 2>&1 < /dev/null &
```

`tts+mtp` 不读取 `SPEC_MODEL`；MTP 权重必须内嵌在 `MODEL` 中。标准
`Qwen/Qwen3-8B` 不含 MTP 权重，应使用 `Qwen3-8B-MTP` 一类兼容 checkpoint：

```bash
nohup env \
  MONITOR=1 RESUME=0 \
  PYTHON_BIN="$PWD/.venv/bin/python" \
  DATA_DIR=./methods/artifacts/datasets \
  BENCHMARKS="aime2026" \
  MODEL=/path/to/Qwen3-8B-MTP \
  SPEC_TOKENS=3 \
  TTS_UPDATE_STRIDE=1 TTS_RESET_PER_REQUEST=1 TTS_LEARNING_RATE=1e-5 \
  RUN_LOG=qwen3_8b_tts_mtp.log \
  bash experiments/run.sh tts+mtp \
  > /dev/null 2>&1 < /dev/null &
```

### 8.1 在 AIME2026 上运行 DFlash 与 DFlash TTS

论文使用 `z-lab/Qwen3-8B-DFlash-b16`，block size 为 16，因此 vLLM 的
`SPEC_TOKENS` 是 15。先用完全相同的 target、采样参数和题目顺序运行冻结
DFlash 基线：

```bash
nohup env \
  MONITOR=1 RESUME=0 \
  PYTHON_BIN="$PWD/.venv/bin/python" \
  DATA_DIR=./methods/artifacts/datasets \
  BENCHMARKS="aime2026" \
  MODEL=./methods/artifacts/models/Qwen3-8B \
  SPEC_MODEL=./methods/artifacts/models/qwen3-8b-dflash \
  MAX_MODEL_LEN=32768 MAX_NUM_BATCHED_TOKENS=32768 \
  MAX_OUTPUT_TOKENS=0 ENABLE_CHUNKED_PREFILL=0 ENABLE_THINKING=1 \
  TEMPERATURE=0.6 TOP_P=0.95 TOP_K=20 MIN_P=0.0 SEED=0 \
  RUN_LOG=qwen3_8b_dflash_aime2026.log \
  bash experiments/run.sh dflash \
  > /dev/null 2>&1 < /dev/null &
```

再运行 DFlash TTS：

```bash
nohup env \
  MONITOR=1 RESUME=0 \
  PYTHON_BIN="$PWD/.venv/bin/python" \
  DATA_DIR=./methods/artifacts/datasets \
  BENCHMARKS="aime2026" \
  MODEL=./methods/artifacts/models/Qwen3-8B \
  SPEC_MODEL=./methods/artifacts/models/qwen3-8b-dflash \
  MAX_MODEL_LEN=32768 MAX_NUM_BATCHED_TOKENS=32768 \
  MAX_OUTPUT_TOKENS=0 ENABLE_CHUNKED_PREFILL=0 ENABLE_THINKING=1 \
  TEMPERATURE=0.6 TOP_P=0.95 TOP_K=20 MIN_P=0.0 SEED=0 \
  TTS_UPDATE_STRIDE=1 TTS_RESET_PER_REQUEST=1 TTS_LEARNING_RATE=1e-5 \
  RUN_LOG=qwen3_8b_dflash_tts_s1_aime2026.log \
  bash experiments/run.sh tts+dflash \
  > /dev/null 2>&1 < /dev/null &
```

`tts+dflash` 会默认选择 `SPEC_TOKENS=15`，并把结果方法名记为 `tts-dflash`；
`tts+eagle3` 对应 `tts-eagle3`。论文没有报告 AIME2026，也没有公开 TTS 的 learning rate，所以上述
`1e-5` 是本地默认值，不是论文给出的超参数。应在同一协议下比较冻结 DFlash 与
DFlash TTS 的 acceptance length 和吞吐，而不能把 AIME2026 数值与论文的
AIME 2024/2025 表格直接比较。`S=1` 最接近算法默认并最大化 acceptance length；
可另跑 `TTS_UPDATE_STRIDE=5` 和 `10` 比较吞吐，但论文的 `S=10` 最优点来自
LiveCodeBench，不保证适用于 AIME2026。

替换数据集时，只需更改 `BENCHMARKS`，例如
`BENCHMARKS="aime2026 gpqa_diamond livecodebench-lite"`，入口和推理参数保持相同。

DFlash TTS 额外常驻训练权重、梯度、AdamW 状态和 context feature cache。
显存不足时可设置 `DFLASH_MAX_CONTEXT_TOKENS=N`，但这会把训练条件截断到最近
`N` 个 token，不再是完整上下文复现。不要在同一张繁忙 GPU 上启动上述命令。

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
  SPEC_TOKENS=7 \
  CHUNK_SIZE=8 \
  TEMPERATURE=0.6 TOP_P=0.95 TOP_K=20 \
  RUN_LOG=qwen3_8b_ospec.log \
  bash experiments/run.sh ospec \
  > /dev/null 2>&1 < /dev/null &
```

当前 OSpec 在 GPU 侧常驻三个 draft learner。在线训练严格单请求有序。

## 9.1 运行 online_draft

TTS 使用 side CUDA stream 上的 GPU eager trainer；OSpec 仍按请求块同步训练。
相关默认值为：

- `TTS_LEARNING_RATE=1e-5`
- `FUSED_OPTIMIZER=1`
- `LOG_UPDATES=0`

`bash experiments/run.sh online_draft` 等价于 `online_draft+eagle3`，并启用 vLLM
内置的异步 CPU trainer。脚本自动设置 `VLLM_ONLINE_DRAFT_TRAIN=1`。
`online_draft+dflash` 和 `online_draft+mtp` 分别选择远程 DFlash/MTP recipe；
三者仍使用同一异步 CPU pipeline。MTP 不设置 `SPEC_MODEL`，且 target 必须包含
内置 `mtp.*` 权重（Qwen3-8B-MTP 或 Qwen3.5）。与所选方法不匹配的
`VLLM_ONLINE_DRAFT_TRAIN` 环境变量会报错。

EAGLE3 CPU trainer 的默认配置为：

- `CHECK_GRADIENTS=0`
- `FROZEN_PARAMETER_PREFIXES=`（空，即不冻结）
- `FUSED_OPTIMIZER=1`
- `ONLINE_DRAFT_MASTER_WEIGHTS=0`
- `ONLINE_DRAFT_PERSISTENT_GRADS=0`
- `ONLINE_DRAFT_CONTINUOUS_TRAINING=0`
- `ONLINE_DRAFT_TRAIN_SCHEDULE=`（空；设置后按累计 round 数固定训练边界）
- `LOG_UPDATES=1`
- `CPU_TRAINING_ATTENTION_BACKEND=flash_attention`
- `VLLM_ONLINE_DRAFT_PROFILE=0`
- `OMP_NUM_THREADS=20`，`MKL_NUM_THREADS` 默认跟随 `OMP_NUM_THREADS`

三种 draft 共用 `CHECK_GRADIENTS`、`FUSED_OPTIMIZER`、
`FROZEN_PARAMETER_PREFIXES`、`LOG_UPDATES` 和 `VLLM_ONLINE_DRAFT_PROFILE`；
`CPU_TRAINING_ATTENTION_BACKEND` 只用于 EAGLE3/MTP。通过 launcher 的
`--master-weights` 可启用 FP32 master weights 与原生 AdamW；对应的 run.sh
开关是 `ONLINE_DRAFT_MASTER_WEIGHTS=1`。`ONLINE_DRAFT_PERSISTENT_GRADS=1`
复用 linear weight 梯度缓冲，`ONLINE_DRAFT_CONTINUOUS_TRAINING=1` 让 CPU
在已有快照等待安装时继续训练，并在 epoch 关闭时发布最新版本。

复现 10-26 的 `position09` 行为时使用：

```bash
MODEL=./methods/artifacts/models/Qwen3-8B \
SPEC_MODEL=./methods/artifacts/models/Qwen3-8B-eagle3 \
ONLINE_DRAFT_LOSS=position \
ONLINE_DRAFT_MASTER_WEIGHTS=1 \
ONLINE_DRAFT_PERSISTENT_GRADS=1 \
ONLINE_DRAFT_CONTINUOUS_TRAINING=1 \
ONLINE_DRAFT_LR_REFERENCE_ROWS=60 \
  bash experiments/run.sh online_draft+eagle3
```

DFlash 在线更新可以使用相同的 `BENCHMARKS` 数据集选择，例如：

```bash
MODEL=./methods/artifacts/models/Qwen3-8B \
SPEC_MODEL=./methods/artifacts/models/qwen3-8b-dflash \
DATA_DIR=./methods/artifacts/datasets \
BENCHMARKS="aime2026 gpqa_diamond livecodebench-lite" \
  bash experiments/run.sh online_draft+dflash
```

该路径默认使用 15 个 draft token。设置 `ONLINE_DRAFT_LOSS=uniform|position|entropy`
选择全 1、固定位置衰减（ρ=0.9）或 teacher entropy（γ=2）loss。

DFlash 还可以覆盖共用 loss 的深度曲线和归一化方式：

- `ONLINE_DRAFT_DFLASH_LOSS_GAMMA=7`：每个 block 内使用
  `exp(-depth / 7)`，不设置时保持共用的 `0.9 ** depth`。
- `ONLINE_DRAFT_DFLASH_LOSS_NORMALIZATION=mean|weights|block`：分别按 token
  数、权重和或每个 block 等权归一化；默认 `mean`。
- `ONLINE_DRAFT_DFLASH_LOSS_REJECTION_WEIGHT=2`：提高每个 block 首个拒绝位置
  （即 correction token）的权重；默认 `1.0`。
- `ONLINE_DRAFT_DFLASH_LOSS_OBJECTIVE=kl|overlap|ce|kl_overlap`：`overlap`
  直接优化 `1 - sum(min(p_teacher, p_draft))`，即 rejection sampling 的单步接受率。
- `ONLINE_DRAFT_DFLASH_LOSS_TEMPERATURE/TOP_K/TOP_P`：按实际 sampler 的
  temperature、top-k 和 top-p 截断 teacher。student 仍在全词表归一化，因此会
  惩罚落在 teacher 支持集之外、只能被拒绝的 draft 概率。
- `ONLINE_DRAFT_DFLASH_LOSS_REFERENCE_WEIGHT=0.5`：加入对冻结初始 DFlash draft
  的 trust-region KL，抑制短上下文频繁更新造成的漂移。
- `ONLINE_DRAFT_DFLASH_LOSS_REFERENCE_DECAY_TOKENS=4096`：按当前序列长度线性
  释放上述锚定；短序列保留约束，长序列恢复自由适应。

历史 DFlash 配方对应：

```bash
ONLINE_DRAFT_LOSS=position \
ONLINE_DRAFT_DFLASH_LOSS_GAMMA=7 \
ONLINE_DRAFT_DFLASH_LOSS_NORMALIZATION=weights \
  bash experiments/run.sh online_draft+dflash
```

单条短上下文（`aime2026[1]`，5 个 seed）上，任何在线更新都只能接近冻结
draft，不能稳定超过它：冻结 AL 为 `4.577`，历史配方为 `4.438`，
full-vocabulary temperature KL 为 `4.562`，sampler-aligned overlap 加
`reference_weight=0.5` 为 `4.557`。短序列只有一条轨迹可供学习，更新次数多
并不等于信息多；如果必须在线更新，优先使用 full-vocabulary temperature KL：

```bash
ONLINE_DRAFT_LOSS=position \
ONLINE_DRAFT_DFLASH_LOSS_GAMMA=7 \
ONLINE_DRAFT_DFLASH_LOSS_NORMALIZATION=weights \
ONLINE_DRAFT_DFLASH_LOSS_OBJECTIVE=kl \
ONLINE_DRAFT_DFLASH_LOSS_TEMPERATURE=0.6 \
ONLINE_DRAFT_DFLASH_LOSS_TOP_K=0 \
  bash experiments/run.sh online_draft+dflash
```

`TOP_K=0` 的 dense teacher 会给每个 token 提供梯度，避免约 20 次短窗口更新
过度拟合少量 top-k 候选。`REFERENCE_WEIGHT=0.5` 可作为更强的抗漂移选项，但
多 seed 结果波动较大，不作为默认。多请求或较长序列仍应从历史配方开始，因为
此时在线训练可以跨请求积累信息（固定 5 题：在线 `3.602`，冻结 `3.485`）。

这些 CPU 训练默认值与 `methods/artifacts/scripts/benchmark_aime_2026_online.py`
一致，可通过环境变量覆盖。两者共用 `online_draft.runtime` 的捕获、传输和训练
实现：后台独立 D2H stream 将投影后的 teacher logits 复制到 pinned CPU buffer，
再由 CPU 构造 FP32 teacher probabilities。默认 BF16 logits 在 `D=7`、
`V_d=32000` 时占 448,000 字节（约 0.448 MB），不包含 hidden states 等其他数据。
confirmed embeddings 直接从 GPU draft prefill 捕获，没有 sampled token IDs 的
CPU→GPU 回传。传输包仍包含 `num_sampled`；原生路径不计算 teacher coverage。

`online_draft` 三种 drafter 都由 `vllm/v1/worker/gpu/online_draft.py` 创建共享
异步 pipeline；bootstrap 只附加统计回调。EAGLE3 TTS/OSpec 使用 `gpu_factory.py`，
DFlash TTS 使用 `methods/dflash/factory.py` 的 CUDA side-stream adapter。

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
aime2026-10: AL=2.825, weight_installs=120, cpu_rounds/train=8.25, tokens=24987, time=720.8s, tokens/s=34.67, prefill=410 tokens/0.112s/3660.71 tokens/s, decode=24986 tokens/287.400s/86.94 tokens/s
```

指标含义：

- `AL`：`1 + accepted draft tokens / speculative rounds`；Base 为 `1.000`。
- `weight_installs`：该请求期间新训练权重实际安装到 GPU draft 推理模型的次数，
  不包括恢复初始权重；不是 optimizer step 数。Base/EAGLE3 为 0。
- `cpu_rounds/train`：CPU 每次成功训练平均使用的 speculative proposal 轮数，
  即成功训练窗口包含的轮数之和除以训练次数。一轮不是一个生成 token；不包含
  warmup 或请求结束时未训练的尾部轮次。仅 CPU `online_draft` 适用，
  TTS/OSpec/Base/EAGLE3 或没有完成 CPU 训练时显示 `N/A`。
- `tokens`：API 返回的 completion token 数。
- `time`：请求端到端时间，包括 HTTP 请求、该请求后台训练/权重安装收尾等待，
  以及请求后的服务端指标读取；不包含 warmup。
- `tokens/s`：`tokens / time`。
- `prefill`：该题的 prompt token 数、vLLM 服务端 prefill 时间和 prefill
  tokens/s；prefill 时间包含首个输出 token。
- `decode`：该题首 token 之后的输出 token 数、vLLM 服务端 decode 时间和
  decode tokens/s。

JSONL 使用 `weight_installs` 和 `cpu_rounds_per_train` 保存上述训练指标，不再
输出含义容易混淆的 `updates` 字段。统计文件缺失、损坏或等待未完成时，
训练指标显示 `N/A`、JSON 写为 `null`，不再误报为 0。CPU warmup 处理结束后
才开始正式计时和计数；这些统计独立于 `LOG_UPDATES`，无需打开逐次训练日志。
OSpec 在 chunk 边界合并并安装权重，一次合并算一次安装，不按三个 learner 的
optimizer step 数累计；尚未到 chunk 边界的题目可以正常显示安装次数为 0。

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

Base、冻结 EAGLE3、冻结 DFlash 和冻结 MTP 支持恢复，默认 `RESUME=1`。这些入口重新执行相同命令
和 `OUTPUT_FILE` 时：

- Base、EAGLE3、DFlash 和 MTP 保留每道已成功题目的结果，只运行失败或尚未完成的题目。
- 损坏的末行、失败记录、请求配置不匹配或已变化 prompt 对应的旧记录不会被
  视为已完成；重复的成功记录会合并为一条。
- 结果文件会先原子整理再追加；run log、server log 和资源监控 CSV 会继续
  追加。

设置 `RUN_LOG` 后，`RESUME=0` 会覆盖 run log、结果、server log 和监控 CSV；
`RESUME=1` 仅供 Base/EAGLE3/DFlash/MTP 按题恢复并追加。外层 shell 应重定向到 `/dev/null`，避免与
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

- TTS 在 8B draft 上会产生明显 GPU 训练负载；可增大
  `TTS_UPDATE_STRIDE` 来降低更新频率。
