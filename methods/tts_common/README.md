# TTS adaptation

Start with `.venv/bin/python -m methods.run tts --learning-rate 1e-5
--update-stride 1`. Each update uses the full proposal canvas and target
distributions captured during verification. TTS runs training on a side CUDA
stream while the draft serves the next `S-1` rounds, then installs the update
before generating the next canvas. Parameters and optimizer reset after each
request unless `--keep-weights` is specified.
TTS owns its GPU training primitives under `methods/tts_common/training`; none of the TTS
entry points imports the `online_draft` package or its CPU asynchronous
pipeline.
The paper's Qwen3-8B LiveCodeBench sweep found its best throughput at
`--update-stride 10` (1.83x relative speedup with 0.88x the `S=1` acceptance
length); the CLI keeps `S=1` as the algorithm default.

Add `--draft-method dflash` to adapt a DFlash block-diffusion drafter instead
of EAGLE-3. DFlash TTS trains a separate copy and installs its trainable
weights at the same stride boundary. See [DFlash methods](../dflash/README.md)
for memory and scope limits, and [method limitations](../README.md).
