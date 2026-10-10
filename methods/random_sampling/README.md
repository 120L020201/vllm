# Random sampling adaptation

Start with `.venv/bin/python -m methods.run random_sampling --probability 0.5
--seed 0`. Eligible rounds use the TTS adaptation's optimizer step. The
per-request Bernoulli RNG is isolated from vLLM's GPU sampler. At probability
one this has the same update policy as the adapted TTS method.
