# TTS adaptation

Start with `.venv/bin/python -m methods.run tts --learning-rate 2e-5
--update-stride 1`. Target distributions and EAGLE3 features captured during
GPU verification drive one CPU confirmed-path forward-KL update on each
eligible round. Parameters and optimizer reset after each request unless
`--keep-weights` is specified. See [method limitations](../README.md).
