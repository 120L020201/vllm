# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""DFlash baseline and its test-time draft updates.

The frozen baseline is vLLM's native DFlash speculator. The `online_draft`,
`tts`, and `random_sampling` entries can additionally adapt the DFlash drafter
through an opt-in worker bootstrap. Online draft training runs on CPU; TTS
training runs on a CUDA side stream.
"""
