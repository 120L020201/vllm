# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Opt-in worker bootstrap; inherited by vLLM spawned workers via PYTHONPATH."""

import os

if os.environ.get("OSD_METHOD") in {"tts", "random_sampling", "ospec"}:
    from methods.utils.factory import install

    install()
