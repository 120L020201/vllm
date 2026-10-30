# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Opt-in worker bootstrap; inherited by vLLM spawned workers via PYTHONPATH."""

import os

_METHOD = os.environ.get("OSD_METHOD")
_DRAFT_METHOD = os.environ.get("OSD_DRAFT_METHOD", "eagle3")

if _METHOD in {"tts", "random_sampling", "ospec"}:
    if _DRAFT_METHOD == "dflash":
        from methods.dflash.factory import install
    else:
        from methods.utils.gpu_factory import install

    install()
elif _METHOD == "online_draft":
    from methods.utils.cpu_stats import install

    install()
