# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Opt-in statistics for the built-in asynchronous CPU draft trainer."""

from functools import wraps

from methods.utils.stats import WorkerStats

_INSTALLED = False
_INTERNAL_REQUEST_PREFIXES = ("_dummy_req_", "_warmup_", "chatcmpl-_warmup_")


def attach_cpu_stats(speculator, pipeline) -> WorkerStats:
    stats = WorkerStats()
    recording = False
    original_propose = speculator.propose
    original_train = pipeline._train_window
    original_install = speculator.install_committed_weights
    original_finish = speculator.finish_request

    @wraps(original_propose)
    def propose(input_batch, *args, **kwargs):
        nonlocal recording
        if not kwargs.get("dummy_run", False) and not kwargs.get("is_profile", False):
            recording = bool(input_batch.req_ids) and not any(
                request_id.startswith(_INTERNAL_REQUEST_PREFIXES)
                for request_id in input_batch.req_ids
            )
            if input_batch.req_ids and not stats.active_requests:
                stats.begin_request()
        return original_propose(input_batch, *args, **kwargs)

    @wraps(original_train)
    def train_window(trainer, window, *args, **kwargs):
        result = original_train(trainer, window, *args, **kwargs)
        if recording:
            stats.record_cpu_train(len(window.rounds))
        return result

    @wraps(original_install)
    def install_weights(*args, **kwargs):
        result = original_install(*args, **kwargs)
        if recording:
            stats.record_weight_install()
        return result

    @wraps(original_finish)
    def finish_request(*args, **kwargs):
        nonlocal recording
        result = original_finish(*args, **kwargs)
        if stats.active_requests:
            stats.end_request()
        recording = False
        return result

    speculator.propose = propose
    pipeline._train_window = train_window
    speculator.install_committed_weights = install_weights
    speculator.finish_request = finish_request
    return stats


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from vllm.v1.worker.gpu_worker import Worker

    original_load = Worker._load_online_draft_model

    @wraps(original_load)
    def load_online_draft_model(self):
        result = original_load(self)
        attach_cpu_stats(self._online_draft_speculator, self._online_draft_pipeline)
        return result

    Worker._load_online_draft_model = load_online_draft_model
    _INSTALLED = True
