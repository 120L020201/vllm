# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Opt-in statistics for the built-in asynchronous CPU draft trainer."""

import time
from functools import wraps

from methods.utils.stats import WorkerStats

_INSTALLED = False
_INTERNAL_REQUEST_PREFIXES = ("_dummy_req_", "_warmup_", "chatcmpl-_warmup_")


def attach_cpu_stats(speculator, pipeline) -> WorkerStats:
    stats = WorkerStats()
    recording = False
    original_propose = speculator.propose
    trainer = pipeline._trainer
    original_train = pipeline._train
    installer = speculator.draft_weight_installer
    original_advance = installer.advance
    original_finish = speculator.finish_request
    original_publish = getattr(pipeline, "_publish_snapshot", None)
    original_close_epoch = getattr(pipeline, "_handle_close_epoch", None)
    published_at: float | None = None

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
    def train(rounds, *args, **kwargs):
        previous_version = trainer.version
        started = time.perf_counter()
        result = original_train(rounds, *args, **kwargs)
        if recording and trainer.version > previous_version:
            trained_rounds = pipeline.update_stats[-1]["trained_rounds"]
            if trained_rounds:
                stats.record_cpu_train(
                    trained_rounds,
                    time.perf_counter() - started,
                )
        return result

    def publish_snapshot(*args, **kwargs):
        nonlocal published_at
        result = original_publish(*args, **kwargs)
        published_at = time.perf_counter()
        return result

    def handle_close_epoch(*args, **kwargs):
        nonlocal published_at
        if recording and published_at is not None:
            stats.record_cpu_install_wait(time.perf_counter() - published_at)
        published_at = None
        return original_close_epoch(*args, **kwargs)

    @wraps(original_advance)
    def advance(snapshot):
        result = original_advance(snapshot)
        if recording and result is not None:
            stats.record_weight_install()
        return result

    @wraps(original_finish)
    def finish_request(*args, **kwargs):
        nonlocal recording, published_at
        result = original_finish(*args, **kwargs)
        if stats.active_requests:
            stats.end_request()
        recording = False
        published_at = None
        return result

    speculator.propose = propose
    pipeline._train = train
    installer.advance = advance
    speculator.finish_request = finish_request
    if original_publish is not None:
        pipeline._publish_snapshot = publish_snapshot
    if original_close_epoch is not None:
        pipeline._handle_close_epoch = handle_close_epoch
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
        runtime = self._online_draft
        attach_cpu_stats(runtime.speculator, runtime.pipeline)
        return result

    Worker._load_online_draft_model = load_online_draft_model
    _INSTALLED = True
