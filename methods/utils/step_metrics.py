# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Record per-round speculative acceptance for every finished request."""

import json
import os

import torch

_INTERNAL_REQUEST_PREFIXES = (
    "_dummy_req_",
    "_warmup_",
    "chatcmpl-_warmup_",
)
_RECORDER: "StepMetricsRecorder | None" = None
_INSTALLED = False


class StepMetricsRecorder:
    """Keep each round's sampled count on the GPU until its request finishes.

    Every verification round appends one ``num_sampled`` element (accepted
    draft tokens plus the bonus) without a host sync; the request's rounds
    are written as one JSON line when vLLM finishes it.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self._requests: dict[str, dict] = {}

    def observe(self, input_batch, num_sampled: torch.Tensor) -> None:
        drafts = input_batch.num_draft_tokens_per_req
        if drafts is None:
            return
        for index, request_id in enumerate(input_batch.req_ids[: input_batch.num_reqs]):
            if request_id.startswith(_INTERNAL_REQUEST_PREFIXES) or not drafts[index]:
                continue
            record = self._requests.setdefault(
                request_id,
                {
                    "prompt_tokens": int(input_batch.prefill_len_np[index]),
                    "draft_tokens": [],
                    "num_sampled": [],
                },
            )
            record["draft_tokens"].append(int(drafts[index]))
            record["num_sampled"].append(num_sampled[index : index + 1].clone())

    def finish(self, request_ids) -> None:
        for request_id in request_ids:
            record = self._requests.pop(request_id, None)
            if record is None:
                continue
            sampled = torch.cat(record["num_sampled"]).tolist()
            line = {
                "request_id": request_id,
                "prompt_tokens": record["prompt_tokens"],
                "draft_tokens": record["draft_tokens"],
                "accepted_tokens": [count - 1 for count in sampled],
            }
            with open(self.path, "a", encoding="utf-8") as output:
                output.write(json.dumps(line, separators=(",", ":")) + "\n")


def install() -> None:
    """Record rounds into ``OSD_STEP_METRICS_FILE`` in the worker process."""
    global _INSTALLED, _RECORDER
    if _INSTALLED:
        return
    _INSTALLED = True
    _RECORDER = StepMetricsRecorder(os.environ["OSD_STEP_METRICS_FILE"])

    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

    original_rejection_call = RejectionSampler.__call__
    original_finish_requests = GPUModelRunner.finish_requests

    def rejection_call(self, logits, input_batch, *args, **kwargs):
        result = original_rejection_call(self, logits, input_batch, *args, **kwargs)
        _RECORDER.observe(input_batch, result.num_sampled)
        return result

    def finish_requests(self, scheduler_output):
        _RECORDER.finish(scheduler_output.finished_req_ids or ())
        return original_finish_requests(self, scheduler_output)

    RejectionSampler.__call__ = rejection_call
    GPUModelRunner.finish_requests = finish_requests
