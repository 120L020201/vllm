# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Capture and update DFlash proposals without editing vLLM sources."""

import concurrent.futures
import logging
import os
from dataclasses import dataclass

import torch
from online_draft.training.trainer import DraftTrainer

from methods.dflash.training import (
    DflashTrainingBatch,
    rebuild_fused_context_buffers,
    target_token_ids,
    train_dflash_round,
)
from methods.random_sampling.gating import BernoulliGate
from methods.tts_common.loss import project_teacher_distribution
from methods.utils.cuda_worker import CudaStreamResult, CudaStreamWorker
from methods.utils.stats import WorkerStats

logger = logging.getLogger(__name__)
_INTERNAL_REQUEST_PREFIXES = (
    "_dummy_req_",
    "_warmup_",
    "chatcmpl-_warmup_",
)


def _clone(tensor: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode(False):
        return tensor.detach().clone()


def _export_trainable_state(trainer: DraftTrainer) -> dict[str, torch.Tensor]:
    state = trainer.model.state_dict()
    return {name: state[name].detach() for name in trainer.trainable_parameter_names}


@dataclass(slots=True)
class _Proposal:
    request_id: str
    context_hidden_states: torch.Tensor | None
    context_positions: torch.Tensor | None
    query_input_ids: torch.Tensor
    query_positions: torch.Tensor


class _ContextCache:
    """Growable ring of accepted context features for one request."""

    def __init__(
        self,
        capacity: int,
        width: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if capacity < 1 or width < 1:
            raise ValueError("context cache capacity and width must be positive")
        self._capacity = capacity
        self._features = torch.empty((capacity, width), dtype=dtype, device=device)
        self._positions = torch.empty(capacity, dtype=torch.long, device=device)
        self._start = 0
        self._end = 0

    @property
    def features(self) -> torch.Tensor:
        return self._features[self._start : self._end]

    @property
    def positions(self) -> torch.Tensor:
        return self._positions[self._start : self._end]

    def clear(self) -> None:
        self._start = 0
        self._end = 0

    def append(self, features: torch.Tensor, positions: torch.Tensor) -> None:
        """Append one accepted chunk, dropping the oldest rows if needed."""
        if features.ndim != 2 or positions.ndim != 1:
            raise ValueError("context features and positions have invalid ranks")
        if features.shape[0] != positions.numel() or features.shape[0] == 0:
            raise ValueError("context features and positions must be nonempty")
        if features.shape[1] != self._features.shape[1]:
            raise ValueError("context feature width changed between rounds")
        if features.device != self._features.device:
            raise ValueError("context features must stay on one device")
        if positions.device != self._positions.device:
            raise ValueError("context positions must stay on one device")
        count = features.shape[0]
        if count > self._capacity:
            features = features[-self._capacity :]
            positions = positions[-self._capacity :]
            count = self._capacity
        if self._end > self._start and int(positions[0]) != (
            int(self._positions[self._end - 1]) + 1
        ):
            raise ValueError("accepted context positions must be contiguous")

        live = self._end - self._start
        drop = max(0, live + count - self._capacity)
        if drop:
            self._start += drop
            live -= drop
        if self._start != 0:
            self._features[:live].copy_(self._features[self._start : self._end].clone())
            self._positions[:live].copy_(
                self._positions[self._start : self._end].clone()
            )
            self._start = 0
            self._end = live
        self._features[self._end : self._end + count].copy_(features)
        self._positions[self._end : self._end + count].copy_(positions)
        self._end += count


@dataclass(slots=True)
class _Verification:
    teacher_probabilities: torch.Tensor
    teacher_coverage: torch.Tensor
    num_sampled: int


@dataclass(frozen=True, slots=True)
class _DflashWeightSnapshot:
    version: int
    state_dict: dict[str, torch.Tensor]


class DflashPolicyBridge:
    """Gate DFlash updates and overlap training on a side stream."""

    def __init__(
        self,
        trainer: DraftTrainer,
        draft_model: torch.nn.Module,
        *,
        method: str,
        max_context_tokens: int = 0,
    ) -> None:
        if method not in {"tts", "random_sampling"}:
            raise ValueError(f"unsupported DFlash method: {method}")
        if max_context_tokens < 0:
            raise ValueError("max_context_tokens must be nonnegative")
        self.trainer = trainer
        self.draft_model = draft_model
        self.method = method
        self.stride = int(os.environ.get("OSD_UPDATE_STRIDE", "1"))
        if self.stride < 1:
            raise ValueError("update stride must be positive")
        self.max_context_tokens = max_context_tokens
        self.gate = None
        if method == "random_sampling":
            self.gate = BernoulliGate(
                float(os.environ.get("OSD_PROBABILITY", "0.5")),
                int(os.environ.get("OSD_SEED", "0")),
                self.stride,
            )
        self.stats = WorkerStats()
        self.log_updates = os.environ.get("OSD_LOG_UPDATES", "0") == "1"
        self._request_id: str | None = None
        self._next_step_id = 0
        self._rounds = 0
        self._side_worker: CudaStreamWorker | None = None
        if method == "tts" and next(trainer.model.parameters()).device.type == "cuda":
            if trainer.model is draft_model:
                raise ValueError("asynchronous DFlash TTS requires a training copy")
            self._side_worker = CudaStreamWorker(
                next(trainer.model.parameters()).device,
                name="dflash_tts_cuda_side_stream",
            )
        self._pending_update: (
            tuple[
                int,
                concurrent.futures.Future[CudaStreamResult[_DflashWeightSnapshot]],
            ]
            | None
        ) = None
        self._reset_snapshot: _DflashWeightSnapshot | None = None

    def observe(self, request_id: str, step_id: int, batch: DflashTrainingBatch) -> int:
        """Schedule or run one verified-round update and return its count."""
        self.stats.begin_request()
        self.stats.enqueue()
        if self._side_worker is not None:
            try:
                return self._observe_async(request_id, step_id, batch)
            except Exception:
                self.stats.complete(0)
                raise
        updates = 0
        try:
            updates = self._observe(request_id, step_id, batch)
        finally:
            self.stats.complete(updates)
        return updates

    def _observe_async(
        self,
        request_id: str,
        step_id: int,
        batch: DflashTrainingBatch,
    ) -> int:
        if self._request_id is None:
            self._request_id = request_id
        if request_id != self._request_id:
            raise ValueError("DFlash methods support one active request per worker")
        if step_id != self._next_step_id:
            raise ValueError(f"expected step {self._next_step_id}, got {step_id}")
        self._next_step_id += 1
        self._rounds += 1
        should_train = self._rounds % self.stride == 0
        if self.gate is not None:
            should_train = self.gate.select(request_id)
        if not should_train:
            self.stats.complete(0)
            return 0
        if self._pending_update is not None:
            raise RuntimeError(
                "the prior DFlash TTS update has not reached its boundary"
            )

        context_hidden_states = batch.context_hidden_states
        context_positions = batch.context_positions
        if self.max_context_tokens and (
            context_hidden_states.shape[0] > self.max_context_tokens
        ):
            context_hidden_states = context_hidden_states[-self.max_context_tokens :]
            context_positions = context_positions[-self.max_context_tokens :]
        training_batch = DflashTrainingBatch(
            context_hidden_states=_clone(context_hidden_states),
            context_positions=_clone(context_positions),
            query_input_ids=_clone(batch.query_input_ids),
            query_positions=_clone(batch.query_positions),
            teacher_probabilities=_clone(batch.teacher_probabilities),
            teacher_coverage=_clone(batch.teacher_coverage),
            rejection_position=batch.rejection_position,
        )
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(training_batch.query_input_ids.device))

        def train_update() -> _DflashWeightSnapshot:
            updates = 0
            try:
                loss = train_dflash_round(
                    self.trainer,
                    self.trainer.model,
                    training_batch,
                    max_context_tokens=self.max_context_tokens,
                )
                updates = 1
                if self.log_updates:
                    logger.info(
                        "%s DFlash TTS update: request=%s step=%d version=%d loss=%.6f",
                        self.method,
                        request_id,
                        step_id,
                        self.trainer.version,
                        loss,
                    )
                return _DflashWeightSnapshot(
                    self.trainer.version,
                    _export_trainable_state(self.trainer),
                )
            finally:
                self.stats.complete(updates)

        future = self._side_worker.submit(
            train_update,
            inputs=training_batch,
            ready=ready,
        )
        self._pending_update = (self._rounds, future)
        return 1

    def maybe_apply_pending_weights(
        self,
        *,
        force: bool = False,
    ) -> _DflashWeightSnapshot | None:
        if self._reset_snapshot is not None:
            snapshot = self._reset_snapshot
            self._reset_snapshot = None
            return snapshot
        if self._side_worker is None or self._pending_update is None:
            return None
        update_round, future = self._pending_update
        if not force and self._rounds - update_round < self.stride - 1:
            return None
        result = future.result()
        result.completed.synchronize()
        self._pending_update = None
        return result.value

    def _observe(
        self,
        request_id: str,
        step_id: int,
        batch: DflashTrainingBatch,
    ) -> int:
        if self._request_id is None:
            self._request_id = request_id
        if request_id != self._request_id:
            raise ValueError("DFlash methods support one active request per worker")
        if step_id != self._next_step_id:
            raise ValueError(f"expected step {self._next_step_id}, got {step_id}")
        self._next_step_id += 1
        self._rounds += 1

        should_train = self._rounds % self.stride == 0
        if self.gate is not None:
            should_train = self.gate.select(request_id)
        if not should_train:
            return 0

        loss = train_dflash_round(
            self.trainer,
            self.draft_model,
            batch,
            max_context_tokens=self.max_context_tokens,
        )
        self.stats.record_weight_install()
        if self.log_updates:
            logger.info(
                "%s DFlash draft update: request=%s step=%d version=%d loss=%.6f",
                self.method,
                request_id,
                step_id,
                self.trainer.version,
                loss,
            )
        return 1

    def reset_request(self, request_id: str) -> None:
        if self._request_id is not None and request_id != self._request_id:
            return
        if self.gate is not None:
            self.gate.reset(request_id)
        if self._side_worker is not None:
            keep_weights = os.environ.get("OSD_KEEP_WEIGHTS", "0") == "1"

            def reset_tts() -> _DflashWeightSnapshot | None:
                if keep_weights:
                    if self._pending_update is None:
                        return None
                elif self.trainer.version == 0:
                    return None
                else:
                    self.trainer.reset()
                    rebuild_fused_context_buffers(self.trainer.model)
                return _DflashWeightSnapshot(
                    self.trainer.version,
                    _export_trainable_state(self.trainer),
                )

            reset_result = self._side_worker.submit(reset_tts).result()
            reset_result.completed.synchronize()
            self._reset_snapshot = reset_result.value
            self._pending_update = None
            self._request_id = None
            self._next_step_id = 0
            self._rounds = 0
            self.stats.end_request()
            return
        if os.environ.get("OSD_KEEP_WEIGHTS", "0") != "1" and self.trainer.version > 0:
            self.trainer.reset()
            rebuild_fused_context_buffers(self.draft_model)
        self._request_id = None
        self._next_step_id = 0
        self._rounds = 0
        self.stats.end_request()

    def shutdown(self) -> None:
        if self._side_worker is not None:
            self._side_worker.shutdown()


class DflashOnlineAdapter:
    """Hold one pending proposal and train it after verification."""

    def __init__(
        self,
        bridge: DflashPolicyBridge,
        draft_model: torch.nn.Module,
        *,
        num_speculative_steps: int,
        max_context_tokens: int = 0,
        max_model_len: int = 32768,
        context_width: int | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> None:
        if num_speculative_steps < 1:
            raise ValueError("num_speculative_steps must be positive")
        if max_context_tokens < 0 or max_model_len < 1:
            raise ValueError("context limits must be nonnegative")
        model = draft_model.model
        if context_width is None:
            context_width = (
                model.fc.weight.shape[1]
                if model.use_aux_hidden_state
                else model.hidden_norm.weight.numel()
            )
        if dtype is None:
            dtype = draft_model.lm_head.weight.dtype
        if device is None:
            device = draft_model.lm_head.weight.device
        capacity = (
            min(max_model_len, max_context_tokens)
            if max_context_tokens
            else (max_model_len)
        )
        self.bridge = bridge
        self.draft_model = draft_model
        self.num_speculative_steps = num_speculative_steps
        self.max_context_tokens = max_context_tokens
        self._draft_vocab_size = int(draft_model.config.draft_vocab_size)
        self._target_token_ids = target_token_ids(draft_model)
        self._context = _ContextCache(
            capacity,
            context_width,
            dtype=dtype,
            device=device,
        )
        self._pending: _Proposal | None = None
        self._verification: _Verification | None = None
        self._request_id: str | None = None
        self._step_id = 0
        self._warned_row_mismatch = False

    def before_propose(self, input_batch) -> None:
        """Train the previously verified round, if one is complete."""
        if self._pending is None:
            return
        if input_batch.req_ids and input_batch.req_ids[0].startswith(
            _INTERNAL_REQUEST_PREFIXES
        ):
            return
        if self._verification is None or input_batch.req_ids != [self._request_id]:
            self._pending = None
            self._verification = None
            return
        batch = self._build_batch()
        request_id = self._request_id
        step_id = self._step_id
        self._pending = None
        self._verification = None
        assert request_id is not None
        assert isinstance(batch, DflashTrainingBatch)
        self.bridge.observe(request_id, step_id, batch)
        snapshot = self.bridge.maybe_apply_pending_weights()
        if snapshot is not None:
            self._apply_snapshot(snapshot)
        self._step_id += 1

    def _apply_snapshot(self, snapshot: _DflashWeightSnapshot) -> None:
        state = self.draft_model.state_dict()
        with torch.no_grad():
            for name, source in snapshot.state_dict.items():
                target = state[name]
                if source.dtype != target.dtype:
                    raise ValueError("DFlash training and inference weights differ")
                if source.shape != target.shape:
                    raise ValueError(f"shape mismatch for DFlash weight {name}")
                target.copy_(
                    source,
                    non_blocking=source.device.type == "cpu" and source.is_pinned(),
                )
        rebuild_fused_context_buffers(self.draft_model)
        if self.draft_model.lm_head.weight.device.type == "cuda":
            torch.cuda.current_stream(
                self.draft_model.lm_head.weight.device
            ).synchronize()
        if snapshot.version > 0:
            self.bridge.stats.record_weight_install()

    def capture_proposal(
        self,
        speculator,
        input_batch,
        *,
        aux_hidden_states: list[torch.Tensor] | None,
        last_hidden_states: torch.Tensor,
        num_rejected: torch.Tensor,
    ) -> None:
        """Capture the context and query block of a live DFlash proposal."""
        if input_batch.num_reqs != 1:
            raise ValueError("DFlash online methods require exactly one request")
        if not input_batch.req_ids or input_batch.req_ids[0].startswith(
            _INTERNAL_REQUEST_PREFIXES
        ):
            return
        valid_context = int(input_batch.num_tokens) - int(num_rejected[0].item())
        if valid_context < 1:
            self._pending = None
            return
        model = speculator.model.model
        if model.use_aux_hidden_state:
            if not aux_hidden_states:
                raise ValueError("DFlash online methods require auxiliary states")
            context = torch.cat(aux_hidden_states, dim=-1)[:valid_context]
            expected_width = model.fc.weight.shape[1]
        else:
            context = last_hidden_states[:valid_context]
            expected_width = model.hidden_norm.weight.numel()
        if context.shape[-1] != expected_width:
            raise ValueError(
                "DFlash context features do not match the drafter input width: "
                f"{context.shape[-1]} != {expected_width}"
            )
        if self._pending is not None:
            raise RuntimeError("capture_proposal requires the previous round to train")
        context_positions = speculator.context_positions[:valid_context]
        assert self._context is not None
        self._context.append(context, context_positions)
        captured_context = self._context.features
        captured_positions = self._context.positions
        num_query_tokens = speculator.num_query_per_req
        self._pending = _Proposal(
            request_id=input_batch.req_ids[0],
            context_hidden_states=captured_context,
            context_positions=captured_positions,
            query_input_ids=_clone(
                speculator.input_buffers.input_ids[:num_query_tokens]
            ).to(dtype=torch.long),
            query_positions=_clone(
                speculator.input_buffers.positions[:num_query_tokens]
            ).to(dtype=torch.long),
        )
        self._request_id = input_batch.req_ids[0]
        self._verification = None

    def capture_verification(self, logits, input_batch, result) -> None:
        """Project the target distribution of the pending block."""
        if self._pending is None:
            return
        if input_batch.req_ids and input_batch.req_ids[0].startswith(
            _INTERNAL_REQUEST_PREFIXES
        ):
            return
        if input_batch.req_ids != [self._request_id]:
            self._pending = None
            self._verification = None
            return
        expected_rows = self.num_speculative_steps + 1
        if logits.shape[0] != expected_rows:
            if not self._warned_row_mismatch:
                logger.warning(
                    "skipping DFlash draft update: expected %d verification "
                    "logits, got %d",
                    expected_rows,
                    logits.shape[0],
                )
                self._warned_row_mismatch = True
            self._pending = None
            self._verification = None
            return
        num_sampled = int(result.num_sampled[0].item())
        if not 1 <= num_sampled <= self.num_speculative_steps + 1:
            self._verification = None
            return
        teacher_logits = logits[: self.num_speculative_steps]
        with torch.inference_mode(False):
            probabilities, coverage = project_teacher_distribution(
                teacher_logits,
                self._target_token_ids,
            )
        self._verification = _Verification(
            teacher_probabilities=probabilities,
            teacher_coverage=coverage.clamp(0.0, 1.0),
            num_sampled=num_sampled,
        )

    def _build_batch(self) -> DflashTrainingBatch:
        assert self._pending is not None
        assert self._verification is not None
        teacher_probabilities = self._verification.teacher_probabilities
        teacher_coverage = self._verification.teacher_coverage
        rejection_position = (
            self._verification.num_sampled - 1
            if self._verification.num_sampled <= self.num_speculative_steps
            else None
        )
        assert self._pending.context_hidden_states is not None
        assert self._pending.context_positions is not None
        batch = DflashTrainingBatch(
            context_hidden_states=self._pending.context_hidden_states,
            context_positions=self._pending.context_positions,
            query_input_ids=self._pending.query_input_ids,
            query_positions=self._pending.query_positions,
            teacher_probabilities=teacher_probabilities,
            teacher_coverage=teacher_coverage,
            rejection_position=rejection_position,
        )
        batch.validate(
            draft_vocab_size=self._draft_vocab_size,
            num_speculative_steps=self.num_speculative_steps,
        )
        return batch

    def reset_request(self, request_id: str) -> None:
        if request_id.startswith(_INTERNAL_REQUEST_PREFIXES):
            return
        if self._request_id is not None and request_id != self._request_id:
            return
        if self._pending is not None and self._verification is not None:
            assert self._request_id is not None
            batch = self._build_batch()
            assert isinstance(batch, DflashTrainingBatch)
            self.bridge.observe(self._request_id, self._step_id, batch)
            self._step_id += 1
        self._pending = None
        self._verification = None
        self._request_id = None
        self._step_id = 0
        if self._context is not None:
            self._context.clear()
        self.bridge.reset_request(request_id)
        snapshot = self.bridge.maybe_apply_pending_weights(force=True)
        if snapshot is not None:
            self._apply_snapshot(snapshot)

    def shutdown(self) -> None:
        self._pending = None
        self._verification = None
        if self._context is not None:
            self._context.clear()
        self.bridge.shutdown()
