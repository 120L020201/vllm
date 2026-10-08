# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import pytest
import torch
from online_draft.runtime.capture_runtime import Eagle3CaptureRuntime
from online_draft.runtime.transport import (
    Eagle3CapturePacket,
    Eagle3CaptureQueue,
)
from online_draft.runtime.vllm_adapter import Eagle3VllmCaptureAdapter


class _CaptureRuntime:
    def __init__(self) -> None:
        self.packets: list[Eagle3CapturePacket] = []

    def submit(self, packet: Eagle3CapturePacket) -> int:
        self.packets.append(packet)
        return len(self.packets)


def _packet(
    step_id: int,
    device: torch.device | str = "cpu",
) -> Eagle3CapturePacket:
    return Eagle3CapturePacket(
        request_id="request-0",
        step_id=step_id,
        source_weight_version=0,
        prompt_context=None,
        proposal_token_input_embeds=torch.zeros(1, 2, device=device),
        proposal_recurrent_hidden_states=torch.zeros(1, 2, device=device),
        teacher_logits=torch.zeros(2, 3, device=device),
        verify_row_indices=torch.tensor([0], device=device),
        target_aux_hidden_states=(torch.zeros(1, 2, device=device),),
        num_sampled=torch.tensor(1, device=device),
        confirmed_prefill_input_embeds=torch.zeros(1, 2, device=device),
        confirmed_prefill_positions=torch.tensor([0], device=device),
        confirmed_prefill_length=torch.tensor(1, device=device),
    )


def test_close_epoch_flushes_packets_before_marker() -> None:
    events: list[object] = []
    queue_sizes: list[int] = []
    closed = Event()

    queue = Eagle3CaptureQueue(
        transfer_chunk_size=8,
        consume_chunk=lambda packets: events.append(
            ("rounds", [packet.step_id for packet in packets])
        ),
        on_finish=lambda request_id: None,
        on_close_epoch=lambda request_id: (
            events.append(("close", request_id)),
            closed.set(),
        ),
        on_queue_size=queue_sizes.append,
    )

    try:
        queue.submit(_packet(0))
        assert len(queue_sizes) == 1
        queue.close_epoch("request-0")

        assert closed.wait(timeout=5)
        assert events == [
            ("rounds", [0]),
            ("close", "request-0"),
        ]
    finally:
        queue.close()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires CUDA",
)
def test_cuda_packet_uses_pinned_d2h_buffers() -> None:
    consumed: list[Eagle3CapturePacket] = []
    copied = Event()

    def consume(
        packets: tuple[Eagle3CapturePacket, ...],
    ) -> None:
        consumed.extend(packets)
        copied.set()

    packet = _packet(0, "cuda")
    packet.ready = torch.cuda.Event()
    packet.ready.record(torch.cuda.current_stream())

    queue = Eagle3CaptureQueue(
        transfer_chunk_size=1,
        consume_chunk=consume,
        on_finish=lambda request_id: None,
        on_close_epoch=lambda request_id: None,
    )

    try:
        queue.submit(packet)

        assert copied.wait(timeout=5)
        assert len(consumed) == 1

        cpu_packet = consumed[0]
        assert cpu_packet.ready is None
        assert cpu_packet.teacher_logits.device.type == "cpu"
        assert cpu_packet.teacher_logits.is_pinned()
        assert cpu_packet.verify_row_indices.is_pinned()
        assert torch.equal(
            cpu_packet.teacher_logits,
            torch.zeros(2, 3),
        )
    finally:
        queue.close()


def test_finish_discards_tail_and_waits_for_callback() -> None:
    consumed: list[tuple[Eagle3CapturePacket, ...]] = []
    finished: list[str] = []
    callback_started = Event()
    release_callback = Event()

    def on_finish(request_id: str) -> None:
        callback_started.set()
        release_callback.wait(timeout=5)
        finished.append(request_id)

    queue = Eagle3CaptureQueue(
        transfer_chunk_size=8,
        consume_chunk=consumed.append,
        on_finish=on_finish,
        on_close_epoch=lambda request_id: None,
    )

    try:
        queue.submit(_packet(0))

        with ThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(
                queue.finish_request,
                "request-0",
            )

            assert callback_started.wait(timeout=5)
            assert not result.done()

            release_callback.set()
            result.result(timeout=5)

        assert consumed == []
        assert finished == ["request-0"]
    finally:
        release_callback.set()
        queue.close()


def test_capture_runtime_forwards_finish() -> None:
    finished: list[str] = []

    runtime = Eagle3CaptureRuntime(
        hidden_size=2,
        num_aux_hidden_states=1,
        draft_vocab_size=3,
        feature_dtype=torch.float32,
        transfer_chunk_size=8,
        on_round=lambda prepared_round: None,
        on_epoch_close=lambda request_id: None,
        on_finish=finished.append,
    )

    try:
        runtime.finish_request("request-0")

        assert finished == ["request-0"]
    finally:
        runtime.close()


def test_next_prefill_submits_previous_proposal() -> None:
    runtime = _CaptureRuntime()
    adapter = Eagle3VllmCaptureAdapter(
        runtime=runtime,  # type: ignore[arg-type]
        target_token_ids=torch.tensor([1, 3]),
        source_weight_version=lambda: 7,
    )
    target_aux = (torch.tensor([[1.0, 2.0], [3.0, 4.0]]),)

    adapter.capture_proposal(
        request_id="request-0",
        verify_row_indices=torch.tensor([1]),
        target_aux_hidden_states=target_aux,
        proposal=SimpleNamespace(
            prefill_input_embeds=torch.tensor([[10.0, 11.0]]),
            prefill_positions=torch.tensor([4]),
            prefill_length=torch.tensor(1),
            draft_token_input_embeds=torch.tensor([[12.0, 13.0]]),
            draft_recurrent_hidden_states=torch.tensor([[14.0, 15.0]]),
        ),
    )
    teacher_logits = torch.tensor([[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]])
    adapter.capture_teacher_logits(teacher_logits)

    next_prefill_input_embeds = torch.tensor([[20.0, 21.0]])
    adapter.capture_prefill(
        request_id="request-0",
        verify_row_indices=torch.tensor([1]),
        target_aux_hidden_states=target_aux,
        num_sampled=1,
        prefill_input_embeds=next_prefill_input_embeds,
        prefill_positions=torch.tensor([5]),
        prefill_length=torch.tensor(1),
    )

    assert len(runtime.packets) == 1
    packet = runtime.packets[0]
    assert packet.source_weight_version == 7
    assert packet.prompt_context is not None
    assert torch.equal(
        packet.prompt_context.prefill_input_embeds,
        torch.tensor([[10.0, 11.0]]),
    )
    assert torch.equal(packet.teacher_logits, teacher_logits[:, [1, 3]])
    assert torch.equal(
        packet.confirmed_prefill_input_embeds,
        next_prefill_input_embeds,
    )
