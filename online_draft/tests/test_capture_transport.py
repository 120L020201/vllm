# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from threading import Event
from types import SimpleNamespace

import torch
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


def _packet(step_id: int) -> Eagle3CapturePacket:
    return Eagle3CapturePacket(
        request_id="request-0",
        step_id=step_id,
        source_weight_version=0,
        prompt_context=None,
        proposal_token_input_embeds=torch.zeros(1, 2),
        proposal_recurrent_hidden_states=torch.zeros(1, 2),
        teacher_logits=torch.zeros(2, 3),
        verify_row_indices=torch.tensor([0]),
        target_aux_hidden_states=(torch.zeros(1, 2),),
        num_sampled=torch.tensor(1),
        confirmed_prefill_input_embeds=torch.zeros(1, 2),
        confirmed_prefill_positions=torch.tensor([0]),
        confirmed_prefill_length=torch.tensor(1),
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
