# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from threading import Event

import torch
from online_draft.runtime.transport import (
    Eagle3CapturePacket,
    Eagle3CaptureQueue,
)


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
