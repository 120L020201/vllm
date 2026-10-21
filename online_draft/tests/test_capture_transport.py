# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import pytest
import torch
from online_draft.runtime.capture.dflash import (
    DFlashCaptureAdapter,
    DFlashCaptureBuffers,
)
from online_draft.runtime.capture.eagle3 import (
    Eagle3CaptureAdapter,
    Eagle3CapturePacket,
)
from online_draft.runtime.transport import CaptureQueue


@pytest.mark.parametrize("accepted", [0, 1, 3])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_dflash_bootstrap_chunks_and_teacher_alignment(accepted, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    rounds = []
    done = Event()
    adapter = DFlashCaptureAdapter(
        buffers=DFlashCaptureBuffers(
            8, 4, 16, 23, dtype=torch.float32, teacher_dtype=torch.float32
        ),
        on_round=rounds.append,
        on_epoch_close=lambda _: done.set(),
        on_finish=lambda _: None,
    )

    def capture(start, count, rejected=0, sampled=0, final=True, request_id="r"):
        adapter.capture_target(
            request_id=request_id,
            fc_output=torch.full((count, 16), float(start), device=device),
            positions=torch.arange(start, start + count, device=device),
            num_rejected=torch.tensor(rejected, dtype=torch.int32, device=device),
            num_sampled=torch.tensor(sampled, dtype=torch.int32, device=device),
            final_prefill=final,
        )

    try:
        capture(0, 2, final=False)
        capture(2, 3, sampled=1)
        input_ids = torch.tensor([2, 22, 22, 22], dtype=torch.int32, device=device)
        query_positions = torch.arange(5, 9, device=device)
        adapter.capture_proposal(input_ids, query_positions, 7)
        input_ids.zero_()
        query_positions.zero_()
        teacher = torch.arange(4 * 23, device=device).reshape(4, 23).float()
        adapter.capture_teacher_logits(teacher)
        teacher.zero_()  # The engine may mutate its logits after the hook.
        capture(5, 4, rejected=3 - accepted, sampled=accepted + 1)
        adapter.close_epoch("r")
        assert done.wait(5)
        adapter.raise_if_failed()
        assert len(rounds) == 1
        prepared = rounds[0]
        assert prepared.source_weight_version == 7
        assert prepared.step_id == 0
        batch = prepared.batch
        assert batch.prompt_fc.shape == (5, 16)
        assert batch.confirmed_fc.shape == (accepted + 1, 16)
        assert batch.input_ids.tolist() == [2, 22, 22, 22]
        assert batch.teacher_probabilities.shape == (min(accepted + 1, 3), 23)
        expected = torch.arange(23).float().softmax(-1)
        torch.testing.assert_close(batch.teacher_probabilities[0], expected)
        assert batch.rejection_position == (accepted if accepted < 3 else None)
        done.clear()
        anchor = 6 + accepted
        adapter.capture_proposal(
            torch.tensor([7, 22, 22, 22], device=device),
            torch.arange(anchor, anchor + 4, device=device),
            8,
        )
        adapter.capture_teacher_logits(torch.zeros(4, 23, device=device))
        capture(anchor, 4, rejected=3, sampled=1)
        adapter.close_epoch("r")
        assert done.wait(5)
        adapter.raise_if_failed()
        # The next D2H reuses staging storage; submitted batches own their data.
        assert batch.positions.tolist() == [5, 6, 7, 8]
        assert batch.confirmed_fc.unique().item() == 5
        assert batch.prompt_fc[:2].sum().item() == 0
        torch.testing.assert_close(batch.teacher_probabilities[0], expected)
        assert rounds[1].batch.prompt_fc is None
        adapter.finish_request("r")
        assert adapter.completed_rounds == 0
        # A new request starts a fresh prompt and continuous history at position 0.
        done.clear()
        capture(0, 2, sampled=1, request_id="next")
        adapter.capture_proposal(
            torch.tensor([3, 22, 22, 22], device=device),
            torch.arange(2, 6, device=device),
            0,
        )
        adapter.capture_teacher_logits(torch.zeros(4, 23, device=device))
        capture(2, 4, rejected=3, sampled=1, request_id="next")
        adapter.close_epoch("next")
        assert done.wait(5)
        adapter.raise_if_failed()
        assert rounds[2].request_id == "next"
        assert rounds[2].step_id == rounds[2].source_weight_version == 0
        assert rounds[2].batch.prompt_fc.shape == (2, 16)
        assert rounds[2].batch.confirmed_fc.unique().item() == 2
        adapter.finish_request("next")
    finally:
        adapter.close()


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

    queue = CaptureQueue(
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

    queue = CaptureQueue(
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

    queue = CaptureQueue(
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


def test_adapter_finishes_request_without_captured_proposal() -> None:
    finished: list[str] = []

    adapter = Eagle3CaptureAdapter(
        target_token_ids=torch.tensor([1, 3]),
        source_weight_version=lambda: 0,
        hidden_size=2,
        num_aux_hidden_states=1,
        draft_vocab_size=2,
        feature_dtype=torch.float32,
        transfer_chunk_size=8,
        on_round=lambda prepared_round: None,
        on_epoch_close=lambda request_id: None,
        on_finish=finished.append,
    )

    try:
        assert adapter.finish_request("request-0") == 0

        assert finished == ["request-0"]
    finally:
        adapter.close()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_next_prefill_builds_aligned_rounds_across_requests(device) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    rounds = []
    closed = Event()
    version = 0
    adapter = Eagle3CaptureAdapter(
        target_token_ids=torch.tensor([1, 3], device=device),
        source_weight_version=lambda: version,
        hidden_size=2,
        num_aux_hidden_states=1,
        draft_vocab_size=2,
        feature_dtype=torch.float32,
        on_round=rounds.append,
        on_epoch_close=lambda _: closed.set(),
        on_finish=lambda _: None,
    )
    try:
        for request_id in ("request-0", "request-1"):
            for step_id, anchor in enumerate((4, 5)):
                closed.clear()
                version = 7 + step_id
                target_aux = (torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device),)
                adapter.capture_proposal(
                    request_id=request_id,
                    verify_row_indices=torch.tensor([1], device=device),
                    target_aux_hidden_states=target_aux,
                    proposal=SimpleNamespace(
                        prefill_input_embeds=torch.tensor(
                            [[10.0, 11.0]], device=device
                        ),
                        prefill_positions=torch.tensor([anchor], device=device),
                        prefill_length=torch.tensor(1, device=device),
                        draft_token_input_embeds=torch.tensor(
                            [[12.0, 13.0]], device=device
                        ),
                        draft_recurrent_hidden_states=torch.tensor(
                            [[14.0, 15.0]], device=device
                        ),
                    ),
                )
                version += 1
                teacher = torch.arange(8, device=device).reshape(2, 4).float()
                adapter.capture_teacher_logits(teacher)
                teacher.zero_()
                confirmed = torch.tensor([[20.0, 21.0]], device=device)
                adapter.capture_prefill(
                    request_id=request_id,
                    verify_row_indices=torch.tensor([1], device=device),
                    target_aux_hidden_states=target_aux,
                    num_sampled=1,
                    prefill_input_embeds=confirmed,
                    prefill_positions=torch.tensor([anchor + 1], device=device),
                    prefill_length=torch.tensor(1, device=device),
                )
                confirmed.zero_()
                adapter.close_epoch(request_id)
                assert closed.wait(5)
                adapter.raise_if_failed()
                prepared = rounds[-1]
                assert prepared.request_id == request_id
                assert prepared.step_id == step_id
                assert prepared.source_weight_version == 7 + step_id
                batch = prepared.batch
                assert batch.prefill_positions.tolist() == [anchor]
                assert batch.prefill_input_embeds.tolist() == (
                    [[10.0, 11.0]] if step_id == 0 else [[20.0, 21.0]]
                )
                assert batch.prefill_aux_hidden_states.tolist() == (
                    [[1.0, 2.0]] if step_id == 0 else [[3.0, 4.0]]
                )
                assert batch.confirmed_input_embeds.tolist() == [[20.0, 21.0]]
                torch.testing.assert_close(
                    batch.teacher_probabilities,
                    torch.tensor([[1.0, 3.0], [5.0, 7.0]]).softmax(-1),
                )
                assert adapter.completed_rounds == step_id + 1
            adapter.finish_request(request_id)
            assert adapter.completed_rounds == 0
        assert len(rounds) == 4
    finally:
        adapter.close()
