# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
import torch
from online_draft.runtime.capture.dflash import (
    DFlashCaptureAdapter,
    DFlashCaptureBuffers,
)
from online_draft.runtime.capture.eagle3 import (
    Eagle3CaptureAdapter,
    Eagle3CaptureBuffers,
    Eagle3CapturePacket,
)
from online_draft.runtime.transport import CaptureQueue, PinnedStaging


def _eagle3_buffers(num_aux_hidden_states: int = 1) -> Eagle3CaptureBuffers:
    return Eagle3CaptureBuffers(
        8,
        2,
        2,
        num_aux_hidden_states,
        2,
        dtype=torch.float32,
        teacher_dtype=torch.float32,
    )


@pytest.mark.parametrize("accepted", [0, 1, 3])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_dflash_bootstrap_chunks_and_teacher_alignment(accepted, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    rounds = []
    done = Event()
    version = 7
    adapter = DFlashCaptureAdapter(
        source_weight_version=lambda: version,
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
        adapter.capture_proposal(input_ids=input_ids, positions=query_positions)
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
        version = 8
        adapter.capture_proposal(
            input_ids=torch.tensor([7, 22, 22, 22], device=device),
            positions=torch.arange(anchor, anchor + 4, device=device),
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
        version = 0
        adapter.capture_proposal(
            input_ids=torch.tensor([3, 22, 22, 22], device=device),
            positions=torch.arange(2, 6, device=device),
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
        proposal=None,
        teacher_logits=torch.zeros(2, 3, device=device),
        input_embeds=torch.zeros(1, 2, device=device),
        positions=torch.tensor([0], device=device),
        aux_hidden_states=torch.zeros(1, 2, device=device),
        metadata=torch.tensor([1, 1], device=device),
        buffers=_eagle3_buffers(),
    )


def test_close_epoch_follows_submitted_packets() -> None:
    events: list[object] = []
    queue_sizes: list[int] = []
    closed = Event()

    queue = CaptureQueue(
        consume=lambda packet: events.append(("round", packet.step_id)),
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
            ("round", 0),
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

    def consume(packet: Eagle3CapturePacket) -> None:
        consumed.append(packet)
        copied.set()

    packet = _packet(0, "cuda")
    packet.ready = torch.cuda.Event()
    packet.ready.record(torch.cuda.current_stream())

    queue = CaptureQueue(
        consume=consume,
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
        assert cpu_packet.metadata.is_pinned()
        assert torch.equal(
            cpu_packet.teacher_logits,
            torch.zeros(2, 3),
        )
    finally:
        queue.close()


def test_finish_follows_submitted_packets_and_waits_for_callback() -> None:
    consumed: list[Eagle3CapturePacket] = []
    finished: list[str] = []
    callback_started = Event()
    release_callback = Event()

    def on_finish(request_id: str) -> None:
        callback_started.set()
        release_callback.wait(timeout=5)
        finished.append(request_id)

    queue = CaptureQueue(
        consume=consumed.append,
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

        assert [packet.step_id for packet in consumed] == [0]
        assert finished == ["request-0"]
    finally:
        release_callback.set()
        queue.close()


def test_adapter_finishes_request_without_captured_proposal() -> None:
    finished: list[str] = []

    adapter = Eagle3CaptureAdapter(
        buffers=_eagle3_buffers(),
        target_token_ids=torch.tensor([1, 3]),
        source_weight_version=lambda: 0,
        on_round=lambda prepared_round: None,
        on_epoch_close=lambda request_id: None,
        on_finish=finished.append,
    )

    try:
        adapter.finish_request("request-0")

        assert finished == ["request-0"]
    finally:
        adapter.close()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_eagle3_rounds_restart_history_for_each_request(device) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    rounds = []
    closed = Event()
    version = 0
    adapter = Eagle3CaptureAdapter(
        buffers=_eagle3_buffers(),
        target_token_ids=torch.tensor([1, 3], device=device),
        source_weight_version=lambda: version,
        on_round=rounds.append,
        on_epoch_close=lambda _: closed.set(),
        on_finish=lambda _: None,
    )

    def capture(request_id, start, rows, confirmed, sampled):
        embeds = torch.arange(start, start + rows, device=device).float()
        adapter.capture_target(
            request_id=request_id,
            input_embeds=embeds[:, None].repeat(1, 2),
            positions=torch.arange(start, start + rows, device=device),
            aux_hidden_states=[embeds[:, None].repeat(1, 2) + 100],
            confirmed_length=torch.tensor(confirmed, device=device),
            num_sampled=torch.tensor(sampled, dtype=torch.int32, device=device),
            final_prefill=True,
        )
        adapter.capture_proposal(
            token_input_embeds=torch.tensor([[12.0, 13.0]], device=device),
            recurrent_hidden_states=torch.tensor([[14.0, 15.0]], device=device),
        )

    try:
        for request_id in ("request-0", "request-1"):
            closed.clear()
            version = 7
            capture(request_id, 0, 4, confirmed=4, sampled=1)
            for step_id, start in enumerate((4, 5)):
                version = 8 + step_id
                teacher = torch.arange(8, device=device).reshape(2, 4).float()
                adapter.capture_teacher_logits(teacher)
                teacher.zero_()
                capture(request_id, start, 3, confirmed=1, sampled=1)
            adapter.close_epoch(request_id)
            assert closed.wait(5)
            adapter.raise_if_failed()
            assert adapter.completed_rounds == 2
            first, second = rounds[-2:]
            assert [first.step_id, second.step_id] == [0, 1]
            assert [first.source_weight_version, second.source_weight_version] == [
                7,
                8,
            ]
            assert first.batch.prefill_positions.tolist() == [0, 1, 2, 3]
            assert first.batch.prefill_aux_hidden_states[:, 0].tolist() == [
                100.0,
                101.0,
                102.0,
                103.0,
            ]
            assert second.batch.prefill_positions.tolist() == [4]
            assert second.batch.prefill_input_embeds.tolist() == [[4.0, 4.0]]
            assert second.batch.confirmed_input_embeds.tolist() == [[5.0, 5.0]]
            torch.testing.assert_close(
                second.batch.teacher_probabilities,
                torch.tensor([[1.0, 3.0], [5.0, 7.0]]).softmax(-1),
            )
            adapter.finish_request(request_id)
            assert adapter.completed_rounds == 0
        assert len(rounds) == 4
    finally:
        adapter.close()


@pytest.mark.parametrize(
    ("tensor", "message"),
    [
        (torch.zeros(2, 3, dtype=torch.bfloat16), "uses torch.bfloat16"),
        (torch.zeros(3, 3), "has 9 elements"),
    ],
)
def test_staging_rejects_lossy_or_oversized_copies(tensor, message) -> None:
    """Staging must not silently cast or truncate captured training data."""
    staging = PinnedStaging({"rows": ((2, 3), torch.float32)})

    with pytest.raises(ValueError, match=message):
        staging.copy("rows", tensor, non_blocking=False)


def test_dflash_rejects_discontinuous_target_history() -> None:
    """A position gap (e.g. a prefix-cache hit) must fail, not train on it."""
    adapter = DFlashCaptureAdapter(
        source_weight_version=lambda: 0,
        buffers=DFlashCaptureBuffers(
            8, 4, 16, 23, dtype=torch.float32, teacher_dtype=torch.float32
        ),
        on_round=lambda _: None,
        on_epoch_close=lambda _: None,
        on_finish=lambda _: None,
    )
    try:
        adapter.capture_target(
            request_id="r",
            final_prefill=True,
            fc_output=torch.zeros(2, 16),
            positions=torch.arange(3, 5),
            num_rejected=torch.tensor(0),
            num_sampled=torch.tensor(1),
        )
        with pytest.raises(ValueError, match="contiguous target positions"):
            adapter.finish_request("r")
    finally:
        with pytest.raises(ValueError):
            adapter.close()
