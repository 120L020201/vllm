# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from threading import Event

import online_draft.training.gemma4_mtp.window as gemma4_mtp_window
import torch
from online_draft.models.gemma4_mtp import (
    Gemma4MTPConfig,
    Gemma4MTPForCausalLM,
    Gemma4MTPLayerConfig,
)
from online_draft.runtime.capture.gemma4_mtp import (
    Gemma4MTPCaptureAdapter,
    Gemma4MTPCaptureBuffers,
)
from online_draft.training.eagle3.batch import Eagle3DistillationBatch
from online_draft.training.eagle3.distill import position_forward_kl_loss
from online_draft.training.gemma4_mtp.window import (
    Gemma4MTPBatch,
    train_gemma4_mtp_window,
)
from online_draft.training.trainer import DraftTrainer, TrainerConfig

BACKBONE, VOCAB, WINDOW = 6, 12, 2


def _model() -> Gemma4MTPForCausalLM:
    torch.manual_seed(0)
    return Gemma4MTPForCausalLM(
        Gemma4MTPConfig(
            backbone_hidden_size=BACKBONE,
            hidden_size=8,
            intermediate_size=16,
            num_attention_heads=2,
            vocab_size=VOCAB,
            rms_norm_eps=1e-6,
            layers=(
                Gemma4MTPLayerConfig(4, 1, WINDOW, kv_index=0),
                Gemma4MTPLayerConfig(8, 1, None, kv_index=1),
            ),
        )
    )


def _history(length: int) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.randn(length, 1, 8), torch.randn(length, 1, 16)


def test_draft_rows_see_target_kv_only_up_to_their_position() -> None:
    """Sliding layers also drop target K/V older than their window."""
    model = _model()
    positions = torch.tensor([3, 3])
    embeds, hidden = torch.randn(2, BACKBONE), torch.randn(2, BACKBONE)
    sliding, full = _history(6)

    def run(sliding, full):
        with torch.no_grad():
            return model(positions, embeds, hidden, (sliding, full))[0]

    reference = run(sliding, full)
    future_sliding, future_full = sliding.clone(), full.clone()
    future_sliding[4:] += 1
    future_full[4:] += 1
    old_sliding = sliding.clone()
    old_sliding[1] += 1
    old_full = full.clone()
    old_full[1] += 1

    torch.testing.assert_close(run(future_sliding, future_full), reference)
    torch.testing.assert_close(run(old_sliding, full), reference)
    assert not torch.allclose(run(sliding, old_full), reference)


def _round(anchor: int, confirmed: int) -> Eagle3DistillationBatch:
    prefix = anchor + 1
    return Eagle3DistillationBatch(
        prefix_positions=torch.arange(prefix),
        prefix_input_embeds=torch.randn(prefix, BACKBONE),
        prefix_aux_hidden_states=torch.randn(prefix, BACKBONE),
        proposal_positions=torch.arange(anchor, anchor + 3),
        draft_token_input_embeds=torch.randn(2, BACKBONE),
        draft_input_hidden_states=torch.randn(2, BACKBONE),
        teacher_probabilities=torch.randn(3, VOCAB).softmax(-1),
        confirmed_positions=torch.arange(prefix, prefix + confirmed),
        confirmed_input_embeds=torch.randn(confirmed, BACKBONE),
        confirmed_aux_hidden_states=torch.randn(confirmed, BACKBONE),
        rejection_position=None if confirmed == 4 else confirmed - 1,
    )


def test_window_appends_target_kv_and_trains_only_query_weights() -> None:
    model = _model()
    trainer = DraftTrainer(model, TrainerConfig(learning_rate=1e-2))
    frozen = {
        name: parameter.clone()
        for name, parameter in model.named_parameters()
        if name.startswith(("lm_head", "model.post_projection"))
    }
    projection = model.model.pre_projection.weight.clone()

    first = Gemma4MTPBatch(_round(2, 2), _history(5))
    cache, result = train_gemma4_mtp_window(trainer, (first,), None)
    second = Gemma4MTPBatch(_round(4, 4), _history(4))
    cache, result = train_gemma4_mtp_window(trainer, (second,), cache)

    assert set(frozen).isdisjoint(trainer.trainable_parameter_names)
    for name, parameter in model.named_parameters():
        if name in frozen:
            assert torch.equal(parameter, frozen[name]), name
    assert not torch.equal(model.model.pre_projection.weight, projection)
    assert result.model_version == 2
    assert result.target_count == 3
    assert [layer.shape[0] for layer in cache] == [9, 9]


def test_window_loss_restarts_depth_on_every_round(monkeypatch) -> None:
    """Gemma4 MTP shares the EAGLE3 depth-decayed loss."""
    depths: list[list[int]] = []

    def record(logits, teacher, proposal_depths):
        depths.append(proposal_depths.tolist())
        return position_forward_kl_loss(logits, teacher, proposal_depths)

    monkeypatch.setattr(gemma4_mtp_window, "position_forward_kl_loss", record)
    trainer = DraftTrainer(_model(), TrainerConfig(learning_rate=1e-2))
    rounds = (
        Gemma4MTPBatch(_round(2, 2), _history(5)),
        Gemma4MTPBatch(_round(4, 4), _history(4)),
    )

    train_gemma4_mtp_window(trainer, rounds, None)

    assert depths == [[0, 1, 0, 1, 2]]


def test_adapter_rounds_carry_prompt_then_confirmed_target_kv() -> None:
    rounds = []
    closed = Event()
    adapter = Gemma4MTPCaptureAdapter(
        buffers=Gemma4MTPCaptureBuffers(
            8,
            2,
            2,
            4,
            ((1, 2), (1, 4)),
            dtype=torch.float32,
            teacher_dtype=torch.float32,
        ),
        source_weight_version=lambda: 0,
        on_round=rounds.append,
        on_epoch_close=lambda _: closed.set(),
        on_finish=lambda _: None,
    )

    def capture(start, rows, confirmed, final_prefill=True):
        values = torch.arange(start, start + rows).float()
        adapter.capture_target(
            request_id="request",
            final_prefill=final_prefill,
            input_embeds=values[:, None].repeat(1, 2),
            positions=torch.arange(start, start + rows),
            aux_hidden_states=[values[:, None].repeat(1, 2)],
            confirmed_length=torch.tensor(confirmed),
            num_sampled=torch.tensor(1),
            sliding_kv=values[:, None, None].repeat(1, 1, 2),
            full_kv=values[:, None, None].repeat(1, 1, 4) + 100,
        )
        adapter.capture_proposal(
            token_input_embeds=torch.zeros(1, 2),
            input_hidden_states=torch.zeros(1, 2),
        )

    try:
        capture(0, 2, 2, final_prefill=False)
        capture(2, 2, 2)
        for start in (4, 5):
            adapter.capture_teacher_logits(torch.zeros(3, 4))
            capture(start, 3, 1)
        adapter.close_epoch("request")
        assert closed.wait(5)
        adapter.raise_if_failed()
    finally:
        adapter.close()

    first, second = (prepared.batch for prepared in rounds)
    assert first.kv[0][:, 0, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert first.kv[1][:, 0, 0].tolist() == [100.0, 101.0, 102.0, 103.0, 104.0]
    assert second.kv[0][:, 0, 0].tolist() == [5.0]
    assert second.round.prefix_positions.tolist() == [4]
