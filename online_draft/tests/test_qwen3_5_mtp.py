# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from online_draft.models.qwen3_5_mtp import Qwen3_5MTPConfig, Qwen3_5MTPForCausalLM
from online_draft.training.eagle3.batch import Eagle3DistillationBatch
from online_draft.training.eagle3.window import train_eagle3_rounds
from online_draft.training.trainer import DraftTrainer, TrainerConfig

HIDDEN, VOCAB = 8, 12


def _model() -> Qwen3_5MTPForCausalLM:
    torch.manual_seed(0)
    model = Qwen3_5MTPForCausalLM(
        Qwen3_5MTPConfig(
            hidden_size=HIDDEN,
            intermediate_size=16,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            rotary_dim=4,
            rope_theta=10000.0,
            rms_norm_eps=1e-6,
            vocab_size=VOCAB,
        )
    )
    # Gemma-style norms start at zero; perturb them so their offset matters.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("norm.weight") or "norm_" in name:
                parameter.normal_(std=0.1)
    return model


def test_cached_forward_matches_full_forward() -> None:
    """Draft decode reuses KV, so splitting a sequence must not change it."""
    model = _model()
    positions = torch.arange(5)
    embeds, hidden = torch.randn(5, HIDDEN), torch.randn(5, HIDDEN)

    with torch.no_grad():
        full = model(positions=positions, input_embeds=embeds, hidden_states=hidden)
        prefix = model(
            positions=positions[:3], input_embeds=embeds[:3], hidden_states=hidden[:3]
        )
        suffix = model(
            positions=positions[3:],
            input_embeds=embeds[3:],
            hidden_states=hidden[3:],
            past_key_values=prefix.past_key_values,
        )

    torch.testing.assert_close(
        torch.cat((prefix.hidden_states, suffix.hidden_states)), full.hidden_states
    )
    assert full.recurrent_hidden_states is full.hidden_states


def test_window_training_updates_mtp_weights_but_not_the_shared_head() -> None:
    """MTP trains through the EAGLE3 window with one target feature row."""
    model = _model()
    trainer = DraftTrainer(model, TrainerConfig(learning_rate=1e-2))
    head, fc = model.lm_head.weight.clone(), model.model.fc.weight.clone()
    batch = Eagle3DistillationBatch(
        prefix_positions=torch.arange(3),
        prefix_input_embeds=torch.randn(3, HIDDEN),
        prefix_aux_hidden_states=torch.randn(3, HIDDEN),
        proposal_positions=torch.arange(2, 5),
        draft_token_input_embeds=torch.randn(2, HIDDEN),
        draft_input_hidden_states=torch.randn(2, HIDDEN),
        teacher_probabilities=torch.randn(3, VOCAB).softmax(-1),
        confirmed_positions=torch.arange(3, 5),
        confirmed_input_embeds=torch.randn(2, HIDDEN),
        confirmed_aux_hidden_states=torch.randn(2, HIDDEN),
        rejection_position=1,
    )

    cache, result = train_eagle3_rounds(trainer, (batch,), None)

    assert "lm_head.weight" not in trainer.trainable_parameter_names
    assert result.model_version == 1
    assert torch.equal(model.lm_head.weight, head)
    assert not torch.equal(model.model.fc.weight, fc)
    assert cache[0][0].shape[1] == 5
