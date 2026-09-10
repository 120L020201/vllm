# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy

import pytest
import torch

from online_eagle3.async_bridge import TrainObservation
from online_eagle3.distillation import qwen3_eagle3_distillation_step
from online_eagle3.loss import (
    distillation_loss,
    project_teacher_distribution,
)
from online_eagle3.qwen3_trainer import Qwen3Eagle3CpuTrainer, Qwen3Eagle3TrainerConfig
from online_eagle3.sync_bridge import Qwen3Eagle3SyncBridge
from online_eagle3.torch_eagle3 import TorchEagle3Config, TorchEagle3ForCausalLM


def _model(layers: int = 1) -> TorchEagle3ForCausalLM:
    return TorchEagle3ForCausalLM(
        TorchEagle3Config(
            hidden_size=8,
            intermediate_size=16,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            num_hidden_layers=layers,
            vocab_size=16,
            draft_vocab_size=8,
            rms_norm_eps=1e-6,
            rope_theta=10000,
        )
    )


def _observation(start=0, prefill=3, confirmed=3, step=0):
    positions = torch.arange(start, start + prefill)
    embeds = torch.randn(prefill, 8)
    return TrainObservation(
        "req",
        step,
        {
            "prefill_positions": positions,
            "prefill_input_embeds": embeds,
            "prefill_aux_hidden_states": torch.randn(prefill, 24),
            "proposal_positions": torch.arange(
                start + prefill - 1, start + prefill + 3
            ),
            "proposal_input_embeds": torch.cat((embeds[-1:], torch.randn(3, 8))),
            "proposal_hidden_states": torch.randn(4, 8),
            "teacher_probs": torch.randn(4, 8).softmax(-1),
            "confirmed_positions": torch.arange(
                start + prefill, start + prefill + confirmed
            ),
            "confirmed_input_embeds": torch.randn(confirmed, 8),
            "confirmed_aux_hidden_states": torch.randn(confirmed, 24),
        },
    )


def _append_reference(model, cache, positions, embeds, auxiliary):
    result = []
    with torch.no_grad():
        model(
            torch.zeros_like(positions),
            positions,
            model.combine_hidden_states(auxiliary),
            embeds,
            past_key_values=cache,
            kv_output=result,
        )
    return tuple(result)


@pytest.mark.parametrize("layers", [1, 2])
def test_cached_attention_matches_full_forward(layers):
    torch.manual_seed(3)
    model = _model(layers)
    ids, positions = torch.arange(6), torch.arange(6)
    embeds, hidden = torch.randn(6, 8), torch.randn(6, 8)
    with torch.no_grad():
        full = model(ids, positions, hidden, embeds)[0]
        cache = []
        model(ids[:4], positions[:4], hidden[:4], embeds[:4], kv_output=cache)
        updated = []
        suffix = model(
            ids[4:],
            positions[4:],
            hidden[4:],
            embeds[4:],
            past_key_values=tuple(cache),
            kv_output=updated,
        )[0]
    torch.testing.assert_close(suffix, full[4:])
    for old, new in zip(cache, updated):
        for before, after in zip(old, new):
            assert torch.equal(before, after[:, :4])


def test_teacher_projection_and_kl_gradient():
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    teacher, coverage = project_teacher_distribution(logits, torch.tensor([1, 2]))
    torch.testing.assert_close(teacher, logits[:, [1, 3]].softmax(-1))
    torch.testing.assert_close(coverage, logits.softmax(-1)[:, [1, 3]].sum(-1))
    student = torch.zeros(1, 2, requires_grad=True)
    loss = distillation_loss(student, teacher)
    loss.backward()
    torch.testing.assert_close(student.grad, student.detach().softmax(-1) - teacher)
    assert loss > 0


@pytest.mark.parametrize("confirmed", [1, 3, 5])
def test_append_uses_updated_weights_and_preserves_history(confirmed):
    torch.manual_seed(4)
    trainer = Qwen3Eagle3CpuTrainer(_model(), Qwen3Eagle3TrainerConfig(lr=0.01))
    baseline = trainer.snapshot()
    observation = _observation(confirmed=confirmed)
    old_model = copy.deepcopy(trainer.model)
    # Match the inference-mode context used by the real GPU runner.
    with torch.inference_mode():
        observation.payload = {k: v.clone() for k, v in observation.payload.items()}
        qwen3_eagle3_distillation_step(trainer, [observation])
    payload = {k: v.clone() for k, v in observation.payload.items()}
    assert trainer.version == 1
    assert trainer.cache_length == 3 + confirmed
    expected = ()
    for prefix in ("prefill", "confirmed"):
        expected = _append_reference(
            trainer.model,
            expected,
            payload[f"{prefix}_positions"],
            payload[f"{prefix}_input_embeds"],
            payload[f"{prefix}_aux_hidden_states"],
        )
    for actual, reference in zip(trainer.kv_cache[0], expected[0]):
        torch.testing.assert_close(actual, reference)
        assert not actual.requires_grad and actual.grad_fn is None
    old_cache = _append_reference(
        old_model,
        (),
        payload["prefill_positions"],
        payload["prefill_input_embeds"],
        payload["prefill_aux_hidden_states"],
    )
    assert not torch.allclose(trainer.kv_cache[0][0][:, :3], old_cache[0][0])

    previous = tuple((k.clone(), v.clone()) for k, v in trainer.kv_cache)
    next_observation = _observation(start=3, prefill=confirmed, confirmed=1, step=1)
    qwen3_eagle3_distillation_step(trainer, [next_observation])
    assert trainer.version == 2
    assert trainer.cache_length == 4 + confirmed
    for old, new in zip(previous[0], trainer.kv_cache[0]):
        assert torch.equal(old, new[:, : old.shape[1]])
    assert trainer.optimizer.state
    trainer.restore_snapshot(baseline)
    assert trainer.cache_length == 0 and trainer.request_id is None
    assert not trainer.optimizer.state


def test_rejected_tail_contributes_gradient_but_not_persistent_history():
    torch.manual_seed(7)
    first = Qwen3Eagle3CpuTrainer(_model())
    second = Qwen3Eagle3CpuTrainer(copy.deepcopy(first.model))
    observation = _observation(confirmed=1)
    changed = copy.deepcopy(observation)
    changed.payload["teacher_probs"][-1] = torch.tensor(
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    qwen3_eagle3_distillation_step(first, [observation])
    qwen3_eagle3_distillation_step(second, [changed])
    assert first.cache_length == second.cache_length == 4
    assert first.last_loss != second.last_loss
    assert not torch.equal(first.model.model.fc.weight, second.model.model.fc.weight)


def test_invalid_observation_fails_before_update():
    trainer = Qwen3Eagle3CpuTrainer(_model())
    observation = _observation()
    with pytest.raises(ValueError, match="S=1"):
        qwen3_eagle3_distillation_step(trainer, [observation, observation])
    observation.payload["confirmed_positions"] += 1
    with pytest.raises(ValueError, match="Confirmed positions"):
        qwen3_eagle3_distillation_step(trainer, [observation])
    assert trainer.version == 0 and trainer.cache_length == 0


def test_s1_bridge_publishes_only_after_append_and_resets():
    trainer = Qwen3Eagle3CpuTrainer(_model())
    bridge = Qwen3Eagle3SyncBridge(
        trainer, qwen3_eagle3_distillation_step, fail_on_error=True
    )
    observation = _observation()
    bridge.observe_step(
        observation.request_id, observation.step_id, observation.payload
    )
    snapshot = bridge.maybe_apply_pending_weights()
    assert snapshot is not None and snapshot.version == 1
    assert trainer.cache_length == 6
    with pytest.raises(ValueError, match="out-of-order"):
        bridge.observe_step(
            observation.request_id, observation.step_id, observation.payload
        )
    assert bridge.maybe_apply_pending_weights() is None
    bridge.reset_request("req")
    assert trainer.cache_length == 0
    assert bridge.maybe_apply_pending_weights().version == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("check_gradients", [True, False])
def test_training_precision_and_gradient_check_config(
    dtype, check_gradients, monkeypatch
):
    trainer = Qwen3Eagle3CpuTrainer(
        _model().to(dtype),
        Qwen3Eagle3TrainerConfig(dtype=dtype, check_gradients=check_gradients),
    )
    original_step = trainer.step

    def check_step():
        grads = [p.grad for p in trainer.model.parameters() if p.grad is not None]
        assert grads and all(grad.dtype == dtype for grad in grads)
        original_step()

    monkeypatch.setattr(trainer, "step", check_step)
    qwen3_eagle3_distillation_step(trainer, [_observation()])
    for state in trainer.optimizer.state.values():
        assert state["exp_avg"].dtype == dtype
        assert state["exp_avg_sq"].dtype == dtype
    assert trainer.kv_cache[0][0].dtype == dtype


@pytest.mark.parametrize("layers", [1, 2])
@pytest.mark.parametrize("with_history", [False, True])
def test_parallel_matches_sequential_fixed_inputs(layers, with_history, monkeypatch):
    torch.manual_seed(11)
    trainer = Qwen3Eagle3CpuTrainer(_model(layers))
    if with_history:
        qwen3_eagle3_distillation_step(trainer, [_observation()])
        observation = _observation(start=3, prefill=3, step=1)
    else:
        observation = _observation()
    payload = observation.payload
    payload["proposal_hidden_states"].requires_grad_()
    reference = copy.deepcopy(trainer.model)
    cache = tuple((k[:, :5], v[:, :5]) for k, v in trainer.kv_cache)
    first = slice(-1, None) if with_history else slice(None)
    new_cache = []
    output, _ = reference(
        torch.zeros_like(payload["prefill_positions"][first]),
        payload["prefill_positions"][first],
        reference.combine_hidden_states(payload["prefill_aux_hidden_states"][first]),
        payload["prefill_input_embeds"][first],
        past_key_values=cache,
        kv_output=new_cache,
    )
    logits = [reference.compute_draft_logits(output[-1:])]
    for index in range(1, 4):
        cache, new_cache = tuple(new_cache), []
        output, _ = reference(
            torch.zeros(1, dtype=torch.long),
            payload["proposal_positions"][index : index + 1],
            payload["proposal_hidden_states"][index : index + 1].detach(),
            payload["proposal_input_embeds"][index : index + 1],
            past_key_values=cache,
            kv_output=new_cache,
        )
        logits.append(reference.compute_draft_logits(output))
    expected_loss = distillation_loss(torch.cat(logits), payload["teacher_probs"])
    expected_loss.backward()

    train_forward_sizes = []

    def capture_forward(_module, _args, kwargs):
        if torch.is_grad_enabled():
            train_forward_sizes.append(kwargs["positions"].numel())

    def check_gradients():
        for actual, expected in zip(trainer.model.parameters(), reference.parameters()):
            if expected.grad is None:
                assert actual.grad is None
            else:
                torch.testing.assert_close(
                    actual.grad, expected.grad, atol=1e-6, rtol=1e-4
                )
        original_step()

    original_step = trainer.step
    monkeypatch.setattr(trainer, "step", check_gradients)
    handle = trainer.model.register_forward_pre_hook(capture_forward, with_kwargs=True)
    try:
        qwen3_eagle3_distillation_step(trainer, [observation])
    finally:
        handle.remove()
    assert train_forward_sizes == [4 if with_history else 6]
    assert trainer.last_loss == pytest.approx(expected_loss.item(), abs=1e-6)
    assert payload["proposal_hidden_states"].grad is None
