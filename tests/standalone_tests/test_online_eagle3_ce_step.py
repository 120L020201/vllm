from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from online_eagle3.async_bridge import TrainObservation
from online_eagle3.ce_step import qwen3_eagle3_ce_step
from online_eagle3.qwen3_trainer import Qwen3Eagle3CpuTrainer


class _ToyCpuDraft(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(vocab_size=8)
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(8, 4)
        self.model.fc = nn.Linear(4, 4)
        self.lm_head = nn.Linear(4, 8, bias=False)
        self.draft_id_to_target_id = nn.Parameter(
            torch.zeros(8, dtype=torch.long), requires_grad=False
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del input_ids, positions
        hidden = self.model.fc(hidden_states + inputs_embeds)
        return hidden, hidden

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.fc(hidden_states)

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)


def _make_observation(num_sampled: int = 2) -> TrainObservation:
    return TrainObservation(
        request_id="req-1",
        step_id=0,
        payload={
            "proposal_input_ids": torch.tensor([1, 2], dtype=torch.int32),
            "proposal_input_embeds": torch.randn(2, 4),
            "proposal_positions": torch.tensor([4, 5], dtype=torch.int64),
            "proposal_hidden_states": torch.randn(2, 4),
            "proposal_aux_hidden_states": torch.randn(1, 4),
            "proposal_num_speculative_tokens": torch.tensor([2], dtype=torch.int32),
            "sampled_token_ids": torch.tensor([[3, 4, 5]], dtype=torch.int32),
            "num_sampled": torch.tensor([num_sampled], dtype=torch.int32),
        },
    )


def test_ce_step_updates_trainable_draft_weights() -> None:
    torch.manual_seed(0)
    model = _ToyCpuDraft()
    trainer = Qwen3Eagle3CpuTrainer(model)
    before = trainer.snapshot()
    before_lm_head = model.lm_head.weight.detach().clone()

    qwen3_eagle3_ce_step(trainer, (_make_observation(),))
    after = trainer.snapshot()

    assert after.version == 1
    assert not torch.equal(
        before.state_dict["model.fc.weight"], after.state_dict["model.fc.weight"]
    )
    assert torch.equal(before_lm_head, model.lm_head.weight)


def test_ce_step_skips_when_verify_result_has_no_active_labels() -> None:
    torch.manual_seed(0)
    model = _ToyCpuDraft()
    trainer = Qwen3Eagle3CpuTrainer(model)
    before = trainer.snapshot()

    qwen3_eagle3_ce_step(trainer, (_make_observation(num_sampled=0),))
    after = trainer.snapshot()

    assert after.version == before.version
    assert torch.equal(
        before.state_dict["model.fc.weight"], after.state_dict["model.fc.weight"]
    )
