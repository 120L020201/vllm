# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn.functional as F
from online_draft.training.eagle3.distill import (
    position_forward_kl_loss,
    project_teacher_distribution,
)


def test_project_teacher_distribution() -> None:
    teacher_logits = torch.tensor(
        [
            [0.0, 1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0, 0.0],
        ],
        requires_grad=True,
    )
    target_token_ids = torch.tensor([0, 2])

    teacher, coverage = project_teacher_distribution(
        teacher_logits,
        target_token_ids,
    )

    selected_logits = teacher_logits.detach().index_select(
        dim=-1,
        index=target_token_ids,
    )
    expected_teacher = selected_logits.softmax(dim=-1)
    expected_coverage = (
        teacher_logits.detach()
        .softmax(dim=-1)
        .index_select(dim=-1, index=target_token_ids)
        .sum(dim=-1)
    )

    torch.testing.assert_close(teacher, expected_teacher)
    torch.testing.assert_close(coverage, expected_coverage)

    assert not teacher.requires_grad
    assert not coverage.requires_grad


def test_position_loss_restarts_depth_per_round_and_averages_rows() -> None:
    """Two rounds of depths 0-2 and 0-1 share weights 1, 0.9, 0.81 by depth."""
    torch.manual_seed(3)
    student_logits = torch.randn(5, 4, requires_grad=True)
    teacher = torch.randn(5, 4).softmax(dim=-1)
    depths = torch.tensor([0, 1, 2, 0, 1])

    actual = position_forward_kl_loss(student_logits, teacher, depths)

    per_position_kl = F.kl_div(
        F.log_softmax(student_logits, dim=-1), teacher, reduction="none"
    ).sum(dim=-1)
    weights = torch.tensor([1.0, 0.9, 0.81, 1.0, 0.9])
    torch.testing.assert_close(actual, (per_position_kl * weights).sum() / 5)
