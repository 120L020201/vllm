# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn.functional as F
from online_draft.training.eagle3.distill import (
    project_teacher_distribution,
    window_forward_kl_loss,
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


def test_window_rejection_mask_weights_multiple_rounds() -> None:
    torch.manual_seed(2)

    student_logits = torch.randn(
        5,
        3,
        requires_grad=True,
    )
    teacher_logits = torch.randn(
        5,
        3,
        requires_grad=True,
    )
    teacher = teacher_logits.softmax(dim=-1)
    rejection_target_mask = torch.tensor([False, True, False, False, True])

    actual = window_forward_kl_loss(
        student_logits=student_logits,
        teacher_probabilities=teacher,
        rejection_target_mask=rejection_target_mask,
    )

    per_position_kl = F.kl_div(
        F.log_softmax(student_logits.float(), dim=-1),
        teacher.detach().float(),
        reduction="none",
    ).sum(dim=-1)
    weights = torch.tensor([1.0, 2.0, 1.0, 1.0, 2.0])
    expected = (per_position_kl * weights).sum() / weights.sum()

    torch.testing.assert_close(actual, expected)

    actual.backward()

    expected_gradient = student_logits.detach().softmax(dim=-1) - teacher.detach()
    expected_gradient *= weights[:, None] / weights.sum()

    torch.testing.assert_close(
        student_logits.grad,
        expected_gradient,
        rtol=1e-5,
        atol=1e-6,
    )
    assert teacher_logits.grad is None
