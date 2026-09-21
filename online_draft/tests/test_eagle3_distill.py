# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
import torch.nn.functional as F
from online_draft.training.eagle3_distill import (
    forward_kl_loss,
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


def test_no_rejection_matches_batchmean_kl() -> None:
    torch.manual_seed(0)

    student_logits = torch.randn(
        1,
        5,
        requires_grad=True,
    )
    teacher = torch.randn(1, 5).softmax(dim=-1)

    actual = forward_kl_loss(
        student_logits,
        teacher,
        rejection_position=None,
    )
    expected = F.kl_div(
        F.log_softmax(student_logits.float(), dim=-1),
        teacher.float(),
        reduction="batchmean",
    )

    torch.testing.assert_close(actual, expected)

    actual.backward()
    assert student_logits.grad is not None


def test_rejection_position_has_double_weight() -> None:
    torch.manual_seed(1)

    student_logits = torch.randn(
        4,
        3,
        requires_grad=True,
    )
    teacher_logits = torch.randn(
        4,
        3,
        requires_grad=True,
    )
    teacher = teacher_logits.softmax(dim=-1)

    actual = forward_kl_loss(
        student_logits,
        teacher,
        rejection_position=2,
    )

    per_position_kl = F.kl_div(
        F.log_softmax(student_logits.float(), dim=-1),
        teacher.detach().float(),
        reduction="none",
    ).sum(dim=-1)
    weights = torch.tensor([1.0, 1.0, 2.0, 1.0])
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


@pytest.mark.parametrize(
    "rejection_position",
    [-1, 4],
)
def test_invalid_rejection_position(
    rejection_position: int,
) -> None:
    student_logits = torch.zeros(4, 3)
    teacher = torch.full((4, 3), 1.0 / 3.0)

    with pytest.raises(
        ValueError,
        match="within the draft sequence",
    ):
        forward_kl_loss(
            student_logits,
            teacher,
            rejection_position=rejection_position,
        )


def test_boolean_rejection_position_is_invalid() -> None:
    student_logits = torch.zeros(2, 3)
    teacher = torch.full((2, 3), 1.0 / 3.0)

    with pytest.raises(
        TypeError,
        match="integer or None",
    ):
        forward_kl_loss(
            student_logits,
            teacher,
            rejection_position=True,
        )
