# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from online_draft.models.qwen3_eagle3 import Eagle3KVCache
from online_draft.runtime.state import Eagle3RequestState


def _make_cache(length: int) -> Eagle3KVCache:
    return (
        (
            torch.zeros(2, length, 4),
            torch.zeros(2, length, 4),
        ),
    )


def test_new_state_starts_without_history() -> None:
    state = Eagle3RequestState(
        request_id="request-1",
        request_generation=0,
    )

    assert state.cache_length == 0
    assert state.last_consumed_step_id is None
    assert state.next_step_id == 0


@pytest.mark.parametrize(
    ("request_id", "request_generation"),
    [
        ("", 0),
        (1, 0),
        ("request-1", -1),
        ("request-1", True),
    ],
)
def test_invalid_request_scope_is_rejected(
    request_id: str,
    request_generation: int,
) -> None:
    with pytest.raises(ValueError):
        Eagle3RequestState(
            request_id=request_id,
            request_generation=request_generation,
        )


def test_commit_advances_cache_and_window_atomically() -> None:
    state = Eagle3RequestState(
        request_id="request-1",
        request_generation=2,
    )
    cache = _make_cache(4)

    state.commit_window(
        request_id="request-1",
        request_generation=2,
        start_step_id=0,
        end_step_id=2,
        persistent_cache=cache,
    )

    assert state.persistent_cache is cache
    assert state.cache_length == 4
    assert state.last_consumed_step_id == 2
    assert state.next_step_id == 3


@pytest.mark.parametrize("start_step_id", [0, 4, -1, True])
def test_duplicate_skipped_or_invalid_window_is_rejected(
    start_step_id: int,
) -> None:
    cache = _make_cache(4)
    state = Eagle3RequestState(
        request_id="request-1",
        request_generation=2,
        persistent_cache=cache,
        last_consumed_step_id=2,
    )

    with pytest.raises(ValueError):
        state.validate_window_scope(
            request_id="request-1",
            request_generation=2,
            start_step_id=start_step_id,
            end_step_id=4,
        )

    assert state.persistent_cache is cache
    assert state.last_consumed_step_id == 2


def test_window_end_must_not_precede_start() -> None:
    state = Eagle3RequestState(
        request_id="request-1",
        request_generation=2,
        persistent_cache=_make_cache(4),
        last_consumed_step_id=0,
    )

    with pytest.raises(
        ValueError,
        match="must not precede",
    ):
        state.validate_window_scope(
            request_id="request-1",
            request_generation=2,
            start_step_id=1,
            end_step_id=0,
        )


@pytest.mark.parametrize(
    ("request_id", "request_generation"),
    [
        ("request-2", 2),
        ("request-1", 3),
    ],
)
def test_mismatched_request_scope_is_rejected(
    request_id: str,
    request_generation: int,
) -> None:
    state = Eagle3RequestState(
        request_id="request-1",
        request_generation=2,
    )

    with pytest.raises(ValueError):
        state.validate_window_scope(
            request_id=request_id,
            request_generation=request_generation,
            start_step_id=0,
            end_step_id=1,
        )


def test_failed_commit_does_not_modify_state() -> None:
    original_cache = _make_cache(4)
    state = Eagle3RequestState(
        request_id="request-1",
        request_generation=2,
        persistent_cache=original_cache,
        last_consumed_step_id=1,
    )

    with pytest.raises(
        ValueError,
        match="must be longer",
    ):
        state.commit_window(
            request_id="request-1",
            request_generation=2,
            start_step_id=2,
            end_step_id=3,
            persistent_cache=_make_cache(4),
        )

    assert state.persistent_cache is original_cache
    assert state.cache_length == 4
    assert state.last_consumed_step_id == 1


def test_differentiable_cache_is_rejected_without_modifying_state() -> None:
    original_cache = _make_cache(4)
    state = Eagle3RequestState(
        request_id="request-1",
        request_generation=2,
        persistent_cache=original_cache,
        last_consumed_step_id=1,
    )
    differentiable_cache = (
        (
            torch.zeros(2, 5, 4, requires_grad=True),
            torch.zeros(2, 5, 4),
        ),
    )

    with pytest.raises(
        ValueError,
        match="must be detached",
    ):
        state.commit_window(
            request_id="request-1",
            request_generation=2,
            start_step_id=2,
            end_step_id=3,
            persistent_cache=differentiable_cache,
        )

    assert state.persistent_cache is original_cache
    assert state.last_consumed_step_id == 1


def test_state_restore_requires_cache_and_step_together() -> None:
    with pytest.raises(
        ValueError,
        match="must be present together",
    ):
        Eagle3RequestState(
            request_id="request-1",
            request_generation=2,
            persistent_cache=_make_cache(4),
        )

    with pytest.raises(
        ValueError,
        match="must be present together",
    ):
        Eagle3RequestState(
            request_id="request-1",
            request_generation=2,
            last_consumed_step_id=0,
        )


def test_reset_clears_request_history() -> None:
    state = Eagle3RequestState(
        request_id="request-1",
        request_generation=2,
        persistent_cache=_make_cache(4),
        last_consumed_step_id=0,
    )

    state.reset()

    assert state.persistent_cache is None
    assert state.cache_length == 0
    assert state.last_consumed_step_id is None
    assert state.next_step_id == 0
