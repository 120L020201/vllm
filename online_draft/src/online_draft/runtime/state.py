# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

from online_draft.models.qwen3_eagle3 import (
    Eagle3KVCache,
)
from online_draft.training.eagle3_cache import (
    PersistentEagle3KVCache,
    persistent_cache_length,
)


@dataclass(slots=True)
class Eagle3RequestState:
    """Long-lived state for one request generation.

    The state owns only persistent detached KV history and request
    metadata. A completed training batch must not be stored here.
    """

    request_id: str
    request_generation: int
    persistent_cache: PersistentEagle3KVCache = None
    last_consumed_step_id: int | None = None

    def __post_init__(self) -> None:
        self._validate_request_id(self.request_id)
        self._validate_request_generation(self.request_generation)

        if self.last_consumed_step_id is not None:
            self._validate_step_id(self.last_consumed_step_id)

        # This validates that any supplied cache is detached and
        # internally consistent before the state takes ownership of it.
        cache_length = persistent_cache_length(self.persistent_cache)

        if self.persistent_cache is not None and cache_length == 0:
            raise ValueError("persistent cache must not be empty")

        has_consumed_step = self.last_consumed_step_id is not None
        has_persistent_history = cache_length > 0
        if has_consumed_step != has_persistent_history:
            raise ValueError(
                "persistent cache and last_consumed_step_id must be present together"
            )

    @property
    def cache_length(self) -> int:
        """Return the current persistent history length."""
        return persistent_cache_length(self.persistent_cache)

    @property
    def next_step_id(self) -> int:
        """Return the only valid next step identifier."""
        if self.last_consumed_step_id is None:
            return 0

        return self.last_consumed_step_id + 1

    def validate_window_scope(
        self,
        *,
        request_id: str,
        request_generation: int,
        start_step_id: int,
        end_step_id: int,
    ) -> None:
        """Validate request identity and a consecutive pending window.

        Args:
            request_id: Request identifier carried by the pending window.
            request_generation: Request lifetime carried by the pending window.
            start_step_id: First GPU round consumed by the window.
            end_step_id: Last GPU round consumed by the window.

        Raises:
            ValueError: If identity or ordering is invalid.
        """
        self.validate_request_scope(
            request_id=request_id,
            request_generation=request_generation,
        )
        self._validate_step_id(start_step_id)
        self._validate_step_id(end_step_id)

        if start_step_id != self.next_step_id:
            raise ValueError(
                "window start_step_id must equal next expected step "
                f"{self.next_step_id}"
            )

        if end_step_id < start_step_id:
            raise ValueError("window end_step_id must not precede start_step_id")

    def validate_request_scope(
        self,
        *,
        request_id: str,
        request_generation: int,
    ) -> None:
        """Validate that metadata belongs to this request lifetime."""
        self._validate_request_id(request_id)
        self._validate_request_generation(request_generation)

        if request_id != self.request_id:
            raise ValueError("observation request_id does not match state")

        if request_generation != self.request_generation:
            raise ValueError("observation generation does not match state")

    def commit_window(
        self,
        *,
        request_id: str,
        request_generation: int,
        start_step_id: int,
        end_step_id: int,
        persistent_cache: Eagle3KVCache,
    ) -> None:
        """Commit one completed training window atomically.

        The state changes only after training and confirmed KV append have
        both succeeded. One commit may consume multiple GPU rounds, and the
        new cache must be strictly longer than the previous history.

        Args:
            request_id: Request identifier carried by the completed window.
            request_generation: Request lifetime carried by the completed window.
            start_step_id: First GPU round consumed by the window.
            end_step_id: Last GPU round consumed by the window.
            persistent_cache: New detached persistent KV cache.

        Raises:
            ValueError: If scope, ordering, or cache growth is invalid.
        """
        self.validate_window_scope(
            request_id=request_id,
            request_generation=request_generation,
            start_step_id=start_step_id,
            end_step_id=end_step_id,
        )

        if persistent_cache is None:
            raise ValueError("committed persistent cache must not be None")

        old_length = self.cache_length
        new_length = persistent_cache_length(persistent_cache)

        if new_length <= old_length:
            raise ValueError(
                "committed persistent cache must be longer than the previous history"
            )

        self.persistent_cache = persistent_cache
        self.last_consumed_step_id = end_step_id

    def reset(self) -> None:
        """Clear request history after request completion or cancellation."""
        self.persistent_cache = None
        self.last_consumed_step_id = None

    def _validate_step_id(
        self,
        step_id: int,
    ) -> None:
        if isinstance(step_id, bool) or not isinstance(step_id, int) or step_id < 0:
            raise ValueError("step_id must be a nonnegative integer")

    @staticmethod
    def _validate_request_id(request_id: str) -> None:
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a nonempty string")

    @staticmethod
    def _validate_request_generation(request_generation: int) -> None:
        if (
            isinstance(request_generation, bool)
            or not isinstance(request_generation, int)
            or request_generation < 0
        ):
            raise ValueError("request_generation must be a nonnegative integer")
