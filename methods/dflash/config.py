# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Launch-time configuration for the DFlash methods."""

from pathlib import Path

BASELINE_METHOD = "dflash"
DRAFT_METHODS = ("eagle3", "dflash", "mtp")
TTS_METHODS = frozenset({"tts", "random_sampling"})
DEFAULT_SPEC_TOKENS = 15
DEFAULT_EAGLE3_SPEC_TOKENS = 7
DEFAULT_MTP_SPEC_TOKENS = 3
_DEFAULT_DRAFT_DIRECTORIES = {
    "eagle3": "qwen3-{size}-eagle3",
    "dflash": "qwen3-{size}-dflash",
}


def resolve_draft_method(method: str, requested: str | None) -> str:
    """Return the draft architecture implied by a launcher method."""
    if requested is not None and requested not in DRAFT_METHODS:
        raise ValueError(f"unsupported draft method: {requested}")
    if method == BASELINE_METHOD:
        if requested not in (None, "dflash"):
            raise ValueError("the dflash method requires a dflash draft")
        return "dflash"
    if method == "mtp":
        if requested not in (None, "mtp"):
            raise ValueError("the mtp method requires an mtp draft")
        return "mtp"
    if method in TTS_METHODS:
        return requested or "eagle3"
    if method == "online_draft":
        return requested or "eagle3"
    if method in {"eagle", "ospec"}:
        if requested not in (None, "eagle3"):
            raise ValueError(f"{method} supports only an eagle3 draft")
        return "eagle3"
    raise ValueError(f"unsupported method: {method}")


def resolve_speculative_tokens(draft_method: str, requested: int | None) -> int:
    """Return the default proposal depth for a draft architecture."""
    if requested is not None:
        return requested
    if draft_method == "dflash":
        return DEFAULT_SPEC_TOKENS
    if draft_method == "mtp":
        return DEFAULT_MTP_SPEC_TOKENS
    return DEFAULT_EAGLE3_SPEC_TOKENS


def default_draft_directory(artifact_root: Path, model_size: str, draft_method: str):
    """Return the default prepared draft checkpoint directory."""
    if draft_method not in _DEFAULT_DRAFT_DIRECTORIES:
        raise ValueError(f"unsupported draft method: {draft_method}")
    name = _DEFAULT_DRAFT_DIRECTORIES[draft_method].format(size=model_size)
    return artifact_root / "models" / name
