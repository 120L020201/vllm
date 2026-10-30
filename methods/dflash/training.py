# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Differentiable block forward and loss for DFlash adaptation.

vLLM's inference forward relies on fused projections and KV-cache side effects,
so training replays the block forward eagerly from the module weights instead
of calling ``model.forward``.
"""

import math
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from online_draft.training.trainer import DraftTrainer, TrainerConfig

from methods.tts_common.loss import window_forward_kl_loss

_FROZEN_PARAMETER_PREFIXES = (
    "lm_head",
    "model.embed_tokens",
    "model.mask_embedding",
    "draft_id_to_target_id",
)
_DFLASH_LOSS_DECAY_GAMMA = {
    8: 4.0,
    10: 5.0,
    16: 7.0,
}


@dataclass(frozen=True, slots=True)
class DflashLossConfig:
    position_decay_rho: float


@dataclass(slots=True)
class DflashTrainingBatch:
    """One verified DFlash proposal round in block-diffusion form.

    Attributes:
        context_hidden_states: Concatenated target auxiliary hidden states for
            the accepted context, or the target hidden states when the drafter
            does not consume auxiliary features.
        context_positions: Absolute positions of the context rows.
        query_input_ids: Bonus token followed by ``num_speculative_steps``
            mask tokens.
        query_positions: Absolute positions of the query block.
        teacher_probabilities: Target next-token probabilities projected onto
            the draft vocabulary for the supervised query rows.
        teacher_coverage: Retained target probability mass per supervised row.
        rejection_position: Zero-based index of the first rejected row, or
            None when the whole block was accepted.
    """

    context_hidden_states: torch.Tensor
    context_positions: torch.Tensor
    query_input_ids: torch.Tensor
    query_positions: torch.Tensor
    teacher_probabilities: torch.Tensor
    teacher_coverage: torch.Tensor
    rejection_position: int | None

    @property
    def target_count(self) -> int:
        return self.teacher_probabilities.shape[0]

    def validate(self, *, draft_vocab_size: int, num_speculative_steps: int) -> None:
        """Validate shapes, positions, and the supervised-row boundary."""
        context_length = self.context_hidden_states.shape[0]
        query_length = self.query_input_ids.numel()
        target_count = self.target_count
        if self.context_hidden_states.ndim != 2 or context_length == 0:
            raise ValueError("context hidden states must be a nonempty matrix")
        if self.context_positions.shape != (context_length,):
            raise ValueError("context positions must match the context length")
        if self.query_input_ids.ndim != 1:
            raise ValueError("query input ids must be one-dimensional")
        if query_length != num_speculative_steps + 1:
            raise ValueError(
                "DFlash requires one bonus token plus num_speculative_steps mask tokens"
            )
        if self.query_positions.shape != (query_length,):
            raise ValueError("query positions must match the query length")
        if target_count < 1 or target_count > num_speculative_steps:
            raise ValueError("target rows must be between 1 and the draft depth")
        if self.teacher_probabilities.shape != (target_count, draft_vocab_size):
            raise ValueError("teacher probabilities must match the draft vocabulary")
        if tuple(self.teacher_coverage.shape) != (target_count,):
            raise ValueError("teacher coverage must match the target rows")
        for name in ("context_positions", "query_positions", "query_input_ids"):
            tensor = getattr(self, name)
            if tensor.dtype != torch.long:
                raise ValueError(f"{name} must use torch.long")
        for name in (
            "context_hidden_states",
            "query_positions",
            "context_positions",
            "teacher_probabilities",
            "teacher_coverage",
        ):
            tensor = getattr(self, name)
            if tensor.device != self.query_input_ids.device:
                raise ValueError(f"{name} must share the query device")
            if tensor.requires_grad:
                raise ValueError(f"{name} must be captured without gradients")
        if self.context_positions.numel() > 1 and not torch.all(
            self.context_positions[1:] == self.context_positions[:-1] + 1
        ):
            raise ValueError("context positions must be contiguous")
        if self.rejection_position is not None:
            if not isinstance(self.rejection_position, int):
                raise ValueError("rejection position must be an integer")
            if not 0 <= self.rejection_position < target_count:
                raise ValueError("rejection position must be a supervised row")
        if self.query_positions.numel() > 1 and not torch.all(
            self.query_positions[1:] == self.query_positions[:-1] + 1
        ):
            raise ValueError("query positions must be contiguous")
        if int(self.query_positions[0]) != int(self.context_positions[-1]) + 1:
            raise ValueError("the query block must start after the context")
        if not torch.isfinite(self.teacher_coverage).all():
            raise ValueError("teacher coverage must be finite")
        if (self.teacher_coverage < 0).any() or (self.teacher_coverage > 1).any():
            raise ValueError("teacher coverage must be within [0, 1]")


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    source = x.float()
    variance = source.square().mean(dim=-1, keepdim=True)
    normalized = source * torch.rsqrt(variance + eps)
    return (normalized * weight.float()).to(x.dtype)


def _linear(
    x: torch.Tensor,
    module: torch.nn.Module,
    *,
    weight_slice: slice | None = None,
) -> torch.Tensor:
    weight = module.weight
    bias = getattr(module, "bias", None)
    if weight_slice is not None:
        weight = weight[weight_slice]
        if bias is not None:
            bias = bias[weight_slice]
    return F.linear(x, weight, bias)


def _apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
) -> torch.Tensor:
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    if is_neox_style:
        first, second = torch.chunk(x, 2, dim=-1)
        rotated_first = first * cos - second * sin
        rotated_second = second * cos + first * sin
        return torch.cat((rotated_first, rotated_second), dim=-1)
    first = x[..., ::2]
    second = x[..., 1::2]
    rotated_first = first * cos - second * sin
    rotated_second = second * cos + first * sin
    return torch.stack((rotated_first, rotated_second), dim=-1).flatten(-2)


def _rope_cache(
    rotary_embedding: torch.nn.Module,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cache = rotary_embedding.cos_sin_cache.index_select(0, positions)
    return cache.chunk(2, dim=-1)


def _attention_mask(
    context_positions: torch.Tensor,
    query_positions: torch.Tensor,
    *,
    causal: bool,
    sliding_window: int | None,
) -> torch.Tensor | None:
    context_length = context_positions.numel()
    query_length = query_positions.numel()
    if not causal and sliding_window is None:
        return None
    key_positions = torch.cat((context_positions, query_positions))
    allowed = torch.ones(
        (query_length, key_positions.numel()),
        dtype=torch.bool,
        device=query_positions.device,
    )
    if sliding_window is not None:
        allowed &= key_positions.unsqueeze(0) > (
            query_positions.unsqueeze(1) - sliding_window
        )
    if causal:
        query_index = torch.arange(
            query_length, dtype=torch.long, device=query_positions.device
        )
        query_keys = query_index.unsqueeze(1) >= query_index.unsqueeze(0)
        allowed[:, context_length:] &= query_keys
    return allowed


def _context_key_value(
    hidden_states: torch.Tensor,
    attention: torch.nn.Module,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_size = attention.q_size
    kv_size = attention.kv_size
    weight_slice = slice(q_size, q_size + 2 * kv_size)
    projected = _linear(hidden_states, attention.qkv_proj, weight_slice=weight_slice)
    key, value = projected.split((kv_size, kv_size), dim=-1)
    shape = (hidden_states.shape[0], attention.num_kv_heads, attention.head_dim)
    key = key.view(shape)
    value = value.view(shape)
    key = _rms_norm(key, attention.k_norm.weight, attention.k_norm.variance_epsilon)
    cos, sin = _rope_cache(attention.rotary_emb, positions)
    key = _apply_rope(key, cos, sin, attention.rotary_emb.is_neox_style)
    return key, value


def _query_key_value(
    hidden_states: torch.Tensor,
    attention: torch.nn.Module,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_size = attention.q_size
    kv_size = attention.kv_size
    projected = _linear(hidden_states, attention.qkv_proj)
    query, key, value = projected.split((q_size, kv_size, kv_size), dim=-1)
    query = query.view(hidden_states.shape[0], attention.num_heads, attention.head_dim)
    key = key.view(hidden_states.shape[0], attention.num_kv_heads, attention.head_dim)
    value = value.view(
        hidden_states.shape[0], attention.num_kv_heads, attention.head_dim
    )
    eps = attention.q_norm.variance_epsilon
    query = _rms_norm(query, attention.q_norm.weight, eps)
    key = _rms_norm(key, attention.k_norm.weight, eps)
    cos, sin = _rope_cache(attention.rotary_emb, positions)
    query = _apply_rope(query, cos, sin, attention.rotary_emb.is_neox_style)
    key = _apply_rope(key, cos, sin, attention.rotary_emb.is_neox_style)
    return query, key, value


def _scaled_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention: torch.nn.Module,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    query_heads = query.shape[1]
    kv_heads = key.shape[1]
    if query_heads % kv_heads:
        raise ValueError("DFlash query heads must be divisible by KV heads")
    if query_heads != kv_heads and query.device.type == "cpu":
        groups = query_heads // kv_heads
        grouped_query = query.permute(1, 0, 2).reshape(
            kv_heads, groups, query.shape[0], query.shape[2]
        )
        grouped_key = key.permute(1, 0, 2).unsqueeze(1)
        grouped_value = value.permute(1, 0, 2).unsqueeze(1)
        scores = torch.matmul(
            grouped_query.float(), grouped_key.float().transpose(-2, -1)
        )
        scores = scores * attention.scaling
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(weights, grouped_value.float())
        return (
            output.permute(2, 0, 1, 3)
            .reshape(query.shape[0], query_heads, query.shape[2])
            .to(query.dtype)
        )

    query = query.transpose(0, 1).unsqueeze(0)
    key = key.transpose(0, 1).unsqueeze(0)
    value = value.transpose(0, 1).unsqueeze(0)
    if mask is not None:
        mask = mask.unsqueeze(0).unsqueeze(0)
    output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=mask,
        dropout_p=0.0,
        scale=attention.scaling,
        enable_gqa=query_heads != kv_heads,
    )
    return output.squeeze(0).transpose(0, 1)


def _mlp(hidden_states: torch.Tensor, mlp: torch.nn.Module) -> torch.Tensor:
    gate_up = _linear(hidden_states, mlp.gate_up_proj)
    gate, up = gate_up.chunk(2, dim=-1)
    return _linear(F.silu(gate) * up, mlp.down_proj)


def validate_dflash_drafter(draft_model: torch.nn.Module) -> None:
    """Fail fast when the resident drafter lacks the expected block structure."""
    model = getattr(draft_model, "model", None)
    if model is None or not hasattr(draft_model, "lm_head"):
        raise TypeError("DFlash drafter must expose .model and .lm_head")
    if not hasattr(model, "_build_fused_kv_buffers"):
        raise TypeError("DFlash drafter must expose fused context KV buffers")
    for name in ("hidden_norm", "norm", "layers"):
        if not hasattr(model, name):
            raise TypeError(f"DFlash drafter is missing .model.{name}")
    if not hasattr(model.config, "rms_norm_eps"):
        raise TypeError("DFlash drafter config is missing rms_norm_eps")
    if getattr(model, "use_aux_hidden_state", True) and not hasattr(model, "fc"):
        raise TypeError("DFlash drafter consumes auxiliary states but has no fc")
    attention_attributes = (
        "q_size",
        "kv_size",
        "num_heads",
        "num_kv_heads",
        "head_dim",
        "scaling",
        "causal",
        "sliding_window",
        "qkv_proj",
        "o_proj",
        "q_norm",
        "k_norm",
        "rotary_emb",
    )
    for layer in model.layers:
        for name in ("input_layernorm", "post_attention_layernorm", "mlp"):
            if not hasattr(layer, name):
                raise TypeError(f"DFlash decoder layer is missing {name}")
        for name in ("gate_up_proj", "down_proj"):
            if not hasattr(layer.mlp, name):
                raise TypeError(f"DFlash MLP is missing {name}")
        for name in attention_attributes:
            if not hasattr(layer.self_attn, name):
                raise TypeError(f"DFlash attention is missing {name}")


def draft_logits(
    draft_model: torch.nn.Module,
    context_features: torch.Tensor,
    context_positions: torch.Tensor,
    query_input_ids: torch.Tensor,
    query_positions: torch.Tensor,
) -> torch.Tensor:
    """Replay one DFlash block forward from the drafter's live weights.

    Args:
        draft_model: The resident ``DFlashQwen3ForCausalLM`` drafter.
        context_features: Target auxiliary features for the context rows.
        context_positions: Absolute positions of the context rows.
        query_input_ids: Bonus token followed by mask tokens.
        query_positions: Absolute positions of the query block.

    Returns:
        Draft-vocabulary logits for the query block, shaped
        ``[num_speculative_steps + 1, draft_vocab_size]``.
    """
    model = draft_model.model
    if context_features.shape[0] != context_positions.numel():
        raise ValueError("context features and positions must match")
    if query_input_ids.numel() != query_positions.numel():
        raise ValueError("query input ids and positions must match")

    eps = model.config.rms_norm_eps
    if model.use_aux_hidden_state:
        context_states = _linear(context_features, model.fc)
    else:
        context_states = context_features
    context_states = _rms_norm(context_states, model.hidden_norm.weight, eps)

    with torch.no_grad():
        if query_input_ids.device.type == "cpu":
            hidden_states = F.embedding(
                query_input_ids,
                model.embed_tokens.weight,
            )
            mask_token_id = getattr(model, "mask_token_id", None)
            if getattr(model, "has_separate_mask_embedding", False) and (
                mask_token_id is not None
            ):
                is_mask = (query_input_ids == mask_token_id).unsqueeze(-1)
                hidden_states = torch.where(
                    is_mask,
                    model.mask_embedding.to(hidden_states.dtype),
                    hidden_states,
                )
        else:
            hidden_states = draft_model.embed_input_ids(query_input_ids)
    for layer in model.layers:
        attention = layer.self_attn
        mask = _attention_mask(
            context_positions,
            query_positions,
            causal=attention.causal,
            sliding_window=attention.sliding_window,
        )
        query, query_key, query_value = _query_key_value(
            _rms_norm(hidden_states, layer.input_layernorm.weight, eps),
            attention,
            query_positions,
        )
        context_key, context_value = _context_key_value(
            context_states,
            attention,
            context_positions,
        )
        key = torch.cat((context_key, query_key))
        value = torch.cat((context_value, query_value))
        attended = _scaled_attention(query, key, value, attention, mask)
        attended = attended.reshape(query.shape[0], -1)
        hidden_states = hidden_states + _linear(attended, attention.o_proj)
        hidden_states = hidden_states + _mlp(
            _rms_norm(hidden_states, layer.post_attention_layernorm.weight, eps),
            layer.mlp,
        )

    hidden_states = _rms_norm(hidden_states, model.norm.weight, eps)
    head_weight = draft_model.lm_head.weight
    logits = F.linear(hidden_states.to(head_weight.dtype), head_weight)
    logits = logits[..., : draft_model.config.draft_vocab_size]
    return logits * _logit_scale(draft_model)


def _logit_scale(draft_model: torch.nn.Module) -> float:
    scale = getattr(draft_model.config, "logit_scale", 1.0)
    if scale is None:
        return 1.0
    return float(scale)


def target_token_ids(draft_model: torch.nn.Module) -> torch.Tensor:
    """Return the target token ID of each draft-vocabulary entry."""
    config = draft_model.config
    device = draft_model.lm_head.weight.device
    ids = torch.arange(config.draft_vocab_size, dtype=torch.long, device=device)
    offsets = getattr(draft_model, "draft_id_to_target_id", None)
    if offsets is not None:
        ids = ids + offsets.to(device=device)
    if ids.unique().numel() != ids.numel():
        raise ValueError("draft-to-target mapping contains duplicate token IDs")
    target_vocab_size = int(getattr(draft_model.lm_head, "org_vocab_size", 0))
    if target_vocab_size and (ids.min() < 0 or ids.max() >= target_vocab_size):
        raise ValueError("draft-to-target mapping exceeds the target vocabulary")
    return ids


def rebuild_fused_context_buffers(draft_model: torch.nn.Module) -> None:
    """Refresh the eager context-KV buffers after an optimizer step."""
    model = draft_model.model
    rebuild = getattr(model, "_build_fused_kv_buffers", None)
    if rebuild is None:
        raise TypeError("DFlash drafter does not expose fused context KV buffers")
    with torch.no_grad():
        rebuild()


def create_dflash_trainer(
    draft_model: torch.nn.Module,
    *,
    learning_rate: float,
    check_gradients: bool | None = None,
) -> DraftTrainer:
    """Build a DFlash trainer.

    Args:
        draft_model: DFlash inference model or its training copy.
        learning_rate: AdamW learning rate.
        check_gradients: Override the trainer's gradient checks.
    """
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning rate must be finite and positive")
    validate_dflash_drafter(draft_model)
    for name, parameter in draft_model.named_parameters():
        if torch.is_inference(parameter):
            raise RuntimeError(
                f"DFlash drafter weight {name} is an inference tensor and cannot "
                "be trained in place"
            )
        if not parameter.is_floating_point():
            if name.endswith("draft_id_to_target_id"):
                parameter.requires_grad_(False)
                continue
            raise ValueError(
                f"DFlash drafter weight {name} is not floating point; quantized "
                "drafters are not supported"
            )
    frozen_prefixes = _FROZEN_PARAMETER_PREFIXES + _env_frozen_prefixes()
    block_size = int(getattr(draft_model.config, "block_size", 16))
    try:
        loss_decay_gamma = _DFLASH_LOSS_DECAY_GAMMA[block_size]
    except KeyError as error:
        raise ValueError(
            f"DFlash TTS has no published loss decay for block size {block_size}"
        ) from error
    master_weights = os.environ.get("OSD_MASTER_WEIGHTS", "0")
    if master_weights not in {"0", "1"}:
        raise ValueError("OSD_MASTER_WEIGHTS must be 0 or 1")
    config = TrainerConfig(
        learning_rate=learning_rate,
        fused=os.environ.get("OSD_FUSED_OPTIMIZER", "1") == "1",
        check_gradients=(True if check_gradients is None else check_gradients),
        frozen_parameter_prefixes=frozen_prefixes,
        master_weights=master_weights == "1",
    )
    trainer = DraftTrainer(draft_model, config)
    trainer.loss_config = DflashLossConfig(
        position_decay_rho=math.exp(-1.0 / loss_decay_gamma),
    )
    trainable = set(trainer.trainable_parameter_names)
    leaked = {name for name in trainable if name.startswith(_FROZEN_PARAMETER_PREFIXES)}
    if leaked:
        raise ValueError(f"shared drafter weights must stay frozen: {sorted(leaked)}")
    draft_model.eval()
    return trainer


def _env_frozen_prefixes() -> tuple[str, ...]:
    raw = os.environ.get("OSD_FROZEN_PARAMETER_PREFIXES", "")
    return tuple(prefix.strip() for prefix in raw.split(",") if prefix.strip())


def train_dflash_round(
    trainer: DraftTrainer,
    draft_model: torch.nn.Module,
    batch: DflashTrainingBatch,
    *,
    max_context_tokens: int = 0,
) -> float:
    """Run one in-place forward-KL update and refresh inference buffers.

    Args:
        trainer: Optimizer and trainable DFlash parameter state.
        draft_model: Model receiving the optimizer update.
        batch: Verified proposal data for this training step.
        max_context_tokens: Optional cap on the most recent context rows.

    Returns:
        The scalar forward-KL loss before the optimizer update.
    """
    if max_context_tokens < 0:
        raise ValueError("max_context_tokens must be nonnegative")
    context_features = batch.context_hidden_states
    context_positions = batch.context_positions
    if max_context_tokens and context_features.shape[0] > max_context_tokens:
        context_features = context_features[-max_context_tokens:]
        context_positions = context_positions[-max_context_tokens:]

    with torch.inference_mode(False), torch.enable_grad():
        logits = draft_logits(
            draft_model,
            context_features,
            context_positions,
            batch.query_input_ids,
            batch.query_positions,
        )
        student = logits[1 : batch.target_count + 1]
        teacher = batch.teacher_probabilities
        proposal_depths = torch.arange(
            batch.target_count,
            dtype=torch.long,
            device=student.device,
        )
        config = trainer.loss_config
        loss = window_forward_kl_loss(
            student_logits=student,
            teacher_probabilities=teacher,
            proposal_depths=proposal_depths,
            position_decay_rho=config.position_decay_rho,
        )
        loss_value = trainer.backward_and_step(loss)
    del loss, logits, student
    if next(draft_model.parameters()).device.type == "cuda":
        rebuild_fused_context_buffers(draft_model)
    return loss_value
