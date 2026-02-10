"""
ALiBi utilities for MarianMT.

ALiBi (Attention with Linear Biases) replaces absolute positional embeddings
with a relative bias added to attention logits.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn
from transformers.models.marian.modeling_marian import MarianAttention


# 说明：实现 ALiBi 斜率/偏置计算，并提供 Marian 自注意力补丁。
def _get_slopes_power_of_2(num_heads: int) -> list[float]:
    start = 2 ** (-(2 ** -(math.log2(num_heads) - 3)))
    ratio = start
    return [start * (ratio**i) for i in range(num_heads)]


def get_alibi_slopes(num_heads: int) -> torch.Tensor:
    """
    Compute ALiBi slopes for each head.

    This follows the reference implementation used in the ALiBi paper.
    """
    if num_heads < 1:
        raise ValueError("num_heads must be >= 1")

    if (num_heads & (num_heads - 1)) == 0:
        slopes = _get_slopes_power_of_2(num_heads)
    else:
        closest = 2 ** math.floor(math.log2(num_heads))
        slopes = _get_slopes_power_of_2(closest)
        extra = _get_slopes_power_of_2(2 * closest)[0::2]
        slopes.extend(extra[: num_heads - closest])

    return torch.tensor(slopes, dtype=torch.float32)


def build_alibi_bias(
    slopes: torch.Tensor,
    tgt_len: int,
    src_len: int,
    device: torch.device,
    dtype: torch.dtype,
    is_causal: bool,
    cache_position: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Build ALiBi bias tensor with shape (1, num_heads, tgt_len, src_len).
    """
    if cache_position is not None:
        if cache_position.dim() == 2:
            cache_position = cache_position[0]
        q_pos = cache_position.to(device=device)
    else:
        q_pos = torch.arange(tgt_len, device=device)

    k_pos = torch.arange(src_len, device=device)

    if is_causal:
        distance = (q_pos[:, None] - k_pos[None, :]).clamp(min=0)
    else:
        distance = (q_pos[:, None] - k_pos[None, :]).abs()

    bias = -distance.to(dtype=dtype)
    slopes = slopes.to(device=device, dtype=dtype).view(1, -1, 1, 1)
    return bias.unsqueeze(0) * slopes


class ZeroPositionalEmbedding(nn.Module):
    """Positional embedding stub that returns zeros for any input shape."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.register_buffer("_device_anchor", torch.empty(0), persistent=False)

    def forward(
        self,
        input_ids_shape: torch.Size,
        past_key_values_length: int = 0,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        seq_len = input_ids_shape[1]
        device = position_ids.device if position_ids is not None else self._device_anchor.device
        return torch.zeros(seq_len, self.embed_dim, device=device)


class AlibiMarianAttention(MarianAttention):
    """
    Marian attention with ALiBi bias for self-attention.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        slopes = get_alibi_slopes(self.num_heads)
        self.register_buffer("alibi_slopes", slopes, persistent=False)

    def forward(self, *args, **kwargs):
        attention_mask = kwargs.get("attention_mask")
        key_value_states = kwargs.get("key_value_states")
        hidden_states = args[0] if args else kwargs.get("hidden_states")

        is_cross_attention = key_value_states is not None
        use_alibi = bool(getattr(self.config, "use_alibi", False))

        if use_alibi and not is_cross_attention:
            tgt_len = hidden_states.shape[1]
            src_len = tgt_len
            cache_position = kwargs.get("cache_position")
            if attention_mask is not None:
                src_len = attention_mask.shape[-1]

            alibi_bias = build_alibi_bias(
                slopes=self.alibi_slopes,
                tgt_len=tgt_len,
                src_len=src_len,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
                is_causal=self.is_causal,
                cache_position=cache_position,
            )
            if attention_mask is None:
                attention_mask = alibi_bias
            else:
                attention_mask = attention_mask + alibi_bias

            kwargs["attention_mask"] = attention_mask

        return super().forward(*args, **kwargs)


def apply_alibi_to_marian(
    model,
    use_on_encoder: bool = True,
    use_on_decoder: bool = True,
):
    """
    Patch a Marian model in-place to use ALiBi for self-attention.
    """
    if getattr(model.config, "model_type", "") != "marian":
        raise ValueError("ALiBi patching is only supported for Marian models.")

    model.config.use_alibi = True
    model.config.alibi_on_encoder = use_on_encoder
    model.config.alibi_on_decoder = use_on_decoder
    model.config._attn_implementation = "eager"

    if use_on_encoder and hasattr(model.model, "encoder"):
        model.model.encoder.embed_positions = ZeroPositionalEmbedding(model.config.d_model)
        for layer in model.model.encoder.layers:
            old_attn = layer.self_attn
            new_attn = AlibiMarianAttention(
                embed_dim=old_attn.embed_dim,
                num_heads=old_attn.num_heads,
                dropout=old_attn.dropout,
                is_decoder=old_attn.is_decoder,
                bias=old_attn.k_proj.bias is not None,
                is_causal=old_attn.is_causal,
                config=old_attn.config,
                layer_idx=old_attn.layer_idx,
            )
            new_attn.load_state_dict(old_attn.state_dict())
            layer.self_attn = new_attn

    if use_on_decoder and hasattr(model.model, "decoder"):
        model.model.decoder.embed_positions = ZeroPositionalEmbedding(model.config.d_model)
        for layer in model.model.decoder.layers:
            old_attn = layer.self_attn
            new_attn = AlibiMarianAttention(
                embed_dim=old_attn.embed_dim,
                num_heads=old_attn.num_heads,
                dropout=old_attn.dropout,
                is_decoder=old_attn.is_decoder,
                bias=old_attn.k_proj.bias is not None,
                is_causal=old_attn.is_causal,
                config=old_attn.config,
                layer_idx=old_attn.layer_idx,
            )
            new_attn.load_state_dict(old_attn.state_dict())
            layer.self_attn = new_attn

    return model

