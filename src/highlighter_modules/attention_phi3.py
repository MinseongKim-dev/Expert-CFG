"""
Attention activation patch for Phi-3/3.5 models.

Implements Expert-CFG paper Eq. 9-10:
  Conditional branch  (first half of CFG batch): attn_weights += log(β) * hl_mask
  Unconditional branch (second half of CFG batch): attn_weights -= (log(β)+2) * hl_mask

Usage:
    patch_phi3_attention(model, beta=3.0)   # once after model load
    set_highlight_mask(binary_mask)          # before each model.generate()
    clear_highlight_mask()                   # after each model.generate()

Compatibility: transformers >= 4.43 (does not rely on transformers' apply_rotary_pos_emb
or repeat_kv, which changed API in 4.44+).
"""

import math
import types

import torch
import torch.nn.functional as F

# Global state set per-sample before model.generate()
_HL_MASK: torch.Tensor | None = None  # [1, prompt_len], float, 0 or 1
_BETA: float = 3.0


def set_highlight_mask(mask: torch.Tensor) -> None:
    global _HL_MASK
    _HL_MASK = mask


def clear_highlight_mask() -> None:
    global _HL_MASK
    _HL_MASK = None


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary(q, k, cos, sin):
    """Apply rotary embeddings. Supports cos/sin shaped [seq, head_dim] or [batch, seq, head_dim]."""
    if cos.ndim == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, head_dim]
        sin = sin.unsqueeze(0).unsqueeze(0)
    else:
        cos = cos.unsqueeze(1)  # [batch, 1, seq, head_dim]
        sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    return (
        hidden_states[:, :, None, :, :]
        .expand(batch, num_kv_heads, n_rep, slen, head_dim)
        .reshape(batch, num_kv_heads * n_rep, slen, head_dim)
    )


def _phi3_hl_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = True,
    cache_position=None,
    **kwargs,
):
    """Replacement forward for Phi3Attention that injects highlight-based attention bias."""

    bsz, q_len, _ = hidden_states.size()

    # --- QKV projection (Phi3 uses a combined qkv_proj) ---
    qkv = self.qkv_proj(hidden_states)
    q_size = self.num_heads * self.head_dim
    kv_size = self.num_key_value_heads * self.head_dim
    query_states = qkv[..., :q_size]
    key_states = qkv[..., q_size : q_size + kv_size]
    value_states = qkv[..., q_size + kv_size :]

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    # --- KV sequence length (compatible with DynamicCache 4.43 and 4.44+) ---
    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if hasattr(past_key_value, 'get_seq_length'):
            kv_seq_len += past_key_value.get_seq_length(self.layer_idx)
        elif hasattr(past_key_value, 'get_usable_length'):
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        elif isinstance(past_key_value, tuple):
            kv_seq_len += past_key_value[0].shape[-2]

    # --- Rotary embeddings ---
    # Use the model's own rotary_emb (self-contained in modeling_phi3_v.py).
    # Do NOT import apply_rotary_pos_emb from transformers — its API changed in 4.44+
    # and is incompatible with the custom RotaryEmbedding's output shape.
    cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)
    query_states, key_states = _apply_rotary(query_states, key_states, cos, sin)

    # --- KV cache update ---
    if past_key_value is not None:
        if hasattr(past_key_value, 'update'):
            ck = {"sin": sin, "cos": cos}
            if cache_position is not None:
                ck["cache_position"] = cache_position
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, ck
            )
        else:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
            past_key_value = (key_states, value_states)

    key_states = _repeat_kv(key_states, self.num_key_value_groups)
    value_states = _repeat_kv(value_states, self.num_key_value_groups)

    # --- Attention weights ---
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    # --- Highlight attention activation (Eq. 9-10) ---
    if _HL_MASK is not None and bsz >= 2:
        bs = bsz // 2
        log_beta = math.log(_BETA)
        delta = log_beta + 2.0

        kv_len = attn_weights.shape[-1]
        prompt_len = _HL_MASK.shape[-1]

        if kv_len <= prompt_len:
            hl = _HL_MASK[..., :kv_len]
        else:
            pad = torch.zeros(1, kv_len - prompt_len, device=_HL_MASK.device, dtype=_HL_MASK.dtype)
            hl = torch.cat([_HL_MASK, pad], dim=-1)

        hl = hl.unsqueeze(1).unsqueeze(1)  # [1, 1, 1, kv_len]

        attn_weights[:bs] = attn_weights[:bs] + log_beta * hl
        attn_weights[bs:] = attn_weights[bs:] - delta * hl

    # --- Standard attention mask (causal + padding) ---
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    # --- Softmax + dropout + output ---
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def patch_phi3_attention(model, beta: float = 3.0) -> int:
    """Monkey-patch all Phi3Attention layers in the model.

    Works with both standard transformers Phi3Attention and custom trust_remote_code
    variants (e.g. Phi3VAttention from modeling_phi3_v.py).

    Args:
        model: AutoModelForCausalLM instance (Phi-3 or Phi-3.5).
        beta: Attention amplification factor (paper default: 3.0).

    Returns:
        Number of layers patched.
    """
    global _BETA
    _BETA = beta

    patched = 0
    for module in model.modules():
        # Match by class name to work with trust_remote_code custom classes
        if (hasattr(module, 'qkv_proj') and hasattr(module, 'o_proj') and hasattr(module, 'rotary_emb')):
            module.forward = types.MethodType(_phi3_hl_forward, module)
            patched += 1

    if patched == 0:
        raise RuntimeError(
            "No Phi3Attention modules found. "
            "Ensure the model is Phi-3/3.5 and loaded with _attn_implementation='eager'."
        )

    print(f"[attention_phi3] Patched {patched} Phi3Attention layers (beta={beta})")
    return patched
