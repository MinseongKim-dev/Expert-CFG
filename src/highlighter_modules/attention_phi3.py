"""
Attention activation patch for Phi-3/3.5 models.

Implements Expert-CFG paper Eq. 9-10:
  Conditional branch  (first half of CFG batch): attn_weights += log(β) * hl_mask
  Unconditional branch (second half of CFG batch): attn_weights -= (log(β)+2) * hl_mask

Usage:
    patch_phi3_attention(model, beta=3.0)   # once after model load
    set_highlight_mask(binary_mask)          # before each model.generate()
    clear_highlight_mask()                   # after each model.generate()
"""

import math
import types

import torch
import torch.nn.functional as F

# Global state set per-sample before model.generate()
_HL_MASK: torch.Tensor | None = None  # [1, prompt_len], float, 0 or 1
_BETA: float = 3.0


def set_highlight_mask(mask: torch.Tensor) -> None:
    """Register highlight mask for the next generation call.

    Args:
        mask: [1, seq_len] float tensor with 1 at highlighted token positions.
    """
    global _HL_MASK
    _HL_MASK = mask


def clear_highlight_mask() -> None:
    global _HL_MASK
    _HL_MASK = None


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
    from transformers.models.phi3.modeling_phi3 import apply_rotary_pos_emb, repeat_kv

    bsz, q_len, _ = hidden_states.size()

    # --- QKV projection (Phi3 uses a combined qkv_proj) ---
    qkv = self.qkv_proj(hidden_states)
    q_size = self.num_heads * self.head_dim
    kv_size = self.num_key_value_heads * self.head_dim
    query_states = qkv[..., :q_size]
    key_states = qkv[..., q_size: q_size + kv_size]
    value_states = qkv[..., q_size + kv_size:]

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    # --- Rotary embeddings ---
    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if hasattr(past_key_value, 'get_usable_length'):
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        elif isinstance(past_key_value, tuple):
            kv_seq_len += past_key_value[0].shape[-2]

    cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

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

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # --- Attention weights ---
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    # --- Highlight attention activation (Eq. 9-10) ---
    if _HL_MASK is not None and bsz >= 2:
        bs = bsz // 2
        log_beta = math.log(_BETA)
        delta = log_beta + 2.0

        kv_len = attn_weights.shape[-1]
        prompt_len = _HL_MASK.shape[-1]

        # Align mask length to current kv length (grows with KV cache during decoding)
        if kv_len <= prompt_len:
            hl = _HL_MASK[..., :kv_len]
        else:
            pad = torch.zeros(1, kv_len - prompt_len, device=_HL_MASK.device, dtype=_HL_MASK.dtype)
            hl = torch.cat([_HL_MASK, pad], dim=-1)

        # [1, 1, 1, kv_len] broadcasts over [bs, num_heads, q_len, kv_len]
        hl = hl.unsqueeze(1).unsqueeze(1)

        # Conditional branch: attend MORE to highlighted tokens
        attn_weights[:bs] = attn_weights[:bs] + log_beta * hl
        # Unconditional branch: attend LESS to highlighted tokens
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

    Must be called after model.load() and before evaluate().
    Requires model loaded with _attn_implementation='eager'.

    Args:
        model: AutoModelForCausalLM instance (Phi-3 or Phi-3.5).
        beta: Attention amplification factor (paper default: 3.0).

    Returns:
        Number of layers patched.
    """
    global _BETA
    _BETA = beta

    try:
        from transformers.models.phi3.modeling_phi3 import Phi3Attention
    except ImportError:
        raise ImportError("transformers Phi3Attention not found. Ensure transformers>=4.36 is installed.")

    patched = 0
    for module in model.modules():
        if isinstance(module, Phi3Attention):
            module.forward = types.MethodType(_phi3_hl_forward, module)
            patched += 1

    if patched == 0:
        raise RuntimeError(
            "No Phi3Attention modules found. "
            "Ensure the model is Phi-3/3.5 and loaded with _attn_implementation='eager'."
        )

    print(f"[attention_phi3] Patched {patched} Phi3Attention layers (beta={beta})")
    return patched
