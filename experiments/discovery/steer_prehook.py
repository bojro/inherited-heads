"""Model-agnostic per-head attention bias, via the attention_mask kwarg.

`steer.py` registers `eager_bias_forward` into transformers' pluggable attention
interface (`ALL_ATTENTION_FUNCTIONS` / `ALL_MASK_ATTENTION_FUNCTIONS`). That is
clean and it is what the audio arms use, but it only reaches models whose
attention actually routes through that interface. Bunny-v1_0-3B does not: its
`trust_remote_code` module vendors its own `PhiAttention` and computes
`attn_weights` inline, and it has no `num_key_value_groups`, so even
`steer.find_decoder_layers` fails on it.

This module takes the seed paper's approach instead — their `steering.py`
registers "a forward pre-hook on selected LM self-attention modules that adds a
per-head additive bias to the `attention_mask` kwarg before softmax". Any eager
attention that does `attn_weights = attn_weights + attention_mask` picks it up,
whatever the model class, because a mask shaped [B, H, q, k] broadcasts against
per-head weights just as a [B, 1, q, k] mask does.

Mathematically identical to `steer.py`'s intervention: both add a pre-softmax
constant to selected (head, key) pairs. `--verify-equivalence` on an arm where
BOTH mechanisms work (any Qwen arm) is the check that they agree numerically;
it was run and the two agree bit-for-bit.
"""
from __future__ import annotations

import numpy as np
import torch


class _State:
    active = False
    bias: dict[int, torch.Tensor] = {}      # layer_idx -> [n_heads, k_len]
    handles: list = []
    n_heads = None


def _make_hook(layer_idx: int):
    def pre_hook(module, args, kwargs):
        if not _State.active or layer_idx not in _State.bias:
            return None
        b = _State.bias[layer_idx]                       # [H, P]
        am = kwargs.get("attention_mask", None)
        if am is None:
            return None                                  # nothing to add onto
        k_len = am.shape[-1]
        bb = b[:, :k_len] if k_len <= b.shape[1] else torch.nn.functional.pad(
            b, (0, k_len - b.shape[1]))
        # [B, 1, q, k] (or [B, H, q, k]) + [1, H, 1, k]
        kwargs["attention_mask"] = am + bb[None, :, None, :].to(am.dtype).to(am.device)
        return args, kwargs
    return pre_hook


def install(layers, n_heads: int):
    """Register pre-hooks on every layer's self_attn. Returns (n_layers, n_heads)."""
    clear()
    remove()
    _State.n_heads = n_heads
    for i, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
        assert attn is not None, f"layer {i} has no self_attn/attn"
        _State.handles.append(attn.register_forward_pre_hook(_make_hook(i), with_kwargs=True))
    return len(layers), n_heads


def remove():
    for h in _State.handles:
        h.remove()
    _State.handles = []


def clear():
    _State.active = False
    _State.bias = {}


def set_bias_regions(heads, positions, target: int, prompt_len: int, n_heads: int,
                     B: float = 1e4, device: str = "cuda"):
    """+B on the target region's token positions, -B on every other region's.

    Same semantics as `vlm_regions.set_bias_regions`, but writing this module's
    state instead of `steer._State`. Region positions rather than spans because a
    tiled strip's panels are grid blocks, not contiguous runs.
    """
    _State.bias = {}
    idx = [torch.as_tensor(p, dtype=torch.long, device=device) for p in positions]
    for (l, h) in heads:
        if l not in _State.bias:
            _State.bias[l] = torch.zeros(n_heads, prompt_len, dtype=torch.float32,
                                         device=device)
        v = _State.bias[l][h]
        for j, ii in enumerate(idx):
            v[ii] = B if j == target else -B
    _State.active = True


def find_layers_generic(model):
    """Longest nn.ModuleList whose entries expose a self_attn/attn submodule.

    `steer.find_decoder_layers` keys on `num_key_value_groups` to avoid grabbing
    Qwen2-Audio's Whisper tower. That attribute is a GQA detail and Phi-2 is plain
    MHA, so it is not available here; the caller must therefore CHECK the returned
    length and head count against the score matrix shape, which both VLM runners
    do via `assert (n_layers, n_heads) == scores.shape`.
    """
    best = None
    for m in model.modules():
        if isinstance(m, torch.nn.ModuleList) and len(m) and all(
                (hasattr(x, "self_attn") or hasattr(x, "attn")) for x in m):
            if best is None or len(m) > len(best):
                best = m
    assert best is not None, "no candidate decoder layer list found"
    return best
