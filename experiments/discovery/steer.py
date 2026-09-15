"""E2 — pre-softmax additive attention bias on chosen backbone heads.

Reproduces the seed paper's intervention (Gandikota & Bau 2026, checked against their code 2026-08-14):
for each selected (layer, head), add +B to the attention logits of the target
segment's audio tokens and -B to the other segments' audio tokens; text
tokens untouched; applied at every query position, prompt and generation
alike. With B = 1e4 the selected heads attend only inside the target span.

Mechanism: transformers 5.x routes every attention module through
ALL_ATTENTION_FUNCTIONS[config._attn_implementation]. We register
"eager_bias" — identical to eager, plus the bias — and switch the backbone's
configs to it. Decoder attention modules are tagged with `_e2_layer_idx`; an
untagged module (e.g. Qwen2-Audio's Whisper tower) gets plain eager. The
addition is done in fp32 so that within-span logit differences survive B
(in bf16, +1e4 would quantise them away and flatten the head over the span).

    install(backbone_model)                       # once per adapter
    set_bias(heads, spans, target, prompt_len, n_heads, B, device)
    ...generate...
    clear()
"""
import torch
import torch.nn.functional as F
try:
    from transformers import AttentionInterface
except ImportError:                      # transformers < 5
    # Bunny runs in .venv-bunny on transformers 4.37, where this class does not
    # exist. That arm steers through steer_prehook.py (the attention_mask
    # pre-hook), which needs nothing from here — but vlm_regions imports this
    # module for set_bias_regions, so the import must not be fatal. install()
    # raises if anything actually tries to use the interface mechanism there.
    AttentionInterface = None
from transformers.models.llama.modeling_llama import repeat_kv


class _State:
    active = False
    bias = {}          # layer_idx -> float32 tensor [n_heads, prompt_len]


def eager_bias_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    # non-GQA modules (e.g. Qwen2-Audio's Whisper tower) have no kv groups
    groups = getattr(module, "num_key_value_groups", 1)
    key_states = repeat_kv(key, groups)
    value_states = repeat_kv(value, groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    li = getattr(module, "_e2_layer_idx", None)
    if _State.active and li is not None and li in _State.bias:
        b = _State.bias[li]                                   # [H, P]
        k_len = attn_weights.shape[-1]
        bb = b[:, :k_len] if k_len <= b.shape[1] else F.pad(b, (0, k_len - b.shape[1]))
        attn_weights = attn_weights.float() + bb[None, :, None, :].to(attn_weights.device)
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


_registered = False


def find_decoder_layers(model):
    """The nn.ModuleList of *text-decoder* layers. Criterion: each layer has a
    self_attn with `num_key_value_groups` (Llama/Qwen2-style GQA attention;
    Whisper's encoder attention has none — Qwen2-Audio's Whisper tower also has
    32 layers, so length alone would be ambiguous). Longest such list wins."""
    best = None
    for m in model.modules():
        if isinstance(m, torch.nn.ModuleList) and len(m) and all(
                hasattr(x, "self_attn") and hasattr(x.self_attn, "num_key_value_groups") for x in m):
            if best is None or len(m) > len(best):
                best = m
    assert best is not None, "no decoder layer list found"
    return best


def install(model):
    """Tag decoder attention modules and switch every config in the tree to eager_bias."""
    global _registered
    if not _registered:
        if AttentionInterface is None:
            raise RuntimeError(
                "steer.install needs transformers>=5 (AttentionInterface). This "
                "environment has an older one — use steer_prehook.install instead.")
        AttentionInterface.register("eager_bias", eager_bias_forward)
        # transformers builds the causal mask per implementation NAME; a name
        # missing from the mask registry gets NO mask (bidirectional prefill).
        # Register ours as the eager mask.
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, eager_mask
        ALL_MASK_ATTENTION_FUNCTIONS.register("eager_bias", eager_mask)
        _registered = True
    layers = find_decoder_layers(model)
    for i, layer in enumerate(layers):
        layer.self_attn._e2_layer_idx = i
    # Switch ONLY the text decoder's config (the object its attention modules
    # and its causal-mask builder read). Switching every config in the tree
    # also re-routed Qwen2-Audio's Whisper tower, whose padding mask is built
    # by name and was silently dropped (0.04 attention drift with bias off).
    cfg = layers[0].self_attn.config
    cfg._attn_implementation = "eager_bias"
    return len(layers), cfg.num_attention_heads


def set_bias(heads, spans, target, prompt_len, n_heads, B=1e4, device="cuda"):
    """heads: iterable of (layer, head). spans: list of (start, end) token index
    spans of the six segments in the prompt. target: segment index to boost."""
    _State.bias = {}
    for (l, h) in heads:
        if l not in _State.bias:
            _State.bias[l] = torch.zeros(n_heads, prompt_len, dtype=torch.float32, device=device)
        v = _State.bias[l][h]
        for j, (s, e) in enumerate(spans):
            v[s:e] = B if j == target else -B
    _State.active = True


def clear():
    _State.active = False
    _State.bias = {}


def top_k_heads(scores, k):
    L, H = scores.shape
    idx = scores.ravel().argsort()[::-1][:k]
    return [(int(i // H), int(i % H)) for i in idx]


def random_k_heads(scores, k, exclude, seed=0):
    """K heads sampled uniformly from the layer band spanned by `exclude` (the
    gaze set), excluding it — the seed paper's random-head control."""
    import numpy as np
    L, H = scores.shape
    band = sorted({l for l, _ in exclude})
    lo, hi = band[0], band[-1]
    pool = [(l, h) for l in range(lo, hi + 1) for h in range(H) if (l, h) not in set(exclude)]
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(pool), size=k, replace=False)
    return [pool[i] for i in pick]


def all_heads(scores):
    L, H = scores.shape
    return [(l, h) for l in range(L) for h in range(H)]
