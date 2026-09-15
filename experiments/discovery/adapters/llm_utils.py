"""Shared LLM-side helpers for the frozen-arm adapters (SALMONN, Ultravox).

Both arms splice continuous audio embeddings into a Llama-family backbone, so
the backbone side is identical: load the causal LM in NF4, run a two-phase
prefill from `inputs_embeds`, return the final-token attention.

Memory tricks for the 8GB card (each is a no-op for the gaze score):
  * `lm_head` is replaced by Identity — the score reads attention, never
    logits. Saves 0.3GB (Vicuna-13B) / 1.05GB (Llama-3.1-8B, untied head).
  * `embed_tokens` is moved to CPU — only the few text tokens per prompt are
    embedded, on CPU, before the (GPU) forward from inputs_embeds. Saves the
    same again.
"""

import torch


def load_causal_lm_4bit(model_id, load_4bit=True, dtype=torch.float16,
                        double_quant=False, keep_lm_head=False,
                        embed_lm_head_on_cpu=False, **extra):
    """Causal LM with eager attention (required for output_attentions).
    Returns (model, embed_tokens_on_cpu).

    embed_lm_head_on_cpu: place embed_tokens and lm_head on the CPU *at load
    time* via device_map (they never occupy the card). Needed for the 13B: NF4
    linears alone are ~6.6GB with double quant (12.7B params x ~0.52 B, absmax
    included); the 0.66GB of embed+head in 16-bit is what pushed the load over
    8GB. bitsandbytes requires llm_int8_enable_fp32_cpu_offload for a device
    map with CPU entries; those two modules then simply live on the CPU."""
    from transformers import AutoModelForCausalLM
    import transformers as _tf
    # transformers 4.x calls it torch_dtype; 5 renamed it to dtype. The Bunny
    # venv is on 4.37 (PREREGISTRATION 17ad), so V18 loads Bunny's own LM here.
    _dt = "torch_dtype" if int(_tf.__version__.split(".")[0]) < 5 else "dtype"
    kwargs = {"attn_implementation": "eager", _dt: dtype,
              "low_cpu_mem_usage": True, **extra}
    if embed_lm_head_on_cpu:
        kwargs["device_map"] = {"model.embed_tokens": "cpu", "lm_head": "cpu",
                                "model.layers": 0, "model.norm": 0, "model.rotary_emb": 0}
    else:
        kwargs["device_map"] = {"": 0}
    if load_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=double_quant,
            llm_int8_enable_fp32_cpu_offload=embed_lm_head_on_cpu,
        )
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).eval()
    if not keep_lm_head:
        model.lm_head = torch.nn.Identity()
    embed = model.get_input_embeddings()
    if embed.weight.device.type == "meta":
        # accelerate CPU-offloaded it: params are on meta, real weights sit in
        # the module's AlignDevicesHook and are loaded on each forward (on CPU).
        # Materialise a plain CPU copy so lookups need no hook machinery.
        # (the hook's execution device is the main GPU, so the probe lands
        # there; bring it back to CPU and drop the GPU copy)
        with torch.no_grad():
            probe = embed(torch.arange(embed.num_embeddings)).to("cpu")   # hook moves ids to the GPU
        embed_cpu = torch.nn.Embedding.from_pretrained(probe.clone(), freeze=True)
        del probe
    else:
        embed_cpu = embed.to("cpu")
    torch.cuda.empty_cache()
    return model, embed_cpu


def embed_text_ids(embed_cpu, ids, device, dtype):
    """ids: 1-D LongTensor (cpu) -> [1, n, d] on device."""
    with torch.no_grad():
        e = embed_cpu(ids.to("cpu")).to(device=device, dtype=dtype)
    return e.unsqueeze(0)


@torch.no_grad()
def two_phase_last_token_attn(model, embeds, seed_cache=False):
    """embeds: [1, L, d] on the model device. Phase 1 runs [0..L-2] with cache
    (no attention output); phase 2 runs the last position alone with
    output_attentions=True -> per layer (1, heads, 1, L). Returns
    np.ndarray [layers, heads, L] (float32)."""
    L = embeds.shape[1]
    attn_mask = torch.ones(1, L, dtype=torch.long, device=embeds.device)
    # seed_cache: Bunny's vendored Phi normalises past_key_values into a Cache
    # only inside `if use_cache:`, so phase 2 (use_cache=False) would receive the
    # legacy tuple phase 1 returned and every attention layer would call
    # .get_usable_length on it. Passing an explicit Cache keeps a Cache object on
    # both sides. OFF by default so the validated audio path is byte-identical.
    first = {}
    if seed_cache:
        from transformers.cache_utils import DynamicCache
        first["past_key_values"] = DynamicCache()
    out = model(inputs_embeds=embeds[:, :-1], attention_mask=attn_mask[:, :-1],
                use_cache=True, **first)
    last = model(inputs_embeds=embeds[:, -1:], attention_mask=attn_mask,
                 past_key_values=out.past_key_values, output_attentions=True,
                 use_cache=False)
    attn = torch.stack([a[0, :, 0, :] for a in last.attentions])  # [layers, heads, L]
    assert attn.shape[-1] == L, (attn.shape, L)
    return attn.float().cpu().numpy()


def absolute_rate_spans(block_start, block_len, boundaries_s, tokens_per_second):
    """Segment (start_s, end_s) -> token index spans at a fixed rate, clamped to
    the located audio block. Same rule as the Qwen2-Audio adapter."""
    block_end = block_start + block_len
    spans = []
    for (s0, s1) in boundaries_s:
        a = block_start + int(round(s0 * tokens_per_second))
        b = block_start + int(round(s1 * tokens_per_second))
        a = min(max(a, block_start), block_end - 1)
        b = min(max(b, a + 1), block_end)
        spans.append((a, b))
    return spans


def load_lm_head_cpu(model_id_or_path, dtype=torch.bfloat16, device="cuda"):
    """lm_head.weight [V, d] from a safetensors checkpoint (hub id or local
    dir) — for greedy decoding when the resident copy was stripped. Placed on
    the GPU in bf16 if it fits (0.3-1.05GB; generation runs without attention
    capture so there is room), else CPU in fp32 (bf16 GEMM on CPU is emulated
    and ~20x slower)."""
    import json, os
    from safetensors import safe_open
    from huggingface_hub import hf_hub_download
    def _p(f):
        return os.path.join(model_id_or_path, f) if os.path.isdir(model_id_or_path) else hf_hub_download(model_id_or_path, f)
    try:
        idx = json.load(open(_p("model.safetensors.index.json")))
        shard = _p(idx["weight_map"]["lm_head.weight"])
    except Exception:
        shard = _p("model.safetensors")
    with safe_open(shard, "pt") as f:
        W = f.get_tensor("lm_head.weight")
    if device == "cuda":
        try:
            return W.to(device="cuda", dtype=dtype)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
    return W.float()


@torch.no_grad()
def greedy_decode_from_embeds(model, embeds, W_cpu, embed_cpu, eos_ids, max_new=60):
    """Greedy decoding for a model whose lm_head is Identity: logits are
    computed on CPU from the last hidden state. Any attention hooks (E2 bias)
    stay active throughout. Returns list of token ids."""
    device, dtype = embeds.device, embeds.dtype
    L = embeds.shape[1]
    out = model(inputs_embeds=embeds, attention_mask=torch.ones(1, L, dtype=torch.long, device=device), use_cache=True)
    past = out.past_key_values
    h = out.logits[0, -1].to(W_cpu.device, W_cpu.dtype)
    toks = []
    for _ in range(max_new):
        tok = int((W_cpu @ h).float().argmax())
        if tok in eos_ids:
            break
        toks.append(tok)
        e = embed_text_ids(embed_cpu, torch.tensor([tok]), device, dtype)
        L += 1
        out = model(inputs_embeds=e, attention_mask=torch.ones(1, L, dtype=torch.long, device=device),
                    past_key_values=past, use_cache=True)
        past = out.past_key_values
        h = out.logits[0, -1].to(W_cpu.device, W_cpu.dtype)
    return toks


@torch.no_grad()
def stream_decode_from_embeds(model, embeds, W_cpu, embed_cpu, eos_ids, max_new=200):
    """Greedy decoding that yields (token_id, attn[L, H, k_len] float32 numpy)
    for every generated token, where attn is that token's own attention (i.e.
    the attention of the position that *predicted* it is the previous yield;
    here we yield the attention computed AT the new token when it is fed back,
    which is what "attention while generating about X" needs: the token that
    belongs to sentence S attends where S is grounded).

    First yield is the last prompt token's attention paired with the first
    generated token (two-phase prefill keeps memory small).
    """
    device, dtype = embeds.device, embeds.dtype
    L = embeds.shape[1]
    ones = lambda n: torch.ones(1, n, dtype=torch.long, device=device)
    out = model(inputs_embeds=embeds[:, :-1], attention_mask=ones(L - 1), use_cache=True)
    past = out.past_key_values
    out = model(inputs_embeds=embeds[:, -1:], attention_mask=ones(L), past_key_values=past,
                use_cache=True, output_attentions=True)
    past = out.past_key_values
    for _ in range(max_new):
        h = out.logits[0, -1].to(W_cpu.device, W_cpu.dtype)
        tok = int((W_cpu @ h).float().argmax())
        attn = torch.stack([a[0, :, 0, :] for a in out.attentions]).float().cpu().numpy()
        if tok in eos_ids:
            return
        yield tok, attn
        e = embed_text_ids(embed_cpu, torch.tensor([tok]), device, dtype)
        L += 1
        out = model(inputs_embeds=e, attention_mask=ones(L), past_key_values=past,
                    use_cache=True, output_attentions=True)
        past = out.past_key_values


@torch.no_grad()
def two_phase_attn_and_value_norms(model, embeds, seed_cache=False):
    """Kobayashi et al. 2020 norm-based analysis, for the gaze score.

    The Gaze Score (and ours) is post-softmax attention alpha. Kobayashi showed
    that measuring ||alpha * f(x)|| instead of alpha REVERSES the classic
    "BERT attends to [SEP]" finding: a head can place large alpha where the
    value vector is tiny, transmitting nothing. Registered in PREREGISTRATION
    17ar as a threat to BOTH our score and the seed paper's.

    Returns (attn, vnorm, ovnorm), each [layers, heads, L] float32:
      attn   — post-softmax alpha of the final prompt token (unchanged)
      vnorm  — ||v_j^h||, the per-position value norm for that head
      ovnorm — ||W_O^h v_j^h||, the value AFTER the head's output projection.
               This is Kobayashi's actual f(x); ||v|| alone ignores that W_O
               is anisotropic. Both are returned because a per-head constant
               cancels under our row normalisation but not under raw mass.

    GQA: v_proj emits n_kv_heads; each is shared by heads_per_kv query heads,
    so the kv-head norms are expanded to query-head positions.
    """
    import numpy as np
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_heads = cfg.num_attention_heads
    n_kv = getattr(cfg, "num_key_value_heads", n_heads)
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // n_heads
    rep = n_heads // n_kv

    layers = model.model.layers
    caught = {}

    def mk_hook(i):
        def hook(mod, inp, out):
            caught.setdefault(i, []).append(out.detach()[0])   # [seq, n_kv*hd]
        return hook

    handles = [layers[i].self_attn.v_proj.register_forward_hook(mk_hook(i))
               for i in range(n_layers)]
    try:
        attn = two_phase_last_token_attn(model, embeds, seed_cache=seed_cache)
    finally:
        for h in handles:
            h.remove()

    L = embeds.shape[1]
    vnorm = np.zeros((n_layers, n_heads, L), dtype=np.float32)
    ovnorm = np.zeros((n_layers, n_heads, L), dtype=np.float32)
    for i in range(n_layers):
        v = torch.cat(caught[i], dim=0)                        # [L, n_kv*hd]
        assert v.shape[0] == L, (v.shape, L)
        v = v.view(L, n_kv, hd).float()                        # [L, n_kv, hd]
        nv = v.norm(dim=-1).T                                  # [n_kv, L]
        vnorm[i] = nv.repeat_interleave(rep, dim=0).cpu().numpy()

        Wo = _dequant(layers[i].self_attn.o_proj)              # [d_model, n_heads*hd]
        Wo = Wo.view(Wo.shape[0], n_heads, hd).float()         # [d, H, hd]
        # per query head h: || Wo[:, h, :] @ v[:, kv(h), :] ||
        vq = v.repeat_interleave(rep, dim=1)                   # [L, H, hd]
        ov = torch.einsum("dhk,lhk->lhd", Wo, vq)              # [L, H, d]
        ovnorm[i] = ov.norm(dim=-1).T.cpu().numpy()
        del v, vq, ov, Wo
    caught.clear()
    torch.cuda.empty_cache()
    return attn, vnorm, ovnorm


def _dequant(linear):
    """bf16/fp16 weight of a Linear, dequantising bitsandbytes NF4 if needed.
    One layer at a time: a 4096x4096 bf16 o_proj is 33MB, which fits alongside
    the NF4 model; holding all 32 would not."""
    w = linear.weight
    qs = getattr(w, "quant_state", None)
    if qs is None:
        return w.data.detach()
    import bitsandbytes.functional as F4
    return F4.dequantize_4bit(w.data, qs).detach()
