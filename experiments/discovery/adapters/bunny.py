"""Adapter: Bunny-v1_0-3B — the seed paper's STRONGEST NULL (their Table 2: 8.3%).

This is V2's target. Their central empirical claim is that gaze heads exist in
some VLMs and are absent in LLaVA and Bunny, attributed to frozen vision
encoders. We test whether a steerable head set is there.

Three things make this adapter different from the Qwen ones, and each is a place
where a silent off-by-N would produce plausible wrong numbers:

1. LLaVA-STYLE SPLICING. `input_ids` carries a single IMAGE_TOKEN_INDEX = -200
   marker; the 729 image embeddings are inserted into `inputs_embeds` by
   `prepare_inputs_labels_for_multimodal`. The image block therefore CANNOT be
   found by masking input_ids the way `qwen2_vl.py` does. We call the splice
   ourselves, so `img_start` is the marker's index and the block is exactly
   729 long — both asserted against the returned embeds.

2. FIXED SQUARE INPUT, PAD ASPECT. siglip-so400m-patch14-384 takes 384x384 and
   the config asks for `image_aspect_ratio: pad`. The bundled processor only
   resizes, so padding to square is done here. A 6:1 horizontal strip then lands
   in ~16% of the frame at ~20 tokens per panel; a 3x2 grid gets ~72. See H13 in
   PREREGISTRATION.md: that difference is a candidate explanation for their null
   that competes with our own, so BOTH presentations are run and reported.

3. 27x27 GRID, NO MERGE. SigLIP emits 729 patch tokens with no spatial merge, so
   the merged grid is 27x27 rather than Qwen's h/2 x w/2.
"""

import sys

import transformers as _tf

_TF4 = int(_tf.__version__.split(".")[0]) < 5      # 4.37 in .venv-bunny

import numpy as np
import torch

IMAGE_TOKEN_INDEX = -200




def _patch_legacy_cache_api():
    """Bunny's vendored Phi code is from the transformers 4.3x era and calls
    four cache methods that transformers 5 removed. They are pure format
    adapters, so restoring them changes no numerics:

      DynamicCache.from_legacy_cache / to_legacy_cache — tuple <-> Cache. The
        vendored forward sets `use_legacy_cache = not isinstance(pkv, Cache)`,
        which is True when we pass None for the first prefill phase, so BOTH
        directions are exercised and must round-trip.
      Cache.get_usable_length — for a DynamicCache this was always just
        get_seq_length(layer_idx).
      Cache.seen_tokens — the old alias for the same thing.
    """
    if _TF4:
        # transformers 4.x has all four natively. Installing the shims there is
        # actively harmful: `seen_tokens` is an INSTANCE attribute set in
        # DynamicCache.__init__, so hasattr(Cache, ...) is False and the shim
        # would add a read-only property that breaks the constructor.
        return
    from transformers.cache_utils import Cache, DynamicCache
    if not hasattr(DynamicCache, "from_legacy_cache"):
        def from_legacy_cache(cls, past_key_values=None):
            if past_key_values is None:
                return cls()
            if isinstance(past_key_values, Cache):
                return past_key_values
            return cls(ddp_cache_data=past_key_values)
        DynamicCache.from_legacy_cache = classmethod(from_legacy_cache)
    if not hasattr(DynamicCache, "to_legacy_cache"):
        DynamicCache.to_legacy_cache = lambda self: tuple(
            (layer.keys, layer.values) for layer in self.layers)
    if not hasattr(Cache, "get_usable_length"):
        Cache.get_usable_length = lambda self, new_seq_length, layer_idx=0: \
            self.get_seq_length(layer_idx)
    if not hasattr(Cache, "seen_tokens"):
        Cache.seen_tokens = property(lambda self: self.get_seq_length())



def _patch_per_head_attention_mask(model_id):
    """Let the vendored PhiAttention accept a PER-HEAD additive mask.

    Steering adds a per-head bias through the `attention_mask` kwarg (the seed
    paper's own mechanism, see steer_prehook.py), which makes the mask
    [bsz, n_heads, q, k]. The vendored attention validates it as [bsz, 1, q, k]
    and raises — but the validation is the only obstacle: the line right after it,
    `attn_weights + attention_mask`, broadcasts a per-head mask correctly.

    Rather than hand-copy 60 lines of their forward into this repo (which would
    silently drift from the checkpoint's code), take their source, relax exactly
    that one comparison, and re-exec it in their module's namespace. The
    assertion on the match count means a change in their code fails loudly here
    instead of being patched into something else.
    """
    import inspect, textwrap
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    cls = get_class_from_dynamic_module("modeling_bunny_phi.PhiAttention", model_id)
    if getattr(cls, "_per_head_mask_patched", False):
        return
    src = textwrap.dedent(inspect.getsource(cls.forward))
    old = "if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):"
    assert src.count(old) == 1, "vendored PhiAttention.forward changed — re-check the patch"
    new = ("if attention_mask.size()[-2:] != (q_len, kv_seq_len) or \\\n"
           "                    attention_mask.size(0) != bsz or \\\n"
           "                    attention_mask.size(1) not in (1, self.num_heads):")
    src = src.replace(old, new)
    ns = sys.modules[cls.__module__].__dict__
    exec(compile(src, "<bunny_per_head_mask_patch>", "exec"), ns)
    cls.forward = ns["forward"]
    cls._per_head_mask_patched = True


def _patch_eager_vision_tower(model_id):
    """Make the SigLIP tower exist BEFORE from_pretrained loads the shards.

    Bunny's remote code builds it with `delay_load=True`, so `self.vision_tower`
    is never created; the checkpoint's 421 `model.vision_tower.vision_tower.*`
    tensors are then dropped as unexpected keys and the first attribute access
    raises `'SigLipVisionTower' object has no attribute 'vision_tower'`. Their
    own loader papers over this by calling `load_model()`, which DOWNLOADS
    google/siglip-so400m-patch14-384 and discards the bundled copy.

    The bundled weights are the POST-TRIM tower — 26 encoder layers (so400m has
    27, and `load_model` deletes the last) and no `head` tensors (it is replaced
    by Identity) — i.e. exactly the structure `load_model` leaves behind. So
    build that structure from config, empty, and let from_pretrained fill it.
    Same weights as their path, no 3.5GB download, and the tensors that ship
    with the checkpoint are actually used.
    """
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    cls = get_class_from_dynamic_module("modeling_bunny_phi.SigLipVisionTower", model_id)
    if getattr(cls, "_eager_patched", False):
        return
    mod = sys.modules[cls.__module__]
    orig_init = cls.__init__

    def __init__(self, vision_tower, vision_tower_cfg, delay_load=False):
        orig_init(self, vision_tower, vision_tower_cfg, delay_load=True)
        vt = mod.SigLipVisionModel(mod.SigLipVisionConfig())
        del vt.vision_model.encoder.layers[-1:]
        vt.vision_model.head = torch.nn.Identity()
        vt.requires_grad_(False)
        vt.eval()
        self.vision_tower = vt
        self.is_loaded = True

    cls.__init__ = __init__
    cls._eager_patched = True


class BunnyAdapter:
    model_id = "BAAI/Bunny-v1_0-3B"
    n_panels = 6
    n_image_tokens = 729          # siglip-so400m-patch14-384: (384/14)^2 = 27^2
    grid_hw = (27, 27)
    merge_size = 1
    # its remote code vendors PhiAttention and bypasses ALL_ATTENTION_FUNCTIONS,
    # so the interface hook cannot reach it; use the mask pre-hook (their approach)
    steer_mechanism = "prehook"
    cell = 256                    # panel cell size before tiling
    gap = 6                       # their DEFAULT_GAP
    target_height = 256           # their DEFAULT_TARGET_HEIGHT (strip presentation)

    QUERY = ("The image is a comic strip of six panels. "
             "What is happening in panel {k} (counting from 1)?")
    NEUTRAL = ("The image is a comic strip of six panels. "
               "Describe what is happening in one of the panels, in one sentence.")

    def __init__(self, load_4bit=True, device="cuda", presentation="grid",
                 region_mode="boxes", image_mode="pad"):
        """presentation: 'grid' (3x2, ~72 tok/panel) or 'strip' (horizontal, ~20).
        Both are run for H13; neither is the default-correct answer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        assert presentation in ("grid", "strip"), presentation
        assert region_mode in ("boxes", "column_bands"), region_mode
        assert image_mode in ("pad", "centercrop"), image_mode
        self.presentation = presentation
        self.region_mode = region_mode
        self.image_mode = image_mode
        kwargs = {"trust_remote_code": True, "attn_implementation": "eager"}
        if load_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                # The vision tower and projector must stay in compute dtype:
                # quantising them would perturb the image features the whole
                # experiment is about. transformers >= 5 anchors these patterns
                # at the START of the module path (re.match) or at its end,
                # where it used to accept any substring -- so the bare names
                # silently matched NOTHING and 156 tower Linears plus both
                # projector Linears were being quantised anyway. Both spellings
                # are passed so either version does the right thing; the
                # assertion after load is what actually guarantees it.
                # "lm_head" MUST be here. Passing llm_int8_skip_modules REPLACES
                # transformers' default keep-list (quantizers/base.py: skip_modules
                # is not None -> modules_to_not_convert = []) instead of extending
                # it, and that default is what normally keeps lm_head in compute
                # dtype. Quantised, Phi-2's head produced "CUDA error: misaligned
                # address" on the logits matmul and, when it did not crash,
                # degenerate greedy output ("_ _ _ _"). Attention capture was
                # unaffected -- lm_head runs after every layer -- so E1 stands and
                # only generation was broken.
                llm_int8_skip_modules=["vision_tower", "mm_projector", "lm_head",
                                       "model.vision_tower", "model.mm_projector"],
            )
        else:
            # Their own inference runs unquantised fp16, and Phi-2 at 2.7B fits
            # on this card. low_cpu_mem_usage must be OFF here: it meta-initialises
            # the module tree, and SigLIP registers `position_ids` as a
            # NON-persistent buffer, so it is absent from the checkpoint and stays
            # uninitialised -- which surfaces as "IndexError: index out of range"
            # inside the vision position embedding. The quantised path escapes
            # this only because bitsandbytes materialises everything on the GPU.
            # low_cpu_mem_usage must be OFF: it meta-initialises the module
            # tree, and SigLIP registers `position_ids` as a NON-persistent
            # buffer, so it is absent from the checkpoint and stays uninitialised
            # -- surfacing as "IndexError: index out of range" inside the vision
            # position embedding. transformers 4.37 refuses device_map without
            # it, so the model is moved to the card after loading instead.
            kwargs["torch_dtype" if _TF4 else "dtype"] = torch.float16
            kwargs["low_cpu_mem_usage"] = False
        self.tok = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        _patch_legacy_cache_api()
        _patch_per_head_attention_mask(self.model_id)
        _patch_eager_vision_tower(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs).eval()
        # bitsandbytes models are already placed and must not be .to()'d; only
        # the unquantised path needs moving, and transformers 4.37 refuses
        # device_map without low_cpu_mem_usage (which we cannot use here).
        if not (getattr(self.model, "is_quantized", False)
                or getattr(self.model, "hf_quantizer", None) is not None):
            self.model = self.model.to(device)
        # its remote code picks an attention class from flash availability; force eager
        self.model.config._attn_implementation = "eager"
        self.device = device
        self.image_processor = self.model.get_vision_tower().image_processor
        self._assert_vision_unquantised()

    def _assert_vision_unquantised(self):
        """A quantised vision tower would still run and still produce a plausible
        null -- exactly the failure V2 exists to rule out -- so this is checked,
        not assumed."""
        bad = [n for n, m in self.model.named_modules()
               if m.__class__.__name__ == "Linear4bit"
               and (".vision_tower." in n or ".mm_projector." in n
                    or n.endswith("lm_head"))]
        assert not bad, (f"{len(bad)} vision/projector/lm_head Linears were quantised "
                         f"(e.g. {bad[0]}) — image features or logits would be wrong")

    # ---- inputs -----------------------------------------------------------
    def _tile(self, panels):
        from vlm_regions import tile_grid, tile_strip
        if self.presentation == "grid":
            canvas, boxes = tile_grid(panels, cols=3, rows=2, cell=self.cell, gap=self.gap)
        else:
            canvas, widths = tile_strip(panels, target_height=self.target_height, gap=self.gap)
            boxes, x = [], 0
            for w in widths:
                boxes.append((x, 0, x + w, self.target_height))
                x += w + self.gap
        return canvas, boxes

    @staticmethod
    def _pad_square(img):
        """`image_aspect_ratio: pad`. The bundled processor only resizes, so the
        padding their inference script does must happen here."""
        from PIL import Image
        w, h = img.size
        if w == h:
            return img, (0, 0), (w, h)
        side = max(w, h)
        out = Image.new("RGB", (side, side), (255, 255, 255))
        off = ((side - w) // 2, (side - h) // 2)
        out.paste(img, off)
        return out, off, (side, side)

    @staticmethod
    def _center_crop_square(img):
        """The preprocessing their Appendix E.3 "panel-preservation fix"
        REPLACES, and which Bunny is NOT listed as having received.

        Fixed-resolution VLM pipelines conventionally crop to a centred square.
        On a 6:1 comic strip that keeps roughly the central sixth of the width
        and discards the rest of the panels from the image entirely. If their
        Bunny run did this, their 8.3% needs no mechanistic explanation at all.
        PREREGISTRATION 17ao item 4; the VLM sweep (17ay item 7) confirms nobody
        has tested preprocessing geometry as the cause of a mechanistic null.

        Returns the same (image, offset, size) triple as `_pad_square` so the
        caller is unchanged; the offset is NEGATIVE here because cropping moves
        canvas coordinates left/up rather than right/down.
        """
        w, h = img.size
        side = min(w, h)
        left, top = (w - side) // 2, (h - side) // 2
        return img.crop((left, top, left + side, top + side)), (-left, -top), (side, side)

    def _tokenize_with_marker(self, text):
        """LLaVA convention: split on <image>, tokenise the pieces, join with -200."""
        parts = text.split("<image>")
        assert len(parts) == 2, "expected exactly one <image> placeholder"
        a = self.tok(parts[0], add_special_tokens=True).input_ids
        b = self.tok(parts[1], add_special_tokens=False).input_ids
        return torch.tensor([a + [IMAGE_TOKEN_INDEX] + b], dtype=torch.long)

    def prepare(self, panels, _sr, query_text):
        """panels: list of 6 PIL.Image -> dict with inputs_embeds and region info."""
        assert len(panels) == self.n_panels, len(panels)
        canvas, boxes = self._tile(panels)
        _fit = self._pad_square if self.image_mode == "pad" else self._center_crop_square
        padded, off, padded_size = _fit(canvas)
        px = self.image_processor.preprocess(padded, return_tensors="pt")["pixel_values"]
        px = px.to(device=self.model.device, dtype=next(self.model.parameters()).dtype)

        prompt = f"USER: <image>\n{query_text} ASSISTANT:"
        ids = self._tokenize_with_marker(prompt).to(self.model.device)
        marker = (ids[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
        assert len(marker) == 1, f"expected one image marker, got {len(marker)}"
        img_start = int(marker[0])

        am = torch.ones_like(ids)
        _, _, attn_mask, _, embeds, _ = self.model.prepare_inputs_labels_for_multimodal(
            ids, None, am, None, None, px)
        assert embeds is not None, "splice returned no inputs_embeds"
        expected = ids.shape[1] - 1 + self.n_image_tokens
        assert embeds.shape[1] == expected, (
            f"spliced sequence is {embeds.shape[1]} but marker+729 implies {expected} — "
            "region indices would be misaligned")

        # boxes were computed on `canvas`; shift them into padded coordinates
        shifted = [(x0 + off[0], y0 + off[1], x1 + off[0], y1 + off[1])
                   for (x0, y0, x1, y1) in boxes]
        self._regions = (img_start, shifted, padded_size)
        return {"inputs_embeds": embeds,
                "attention_mask": attn_mask if attn_mask is not None else torch.ones(
                    embeds.shape[:2], dtype=torch.long, device=embeds.device),
                "input_ids": ids}          # kept for length bookkeeping only

    def panel_regions(self, inputs):
        """-> (img_start, region_ids, positions) over the spliced sequence.

        `region_mode` selects HOW panels are mapped onto image tokens, and on a
        pad-to-square model the two choices differ enormously:

        "boxes" (default) — a token belongs to the panel whose box contains its
            grid-cell centre. Padding belongs to no panel and is excluded. This
            generalises the seed paper's own `bbox_to_token_positions` rule.

        "column_bands" — their released `assign_panels_to_tokens` verbatim: split
            the grid's COLUMNS proportionally to panel widths and tile that split
            down EVERY row (`np.tile(col_to_panel, t * merged_h)`). That is exact
            for Qwen, whose native-resolution grid has no padding. Ported to a
            fixed-square model it is not: their 6.12:1 strip occupies 16.3% of a
            padded square, so ~85% of each panel's tokens are blank padding.
            Their code is Qwen-only (zero mentions of Bunny/LLaVA/SigLIP), so the
            Bunny number in their Table 2 came from code that was never released
            and this is the natural port of what WAS released. See V19 / 17ag.
        """
        from vlm_regions import (assign_panels_to_grid_blocks, assign_panels_to_tokens,
                                 region_positions)
        img_start, boxes, padded_size = self._regions
        mh, mw = self.grid_hw
        if self.region_mode == "column_bands":
            widths = [int(x1 - x0) for (x0, _y0, x1, _y1) in boxes]
            region_ids, _, _ = assign_panels_to_tokens(
                [[1, mh, mw]], widths, spatial_merge=1)
        else:
            region_ids = assign_panels_to_grid_blocks(mh, mw, boxes, padded_size,
                                                      pad_to_square=False)
        assert region_ids.shape[0] == self.n_image_tokens, region_ids.shape
        pos = region_positions(img_start, region_ids, self.n_panels)
        self.panel_token_counts = [len(p) for p in pos]
        if self.image_mode == "pad":
            assert all(len(p) > 0 for p in pos), self.panel_token_counts
        # under "centercrop" an empty panel is the finding, not a bug: a panel
        # cropped out of the image has no tokens to bias, so steering toward it
        # is a no-op and the model keeps describing whatever survived the crop.
        return img_start, region_ids, pos

    # ---- attention --------------------------------------------------------
    @torch.no_grad()
    def _two_phase_last_token_attn(self, inputs):
        """Same two-phase prefill as llm_utils, but Bunny needs its own copy.

        The vendored Phi normalises `past_key_values` into a Cache only INSIDE
        `if use_cache:`, and it only converts back to a legacy tuple when the
        caller handed it something that was not already a Cache. llm_utils
        passes None then runs phase 2 with use_cache=False, so phase 2 would
        receive the raw tuple phase 1 returned and every attention layer would
        call `.get_usable_length` on it. Seeding phase 1 with an explicit
        DynamicCache keeps a Cache object on both sides, and phase 2 keeps
        use_cache=True so the vendored code takes the path generation uses.
        """
        from transformers.cache_utils import DynamicCache
        embeds = inputs["inputs_embeds"]
        L = embeds.shape[1]
        am = torch.ones(1, L, dtype=torch.long, device=embeds.device)
        out = self.model(inputs_embeds=embeds[:, :-1], attention_mask=am[:, :-1],
                         past_key_values=DynamicCache(), use_cache=True)
        last = self.model(inputs_embeds=embeds[:, -1:], attention_mask=am,
                          past_key_values=out.past_key_values,
                          output_attentions=True, use_cache=True)
        attn = torch.stack([a[0, :, 0, :] for a in last.attentions])
        assert attn.shape[-1] == L, (attn.shape, L)
        return attn.float().cpu().numpy()

    @torch.no_grad()
    def forward_last_token_attn(self, inputs):
        """The attention the gaze score reads.
    NOTE (2026-08-17, PREREGISTRATION 17aa). Production uses the SINGLE PASS,
    not the two-phase prefill the audio adapters use. The two agree on the
    computation — identical top-5 next tokens, logit cosine .9985 — but the
    attention TENSORS they report diverge at layers 0 and 27 of 28 under bf16
    (max .92 on a row), which propagates to an 81/100 top-100 head-set
    difference. The single pass is the reference: it is one forward with no
    cache, and it is what the seed paper's own code does. It costs .47GB of
    attention here (28x28x549^2 in bf16) and ~1.2GB for Bunny, both affordable;
    the audio arms need two-phase only because their sequences are far longer.
    `_two_phase_last_token_attn` is kept for the record and for the self-test.
        """
        return self.single_pass_last_token_attn(inputs)

    @torch.no_grad()
    def single_pass_last_token_attn(self, inputs):
        out = self.model(inputs_embeds=inputs["inputs_embeds"],
                         attention_mask=inputs["attention_mask"],
                         output_attentions=True, use_cache=False)
        return torch.stack([a[0, :, -1, :] for a in out.attentions]).float().cpu().numpy()

    # ---- generation -------------------------------------------------------
    @torch.no_grad()
    def generate(self, inputs, max_new=50):
        """Greedy decode by hand rather than through `generate`.

        transformers 5 stopped having `PreTrainedModel` inherit `GenerationMixin`,
        and Bunny's vendored class predates that, so `self.model.generate` does
        not exist. Re-adding the mixin would pull in its cache bookkeeping, which
        depends on more methods this transformers removed (`seen_tokens`,
        `get_max_length`) — a longer chain of shims than the loop itself. This
        also matches how the SALMONN and Ultravox arms decode, and it keeps any
        attention bias hook active for every generated token, which is the whole
        point during a steering run.
        """
        from transformers.cache_utils import DynamicCache
        embeds = inputs["inputs_embeds"]
        device, dtype = embeds.device, embeds.dtype
        n = embeds.shape[1]
        ones = lambda k: torch.ones(1, k, dtype=torch.long, device=device)
        embed_tokens = self.model.get_model().embed_tokens
        eos = {self.tok.eos_token_id}
        out = self.model(inputs_embeds=embeds, attention_mask=ones(n),
                         past_key_values=DynamicCache(), use_cache=True)
        toks = []
        for _ in range(max_new):
            tok = int(out.logits[0, -1].argmax())
            if tok in eos:
                break
            toks.append(tok)
            e = embed_tokens(torch.tensor([[tok]], device=device)).to(dtype)
            n += 1
            out = self.model(inputs_embeds=e, attention_mask=ones(n),
                             past_key_values=out.past_key_values, use_cache=True)
        return self.tok.decode(toks, skip_special_tokens=True)

    @torch.no_grad()
    def stream_generate(self, inputs, max_new=200):
        """Yields (token_id, attn[L, H, k_len]) for each generated token — the
        attention of the forward pass that PRODUCED it, matching Qwen2-VL's
        stream_generate and llm_utils.stream_decode_from_embeds.

        Two-phase prefill, unlike `forward_last_token_attn`'s single pass: with
        ~800 positions a full [H, n, n] attention tensor is ~2.6GB across 32
        layers and does not fit alongside the fp16 model. Phase 1 runs
        [0..L-2] with the cache and NO attention output; phase 2 runs the last
        prompt token alone, so every attention tensor after that is [H, 1, k].
        An explicit DynamicCache is passed to both phases for the same reason
        `generate` does: Bunny's vendored Phi predates the transformers cache
        API and mixes legacy tuples with Cache objects otherwise.
        """
        from transformers.cache_utils import DynamicCache
        embeds = inputs["inputs_embeds"]
        device, dtype = embeds.device, embeds.dtype
        n = embeds.shape[1]
        ones = lambda k: torch.ones(1, k, dtype=torch.long, device=device)
        embed_tokens = self.model.get_model().embed_tokens
        eos = {self.tok.eos_token_id}
        out = self.model(inputs_embeds=embeds[:, :-1], attention_mask=ones(n - 1),
                         past_key_values=DynamicCache(), use_cache=True)
        out = self.model(inputs_embeds=embeds[:, -1:], attention_mask=ones(n),
                         past_key_values=out.past_key_values, use_cache=True,
                         output_attentions=True)
        for _ in range(max_new):
            tok = int(out.logits[0, -1].argmax())
            attn = torch.stack([a[0, :, -1, :] for a in out.attentions]).float().cpu().numpy()
            if tok in eos:
                return
            yield tok, attn
            e = embed_tokens(torch.tensor([[tok]], device=device)).to(dtype)
            n += 1
            out = self.model(inputs_embeds=e, attention_mask=ones(n),
                             past_key_values=out.past_key_values, use_cache=True,
                             output_attentions=True)

    def token_pieces(self, toks):
        return [self.tok.decode([x], skip_special_tokens=True) for x in toks]

    def decode_tokens(self, toks):
        return self.tok.decode(toks, skip_special_tokens=True)
