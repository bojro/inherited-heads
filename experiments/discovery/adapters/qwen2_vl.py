"""Adapter: Qwen2-VL-7B-Instruct — the POSITIVE CONTROL for experiment V2.

Why this exists. Our headline claim is that the seed paper's gaze score is
confounded with attention MASS. We cannot assert that from audio alone,
so V2 runs both score definitions in the seed paper's own modality, on their own
comic-strip data. Qwen2-VL-7B is the control: their Table 2 reports 66.2%
steering for it, so if our pipeline cannot reproduce a working head set HERE,
any null we later measure on Bunny-3B is uninterpretable.

Presentation matches theirs: ONE horizontally tiled strip, panels at height 256
with 6px gaps (their DEFAULT_TARGET_HEIGHT / DEFAULT_GAP), and each image token
assigned to a panel by its COLUMN in the merged patch grid. Panel tokens are
therefore not contiguous row-major, which is why `discovery/vlm_regions.py`
exists — it provides region-indexed aggregation and a region-indexed bias that
reuse `steer.py`'s hook without editing it. An earlier draft fed six separate
images to keep spans contiguous; that was easier but it is a different
experiment, and V2's entire purpose is testing THEIR method on THEIR data.

Token budget: panels land at ~336x336 after the processor's max_pixels clamp,
giving (336/28)^2 = 144 tokens each and ~864 for the strip — deliberately close
to Qwen2-Audio's 738, so the two modalities are compared at similar sequence
length rather than at similar pixel count.

M-RoPE caveat. Qwen2-VL derives 3-D position ids from the image grid, so the
two-phase prefill CANNOT let the model infer positions for the final token: it
would place it at the wrong index. `get_rope_index` is called once over the full
sequence and the result sliced, which is what `--self-test` verifies against a
single-pass forward.
"""

import numpy as np
import torch


class Qwen2VLAdapter:
    model_id = "Qwen/Qwen2-VL-7B-Instruct"
    n_panels = 6
    steer_mechanism = "interface"   # routes through ALL_ATTENTION_FUNCTIONS
    target_height = 256      # their DEFAULT_TARGET_HEIGHT
    gap = 6                  # their DEFAULT_GAP
    # one query per panel, mirroring the audio wording as closely as the
    # modality allows (run_discovery's is "What does the speaker of segment {k}
    # (counting from 1) talk about?")
    QUERY = ("The images are six panels of a comic strip, shown in order. "
             "What happens in panel {k} (counting from 1)?")
    NEUTRAL = ("The images are six panels of a comic strip, shown in order. "
               "Describe what happens in one of the panels, in one sentence.")

    def __init__(self, load_4bit=True, device="cuda", layout="hstrip", max_pixels=1_000_000,
                 min_pixels=256 * 28 * 28):
        # max_pixels must NOT downscale the tiled strip. A 6x256px strip is
        # ~400k px; clamping to 336*336 (the value that suited an earlier
        # six-separate-images draft) collapses it to a 4x29 merged grid = 116
        # tokens, ~19 per panel, at which point a null result would say more
        # about resolution than about the score. At >= the strip's native pixel
        # count the grid saturates at 18x112 -> 504 tokens, 84 per panel, which
        # is what their setup produces with the processor's own default.
        assert layout in ("hstrip", "vstrip"), layout
        self.layout = layout
        from transformers import AutoTokenizer, Qwen2VLForConditionalGeneration
        kwargs = {"attn_implementation": "eager"}   # required for output_attentions
        if load_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            kwargs["dtype"] = torch.bfloat16

        # AutoProcessor is deliberately NOT used. In transformers 5.15 it pulls in
        # Qwen2VLVideoProcessor, which hard-requires torchvision, and the isinstance
        # check in ProcessorMixin compares against a torchvision-gated placeholder
        # so no stub satisfies it. The image processor and tokenizer are enough:
        # the only thing the processor adds is placeholder expansion, which is a
        # closed-form function of the patch grid and is asserted in prepare().
        try:
            from transformers import Qwen2VLImageProcessorPil as _IP
        except ImportError:                       # older transformers
            from transformers import Qwen2VLImageProcessor as _IP
        self.image_processor = _IP.from_pretrained(
            self.model_id, min_pixels=min_pixels, max_pixels=max_pixels)
        self.tok = AutoTokenizer.from_pretrained(self.model_id)
        self.merge_size = self.image_processor.merge_size
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id, **kwargs).eval()
        self.device = device
        cfg = self.model.config
        self.image_token_id = cfg.image_token_id

    # ---- inputs -----------------------------------------------------------
    def prepare(self, images, _sr, query_text):
        """images: list of 6 PIL.Image in panel order -> one tiled-strip input.

        The unused middle argument keeps the signature shaped like the audio
        adapters' prepare(wav, sr, query_text).

        Placeholder expansion is done here rather than by AutoProcessor: in
        transformers 5.15 AutoProcessor pulls in Qwen2VLVideoProcessor, which
        hard-requires torchvision, and ProcessorMixin's isinstance check compares
        against a torchvision-gated placeholder that no stub satisfies. The only
        thing the processor adds is the expansion, which is prod(grid_thw) /
        merge_size**2 tokens — asserted against the located block below.
        """
        from vlm_regions import tile_column, tile_strip
        assert len(images) == self.n_panels, len(images)
        if self.layout == "vstrip":
            # V20: same panels, same task, stacked top-to-bottom. Row-major
            # flattening then makes panel k a CONTIGUOUS token run instead of
            # 24 interleaved ones.
            strip, widths = tile_column(images, target_width=self.target_height,
                                        gap=self.gap)
        else:
            strip, widths = tile_strip(images, target_height=self.target_height,
                                       gap=self.gap)
        ie = self.image_processor(images=[strip], return_tensors="pt")
        grid = ie["image_grid_thw"]
        n_tok = int(grid[0].prod() // (self.merge_size ** 2))
        text = self.tok.apply_chat_template(
            [{"role": "user", "content": "<|vision_start|>"
              + "<|image_pad|>" * n_tok + "<|vision_end|>" + query_text}],
            add_generation_prompt=True, tokenize=False)
        enc = self.tok(text, return_tensors="pt", add_special_tokens=False)
        inputs = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                  "pixel_values": ie["pixel_values"], "image_grid_thw": grid,
                  # transformers >= 5.15 requires this alongside input_ids
                  # whenever a grid is present; the processor we bypass is what
                  # normally supplies it
                  "mm_token_type_ids": self._mm_token_type_ids(enc["input_ids"])}
        self._panel_widths = widths
        self._expected_n_tok = n_tok
        return {k: (v.to(self.model.device) if hasattr(v, "to") else v)
                for k, v in inputs.items()}

    def panel_regions(self, inputs):
        """-> (img_start, region_ids, positions) for the tiled strip.

        region_ids has one entry per image token; positions[k] lists the absolute
        sequence indices belonging to panel k. Assignment is ported from their
        regions.py and verified identical to it.
        """
        from vlm_regions import assign_panels_to_tokens, region_positions
        ids = inputs["input_ids"][0]
        pos = (ids == self.image_token_id).nonzero(as_tuple=True)[0]
        if len(pos) == 0:
            raise RuntimeError("no image tokens located — placeholder not expanded")
        img_start, img_end = int(pos[0]), int(pos[-1]) + 1
        n_located = img_end - img_start
        if n_located != len(pos):
            raise RuntimeError(f"image tokens are not one contiguous block: "
                               f"{n_located} span vs {len(pos)} tokens")
        exp = getattr(self, "_expected_n_tok", None)
        if exp is not None and n_located != exp:
            raise RuntimeError(f"located {n_located} image tokens but the patch grid "
                               f"implies {exp} — regions would be misaligned")
        if self.layout == "vstrip":
            from vlm_regions import assign_panels_to_rows
            region_ids, grid_shape, ranges = assign_panels_to_rows(
                inputs["image_grid_thw"].cpu().numpy(), self._panel_widths, self.merge_size)
        else:
            region_ids, grid_shape, ranges = assign_panels_to_tokens(
                inputs["image_grid_thw"].cpu().numpy(), self._panel_widths, self.merge_size)
        region_ids = region_ids[:n_located]
        return img_start, region_ids, region_positions(img_start, region_ids, self.n_panels)

    # ---- attention --------------------------------------------------------
    def _mm_token_type_ids(self, ids):
        """0 = text, 1 = image, 2 = video, per token. transformers >= 5.15 makes
        this a required argument of get_rope_index instead of deriving it from
        the token ids internally; the processor normally supplies it, and this
        adapter bypasses AutoProcessor (it pulls in torchvision), so build it."""
        t = torch.zeros_like(ids, dtype=torch.int32)
        cfg = self.model.config
        for attr, code in (("image_token_id", 1), ("video_token_id", 2)):
            tid = getattr(cfg, attr, None)
            if tid is None:
                tid = getattr(getattr(cfg, "text_config", None), attr, None)
            if tid is not None:
                t[ids == tid] = code
        assert (t == 1).any(), "no image tokens found — image_token_id is wrong"
        return t

    def _position_ids(self, inputs):
        """Full 3-D M-RoPE position ids for the sequence. Must be computed over
        the WHOLE sequence and sliced; letting the model infer them for a lone
        final token puts it at the wrong position."""
        import inspect
        ids = inputs["input_ids"]
        kw = dict(image_grid_thw=inputs.get("image_grid_thw"),
                  video_grid_thw=inputs.get("video_grid_thw"),
                  attention_mask=inputs.get("attention_mask"))
        if "mm_token_type_ids" in inspect.signature(
                self.model.model.get_rope_index).parameters:
            kw["mm_token_type_ids"] = self._mm_token_type_ids(ids)
        pos, _ = self.model.model.get_rope_index(ids, **kw)
        return pos                                  # [3, 1, L]

    @torch.no_grad()
    def _two_phase_last_token_attn(self, inputs):
        """Two-phase prefill: bulk with cache, final token with attentions.
        Returns [layers, heads, L] float32."""
        ids = inputs["input_ids"]
        L = ids.shape[1]
        am = inputs.get("attention_mask", torch.ones_like(ids))
        pos = self._position_ids(inputs)
        mm = inputs.get("mm_token_type_ids")
        # The vision kwargs belong to phase 1 ONLY: by phase 2 the image is
        # already in the KV cache, and re-passing pixel_values would make the
        # model look for 504 image placeholders inside a 1-token input.
        vis = {k: inputs[k] for k in ("pixel_values", "image_grid_thw",
                                      "pixel_values_videos", "video_grid_thw")
               if k in inputs}
        out = self.model(input_ids=ids[:, :-1], attention_mask=am[:, :-1],
                         position_ids=pos[..., :-1], use_cache=True,
                         mm_token_type_ids=None if mm is None else mm[:, :-1],
                         **vis)
        last = self.model(input_ids=ids[:, -1:], attention_mask=am,
                          position_ids=pos[..., -1:],
                          past_key_values=out.past_key_values,
                          output_attentions=True, use_cache=False)
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
        """Reference path for --self-test only: one forward over the whole
        sequence, then take the final row. Materialises [L,H,seq,seq], so it is
        for short sequences and validation, never for a real run."""
        out = self.model(**inputs, output_attentions=True, use_cache=False)
        return torch.stack([a[0, :, -1, :] for a in out.attentions]).float().cpu().numpy()

    # ---- generation -------------------------------------------------------
    @torch.no_grad()
    def generate(self, inputs, max_new=60):
        out = self.model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
        new = out[0, inputs["input_ids"].shape[1]:]
        return self.tok.decode(new, skip_special_tokens=True)

    @torch.no_grad()
    def stream_generate(self, inputs, max_new=220):
        """Yields (token_id, attn[L, H, k_len]) for each generated token — the
        attention of the forward pass that PRODUCED that token, matching
        llm_utils.stream_decode_from_embeds.

        Prefill lets the model compute its own position ids (that also caches
        `rope_deltas`). Decode must pass them EXPLICITLY: transformers' cached
        branch builds them as `attention_mask.cumsum(-1) - 1` and never slices to
        the current step, so a 1-token input with a full-length mask produces
        length-L positions and the forward dies on a shape mismatch. Generated
        text sits after the image, where all three M-RoPE channels are equal and
        advance by one per token, so continuing from the last prompt position is
        exact rather than an approximation.
        """
        dev = inputs["input_ids"].device
        nxt = self._position_ids(inputs)[..., -1:].clone()      # [3, 1, 1]
        out = self.model(**inputs, use_cache=True, output_attentions=True)
        past, n = out.past_key_values, inputs["input_ids"].shape[1]
        eos = {self.tok.eos_token_id,
               self.tok.convert_tokens_to_ids("<|im_end|>")} - {None}
        for _ in range(max_new):
            tok = int(out.logits[0, -1].argmax())
            attn = torch.stack([a[0, :, -1, :] for a in out.attentions]).float().cpu().numpy()
            if tok in eos:
                return
            yield tok, attn
            n += 1
            nxt = nxt + 1
            out = self.model(input_ids=torch.tensor([[tok]], device=dev),
                             attention_mask=torch.ones(1, n, dtype=torch.long, device=dev),
                             position_ids=nxt,
                             past_key_values=past, use_cache=True, output_attentions=True)
            past = out.past_key_values

    def token_pieces(self, toks):
        return [self.tok.decode([x], skip_special_tokens=True) for x in toks]

    def decode_tokens(self, toks):
        return self.tok.decode(toks, skip_special_tokens=True)
