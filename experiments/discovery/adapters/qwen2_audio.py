"""Adapter: Qwen2-Audio-7B-Instruct (positive arm).

Audio enters the backbone as continuous features at ~25 tok/s (40ms/frame,
verified from the tech report). The processor expands the <|AUDIO|> placeholder
so that input_ids contains one audio_token_id per audio frame — the audio-token
block is located by masking input_ids, and segment spans map proportionally by
time within that block.

VALIDATE ON GPU BOX: the placeholder-expansion behavior is version-dependent
in transformers; `python run_discovery.py --self-test` checks that the located
audio-token count matches duration * 25Hz within tolerance before any real run.
"""

import numpy as np
import torch


class Qwen2AudioAdapter:
    model_id = "Qwen/Qwen2-Audio-7B-Instruct"
    tokens_per_second = 25.0

    def __init__(self, load_4bit=True, device="cuda"):
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
        kwargs = {"attn_implementation": "eager"}  # required for attention output
        if load_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            kwargs["torch_dtype"] = torch.bfloat16
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            self.model_id, **kwargs
        ).eval()
        self.device = device
        self.audio_token_id = self.model.config.audio_token_index
        self.encoder_window_s = self.processor.feature_extractor.chunk_length
        self._warned_truncation = False

    def prepare(self, wav, sr, query_text):
        conversation = [{
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": "placeholder"},
                {"type": "text", "text": query_text},
            ],
        }]
        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        inputs = self.processor(
            text=text, audio=[wav], sampling_rate=sr, return_tensors="pt"
        )
        return {k: v.to(self.model.device) for k, v in inputs.items()}

    def segment_token_spans(self, inputs, boundaries_s):
        ids = inputs["input_ids"][0]
        audio_pos = (ids == self.audio_token_id).nonzero(as_tuple=True)[0]
        if len(audio_pos) == 0:
            raise RuntimeError("no audio tokens located — check placeholder expansion")
        block_start, block_len = int(audio_pos[0]), len(audio_pos)
        block_end = block_start + block_len

        # Map by ABSOLUTE token rate, not proportionally over the strip duration.
        # Qwen2AudioProcessor forces padding="max_length" into the Whisper
        # feature extractor, whose window is chunk_length seconds (30s). Audio
        # past that is truncated, so the emitted block is
        # min(duration, 30s) * 25 tokens — NOT duration * 25. Dividing the block
        # proportionally by the full strip duration therefore assigns the wrong
        # rate (e.g. 750 tokens / 33.5s = 22.4 tok/s) and drifts every span late.
        # The encoder rate is fixed, so token index == time * 25 within the
        # window; clamp anything beyond it.
        spans = []
        for (s0, s1) in boundaries_s:
            a = block_start + int(round(s0 * self.tokens_per_second))
            b = block_start + int(round(s1 * self.tokens_per_second))
            a = min(max(a, block_start), block_end - 1)
            b = min(max(b, a + 1), block_end)
            spans.append((a, b))

        audible_s = block_len / self.tokens_per_second
        # tolerance of one token: 29.5s * 25 = 737.5 -> the processor emits 737,
        # which is rounding, not truncation
        one_tok = 1.0 / self.tokens_per_second
        if boundaries_s[-1][1] > audible_s + one_tok and not self._warned_truncation:
            self._warned_truncation = True
            lost = [i for i, (s0, _) in enumerate(boundaries_s) if s0 >= audible_s]
            print(
                f"WARNING: strip is {boundaries_s[-1][1]:.1f}s but the encoder "
                f"window holds only {audible_s:.1f}s ({block_len} tokens) — audio "
                f"past {audible_s:.1f}s is truncated by the processor. "
                f"Segments starting after the window: {lost or 'none'}; "
                f"segments spanning it are clamped and partially observed."
            )
        return spans

    @torch.no_grad()
    def forward_last_token_attn(self, inputs):
        """Two-phase prefill: bulk with cache, final token with attentions."""
        ids = inputs["input_ids"]
        bulk = {**inputs, "input_ids": ids[:, :-1],
                "attention_mask": inputs["attention_mask"][:, :-1]}
        out = self.model(**bulk, use_cache=True)
        last = self.model(
            input_ids=ids[:, -1:],
            attention_mask=inputs["attention_mask"],
            past_key_values=out.past_key_values,
            output_attentions=True,
            use_cache=False,
        )
        # each layer: (1, heads, 1, L) -> [layers, heads, L]
        attn = torch.stack([a[0, :, 0, :] for a in last.attentions])
        return attn.float().cpu().numpy()

    @torch.no_grad()
    @torch.no_grad()
    def next_token_logits(self, inputs):
        """Full next-token logits for the prompt as given — no generation.

        V16's readout. The V15 gates showed these models will not SAY a gender
        or an emotion, but a forced comparison between two candidate tokens
        needs only the logits, so it can have signal where free generation has
        none. Different measurement, gated separately.
        """
        out = self.model(**inputs, use_cache=False)
        return out.logits[0, -1].float().cpu()

    def generate(self, inputs, max_new=60):
        """Greedy text answer via HF generate (lm_head is resident)."""
        out = self.model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
        new = out[0, inputs["input_ids"].shape[1]:]
        return self.processor.tokenizer.decode(new, skip_special_tokens=True)

    @torch.no_grad()
    def stream_generate(self, inputs, max_new=200):
        """Yields (token_id, attn[L,H,k_len]) per generated token. lm_head is
        resident, so logits come from the model; two-phase prefill first."""
        ids = inputs["input_ids"]; L = ids.shape[1]
        dev = ids.device
        ones = lambda n: torch.ones(1, n, dtype=torch.long, device=dev)
        bulk = {**inputs, "input_ids": ids[:, :-1], "attention_mask": ones(L - 1)}
        out = self.model(**bulk, use_cache=True)
        out = self.model(input_ids=ids[:, -1:], attention_mask=ones(L), past_key_values=out.past_key_values,
                         use_cache=True, output_attentions=True)
        eos = {self.processor.tokenizer.eos_token_id, self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")}
        for _ in range(max_new):
            tok = int(out.logits[0, -1].argmax())
            attn = torch.stack([a[0, :, 0, :] for a in out.attentions]).float().cpu().numpy()
            if tok in eos:
                return
            yield tok, attn
            L += 1
            out = self.model(input_ids=torch.tensor([[tok]], device=dev), attention_mask=ones(L),
                             past_key_values=out.past_key_values, use_cache=True, output_attentions=True)

    def decode_tokens(self, toks):
        return self.processor.tokenizer.decode(toks, skip_special_tokens=True)

    def token_pieces(self, toks):
        t = self.processor.tokenizer
        return [t.decode([x], skip_special_tokens=True) for x in toks]
