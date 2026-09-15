"""Adapter: Ultravox v0.6 (fixie-ai/ultravox-v0_6-llama-3_1-8b) — frozen-backbone arm.

Architecture (HF card + config.json + ultravox_model.py, read 2026-08-16):
  whisper-large-v3-turbo ENCODER (fine-tuned) -> stack 8 consecutive 50Hz frames
  -> SwiGLU projector (RMSNorm-pre, 10240->4096, SwiGLU->2048, RMSNorm-mid,
  2048->4096) -> spliced into Llama-3.1-8B-Instruct (FROZEN, never trained on
  audio) in place of a run of placeholder tokens. No LoRA (r=0 in the released
  config). Rate: Whisper 100Hz mel -> conv stride 2 -> 50Hz -> /8 = 6.25 tok/s;
  token count = ceil(mel_frames / 16). Encoder window 30s (3000 mel frames).

Why this adapter is composed by hand instead of `AutoModel.from_pretrained(...,
trust_remote_code=True)`:
  1. The custom loader passes only device_map/torch_dtype/attn_implementation
     through to the inner `AutoModelForCausalLM.from_pretrained(text_model_id)`,
     never quantization_config, so the 8B backbone would land in bf16 (16GB) —
     impossible on this 8GB card. Loading Llama ourselves lets it be NF4.
  2. Their `ModifiedWhisperEncoder.forward` was written against transformers
     4.51 and indexes `layer_outputs[0]`; in transformers 5.x
     `WhisperEncoderLayer` returns a bare tensor, so it silently breaks.
  So: standard `WhisperEncoder` from the installed transformers, weights from
  the Ultravox checkpoint (`audio_tower.*`, 487 tensors, strict load); the
  projector re-implemented from their code (4 tensors, strict); Llama-3.1-8B
  from meta-llama in NF4. Processing mirrors `UltravoxProcessor.__call__`
  (padding="longest", token_len = ceil(audio_lens / (2*8)), placeholder run
  filled with the eos id and overwritten by audio embeddings).

VALIDATED via `run_discovery.py --self-test --model ultravox`: implied
rate on a 10s excerpt vs 6.25 tok/s, full-strip count, strip fits window.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from adapters.llm_utils import (absolute_rate_spans, embed_text_ids,
                                greedy_decode_from_embeds, load_causal_lm_4bit,
                                load_lm_head_cpu, stream_decode_from_embeds,
                                two_phase_last_token_attn)


class _SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class _RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight.float() * x).to(dt)


class _Projector(nn.Module):
    """UltravoxProjector, projector_act=swiglu, projector_ln_mid=True."""

    def __init__(self, enc_dim=1280, stack=8, hidden=4096, out=4096):
        super().__init__()
        self.stack = stack
        self.ln_pre = _RMSNorm(enc_dim * stack)
        self.linear_1 = nn.Linear(enc_dim * stack, hidden, bias=False)
        self.act = _SwiGLU()
        self.ln_mid = _RMSNorm(hidden // 2)
        self.linear_2 = nn.Linear(hidden // 2, out, bias=False)

    def forward(self, x):  # [B, T, C]
        B, T, C = x.shape
        T_pad = math.ceil(T / self.stack) * self.stack
        x = F.pad(x, (0, 0, 0, T_pad - T)).view(B, T_pad // self.stack, C * self.stack)
        x = self.ln_pre(x)
        x = self.act(self.linear_1(x))
        x = self.ln_mid(x)
        return self.linear_2(x)


class UltravoxAdapter:
    model_id = "fixie-ai/ultravox-v0_6-llama-3_1-8b"
    text_model_id = "meta-llama/Llama-3.1-8B-Instruct"
    audio_model_id = "openai/whisper-large-v3-turbo"
    tokens_per_second = 6.25          # 50Hz / stack_factor 8
    encoder_window_s = 30.0           # 3000 mel frames
    stack_factor = 8
    encoder_ds_factor = 2

    def __init__(self, load_4bit=True, device="cuda", load_llm=True):
        self._init_audio_side(device)
        if load_llm:
            self._init_llm(load_4bit)

    def _init_audio_side(self, device):
        from safetensors import safe_open
        from huggingface_hub import hf_hub_download
        from transformers import (AutoConfig, AutoTokenizer,
                                  WhisperFeatureExtractor)
        from transformers.models.whisper.modeling_whisper import WhisperEncoder

        self.device = torch.device(device)
        self.dtype = torch.bfloat16   # Llama-3.1 is bf16-native; matches the NF4 compute dtype

        cfg = AutoConfig.from_pretrained(self.model_id, trust_remote_code=True)
        assert cfg.stack_factor == self.stack_factor and cfg.projector_act == "swiglu" \
            and cfg.projector_ln_mid, "unexpected Ultravox config"
        # tokenizer shipped with the checkpoint (Llama-3.1 template + <|audio|>)
        self.tok = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        self.filler_id = self.tok.eos_token_id      # UltravoxProcessor's audio_token_replacement
        self.fe = WhisperFeatureExtractor.from_pretrained(self.audio_model_id)
        assert self.fe.feature_size == cfg.audio_config.num_mel_bins == 128

        # --- audio tower: standard encoder class, Ultravox (fine-tuned) weights ---
        enc = WhisperEncoder(cfg.audio_config).eval()
        ckpt = hf_hub_download(self.model_id, "model.safetensors")
        enc_sd, proj_sd = {}, {}
        with safe_open(ckpt, "pt") as f:
            for k in f.keys():
                if k.startswith("audio_tower."):
                    enc_sd[k[len("audio_tower."):]] = f.get_tensor(k)
                elif k.startswith("multi_modal_projector."):
                    proj_sd[k[len("multi_modal_projector."):]] = f.get_tensor(k)
        missing, unexpected = enc.load_state_dict(enc_sd, strict=False)
        assert not missing and not unexpected, (missing, unexpected)
        self.enc = enc.to(self.device, self.dtype)

        self.proj = _Projector(enc_dim=cfg.audio_config.d_model, stack=cfg.stack_factor,
                               hidden=cfg.hidden_size, out=cfg.text_config.hidden_size)
        self.proj.load_state_dict(proj_sd, strict=True)
        self.proj = self.proj.to(self.device, self.dtype).eval()

    def _init_llm(self, load_4bit):
        # --- frozen backbone, NF4 ---
        self.llm, self.embed_cpu = load_causal_lm_4bit(
            self.text_model_id, load_4bit=load_4bit, dtype=self.dtype)
        self.n_layers = self.llm.config.num_hidden_layers

    # ---- audio ----
    @torch.no_grad()
    def _encode(self, wav, sr):
        assert sr == 16000
        wav = np.asarray(wav, dtype=np.float32)
        hop = self.fe.hop_length
        if len(wav) < 2 * hop:
            wav = np.pad(wav, (0, 2 * hop - len(wav)))
        x = self.fe([wav], sampling_rate=sr, padding="longest", pad_to_multiple_of=hop,
                    truncation=False, return_attention_mask=True, return_tensors="pt")
        feats = x.input_features.to(self.device, self.dtype)          # [1, 128, T_mel]
        audio_len = int(x.attention_mask.sum())
        if feats.shape[-1] > 3000:
            raise RuntimeError(f"audio {feats.shape[-1]} mel frames > 30s window; chunking not implemented")
        # ModifiedWhisperEncoder.forward, minus layerdrop/streaming mask; mask is
        # all-valid for a single un-padded clip so it is omitted.
        e = self.enc
        h = F.gelu(e.conv1(feats))
        h = F.gelu(e.conv2(h)).permute(0, 2, 1)                       # [1, T_enc, 1280]
        h = h + e.embed_positions.weight[: h.shape[1]]
        for layer in e.layers:
            h = layer(h, None)
        h = e.layer_norm(h)
        emb = self.proj(h)                                            # [1, ceil(T_enc/8), 4096]
        n_tok = math.ceil(audio_len / (self.encoder_ds_factor * self.stack_factor))
        assert emb.shape[1] == n_tok, (emb.shape, n_tok)
        return emb, n_tok

    # ---- interface ----
    def prepare(self, wav, sr, query_text):
        text = self.tok.apply_chat_template(
            [{"role": "user", "content": "<|audio|>\n" + query_text}],
            add_generation_prompt=True, tokenize=False)
        left, right = text.split("<|audio|>")
        left_ids = self.tok(left, add_special_tokens=False).input_ids
        right_ids = self.tok(right, add_special_tokens=False).input_ids
        audio_emb, n_tok = self._encode(wav, sr)
        ids = torch.tensor(left_ids + [self.filler_id] * n_tok + right_ids)
        return {"input_ids": ids.unsqueeze(0), "audio_embeds": audio_emb,
                "block_start": len(left_ids), "block_len": n_tok}

    def segment_token_spans(self, inputs, boundaries_s):
        if inputs["block_len"] == 0:
            raise RuntimeError("no audio tokens located")
        return absolute_rate_spans(inputs["block_start"], inputs["block_len"],
                                   boundaries_s, self.tokens_per_second)

    def _embeds(self, inputs):
        ids = inputs["input_ids"][0]
        embeds = embed_text_ids(self.embed_cpu, ids, self.device, self.dtype)  # [1, L, d]
        s, n = inputs["block_start"], inputs["block_len"]
        embeds[:, s:s + n] = inputs["audio_embeds"].to(embeds.dtype)
        return embeds

    @torch.no_grad()
    def forward_last_token_attn(self, inputs):
        return two_phase_last_token_attn(self.llm, self._embeds(inputs))

    @torch.no_grad()
    def generate(self, inputs, max_new=60):
        """Greedy text answer (lm_head evaluated on CPU)."""
        if not hasattr(self, "_W"):
            self._W = load_lm_head_cpu(self.text_model_id)
            self._eos = {self.tok.eos_token_id, self.tok.convert_tokens_to_ids("<|eot_id|>")}
        toks = greedy_decode_from_embeds(self.llm, self._embeds(inputs), self._W, self.embed_cpu, self._eos, max_new)
        return self.tok.decode(toks, skip_special_tokens=True)

    @torch.no_grad()
    def next_token_logits(self, inputs):
        """Full next-token logits for the prompt as given — no generation.

        V16's readout. The V15 gates showed these models will not SAY a gender
        or an emotion, but a forced comparison between two candidate tokens
        needs only the logits, so it can have signal where free generation has
        none. Different measurement, gated separately.
        """
        if not hasattr(self, "_W"):
            self._W = load_lm_head_cpu(self.text_model_id)
            self._eos = {self.tok.eos_token_id, self.tok.convert_tokens_to_ids("<|eot_id|>")}
        e = self._embeds(inputs)
        out = self.llm(inputs_embeds=e,
                       attention_mask=torch.ones(1, e.shape[1], dtype=torch.long, device=e.device),
                       use_cache=False)
        h = out.logits[0, -1].to(self._W.device, self._W.dtype)
        return (self._W @ h).float().cpu()

    def stream_generate(self, inputs, max_new=200):
        """Yields (token_id, attn[L,H,k_len]) per generated token (see llm_utils)."""
        if not hasattr(self, "_W"):
            self._W = load_lm_head_cpu(self.text_model_id)
            self._eos = {self.tok.eos_token_id, self.tok.convert_tokens_to_ids("<|eot_id|>")}
        yield from stream_decode_from_embeds(self.llm, self._embeds(inputs), self._W, self.embed_cpu, self._eos, max_new)

    def decode_tokens(self, toks):
        return self.tok.decode(toks, skip_special_tokens=True)

    def token_pieces(self, toks):
        return [self.tok.decode([x], skip_special_tokens=True) for x in toks]
