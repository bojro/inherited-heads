"""Adapter: SALMONN-13B v1 (tsinghua-ee/SALMONN) — the fully-frozen arm.

Architecture (arXiv:2310.13289 + HF model.py, read 2026-08-16):
  Whisper-large-v2 ENCODER (frozen, fp32) and BEATs iter3+ AS2M cpt2 (frozen)
  -> per-stream LayerNorm -> concat along channels (1280 + 768 = 2048) at the
  Whisper frame rate (50Hz, 1500 frames for the 30s-padded input; BEATs' ~1472
  frames are zero-padded up to 1500) -> WINDOW-LEVEL Q-FORMER: the frame
  sequence is cut into non-overlapping windows of L = round(1500 * 0.3333 / 30)
  = 17 frames (0.34s), one trainable query per window, a 2-layer BERT with
  cross-attention -> 88 outputs per 30s -> Linear 768 -> 5120 -> spliced into
  Vicuna-13B v1.1 (FROZEN) with LoRA r=8, alpha=32 on q_proj/v_proj.
  Prompt: "USER: <Speech>{88 audio embeds}</Speech> {text}\nASSISTANT:" after BOS.

Index -> time map. Whisper output frames are 20ms, so Q-Former window i covers
[0.34 i, 0.34 (i+1)) s: token index == time / 0.34 -> tokens_per_second = 2.941
(the paper's "88 tokens per 30s" = 2.93 is the same thing rounded by the last
partial window). The block is ALWAYS 88 tokens because the feature extractor
pads to 30s; a 29.5s strip occupies windows 0..86 and window 87 is padding.

Memory on the 8GB card. Vicuna-13B in NF4 is ~6.6GB of linears + 0.66GB of
embed/lm_head. Encoders (Whisper 2.5GB fp32, BEATs 0.36GB) cannot share the
card with it, so the adapter is two-phase:
  * `precompute(strips)` runs the audio side (encoders + Q-Former + proj) on the
    GPU for every strip, caches the [88, 5120] embeddings on CPU (keyed by the
    waveform), then frees the encoders.
  * the LLM loads lazily on the first forward, with lm_head replaced by
    Identity and embed_tokens kept on CPU (see llm_utils) — both are irrelevant
    to the attention we read.
  A cache miss after the LLM is up falls back to running the encoders on CPU
  (slow, correct). `--self-test` never loads the LLM.

Faithfulness notes: encoders and Q-Former run in fp32 exactly as model.py does
(that code casts only the LLM to fp16); the LLM here is NF4 with bf16 compute
(model.py: fp16). Q-Former code is vendored under salmonn_vendor/ with three
compat patches for transformers 5.x (see its header).
"""

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from adapters.llm_utils import (absolute_rate_spans, embed_text_ids,
                                greedy_decode_from_embeds, load_causal_lm_4bit,
                                load_lm_head_cpu, stream_decode_from_embeds,
                                two_phase_last_token_attn)

_VENDOR = Path(__file__).resolve().parent / "salmonn_vendor"
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))


def _wav_key(wav):
    a = np.ascontiguousarray(np.asarray(wav, dtype=np.float32))
    return hashlib.sha1(a.tobytes()).hexdigest()


class SalmonnAdapter:
    ckpt_repo, ckpt_file = "tsinghua-ee/SALMONN", "salmonn_v1.pth"
    whisper_id = "openai/whisper-large-v2"
    beats_repo = "WeiChihChen/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2"
    beats_file = "BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"   # HF mirror of the OneDrive file, same name & size (363,145,291 B)
    vicuna_id = "lmsys/vicuna-13b-v1.1"
    # transformers 5.x refuses .bin checkpoints on torch < 2.6; vicuna-13b-v1.1
    # ships only .bin. Re-shard once with discovery/tools/reshard_to_safetensors.py
    # and point here (tokenizer files are copied alongside).
    vicuna_path = os.environ.get("SALMONN_VICUNA_PATH",
                                 os.path.expanduser("~/models/vicuna-13b-v1.1-safetensors"))

    second_per_frame = 0.333333        # model.py defaults (window length & stride)
    second_stride = 0.333333
    window_frames = 17                 # round(1500 * 0.333333 / 30)
    tokens_per_second = 1.0 / (17 * 0.02)   # 2.941 tok/s
    encoder_window_s = 30.0
    n_audio_tokens = 88                # (1500 - 17) // 17 + 1
    lora_rank, lora_alpha, lora_dropout = 8, 32, 0.1
    prompt_pattern = "USER: <Speech><SpeechHere></Speech> {}\nASSISTANT:"

    def __init__(self, load_4bit=True, device="cuda"):
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer, WhisperFeatureExtractor
        from qformer.Qformer import BertConfig, BertLMHeadModel

        self.device = torch.device(device)
        self.load_4bit = load_4bit
        self.llm_dtype = torch.bfloat16
        self.enc_dtype = torch.float32

        # ---- SALMONN's own weights (Q-Former, proj, LNs, query, LoRA) ----
        ck = torch.load(hf_hub_download(self.ckpt_repo, self.ckpt_file),
                        map_location="cpu", weights_only=False)["model"]
        self._sd = ck
        self.fe = WhisperFeatureExtractor.from_pretrained(self.whisper_id)
        # Q-Former: init_speech_Qformer(num_query_token=1, speech_width=1280+768, layers=2)
        cfg = BertConfig()
        cfg.num_hidden_layers = 2
        cfg.encoder_width = 1280 + 768
        cfg.add_cross_attention = True
        cfg.cross_attention_freq = 1
        cfg.query_length = 1
        self.qformer = BertLMHeadModel(config=cfg).eval()
        q_sd = {k[len("speech_Qformer."):]: v for k, v in ck.items() if k.startswith("speech_Qformer.")}
        missing, unexpected = self.qformer.load_state_dict(q_sd, strict=False)
        # The checkpoint holds the 55 tensors the query-only path touches. Never
        # touched (so absent from the checkpoint and left at init): word/position
        # embeddings (no input_ids are ever given — BertEmbeddings.forward uses
        # query_embeds directly), the text-token FFN `intermediate.*`/`output.*`
        # (query positions go through `intermediate_query`/`output_query`), and
        # the `cls` LM head.
        unused_ok = ("bert.embeddings.word_embeddings.", "bert.embeddings.position_embeddings.",
                     ".intermediate.dense.", ".output.dense.", ".output.LayerNorm.", "cls.")
        bad = [m for m in missing if not any(u in m for u in unused_ok)]
        assert not bad and not unexpected, (bad, unexpected)
        assert len(q_sd) == 55, len(q_sd)
        self.query_tokens = nn.Parameter(ck["speech_query_tokens"].clone(), requires_grad=False)  # [1,1,768]
        self.ln_speech = nn.LayerNorm(1280); self.ln_speech.load_state_dict(
            {"weight": ck["ln_speech.weight"], "bias": ck["ln_speech.bias"]})
        self.ln_audio = nn.LayerNorm(768); self.ln_audio.load_state_dict(
            {"weight": ck["ln_audio.weight"], "bias": ck["ln_audio.bias"]})
        self.proj = nn.Linear(768, 5120); self.proj.load_state_dict(
            {"weight": ck["speech_llama_proj.weight"], "bias": ck["speech_llama_proj.bias"]})
        for m in (self.qformer, self.ln_speech, self.ln_audio, self.proj):
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)

        self.tok = AutoTokenizer.from_pretrained(self.vicuna_id, use_fast=True)
        left, right = self.prompt_pattern.split("<SpeechHere>")
        self._left_ids = self.tok(left, add_special_tokens=False).input_ids
        self._right_tpl = right
        self.bos_id = self.tok.bos_token_id
        self.filler_id = self.tok.unk_token_id if self.tok.unk_token_id is not None else 0

        self._enc = None          # (whisper_encoder, beats) once loaded
        self._enc_device = None
        self._audio_cache = {}
        self.llm = None
        self.embed_cpu = None

    # ------------------------------------------------------------------ audio side
    def _load_encoders(self, device):
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open
        from transformers import WhisperConfig
        from transformers.models.whisper.modeling_whisper import WhisperEncoder
        from beats.BEATs import BEATs, BEATsConfig

        device = torch.device(device)
        wcfg = WhisperConfig.from_pretrained(self.whisper_id)
        with torch.device(device):
            enc = WhisperEncoder(wcfg)
        enc = enc.to(self.enc_dtype).eval()
        path = hf_hub_download(self.whisper_id, "model.safetensors")
        sd = {}
        with safe_open(path, "pt") as f:
            for k in f.keys():
                if k.startswith("model.encoder."):
                    sd[k[len("model.encoder."):]] = f.get_tensor(k).to(device, self.enc_dtype)
        missing, unexpected = enc.load_state_dict(sd, strict=False)
        assert not unexpected and not missing, (missing, unexpected)
        del sd

        bck = torch.load(hf_hub_download(self.beats_repo, self.beats_file),
                         map_location="cpu", weights_only=False)
        beats = BEATs(BEATsConfig(bck["cfg"]))
        beats.load_state_dict(bck["model"])
        # SALMONN's checkpoint carries (frozen) BEATs relative_attention_bias
        # tensors; model.py loads them over the base checkpoint (strict=False)
        b_over = {k[len("beats."):]: v for k, v in self._sd.items() if k.startswith("beats.")}
        r = beats.load_state_dict(b_over, strict=False)
        assert not r.unexpected_keys, r.unexpected_keys
        beats = beats.to(device, self.enc_dtype).eval()
        for p in list(enc.parameters()) + list(beats.parameters()):
            p.requires_grad_(False)

        for m in (self.qformer, self.ln_speech, self.ln_audio, self.proj):
            m.to(device, self.enc_dtype)
        self.query_tokens.data = self.query_tokens.data.to(device, self.enc_dtype)
        self._enc, self._enc_device = (enc, beats), device

    def _free_encoders(self):
        self._enc = None
        for m in (self.qformer, self.ln_speech, self.ln_audio, self.proj):
            m.to("cpu")
        self.query_tokens.data = self.query_tokens.data.to("cpu")
        self._enc_device = None
        torch.cuda.empty_cache()

    @torch.no_grad()
    def _encode(self, wav, sr):
        """model.py `generate` up to the projected speech embeds -> [88, 5120] fp32 CPU."""
        key = _wav_key(wav)
        if key in self._audio_cache:
            return self._audio_cache[key]
        if self._enc is None:
            # GPU if the LLM is not resident yet, else CPU fallback
            self._load_encoders("cpu" if self.llm is not None else self.device)
        enc, beats = self._enc
        dev = self._enc_device
        assert sr == 16000
        wav = np.asarray(wav, dtype=np.float32)
        if len(wav) > 30 * sr:
            wav = wav[: 30 * sr]
        spec = self.fe(wav, return_tensors="pt", sampling_rate=16000).input_features.to(dev, self.enc_dtype)  # [1,80,3000]
        speech = enc(spec, return_dict=True).last_hidden_state                       # [1,1500,1280]
        raw = torch.from_numpy(wav).to(dev, self.enc_dtype).unsqueeze(0)
        audio, _ = beats.extract_features(raw, padding_mask=torch.zeros(raw.shape, device=dev).bool(),
                                          feature_only=True)                        # [1,T_b,768]
        speech = self.ln_speech(speech)
        audio = self.ln_audio(audio)
        audio = F.pad(audio, (0, 0, 0, speech.size(1) - audio.size(1)))
        x = torch.cat([speech, audio], dim=-1)                                       # [1,1500,2048]
        B, T, C = x.shape
        kernel = round(T * self.second_per_frame / 30.0)
        stride = round(T * self.second_stride / 30.0)
        assert T == 1500 and kernel == self.window_frames == stride, (T, kernel, stride)
        xt = x.transpose(1, 2).unsqueeze(2)                                          # [1,C,1,T]
        win = F.unfold(xt, kernel_size=(1, kernel), dilation=1, padding=0, stride=(1, stride))
        _, _, L = win.shape
        win = win.view(B, -1, kernel, L).permute(0, 3, 2, 1).reshape(-1, kernel, C)  # [B*L, 17, C]
        assert L == self.n_audio_tokens, L
        att = torch.ones(win.shape[:-1], dtype=torch.long, device=dev)
        q = self.query_tokens.expand(win.shape[0], -1, -1)
        out = self.qformer.bert(query_embeds=q, encoder_hidden_states=win,
                                encoder_attention_mask=att, return_dict=True)
        emb = self.proj(out.last_hidden_state).view(B, -1, 5120)[0]                 # [88, 5120]
        emb = emb.float().cpu()
        self._audio_cache[key] = emb
        return emb

    def precompute(self, strips):
        """Encode every strip on the GPU before the LLM takes the card."""
        if self.llm is not None:
            return
        self._load_encoders(self.device)
        n = 0
        for wav, sr, _ in strips:
            self._encode(wav, sr); n += 1
        self._free_encoders()
        print(f"[salmonn] precomputed audio embeddings for {n} strips; encoders freed")

    # ------------------------------------------------------------------ LLM side
    def _ensure_llm(self):
        if self.llm is not None:
            return
        from peft import LoraConfig, TaskType, get_peft_model
        if self._enc is not None and self._enc_device.type == "cuda":
            self._free_encoders()
        src = self.vicuna_path if os.path.isdir(self.vicuna_path) else self.vicuna_id
        # 13B on 8GB: double-quant NF4 (~6.6GB of linears) and embed/lm_head
        # kept off the card from the start; a plain NF4 load OOMs at 7.1GB.
        base, self.embed_cpu = load_causal_lm_4bit(src, load_4bit=self.load_4bit,
                                                   dtype=self.llm_dtype,
                                                   double_quant=True,
                                                   embed_lm_head_on_cpu=True)
        peft_cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=True,
                              r=self.lora_rank, lora_alpha=self.lora_alpha,
                              lora_dropout=self.lora_dropout, target_modules=None)
        model = get_peft_model(base, peft_cfg)
        lora_sd = {k[len("llama_model."):]: v.to(self.llm_dtype)
                   for k, v in self._sd.items() if k.startswith("llama_model.") and "lora_" in k}
        own = model.state_dict()
        hit = {k: v for k, v in lora_sd.items() if k in own}
        assert len(hit) == len(lora_sd) == 160, (len(hit), len(lora_sd), list(lora_sd)[:2], [k for k in own if 'lora' in k][:2])
        r = model.load_state_dict(hit, strict=False)
        assert not r.unexpected_keys
        model.eval()
        self.llm = model
        self.n_layers = base.config.num_hidden_layers
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ interface
    def prepare(self, wav, sr, query_text):
        emb = self._encode(wav, sr)                                   # [88, 5120]
        right_ids = self.tok(self._right_tpl.format(query_text), add_special_tokens=False).input_ids
        block_start = 1 + len(self._left_ids)
        ids = [self.bos_id] + self._left_ids + [self.filler_id] * emb.shape[0] + right_ids
        return {"input_ids": torch.tensor(ids).unsqueeze(0), "audio_embeds": emb,
                "block_start": block_start, "block_len": emb.shape[0]}

    def segment_token_spans(self, inputs, boundaries_s):
        if inputs["block_len"] == 0:
            raise RuntimeError("no audio tokens located")
        return absolute_rate_spans(inputs["block_start"], inputs["block_len"],
                                   boundaries_s, self.tokens_per_second)

    def _embeds(self, inputs):
        self._ensure_llm()
        ids = inputs["input_ids"][0]
        embeds = embed_text_ids(self.embed_cpu, ids, self.device, self.llm_dtype)
        s, n = inputs["block_start"], inputs["block_len"]
        embeds[:, s:s + n] = inputs["audio_embeds"].to(self.device, self.llm_dtype)
        return embeds

    @torch.no_grad()
    def forward_last_token_attn(self, inputs):
        embeds = self._embeds(inputs)        # loads the LLM lazily — must run before self.llm is read
        return two_phase_last_token_attn(self.llm, embeds)

    @torch.no_grad()
    @torch.no_grad()
    def next_token_logits(self, inputs):
        """Full next-token logits for the prompt as given — no generation.

        V16's readout. The V15 gates showed these models will not SAY a gender
        or an emotion, but a forced comparison between two candidate tokens
        needs only the logits, so it can have signal where free generation has
        none. Different measurement, gated separately.
        """
        embeds = self._embeds(inputs)
        if not hasattr(self, "_W"):
            src = self.vicuna_path if os.path.isdir(self.vicuna_path) else self.vicuna_id
            self._W = load_lm_head_cpu(src)
            self._eos = {self.tok.eos_token_id}
            self._sd = None
        out = self.llm(inputs_embeds=embeds,
                       attention_mask=torch.ones(1, embeds.shape[1], dtype=torch.long,
                                                 device=embeds.device),
                       use_cache=False)
        h = out.logits[0, -1].to(self._W.device, self._W.dtype)
        return (self._W @ h).float().cpu()

    def generate(self, inputs, max_new=60):
        """Greedy text answer (lm_head evaluated on CPU)."""
        embeds = self._embeds(inputs)
        if not hasattr(self, "_W"):
            src = self.vicuna_path if os.path.isdir(self.vicuna_path) else self.vicuna_id
            self._W = load_lm_head_cpu(src)
            self._eos = {self.tok.eos_token_id}
            self._sd = None   # drop the 400MB checkpoint dict; everything is loaded
        toks = greedy_decode_from_embeds(self.llm, embeds, self._W, self.embed_cpu, self._eos, max_new)
        return self.tok.decode(toks, skip_special_tokens=True)

    def stream_generate(self, inputs, max_new=200):
        embeds = self._embeds(inputs)
        if not hasattr(self, "_W"):
            src = self.vicuna_path if os.path.isdir(self.vicuna_path) else self.vicuna_id
            self._W = load_lm_head_cpu(src)
            self._eos = {self.tok.eos_token_id}
            self._sd = None
        yield from stream_decode_from_embeds(self.llm, embeds, self._W, self.embed_cpu, self._eos, max_new)

    def decode_tokens(self, toks):
        return self.tok.decode(toks, skip_special_tokens=True)

    def token_pieces(self, toks):
        return [self.tok.decode([x], skip_special_tokens=True) for x in toks]
