"""E3 — mid-generation retargeting (experiment E3; seed paper §6.2).

E2 fixes one target for a whole generation. E3 asks the harder question: can
the target be SWITCHED while the model is talking, and does the model follow?
The seed paper reports the model wrapping up the current panel and moving to
the new one within a few tokens (rank correlation rho=0.87 with the schedule).

Protocol: bias to segment A, decode `switch_at` tokens, then re-point the bias
to segment B and decode `switch_at` more. Judge the two halves separately.
  follow_rate  = P(second half judged B)
  stay_rate    = P(first half judged A)
  latency      = tokens after the switch before the first content word
                 belonging to B appears
A model that merely had its input masked at prefill cannot follow a switch
made during decoding, so this also discriminates steering from input masking.

    python run_retarget.py --model qwen2-audio --scores <gaze.npz> --k 100 \
        --n-items 20 --item-offset 100 --out results/e3/qwen2_audio
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch

import steer
from judge import attribute, _tokens
from run_discovery import load_strips, make_adapter
from run_steer import NEUTRAL

# The one-sentence NEUTRAL prompt makes retargeting untestable: the model finishes
# in ~20-25 tokens and the "second half" is empty, so follow_rate measures "it had
# stopped talking", not "the lever cannot be re-pointed". The seed paper used ~300
# tokens of narration with a switch every 50. This prompt is for that regime.
NARRATE = ("The audio contains six segments spoken by different people, separated "
           "by silences. Narrate in detail what the speakers talk about, one after "
           "another, describing each at length.")
PROMPT_STYLES = {"neutral": NEUTRAL, "narrate": NARRATE}
from adapters.llm_utils import embed_text_ids, load_lm_head_cpu


@torch.no_grad()
def decode_with_switch(adapter, inputs, spans, n_heads, heads, tgt_a, tgt_b, switch_at, B, max_new):
    """Greedy decode; re-point the bias at `switch_at`. Returns (ids_a, ids_b)."""
    P = int(inputs["input_ids"].shape[1])
    steer.set_bias(heads, spans, tgt_a, P, n_heads, B)
    if hasattr(adapter, "stream_generate"):
        gen = adapter.stream_generate(inputs, max_new=max_new)
        a_ids, b_ids = [], []
        for i, (tok, _) in enumerate(gen):
            if i == switch_at:
                steer.set_bias(heads, spans, tgt_b, P, n_heads, B)
            (a_ids if i < switch_at else b_ids).append(tok)
            if len(a_ids) + len(b_ids) >= 2 * switch_at:
                break
        return a_ids, b_ids
    raise RuntimeError("adapter lacks stream_generate")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strips", type=Path, default=Path("../testbed/strips"))
    ap.add_argument("--model", default="qwen2-audio")
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--B", type=float, default=1e4)
    ap.add_argument("--switch-at", type=int, default=25)
    ap.add_argument("--prompt-style", choices=("neutral", "narrate"), default="neutral",
                    help="'narrate' elicits sustained output so a mid-generation "
                         "switch has something to land in")
    ap.add_argument("--max-new", type=int, default=0,
                    help="0 = 2*switch_at (the original behaviour)")
    ap.add_argument("--n-items", type=int, default=20)
    ap.add_argument("--item-offset", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--full-precision", action="store_true")
    a = ap.parse_args()
    import pandas as pd
    meta = pd.read_parquet(a.strips / "metadata.parquet")
    scores = np.load(a.scores)["scores"]
    adapter = make_adapter(a.model, load_4bit=not a.full_precision)
    strips = list(load_strips(a.strips, a.n_items, a.item_offset))
    if hasattr(adapter, "precompute"):
        adapter.precompute(strips)
    if hasattr(adapter, "_ensure_llm"):
        adapter._ensure_llm()
    backbone = getattr(adapter, "llm", None) or adapter.model
    wav0, sr0, _ = strips[0]
    PROMPT = PROMPT_STYLES[a.prompt_style]
    inp0 = adapter.prepare(wav0, sr0, PROMPT)
    ref = adapter.forward_last_token_attn(inp0)
    n_layers, n_heads = steer.install(backbone)
    steer.clear()
    dmax = float(np.abs(ref - adapter.forward_last_token_attn(inp0)).max())
    print(f"[hook check] max |attn_ref - attn_hooked| = {dmax:.2e}", flush=True)
    assert dmax < 1e-2, "eager_bias with bias off must equal eager"
    assert (n_layers, n_heads) == scores.shape, ((n_layers, n_heads), scores.shape)
    heads = steer.top_k_heads(scores, a.k)
    rng = np.random.default_rng(0)

    a.out.mkdir(parents=True, exist_ok=True)
    stay = follow = n = 0
    lat = []
    with open(a.out / "answers.jsonl", "w") as f:
        for item, (wav, sr, bounds) in enumerate(strips):
            tr = meta[meta.item_id == item + a.item_offset].sort_values("seg_idx").transcript.tolist()
            inputs = adapter.prepare(wav, sr, PROMPT)
            spans = adapter.segment_token_spans(inputs, bounds)
            # a derangement: every pair (A,B) with A != B, sampled once per item
            tgt_a = int(rng.integers(0, 6))
            tgt_b = int((tgt_a + 1 + rng.integers(0, 5)) % 6)
            ai, bi = decode_with_switch(adapter, inputs, spans, n_heads, heads,
                                        tgt_a, tgt_b, a.switch_at, a.B,
                                        a.max_new or 2 * a.switch_at)
            steer.clear()
            txt_a, txt_b = adapter.decode_tokens(ai), adapter.decode_tokens(bi)
            ja, _ = attribute(txt_a, tr)
            jb, _ = attribute(txt_b, tr)
            stay += int(ja == tgt_a); follow += int(jb == tgt_b); n += 1
            # latency: first token index after the switch whose word is in B's transcript
            bwords = set(_tokens(tr[tgt_b]))
            pieces = adapter.token_pieces(bi)
            lt = next((i for i, p in enumerate(pieces) if set(_tokens(p)) & bwords), None)
            if lt is not None:
                lat.append(lt)
            f.write(json.dumps(dict(item=item + a.item_offset, n_tok_a=len(ai), n_tok_b=len(bi),
                                    target_a=tgt_a, target_b=tgt_b,
                                    judged_a=ja, judged_b=jb, text_a=txt_a, text_b=txt_b,
                                    latency=lt)) + "\n")
            if n % 5 == 0:
                print(f"  {n} items: stay {stay/n:.2f} follow {follow/n:.2f}", flush=True)
    rep = dict(model=a.model, k=a.k, switch_at=a.switch_at, n_items=n,
               prompt_style=a.prompt_style, max_new=a.max_new or 2 * a.switch_at,
               item_offset=a.item_offset, chance=1 / 6,
               stay_rate=stay / n, follow_rate=follow / n,
               median_latency_tokens=float(np.median(lat)) if lat else None,
               n_with_latency=len(lat))
    (a.out / "report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
