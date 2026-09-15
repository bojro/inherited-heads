"""Gaze discovery on a pure-TEXT analogue of the strip task (inheritance test).

Same protocol as run_discovery.py, but the six segments are text passages
rather than audio spans, and the model is the bare text backbone with no audio
path at all. Everything else is identical: one forward pass per segment query,
final-prompt-token attention summed over each segment's TOKEN span, row-
normalised (queried x attended) matrix, gaze score = mean diagonal, chance 1/6,
plus a shuffled-label baseline.

The point: if a frozen-backbone audio model's gaze heads coincide with the
heads that track segments in pure text, the tracking was inherited from text
pretraining, not built by the audio adapter.

    python run_text_discovery.py --model meta-llama/Llama-3.1-8B-Instruct \
        --text testbed/text_strips --n-items 50 --out results/text/llama31_8b
"""
import argparse, json, re
from pathlib import Path

import numpy as np
import torch

from core import concentration_report, gaze_scores, segment_attention_matrix
from adapters.llm_utils import load_causal_lm_4bit, embed_text_ids, two_phase_last_token_attn

VICUNA_SYSTEM = ("A chat between a curious user and an artificial intelligence "
                 "assistant. The assistant gives helpful, detailed, and polite "
                 "answers to the user's questions.")

QUERY = ("The text above contains six segments written by different people. "
         "What does the writer of segment {k} (counting from 1) talk about?")


def _nf4_footprint_gb(cfg, double_quant, embed_on_cpu=False):
    """Estimated resident GB for a 4-bit load: NF4 linears + fp16 embed/head.

    NF4 stores 4 bits per weight plus a per-block absmax: 4 + 32/64 = 4.5 bits
    without double quant, ~4.127 with it. embed_tokens and lm_head are NOT
    quantised by bitsandbytes and stay in 16-bit.
    """
    h, L, V = cfg.hidden_size, cfg.num_hidden_layers, cfg.vocab_size
    inter = getattr(cfg, "intermediate_size", 4 * h)
    kvh = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
    kv = h * (kvh / cfg.num_attention_heads)
    per_layer = h * h + 2 * h * kv + h * h + 3 * h * inter      # q, k+v, o, mlp
    embed = 0 if embed_on_cpu else V * h * (1 if getattr(cfg, "tie_word_embeddings", False) else 2)
    bits = 4.127 if double_quant else 4.5
    return (L * per_layer * bits / 8 + embed * 2) / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--text", type=Path, default=Path("../testbed/text_strips"))
    ap.add_argument("--n-items", type=int, default=50)
    ap.add_argument("--item-offset", type=int, default=0,
                    help=">=100 selects held-out passages (PREREGISTRATION.md)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--full-precision", action="store_true")
    ap.add_argument("--trust-remote-code", action="store_true",
                    help="needed for Bunny's LM (V18): its class ships with the "
                         "checkpoint. H12 taught us to use the ACTUAL backbone, "
                         "not a stock lookalike, so we load Bunny's own fine-tuned "
                         "Phi-2 rather than microsoft/phi-2.")
    ap.add_argument("--vram-budget-gb", type=float, default=6.6,
                    help="above this estimated NF4 footprint, offload embed/lm_head "
                         "to CPU and enable double quant (8GB card, headroom for "
                         "eager attention over ~1.5k tokens)")
    a = ap.parse_args()

    from transformers import AutoConfig, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=a.trust_remote_code)
    # 13B in NF4 without double quant is ~7.1GB of linears plus 0.66GB of
    # fp16 embed+head = over an 8GB card, and bitsandbytes reports that
    # allocation failure as "CUDA driver error: device not ready" rather than
    # OutOfMemoryError -- which is what killed H11 three times (Deviations
    # 2026-08-17z). Above the budget, quantise harder and keep embed/lm_head
    # off the card entirely (llm_utils does this for SALMONN's Vicuna-13B).
    cfg = AutoConfig.from_pretrained(a.model, trust_remote_code=a.trust_remote_code)
    big = (not a.full_precision) and _nf4_footprint_gb(cfg, False) > a.vram_budget_gb
    if big:
        print(f"  large model: plain NF4 est {_nf4_footprint_gb(cfg, False):.2f}GB > "
              f"{a.vram_budget_gb}GB budget -> double quant + embed/head on CPU, "
              f"est {_nf4_footprint_gb(cfg, True, embed_on_cpu=True):.2f}GB resident",
              flush=True)
    model, embed_cpu = load_causal_lm_4bit(a.model, load_4bit=not a.full_precision,
                                           dtype=torch.bfloat16,
                                           double_quant=big,
                                           embed_lm_head_on_cpu=big,
                                           **({"trust_remote_code": True} if a.trust_remote_code else {}))
    dev = torch.device("cuda")
    rows = [json.loads(l) for l in open(a.text / "text_strips.jsonl") if l.strip()]
    rows = rows[a.item_offset : a.item_offset + a.n_items]
    if len(rows) < a.n_items:
        raise SystemExit(f"need {a.n_items} strips from offset {a.item_offset}, "
                         f"got {len(rows)} — rebuild testbed/build_text_strips.py with a larger --n-items")

    acc, acc_shuf, n = None, None, 0
    rng = np.random.default_rng(0)
    for r in rows:
        per_query = []
        spans = None
        for k in range(6):
            msg = r["body"] + "\n\n" + QUERY.format(k=k + 1)
            if getattr(tok, "chat_template", None):
                text = tok.apply_chat_template([{"role": "user", "content": msg}],
                                               add_generation_prompt=True, tokenize=False)
            else:
                # Vicuna-1.1 ships no chat_template (H11 died here after the
                # memory fix). Its documented format is the FastChat "v1"
                # conversation: system preamble, then "USER: ... ASSISTANT:".
                # The gaze score reads the FINAL prompt token, so the wrapper
                # must be the one the model was tuned on or that token is
                # off-distribution and the whole ranking shifts.
                text = VICUNA_SYSTEM + " USER: " + msg + " ASSISTANT:"
            ids = tok(text, add_special_tokens=False).input_ids
            if spans is None:
                # locate each "Segment i: ..." block by tokenising the prefix
                spans = []
                for i, seg in enumerate(r["segments"]):
                    marker = f"Segment {i+1}: {seg}"
                    j = text.find(marker)
                    pre = len(tok(text[:j], add_special_tokens=False).input_ids)
                    full = len(tok(text[:j + len(marker)], add_special_tokens=False).input_ids)
                    spans.append((pre, max(full, pre + 1)))
            embeds = embed_text_ids(embed_cpu, torch.tensor(ids), dev, torch.bfloat16)
            attn = two_phase_last_token_attn(model, embeds,
                                         seed_cache=a.trust_remote_code)      # [L, H, seq]
            per_query.append(segment_attention_matrix(attn, spans))
        matrices, _ = gaze_scores(per_query)
        acc = matrices if acc is None else acc + matrices
        perm = rng.permutation(6)
        ms, _ = gaze_scores([per_query[i] for i in perm])
        acc_shuf = ms if acc_shuf is None else acc_shuf + ms
        n += 1
        if n % 10 == 0:
            print(f"  {n} items", flush=True)
    matrices = acc / n
    scores = np.einsum("lhqq->lhq", matrices).mean(axis=-1)
    ms = acc_shuf / n
    shuf_scores = np.einsum("lhqq->lhq", ms).mean(axis=-1)
    rep, rep_s = concentration_report(scores), concentration_report(shuf_scores)
    a.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out / "scores.npz", scores=scores, matrices=matrices,
                        shuffled_scores=shuf_scores)
    summary = dict(model=a.model, modality="text", n_items=n,
                   item_offset=a.item_offset,
                   item_ids=[r["item_id"] for r in rows], query=QUERY,
                   top100_mean=rep["top100_mean"], median=rep["median"],
                   shuffled_top100_mean=rep_s["top100_mean"],
                   separation_ratio=rep["top100_mean"] / max(rep["median"], 1e-9),
                   real_vs_shuffled_top100=rep["top100_mean"] / max(rep_s["top100_mean"], 1e-9),
                   n_heads_gt_025=int((scores > 0.25).sum()),
                   top100_layer_hist=rep["top100_layer_hist"].tolist())
    (a.out / "report.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "top100_layer_hist"}, indent=2))


if __name__ == "__main__":
    main()
