"""E1' — generation-conditioned gaze score (closes the comprehension loophole).

E1 scores attention at the last prompt token of a QUERY about segment k, so a
model that cannot parse "segment k" cannot score even if it has the
mechanism. Here, instead, the model is asked to describe all six speakers in
order; every generated token is attributed to the segment its sentence
describes (judge.attribute on the sentence, transcripts as ground truth); the
per-head (described x attended) matrix is built from the tokens' own
attention; score = mean diagonal over the segments actually described.
This is the seed paper's operationalisation (attention while *describing*
panel R), model-agnostic w.r.t. instruction following.

    python run_gencond.py --model qwen2-audio --n-items 50 --out results/gencond/qwen2_audio
"""
import argparse, json, re
from pathlib import Path

import numpy as np

from core import concentration_report, segment_attention_matrix
from judge import attribute
from run_discovery import load_strips, make_adapter

PROMPT = ("The audio contains six segments spoken by different people, "
          "separated by silences. Describe what each of the six speakers "
          "talks about, one sentence per speaker, in order.")

_SENT_END = re.compile(r"(?<=[.!?])\s+|\n+")


def sentences_with_spans(text):
    """[(start_char, end_char, sentence)] over the generated text."""
    out, pos = [], 0
    for part in _SENT_END.split(text):
        if not part:
            continue
        i = text.find(part, pos)
        if i < 0:
            i = pos
        out.append((i, i + len(part), part))
        pos = i + len(part)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strips", type=Path, default=Path("../testbed/strips"))
    ap.add_argument("--model", default="qwen2-audio")
    ap.add_argument("--n-items", type=int, default=50)
    ap.add_argument("--item-offset", type=int, default=0, help=">=50 selects held-out data")
    ap.add_argument("--max-new", type=int, default=220)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--full-precision", action="store_true")
    a = ap.parse_args()
    import pandas as pd
    meta = pd.read_parquet(a.strips / "metadata.parquet")
    adapter = make_adapter(a.model, load_4bit=not a.full_precision)
    strips = list(load_strips(a.strips, a.n_items, a.item_offset))
    if hasattr(adapter, "precompute"):
        adapter.precompute(strips)
    a.out.mkdir(parents=True, exist_ok=True)

    acc_M, acc_cnt = None, None          # sum of row-normalised matrices; count of rows present
    acc_S = None                          # shuffled-label baseline (permute which segment each sentence "describes")
    rng = np.random.default_rng(0)
    per_item, coverage, log = [], [], []
    for item, (wav, sr, bounds) in enumerate(strips):
        tr = meta[meta.item_id == item + a.item_offset].sort_values("seg_idx").transcript.tolist()
        inputs = adapter.prepare(wav, sr, PROMPT)
        spans = adapter.segment_token_spans(inputs, bounds)
        toks, masses = [], []
        for tok, attn in adapter.stream_generate(inputs, max_new=a.max_new):
            toks.append(tok)
            masses.append(segment_attention_matrix(attn, spans))      # [L, H, 6]
        if not toks:
            coverage.append(0); continue
        pieces = adapter.token_pieces(toks)
        text = "".join(pieces)
        # char offset of each token
        offs, c = [], 0
        for p in pieces:
            offs.append(c); c += len(p)
        # sentence -> segment
        tok_seg = np.full(len(toks), -1)
        sent_log = []
        for (s0, s1, sent) in sentences_with_spans(text):
            j, _ = attribute(sent, tr)
            sent_log.append((j, sent))
            for ti, o in enumerate(offs):
                if s0 <= o < s1:
                    tok_seg[ti] = j
        def build(labels):
            M = np.zeros(masses[0].shape[:2] + (6, 6), dtype=np.float64)   # [L,H,described,attended]
            present = np.zeros(6, dtype=bool)
            for d in range(6):
                sel = np.where(labels == d)[0]
                if len(sel) == 0:
                    continue
                present[d] = True
                m = np.stack([masses[i] for i in sel]).mean(0)             # [L,H,6]
                denom = m.sum(-1, keepdims=True)
                M[:, :, d, :] = np.divide(m, denom, out=np.zeros_like(m), where=denom > 0)
            return M, present
        M, present = build(tok_seg)
        # shuffled baseline: permute the segment identities of the described sentences
        ids = sorted(set(tok_seg[tok_seg >= 0].tolist()))
        perm = dict(zip(ids, rng.permutation(ids)))
        S, _ = build(np.array([perm.get(x, -1) for x in tok_seg]))
        acc_S = S if acc_S is None else acc_S + S
        acc_M = M if acc_M is None else acc_M + M
        acc_cnt = present.astype(int) if acc_cnt is None else acc_cnt + present
        per_item.append(M.astype(np.float32)); coverage.append(int(present.sum()))
        log.append(dict(item=item + a.item_offset, text=text, sentences=sent_log, n_tokens=len(toks),
                        n_attributed_tokens=int((tok_seg >= 0).sum()), segments_described=present.tolist()))
        if item % 10 == 9:
            print(f"[{a.model}] {item+1} items, mean segments described {np.mean(coverage):.2f}", flush=True)

    # average over items per described row (rows present), then mean diagonal
    Mavg = acc_M / np.maximum(acc_cnt, 1)[None, None, :, None]
    diag = np.einsum("lhqq->lhq", Mavg)
    scores = (diag * (acc_cnt > 0)[None, None, :]).sum(-1) / max((acc_cnt > 0).sum(), 1)
    rep = concentration_report(scores)
    Savg = acc_S / np.maximum(acc_cnt, 1)[None, None, :, None]
    sdiag = np.einsum("lhqq->lhq", Savg)
    shuf_scores = (sdiag * (acc_cnt > 0)[None, None, :]).sum(-1) / max((acc_cnt > 0).sum(), 1)
    rep_s = concentration_report(shuf_scores)
    np.savez_compressed(a.out / "scores.npz", scores=scores, matrices=Mavg, per_item=np.stack(per_item),
                        row_counts=acc_cnt, shuffled_scores=shuf_scores)
    (a.out / "generations.jsonl").write_text("\n".join(json.dumps(x) for x in log))
    summary = dict(model=a.model, n_items=len(strips), prompt=PROMPT,
                   mean_segments_described=float(np.mean(coverage)),
                   segments_described_hist=np.bincount(coverage, minlength=7).tolist(),
                   row_counts=acc_cnt.tolist(),
                   top100_mean=rep["top100_mean"], median=rep["median"],
                   separation_ratio=rep["top100_mean"] / max(rep["median"], 1e-9),
                   top10_mean=rep["top10_mean"], n_heads_gt_025=int((scores > 0.25).sum()),
                   shuffled_top100_mean=rep_s["top100_mean"],
                   real_vs_shuffled_top100=rep["top100_mean"] / max(rep_s["top100_mean"], 1e-9),
                   top100_layer_hist=rep["top100_layer_hist"].tolist())
    (a.out / "report.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
