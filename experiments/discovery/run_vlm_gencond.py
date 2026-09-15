"""D4-vision — generation-conditioned gaze score for a VLM.

The audio arms showed that WHEN you measure changes the answer: Ultravox reads
separation 1.096 at the final prompt token (flat enough to be called "no gaze
heads") and 2.812 when scored from the attention of the tokens actually
describing each segment. That is recommendation #4, and so far it exists only in
audio. Qwen2-VL sits at 1.240 at prefill — the same suspicious range — so this
is a prediction, registered before the run: scoring during generation should
lift it substantially.

Same operationalisation as run_gencond.py, with panels for segments and the
dataset's own captions as the judge's ground truth: ask for all six panels in
order, attribute each generated sentence to a panel, and build the (described x
attended) matrix from those tokens' own attention.

    python run_vlm_gencond.py --model qwen2-vl --n-items 50 --item-offset 100 \
        --out results/vlm/gencond/qwen2_vl
"""
import argparse, json, re
from pathlib import Path

import numpy as np

from core import concentration_report
from judge import attribute
from vlm_regions import region_masses
from run_vlm_discovery import load_comics, make_adapter
from run_gencond import sentences_with_spans

PROMPT = ("This is a comic strip of six panels, in order. Describe what happens "
          "in each of the six panels, one sentence per panel, in order.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2-vl")
    ap.add_argument("--comics", type=Path, default=Path("testbed/comics"))
    ap.add_argument("--n-items", type=int, default=50)
    ap.add_argument("--item-offset", type=int, default=100)
    ap.add_argument("--max-new", type=int, default=220)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--full-precision", action="store_true")
    a = ap.parse_args()

    import pandas as pd
    meta = pd.read_parquet(a.comics / "metadata.parquet")
    adapter = make_adapter(a.model, load_4bit=not a.full_precision)
    strips = list(load_comics(a.comics, a.n_items, a.item_offset))
    a.out.mkdir(parents=True, exist_ok=True)

    acc_M = acc_S = acc_cnt = None
    rng = np.random.default_rng(0)
    per_item, coverage, log = [], [], []
    for item, (imgs, _sr, _p) in enumerate(strips):
        iid = item + a.item_offset
        caps = meta[meta.item_id == iid].sort_values("panel_idx").caption.tolist()
        inputs = adapter.prepare(imgs, None, PROMPT)
        img_start, region_ids, _pos = adapter.panel_regions(inputs)
        toks, masses = [], []
        for tok, attn in adapter.stream_generate(inputs, max_new=a.max_new):
            toks.append(tok)
            masses.append(region_masses(attn, img_start, region_ids, 6))   # [L,H,6]
        if not toks:
            coverage.append(0); continue
        pieces = adapter.token_pieces(toks)
        text = "".join(pieces)
        offs, c = [], 0
        for p in pieces:
            offs.append(c); c += len(p)
        tok_seg = np.full(len(toks), -1)
        sent_log = []
        for (s0, s1, sent) in sentences_with_spans(text):
            j, _ = attribute(sent, caps)
            sent_log.append((j, sent))
            for ti, o in enumerate(offs):
                if s0 <= o < s1:
                    tok_seg[ti] = j

        def build(labels):
            M = np.zeros(masses[0].shape[:2] + (6, 6), dtype=np.float64)
            present = np.zeros(6, dtype=bool)
            for d in range(6):
                sel = np.where(labels == d)[0]
                if len(sel) == 0:
                    continue
                present[d] = True
                m = np.stack([masses[i] for i in sel]).mean(0)
                denom = m.sum(-1, keepdims=True)
                M[:, :, d, :] = np.divide(m, denom, out=np.zeros_like(m), where=denom > 0)
            return M, present

        M, present = build(tok_seg)
        ids = sorted(set(tok_seg[tok_seg >= 0].tolist()))
        perm = dict(zip(ids, rng.permutation(ids)))
        S, _ = build(np.array([perm.get(x, -1) for x in tok_seg]))
        acc_M = M if acc_M is None else acc_M + M
        acc_S = S if acc_S is None else acc_S + S
        acc_cnt = present.astype(int) if acc_cnt is None else acc_cnt + present
        per_item.append(M.astype(np.float32)); coverage.append(int(present.sum()))
        log.append(dict(item=iid, text=text, sentences=sent_log, n_tokens=len(toks),
                        n_attributed_tokens=int((tok_seg >= 0).sum()),
                        panels_described=present.tolist()))
        if item % 10 == 9:
            print(f"[{a.model}] {item+1} items, mean panels described "
                  f"{np.mean(coverage):.2f}", flush=True)

    Mavg = acc_M / np.maximum(acc_cnt, 1)[None, None, :, None]
    scores = (np.einsum("lhqq->lhq", Mavg) * (acc_cnt > 0)[None, None, :]).sum(-1) / max((acc_cnt > 0).sum(), 1)
    Savg = acc_S / np.maximum(acc_cnt, 1)[None, None, :, None]
    shuf = (np.einsum("lhqq->lhq", Savg) * (acc_cnt > 0)[None, None, :]).sum(-1) / max((acc_cnt > 0).sum(), 1)
    rep, rep_s = concentration_report(scores), concentration_report(shuf)
    np.savez_compressed(a.out / "scores.npz", scores=scores, matrices=Mavg,
                        per_item=np.stack(per_item), row_counts=acc_cnt, shuffled_scores=shuf)
    (a.out / "generations.jsonl").write_text("\n".join(json.dumps(x) for x in log))
    summary = dict(model=a.model, modality="image", n_items=len(strips),
                   item_offset=a.item_offset, prompt=PROMPT,
                   mean_panels_described=float(np.mean(coverage)),
                   panels_described_hist=np.bincount(coverage, minlength=7).tolist(),
                   row_counts=acc_cnt.tolist(),
                   top100_mean=rep["top100_mean"], median=rep["median"],
                   separation_ratio=rep["top100_mean"] / max(rep["median"], 1e-9),
                   n_heads_gt_025=int((scores > 0.25).sum()),
                   shuffled_top100_mean=rep_s["top100_mean"],
                   real_vs_shuffled_top100=rep["top100_mean"] / max(rep_s["top100_mean"], 1e-9),
                   top100_layer_hist=rep["top100_layer_hist"].tolist())
    (a.out / "report.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
