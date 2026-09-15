"""Unsteered generation + judge: for each strip and each segment query, ask
"what does the speaker of segment k talk about", decode greedily, attribute
the answer to a segment with judge.attribute, and report accuracy against k.

    python run_generate.py --model qwen2-audio --n-items 50 --out results/qwen2_audio_gen

Outputs answers.jsonl (item, k, answer, judged, scores) and gen_report.json
(accuracy, unattributed rate, confusion matrix queried x judged). Chance =
1/6. This is E2's unsteered baseline and the direct test of whether a model
understands "segment k" at all (the comprehension loophole on E1 nulls).
"""
import argparse, json
from pathlib import Path

import numpy as np

from judge import attribute
from run_discovery import QUERIES, load_strips, make_adapter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strips", type=Path, default=Path("../testbed/strips"))
    ap.add_argument("--model", default="qwen2-audio")
    ap.add_argument("--n-items", type=int, default=50)
    ap.add_argument("--item-offset", type=int, default=0, help=">=50 selects held-out data")
    ap.add_argument("--prompt", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=60)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--full-precision", action="store_true")
    a = ap.parse_args()
    import pandas as pd
    meta = pd.read_parquet(a.strips / "metadata.parquet")
    query = QUERIES[a.prompt]
    adapter = make_adapter(a.model, load_4bit=not a.full_precision)
    strips = list(load_strips(a.strips, a.n_items, a.item_offset))
    if hasattr(adapter, "precompute"):
        adapter.precompute(strips)
    a.out.mkdir(parents=True, exist_ok=True)
    conf = np.zeros((6, 7), dtype=int)   # queried x judged (col 6 = unattributed)
    rows = []
    with open(a.out / "answers.jsonl", "w") as f:
        for item, (wav, sr, bounds) in enumerate(strips):
            tr = meta[meta.item_id == item + a.item_offset].sort_values("seg_idx").transcript.tolist()
            for k in range(6):
                inputs = adapter.prepare(wav, sr, query.format(k=k + 1))
                ans = adapter.generate(inputs, max_new=a.max_new)
                j, sc = attribute(ans, tr)
                conf[k, j if j >= 0 else 6] += 1
                rows.append(dict(item=item + a.item_offset, k=k, judged=j, answer=ans, scores=[round(x, 2) for x in sc]))
                f.write(json.dumps(rows[-1]) + "\n")
            if item % 10 == 9:
                acc = np.trace(conf[:, :6]) / conf.sum()
                print(f"[{a.model}] {item+1} items: acc {acc:.3f}, unattributed {conf[:,6].sum()/conf.sum():.3f}", flush=True)
    n = conf.sum()
    rep = dict(model=a.model, n_items=len(strips), prompt=a.prompt, query=query,
               accuracy=float(np.trace(conf[:, :6]) / n),
               accuracy_among_attributed=float(np.trace(conf[:, :6]) / max(conf[:, :6].sum(), 1)),
               unattributed_rate=float(conf[:, 6].sum() / n),
               per_query_accuracy=[float(conf[k, k] / conf[k].sum()) for k in range(6)],
               confusion_queried_x_judged=conf.tolist(),
               judged_col_marginal=conf.sum(0).tolist())
    (a.out / "gen_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
