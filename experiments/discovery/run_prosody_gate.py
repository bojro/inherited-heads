"""V15 capability gate: can the model do the acoustic task AT ALL when told which
segment to describe?

Every V15 steering number is uninterpretable without this. If a model cannot name
the emotion of segment 3 when the prompt literally says "segment 3", then a
steering accuracy at chance says nothing about whether heads route acoustic
information — it only says the model cannot answer the question. That is a
capability floor, not a null result, and the two must not be confused.

This is deliberately the EASY version of the task: the segment is named, no
steering, no head selection. Accuracy here is the ceiling everything else is
measured against.

    python run_prosody_gate.py --model qwen2-audio --strips testbed/prosody_emotion \
        --axis emotion --n-items 10 --item-offset 100 --out results/v15/gate/...
"""
import argparse, json
from pathlib import Path

import numpy as np

from run_discovery import load_strips, make_adapter, QUERIES
from judge_prosody import attribute_emotion, attribute_gender

AXIS_QUERY = {"speaker": QUERIES[3], "emotion": QUERIES[4]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2-audio")
    ap.add_argument("--strips", type=Path, required=True)
    ap.add_argument("--axis", choices=["speaker", "emotion"], required=True)
    ap.add_argument("--n-items", type=int, default=10)
    ap.add_argument("--item-offset", type=int, default=100)
    ap.add_argument("--max-new", type=int, default=40)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--full-precision", action="store_true")
    a = ap.parse_args()

    import pandas as pd
    meta = pd.read_parquet(a.strips / "metadata.parquet")
    adapter = make_adapter(a.model, load_4bit=not a.full_precision)
    strips = list(load_strips(a.strips, a.n_items, a.item_offset))
    if hasattr(adapter, "precompute"):
        adapter.precompute(strips)
    q = AXIS_QUERY[a.axis]
    chance = 0.5 if a.axis == "speaker" else 1 / 6

    a.out.mkdir(parents=True, exist_ok=True)
    n = ok = unattr = 0
    with open(a.out / "answers.jsonl", "w") as f:
        for item, (wav, sr, _bounds) in enumerate(strips):
            g = meta[meta.item_id == item + a.item_offset].sort_values("seg_idx")
            labels, genders = g.seg_label.tolist(), g.gender.tolist()
            for k in range(6):
                ans = adapter.generate(adapter.prepare(wav, sr, q.format(k=k + 1)),
                                       max_new=a.max_new)
                if a.axis == "emotion":
                    j, _ = attribute_emotion(ans, labels)
                    correct, pred = (j == k) if j >= 0 else None, j
                else:
                    correct, pred = attribute_gender(ans, genders[k])
                n += 1
                if correct is None:
                    unattr += 1
                elif correct:
                    ok += 1
                f.write(json.dumps(dict(item=item + a.item_offset, seg=k,
                                        correct=correct, pred=pred, answer=ans)) + "\n")
            print(f"  {item+1} items: acc {ok/n:.3f} unattributed {unattr/n:.3f} "
                  f"(chance {chance:.3f})", flush=True)

    rep = dict(model=a.model, axis=a.axis, strips=str(a.strips), n_items=len(strips),
               item_offset=a.item_offset, n_trials=n, chance=chance,
               accuracy=ok / n, unattributed_rate=unattr / n,
               # The gate is about capability, so an answer the judge cannot read
               # counts against the model here rather than being excluded.
               accuracy_of_attributed=(ok / (n - unattr)) if n > unattr else None)
    (a.out / "gate_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
