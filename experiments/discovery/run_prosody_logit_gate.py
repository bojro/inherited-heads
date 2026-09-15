"""V16 gate — can the model report a segment's acoustic attribute at all, when
it never has to SAY the word?

The V15 free-generation gates failed everywhere: Ultravox names no gender in 93%
of answers, SALMONN names one every time and is exactly at chance. Those are
consistent with genuine inability AND with a prompt/format mismatch, and one
prompt cannot separate them. This is a structurally different readout: prompt
the model up to the decision point and compare the LOGITS of two candidate
continuations. No generation, no judge, no lexicon.

If this is also at chance, two very different measurements agree and the
capability claim is safe. If it has signal, V16's interchange patching becomes
possible with this as its readout.

    python run_prosody_logit_gate.py --model qwen2-audio \
        --strips testbed/prosody_speaker --axis speaker --n-items 20 --out ...
"""
import argparse, json
from pathlib import Path

import numpy as np

from run_discovery import load_strips, make_adapter

PROMPTS = {
    "speaker": ("The audio contains six segments, separated by silences, each a "
                "different speaker reading the same sentence. Listening only to "
                "segment {k}, the speaker is a"),
    "emotion": ("The audio contains six segments, separated by silences, each the "
                "same sentence spoken with a different emotion. Listening only to "
                "segment {k}, the speaker sounds"),
}
CANDIDATES = {
    "speaker": {"male": " man", "female": " woman"},
    "emotion": {"neutral": " neutral", "calm": " calm", "happy": " happy",
                "sad": " sad", "angry": " angry", "fearful": " afraid",
                "disgust": " disgusted", "surprised": " surprised"},
}


def candidate_ids(tok, words):
    """First token id of each candidate continuation. Comparing first tokens is
    valid only if they are DISTINCT; asserted, because two candidates sharing a
    first token would make the comparison vacuous."""
    ids = {}
    for label, w in words.items():
        t = tok(w, add_special_tokens=False).input_ids
        if not t:
            raise SystemExit(f"candidate {w!r} tokenised to nothing")
        ids[label] = int(t[0])
    if len(set(ids.values())) != len(ids):
        raise SystemExit(f"candidate first tokens collide: {ids}")
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2-audio")
    ap.add_argument("--strips", type=Path, required=True)
    ap.add_argument("--axis", choices=["speaker", "emotion"], required=True)
    ap.add_argument("--n-items", type=int, default=20)
    ap.add_argument("--item-offset", type=int, default=100)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--full-precision", action="store_true")
    a = ap.parse_args()

    import pandas as pd
    meta = pd.read_parquet(a.strips / "metadata.parquet")
    adapter = make_adapter(a.model, load_4bit=not a.full_precision)
    strips = list(load_strips(a.strips, a.n_items, a.item_offset))
    if hasattr(adapter, "precompute"):
        adapter.precompute(strips)
    tok = getattr(adapter, "tok", None) or adapter.processor.tokenizer
    cids = candidate_ids(tok, CANDIDATES[a.axis])
    prompt = PROMPTS[a.axis]

    a.out.mkdir(parents=True, exist_ok=True)
    n = ok = 0
    rows = []
    for item, (wav, sr, _b) in enumerate(strips):
        g = meta[meta.item_id == item + a.item_offset].sort_values("seg_idx")
        truth = g.gender.tolist() if a.axis == "speaker" else g.emotion.tolist()
        present = sorted(set(truth))            # emotion: only the 6 in THIS strip
        keys = [k for k in cids if k in present]
        for k in range(6):
            lg = adapter.next_token_logits(adapter.prepare(wav, sr, prompt.format(k=k + 1)))
            sub = {lab: float(lg[cids[lab]]) for lab in keys}
            pred = max(sub, key=sub.get)
            n += 1; ok += int(pred == truth[k])
            rows.append(dict(item=item + a.item_offset, seg=k, truth=truth[k],
                             pred=pred, logits={l: round(v, 3) for l, v in sub.items()}))
        print(f"  {item+1} items: acc {ok/n:.3f} (chance {1/len(keys):.3f})", flush=True)

    chance = 1 / len(keys)
    rep = dict(model=a.model, axis=a.axis, readout="logit-difference",
               strips=str(a.strips), n_items=len(strips), n_trials=n,
               chance=chance, accuracy=ok / n,
               candidates=CANDIDATES[a.axis])
    (a.out / "gate_report.json").write_text(json.dumps(rep, indent=2))
    (a.out / "trials.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
