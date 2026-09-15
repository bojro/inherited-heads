"""Alternative head ranking: total attention mass on the audio block.

The decisive alternative explanation for E2. If biasing the top-K *gaze-score*
heads steers only because those heads happen to put the most attention on audio
at all (so clamping them is really input masking), then ranking heads by total
audio mass — a quantity with no notion of *which* segment is queried — should
steer just as well. If the gaze ranking wins, the score is finding something
about tracking, not merely audio-attentiveness.

One forward pass per strip (neutral prompt, no per-segment queries). Writes a
scores.npz whose "scores" key is [layers, heads] mean audio mass, so it drops
straight into run_steer.py --scores.

    python rank_audio_mass.py --model qwen2-audio --n-items 20 --out results/mass/qwen2_audio
"""
import argparse
import json
from pathlib import Path

import numpy as np

from run_discovery import load_strips, make_adapter
from run_steer import NEUTRAL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strips", type=Path, default=Path("../testbed/strips"))
    ap.add_argument("--model", default="qwen2-audio")
    ap.add_argument("--n-items", type=int, default=20)
    ap.add_argument("--item-offset", type=int, default=0, help=">=50 selects held-out data")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--full-precision", action="store_true")
    a = ap.parse_args()
    adapter = make_adapter(a.model, load_4bit=not a.full_precision)
    strips = list(load_strips(a.strips, a.n_items, a.item_offset))
    if hasattr(adapter, "precompute"):
        adapter.precompute(strips)
    acc = None
    for wav, sr, bounds in strips:
        inputs = adapter.prepare(wav, sr, NEUTRAL)
        spans = adapter.segment_token_spans(inputs, bounds)
        attn = adapter.forward_last_token_attn(inputs)          # [L, H, seq]
        lo, hi = spans[0][0], spans[-1][1]
        mass = attn[:, :, lo:hi].sum(-1)                        # [L, H]
        acc = mass if acc is None else acc + mass
    scores = acc / len(strips)
    a.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out / "scores.npz", scores=scores)
    flat = np.sort(scores.ravel())[::-1]
    rep = dict(model=a.model, n_items=len(strips), metric="total_audio_attention_mass",
               top1=float(flat[0]), top100_mean=float(flat[:100].mean()),
               median=float(np.median(flat)), min=float(flat[-1]))
    (a.out / "report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
