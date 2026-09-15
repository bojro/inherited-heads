"""Depth-matched mass split of the gaze set (PREREGISTRATION.md Deviations 2).

gaze_lowmass vs gaze_highmass differ in mean layer (21.5 vs 23.5 in Qwen), so
part of their steering gap may be depth, not audio mass. This builds a pair
whose layer histograms are IDENTICAL BY CONSTRUCTION: inside each layer
separately, the gaze heads of that layer are split at that layer's own median
audio mass. Every layer contributes the same COUNT to both sets, so any
remaining difference cannot be depth.

Odd counts leave a median head that belongs to neither half; it is dropped from
both sets, so the two sets stay exactly equal in size and histogram.

    python make_depth_matched.py --gaze results/<arm>/scores.npz \
        --mass results/mass/<arm>/scores.npz --k 100 --out results/sets/<arm>
"""
import argparse, json
from collections import defaultdict
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--gaze", type=Path, required=True)
ap.add_argument("--mass", type=Path, required=True)
ap.add_argument("--k", type=int, default=100)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()

g = np.load(a.gaze)["scores"]
m = np.load(a.mass)["scores"]
L, H = g.shape
gi = np.argsort(g.ravel())[::-1][: a.k]

by_layer = defaultdict(list)
for i in gi.tolist():
    by_layer[i // H].append(i)

low, high, dropped = [], [], []
for lay in sorted(by_layer):
    heads = sorted(by_layer[lay], key=lambda i: m.ravel()[i])   # ascending mass
    h = len(heads) // 2
    low.extend(heads[:h])
    high.extend(heads[len(heads) - h:])
    if len(heads) % 2:
        dropped.append(heads[h])

assert len(low) == len(high)
lo_lay = sorted(np.array(low) // H)
hi_lay = sorted(np.array(high) // H)
assert lo_lay == hi_lay, "layer histograms must match exactly"

a.out.mkdir(parents=True, exist_ok=True)
meta = {"k": a.k, "shape": [L, H], "size": len(low), "n_dropped_median": len(dropped),
        "layer_hist_shared": {int(l): lo_lay.count(l) for l in sorted(set(lo_lay))}}
for name, idx in (("gaze_lowmass_dm", low), ("gaze_highmass_dm", high)):
    s = np.zeros(L * H, dtype=np.float32); s[idx] = 1.0
    np.savez_compressed(a.out / f"{name}.npz", scores=s.reshape(L, H))
    lay = np.array(idx) // H
    meta[name] = dict(size=len(idx), mean_gaze=float(g.ravel()[idx].mean()),
                      mean_mass=float(m.ravel()[idx].mean()),
                      layer_mean=float(lay.mean()))
(a.out / "sets_depthmatched.json").write_text(json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2))
