"""Head-set files that isolate gaze score from audio-attention mass.

Naive version (gaze-top-K minus mass-top-K, vs the reverse) is confounded: the
mass ranking draws heads from every layer including the earliest, while the
gaze ranking sits in a mid-late band. A win for the gaze set could then just
mean "late layers steer, early layers don't". Two better sets:

WITHIN-GAZE SPLIT (the clean test — no layer confound by construction).
  Take the gaze top-K and split it at the median audio mass:
    gaze_lowmass  / gaze_highmass  (K/2 each, same gaze provenance, same band)
  If both steer alike, audio mass is not what makes gaze heads work.

LAYER-MATCHED MASS CONTROL.
  Rank by audio mass but only inside the gaze set's layer band, excluding gaze
  heads: mass_bandmatched. This is the fair "audio-attentive but not gaze"
  comparison for gaze_lowmass.

Each file is a scores.npz that is 1.0 on the chosen heads and 0 elsewhere, so
`run_steer.py --condition top --k <size>` selects exactly them.
"""
import argparse, json
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
gset = set(gi.tolist())
gl = gi // H
lo, hi = int(gl.min()), int(gl.max())

gm = m.ravel()[gi]
order = np.argsort(gm)                      # ascending audio mass within the gaze set
half = a.k // 2
sets = {"gaze_lowmass": gi[order[:half]].tolist(),
        "gaze_highmass": gi[order[-half:]].tolist()}

band = [i for i in range(L * H) if lo <= i // H <= hi and i not in gset]
band = sorted(band, key=lambda i: -m.ravel()[i])[:half]
sets["mass_bandmatched"] = band

a.out.mkdir(parents=True, exist_ok=True)
meta = {"k": a.k, "gaze_band": [lo, hi], "shape": [L, H],
        "all_head_median_mass": float(np.median(m))}
for name, idx in sets.items():
    s = np.zeros(L * H, dtype=np.float32); s[idx] = 1.0
    np.savez_compressed(a.out / f"{name}.npz", scores=s.reshape(L, H))
    lay = np.array(idx) // H
    meta[name] = dict(size=len(idx), mean_gaze=float(g.ravel()[idx].mean()),
                      mean_mass=float(m.ravel()[idx].mean()),
                      layer_min=int(lay.min()), layer_max=int(lay.max()),
                      layer_mean=float(lay.mean()))
(a.out / "sets.json").write_text(json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2))
