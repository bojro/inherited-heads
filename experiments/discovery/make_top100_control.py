"""Build the control the sufficiency result was missing: a random draw taken
from INSIDE the discovered top-100.

Every control run so far (`*_randmatched*`) is drawn from outside the top-K with
the subset's per-layer counts. That controls DEPTH. It does not control for
being in the top-100 at all, so "the shared subset carries the effect" is
currently confounded with "the top-100 carries the effect". This draws from
within the top-100 so the only thing that varies is how much of the draw is
shared with the text backbone.

Two cells per arm, and the second is the one with power:

  A. size = |shared|. This is the literal missing cell, but it is structurally
     near-saturated: on Ultravox any 71 of 100 must contain at least 42 of the
     71 shared heads, so a null here is weak evidence. Run and reported anyway,
     because it is the natural control for this question.
  B. size = |only|. Compared against the ALREADY-RUN audio-only subset of the
     same size, which contains zero shared heads. Both sets sit inside the
     top-100; they differ only in shared fraction. This is the powered contrast.

Expected shared count of each draw is printed and stored, so the result is read
against the mixture actually drawn rather than an idealised one.

    python make_top100_control.py --e1 results/ultravox/scores.npz \
        --shared results/sets/ultravox_shared71.npz \
        --out results/sets/t100 --tag ultravox --seeds 0 1 2
"""
import argparse, json
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--e1", type=Path, required=True, help="E1 scores.npz the top-100 comes from")
ap.add_argument("--shared", type=Path, required=True, help="the shared-subset mask npz")
ap.add_argument("--k", type=int, default=100)
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--tag", required=True)
a = ap.parse_args()

g = np.load(a.e1)["scores"]
L, H = g.shape
top = np.argsort(g.ravel())[::-1][: a.k]
shared = set(np.flatnonzero(np.load(a.shared)["scores"].ravel() > 0).tolist())
assert shared <= set(top.tolist()), "shared set is not inside the top-K it was split from"
n_shared, n_only = len(shared), a.k - len(shared)

a.out.mkdir(parents=True, exist_ok=True)
meta = {"e1": str(a.e1), "shared": str(a.shared), "k": a.k, "shape": [L, H],
        "n_shared": n_shared, "n_only": n_only, "cells": {}}

for size, cell in ((n_shared, "A"), (n_only, "B")):
    # hypergeometric floor: the fewest shared heads any draw of this size can contain
    floor = max(0, size - n_only)
    meta["cells"][cell] = {"size": size, "expected_shared": size * n_shared / a.k,
                           "min_possible_shared": floor, "draws": {}}
    for sd in a.seeds:
        idx = sorted(np.random.default_rng(1000 + sd).choice(top, size, replace=False).tolist())
        arr = np.zeros(L * H, dtype=np.float32)
        arr[idx] = 1.0
        name = f"{a.tag}_t100rand{size}_s{sd}"
        np.savez(a.out / f"{name}.npz", scores=arr.reshape(L, H))
        got = len(set(idx) & shared)
        lays = [i // H for i in idx]
        meta["cells"][cell]["draws"][name] = {
            "seed": sd, "size": size, "shared_in_draw": got,
            "shared_frac": got / size, "layer_mean": float(np.mean(lays))}
        print(f"  {name:34s} n={size:3d}  shared {got:3d}/{size} "
              f"({got/size:.0%}, expected {size*n_shared/a.k:.1f})  mean layer {np.mean(lays):.1f}")

(a.out / f"{a.tag}_t100_control.json").write_text(json.dumps(meta, indent=2))
