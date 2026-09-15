"""Build a layer-HISTOGRAM-matched random head set, saved in the score-array
encoding run_steer.py expects (exactly K nonzero entries, so `--condition top
--k K` selects them).

Why this exists as a separate control from `steer.random_k_heads`: that samples
UNIFORMLY from the contiguous layer BAND the gaze set spans. This samples with
the gaze set's own per-layer COUNTS. The two are not interchangeable and on
Ultravox they disagree sharply (band-matched K=100 steers .453; histogram-
matched K=71 steers .150), so the paper must say which control a number came
from. See PREREGISTRATION 17au.
"""
import argparse, collections, json
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, required=True, help="the gaze ranking to match")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    g = np.load(a.scores)["scores"]
    L, H = g.shape
    top = np.argsort(g.ravel())[::-1][:a.k]
    gaze = {(int(i) // H, int(i) % H) for i in top}
    hist = collections.Counter(l for l, _ in gaze)

    rng = np.random.default_rng(a.seed)
    out = np.zeros((L, H), dtype=np.float32)
    picked = []
    for layer, count in sorted(hist.items()):
        pool = [h for h in range(H) if (layer, h) not in gaze]
        if len(pool) < count:
            raise RuntimeError(f"layer {layer}: need {count} heads, only {len(pool)} free")
        for h in rng.choice(pool, size=count, replace=False):
            picked.append((layer, int(h)))
    # descending ranks so `--condition top --k K` recovers exactly this set
    for rank, (l, h) in enumerate(picked):
        out[l, h] = float(len(picked) - rank)
    assert int((out > 0).sum()) == a.k == len(picked), (int((out > 0).sum()), a.k)
    assert not (set(picked) & gaze), "matched set overlaps the gaze set"

    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, scores=out)
    json.dump({"k": a.k, "seed": a.seed, "source": str(a.scores),
               "layer_histogram": {str(k): v for k, v in sorted(hist.items())}},
              open(a.out.with_suffix(".json"), "w"), indent=2)
    print(f"  wrote {a.out}  K={a.k} seed={a.seed} layers={min(hist)}-{max(hist)}")


if __name__ == "__main__":
    main()
