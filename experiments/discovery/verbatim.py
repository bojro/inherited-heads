"""How much of a steered answer is COPIED from the segment transcript?

The monotonicity trap: if steering only ever produces a
transcription of the targeted span, the result is consistent with rediscovering
ASR alignment heads rather than finding an attentional lever. The judge cannot
see this — it scores WHICH segment an answer matches, so verbatim replay and
genuine paraphrase score identically.

Two measures per answer, against the transcript of the segment it was judged to
be about:
  copy_rate     fraction of the answer's content words that also occur there
                (bag of words; catches copying that has been reordered)
  frac_median   longest run of consecutive content words shared with the
                transcript, AS A FRACTION of that segment's length. Segments
                average ~10 content words, so absolute run length is
                uninterpretable: a run of 8 is near-total replay.
  replay_rate   share of answers whose longest run covers >=70% of the segment
  paraphrase_rate  share covering <=30%

    python verbatim.py results/e2/qwen2_audio_top100 [more dirs...]
"""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from judge import _tokens


def longest_common_run(a, b):
    """Longest run of consecutive tokens of `a` appearing consecutively in `b`."""
    if not a or not b:
        return 0
    best = 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            if x == y:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


def analyse(d, meta):
    d = Path(d)
    f = d / "answers.jsonl"
    if not f.exists():
        return None
    rows = [json.loads(l) for l in open(f)]
    copy, runs = [], []
    for r in rows:
        seg = r["judged"]
        if seg is None or seg < 0:          # unattributed
            continue
        g = meta[(meta.item_id == r["item"]) & (meta.seg_idx == seg)]
        if g.empty:
            continue
        tr = _tokens(g.iloc[0].transcript)
        ans = _tokens(r["answer"])
        if not ans:
            continue
        copy.append(len(set(ans) & set(tr)) / len(set(ans)))
        # normalise by the SEGMENT length: segments run ~10 content words, so a
        # run of 8 is near-total replay, not "a phrase". Absolute run length is
        # uninterpretable on its own.
        runs.append(longest_common_run(ans, tr) / max(len(tr), 1))
    if not runs:
        return None
    runs = np.array(runs)
    return dict(n=len(runs), copy_rate=float(np.mean(copy)),
                frac_median=float(np.median(runs)), frac_p90=float(np.quantile(runs, .9)),
                replay_rate=float((runs >= 0.7).mean()),
                paraphrase_rate=float((runs <= 0.3).mean()))


if __name__ == "__main__":
    meta = pd.read_parquet(Path(__file__).parent.parent / "testbed/strips/metadata.parquet")
    print(f"{'run':38} {'n':>4} {'copy':>6} {'frac_med':>9} {'replay':>7} {'paraphr':>8}")
    for d in sys.argv[1:]:
        r = analyse(d, meta)
        if r is None:
            print(f"{Path(d).name:38}   (no usable answers)")
            continue
        print(f"{Path(d).name:38} {r['n']:4d} {r['copy_rate']:6.3f} "
              f"{r['frac_median']:9.2f} {r['replay_rate']:7.2f} {r['paraphrase_rate']:8.2f}")
