"""V21 — norm-weighted (||alpha*v||) variants of BOTH gaze scores.

PREREGISTRATION 17ar registers this as a threat to our own result. The Gaze
Score is post-softmax attention alpha; so is ours. Kobayashi et al. 2020 showed
that re-measuring with ||alpha * f(x)|| REVERSES the classic finding that BERT
attends to [SEP], because a head can place large alpha on positions whose value
vectors are negligible. A head could therefore spread alpha perfectly over the
queried segment while transmitting nothing from it.

Pre-committed reading, written before the run:
  * If OUR ranking is stable under norm weighting (high top-100 overlap with the
    alpha ranking) and the RAW ranking still fails, the attention-is-not-
    explanation objection is closed for this result.
  * If our ranking is NOT stable, that is reported as a limitation and the
    normalisation claim weakens to "alpha-normalisation and norm-weighting
    disagree; neither is validated as the correct instrument".
  * If the RAW ranking is RESCUED by norm weighting -- i.e. raw ||alpha*v||
    recovers heads that steer -- then the seed paper's criterion was right and
    only its measurement was wrong. That outcome is a finding AGAINST our
    headline and gets reported as such.

Rankings are computed on items 0-49 (EXPLORATORY), the same data the frozen
alpha rankings came from, so the comparison is like-for-like. Steering is run
separately on held-out items >= 100.
"""

import argparse, json
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent))
from core import gaze_scores, segment_attention_matrix
from run_discovery import QUERIES, load_strips, make_adapter
from adapters.llm_utils import two_phase_attn_and_value_norms

WEIGHTINGS = ["alpha", "alpha_vnorm", "alpha_ovnorm"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ultravox")
    ap.add_argument("--strips", type=Path, default=Path("../testbed/strips"))
    ap.add_argument("--n-items", type=int, default=50)
    ap.add_argument("--item-offset", type=int, default=0)
    ap.add_argument("--prompt", type=int, default=0, help="index into QUERIES")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--full-precision", action="store_true")
    a = ap.parse_args()

    query = QUERIES[a.prompt]
    adapter = make_adapter(a.model, load_4bit=not a.full_precision)
    a.out.mkdir(parents=True, exist_ok=True)

    acc = {f"{w}_{n}": None for w in WEIGHTINGS for n in ("norm", "raw")}
    n_items = 0
    for wav, sr, bounds in load_strips(a.strips, a.n_items, a.item_offset):
        per_query = {w: [] for w in WEIGHTINGS}
        for k in range(len(bounds)):
            inputs = adapter.prepare(wav, sr, query.format(k=k + 1))
            spans = adapter.segment_token_spans(inputs, bounds)
            with torch.no_grad():
                attn, vnorm, ovnorm = two_phase_attn_and_value_norms(
                    adapter.llm, adapter._embeds(inputs))
            for w, weighted in (("alpha", attn),
                                ("alpha_vnorm", attn * vnorm),
                                ("alpha_ovnorm", attn * ovnorm)):
                per_query[w].append(segment_attention_matrix(weighted, spans))
        for w in WEIGHTINGS:
            for nm, flag in (("norm", True), ("raw", False)):
                m, _ = gaze_scores(per_query[w], normalize=flag)
                key = f"{w}_{nm}"
                acc[key] = m if acc[key] is None else acc[key] + m
        n_items += 1
        print(f"  {n_items} items", flush=True)

    scores = {}
    for key, m in acc.items():
        m = m / n_items
        s = np.einsum("lhqq->lhq", m).mean(axis=-1)
        scores[key] = s
        np.savez(a.out / f"{key}.npz", scores=s, matrices=m, n_items=n_items)

    top = lambda s, k=100: set(np.argsort(s.ravel())[::-1][:k].tolist())
    ref_norm = top(scores["alpha_norm"])
    ref_raw = top(scores["alpha_raw"])
    report = {"model": a.model, "n_items": n_items,
              "item_offset": a.item_offset, "keys": list(scores)}
    print("\n  top-100 overlap with the FROZEN alpha rankings")
    for key, s in scores.items():
        ov_n, ov_r = len(top(s) & ref_norm), len(top(s) & ref_raw)
        report[key] = {"overlap_with_alpha_norm": ov_n, "overlap_with_alpha_raw": ov_r}
        print(f"    {key:20s}  vs ours(alpha_norm) {ov_n:3d}/100   vs theirs(alpha_raw) {ov_r:3d}/100")
    json.dump(report, open(a.out / "report.json", "w"), indent=2)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
