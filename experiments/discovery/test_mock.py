"""Harness validation without any model or GPU (CPU only).

Builds a synthetic adapter with 4 layers x 8 heads over fake 6-segment strips.
Heads (1,2) and (3,5) are planted "gaze heads": their final-token attention
concentrates on the queried segment. Everything else attends uniformly at a
comparable total mass over audio tokens. PASS iff discovery ranks exactly the
planted heads on top, they beat the shuffled baseline, and uniform heads don't.
"""

import numpy as np

from core import run_discovery, concentration_report

LAYERS, HEADS, SEQ, N_SEG = 4, 8, 600, 6
PLANTED = [(1, 2), (3, 5)]


class MockAdapter:
    tokens_per_second = 25.0

    def prepare(self, wav, sr, query_text):
        k = int(query_text.split("#")[1])
        return {"query_seg": k}

    def segment_token_spans(self, inputs, boundaries_s):
        block = SEQ - 100  # audio tokens occupy [50, 550)
        per = block // N_SEG
        return [(50 + i * per, 50 + (i + 1) * per) for i in range(N_SEG)]

    def forward_last_token_attn(self, inputs):
        k = inputs["query_seg"]
        spans = self.segment_token_spans(None, None)
        attn = np.full((LAYERS, HEADS, SEQ), 1.0 / SEQ)
        for (l, h) in PLANTED:
            attn[l, h] = 0.0005 / SEQ
            s, e = spans[k]
            attn[l, h, s:e] = 0.9 / (e - s)   # planted: locks onto queried segment
        return attn / attn.sum(axis=-1, keepdims=True)


def run():
    strips = [(None, 16000, [(i * 5.7, i * 5.7 + 5.0) for i in range(N_SEG)])] * 3
    real, shuf = run_discovery(
        MockAdapter(), strips, lambda k: f"seg#{k}", shuffle_baseline_seed=0
    )
    order = np.dstack(np.unravel_index(np.argsort(real.scores.ravel())[::-1],
                                       real.scores.shape))[0]
    top = {tuple(x) for x in order[: len(PLANTED)]}
    rep = concentration_report(real.scores)
    planted_scores = [real.scores[l, h] for l, h in PLANTED]
    uniform_median = np.median(real.scores)
    # planted heads put ~all attention on the queried segment, so after
    # row-normalization over segments their diagonal is ~1.0; uniform heads 1/6
    checks = {
        "planted heads ranked top": top == set(PLANTED),
        "planted score ~1.0": all(s > 0.95 for s in planted_scores),
        "uniform heads ~1/6": abs(uniform_median - 1 / 6) < 0.02,
        "planted/uniform separation > 4": min(planted_scores) / uniform_median > 4,
        "planted beat shuffled": real.scores[1, 2] > shuf.scores[1, 2] + 0.3,
        "concentration report runs": rep["n_heads"] == LAYERS * HEADS,
    }
    for name, ok in checks.items():
        print(("PASS" if ok else "FAIL"), "-", name)
    if not all(checks.values()):
        raise SystemExit(1)
    print("mock harness validation: ALL PASS")


if __name__ == "__main__":
    run()
