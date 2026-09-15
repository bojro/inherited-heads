"""Degeneracy metrics for steered generations (the 'destroys generation' axis).

Steering accuracy alone cannot tell a clean redirect from a broken one: a
model stuck in "the one the one the one" still repeats target-segment words
and scores as correct under the overlap judge. Reported per run:
  empty_rate    fraction of answers with no text
  distinct2     unique bigrams / total bigrams, averaged over answers (1.0 =
                no repetition; < ~0.6 is visibly degenerate)
  max_rep_run   mean longest run of one repeated token
    python fluency.py results/e2/*/answers.jsonl
"""
import json
import sys
from pathlib import Path


def metrics(answers):
    import re
    d2, runs, empty = [], [], 0
    for a in answers:
        w = re.findall(r"\w+", a.lower())
        if not w:
            empty += 1
            continue
        bg = list(zip(w, w[1:]))
        d2.append(len(set(bg)) / len(bg) if bg else 1.0)
        best = cur = 1
        for i in range(1, len(w)):
            cur = cur + 1 if w[i] == w[i - 1] else 1
            best = max(best, cur)
        runs.append(best)
    n = len(answers)
    return dict(n=n, empty_rate=empty / n if n else 0.0,
                distinct2=sum(d2) / len(d2) if d2 else 0.0,
                max_rep_run=sum(runs) / len(runs) if runs else 0.0)


if __name__ == "__main__":
    print(f"{'run':30} {'n':>4} {'empty':>6} {'distinct2':>10} {'rep_run':>8}")
    for p in sorted(sys.argv[1:]):
        ans = [json.loads(l)["answer"] for l in open(p)]
        m = metrics(ans)
        print(f"{Path(p).parent.name:30} {m['n']:4d} {m['empty_rate']:6.2f} {m['distinct2']:10.3f} {m['max_rep_run']:8.2f}")
