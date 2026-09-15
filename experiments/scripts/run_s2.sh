#!/bin/bash
# S2 -- the control the sufficiency result was missing.
#
# Every existing control (*_randmatched*) is drawn from OUTSIDE the top-100 with
# the subset's per-layer counts, so it separates DEPTH. It does not separate
# "shared with the text backbone" from "merely in the discovered top-100", which
# is the objection the sufficiency claim actually has to answer.
#
# Two cells per arm. Cell B carries the power and runs first within each model.
#   B  size = |audio-only|, against the ALREADY-RUN audio-only set of the same
#      size. Both sit inside the top-100; they differ only in shared fraction
#      (0% vs ~70%). A gap here is sharedness, not depth and not top-100
#      membership.
#   A  size = |shared|, the cell matched on size. Structurally near
#      saturated -- on Ultravox any 71 of 100 must contain at least 42 shared
#      heads -- so a null here is weak evidence and is reported as such.
#
# Three seeds per cell, because a single seed of a random control has already
# been shown on this project to span .147-.663 and is not a usable number.
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
G=discovery/results/sets/t100
echo "=== S2 starting $(date +%H:%M:%S) ==="

run () {  # run <model> <set-name> <k>
  local o=discovery/results/t100/$2
  if [ -f $o/steer_report.json ]; then echo "  skip $2 (done)"; return; fi
  echo "=== $2 (K=$3) start $(date +%H:%M:%S) ==="
  .venv/bin/python discovery/run_steer.py --model $1 --strips testbed/strips \
    --scores $G/$2.npz --condition top --k $3 --n-items 20 --item-offset 100 \
    --out $o 2>&1 | grep -E 'hook check|items:|"accuracy"|"unattributed"|Error|Traceback'
  echo "=== $2 end $(date +%H:%M:%S) ==="
}

for sd in 0 1 2; do run ultravox    ultravox_t100rand29_s$sd 29; done
for sd in 0 1 2; do run ultravox    ultravox_t100rand71_s$sd 71; done
for sd in 0 1 2; do run qwen2-audio qwen2_audio_t100rand34_s$sd 34; done
for sd in 0 1 2; do run qwen2-audio qwen2_audio_t100rand66_s$sd 66; done
for sd in 0 1 2; do run salmonn     salmonn_t100rand26_s$sd 26; done
for sd in 0 1 2; do run salmonn     salmonn_t100rand74_s$sd 74; done

echo "=== S2 summary $(date +%H:%M:%S) ==="
.venv/bin/python - <<'PYEOF'
import json, os
R = "discovery/results"
def acc(p):
    p = os.path.join(R, p, "steer_report.json")
    return json.load(open(p))["accuracy"] if os.path.exists(p) else None
def f(v): return "  MISSING" if v is None else f"{v:9.3f}"
for tag, sh, on, only_dir in (("ultravox", 71, 29, "crosstask/ultravox_audioonly29"),
                              ("qwen2_audio", 66, 34, "crosstask/qwen2_audio_only34"),
                              ("salmonn", 74, 26, "crosstask/salmonn_only26")):
    print(f"\n--- {tag} ---   full top-100 {f(acc(f'confirm/{tag}_top100'))}")
    for cell, n, ref, reflab in (("B", on, only_dir, f"audio-only {on} (0% shared)"),
                                 ("A", sh, f"crosstask/{tag}_shared{sh}", f"shared {sh} (100% shared)")):
        rs = [acc(f"t100/{tag}_t100rand{n}_s{s}") for s in range(3)]
        got = [r for r in rs if r is not None]
        m = f"{sum(got)/len(got):9.3f}" if got else "  MISSING"
        rng = f"{min(got):.3f}-{max(got):.3f}" if len(got) > 1 else "-"
        print(f"  cell {cell}  random {n} from top-100  mean {m}  over {len(got)} seeds  range {rng}")
        print(f"           vs {reflab:26s} {f(acc(ref))}")
PYEOF
echo S2_DONE
