#!/bin/bash
# Two experiments an area-chair review named as required for main-track acceptance.
#
# AC1 - CAUSAL CROSS-TASK TEST. The inheritance claim currently rests on set
# overlap, which is correlational: "62-74 of 100 overlap" is consistent with both
# rankings loading on the same generic high-norm heads. This asks whether the
# SHARED heads are the ones that do the work. Ultravox's audio top-100 splits
# into 71 shared with its text backbone's top-100 and 29 audio-only. Steer each
# subset, against random controls drawn from the SAME LAYER HISTOGRAM so a
# difference cannot be a depth artefact. Prediction, fixed here before the run:
# shared-71 approaches the full top-100 (.990) while audio-only-29 does not
# exceed its matched control. If audio-only-29 steers just as well, the shared
# set is not privileged and the inheritance claim stays correlational -- that
# outcome gets reported.
#
# AC2 - DOES MEASUREMENT POINT EXPLAIN THE BUNNY GAP? Their score reads the
# FINAL PROMPT TOKEN. We already show that point understates: Qwen2-VL reads
# 1.240 at prefill and 2.631 during generation. If Bunny's structure likewise
# only appears during generation, then head selection at prefill picks the wrong
# heads and their 8.3% is explained by measurement point rather than by anything
# about Bunny. This is the obvious open question and we do not yet have an answer.
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== AC batch starting $(date +%H:%M) ==="

# ---- AC1: which subset carries the steering? -------------------------------
G=discovery/results/sets
for spec in "shared71 71" "audioonly29 29" "randmatched71 71" "randmatched29 29"; do
  set -- $spec
  o=discovery/results/crosstask/ultravox_$1
  [ -f $o/steer_report.json ] || { echo "=== AC1 ultravox $1 (K=$2) $(date +%H:%M) ==="
    .venv/bin/python discovery/run_steer.py --model ultravox --strips testbed/strips \
      --scores $G/ultravox_$1.npz --condition top --k $2 --n-items 20 --item-offset 100 \
      --out $o 2>&1 | grep -E 'hook check|items:|"accuracy"|"unattributed"|Error|Traceback'; }
done
echo "=== AC1 summary $(date +%H:%M) ==="
.venv/bin/python - <<'PYEOF'
import json, os
base = "discovery/results"
ref = json.load(open(f"{base}/confirm/ultravox_top100/steer_report.json"))["accuracy"]
print(f"  {'set':16s} {'K':>4s} {'accuracy':>9s}")
print(f"  {'full top-100':16s} {100:>4d} {ref:9.3f}   (reference)")
for name, k in (("shared71",71), ("randmatched71",71), ("audioonly29",29), ("randmatched29",29)):
    p = f"{base}/crosstask/ultravox_{name}/steer_report.json"
    if os.path.exists(p):
        print(f"  {name:16s} {k:>4d} {json.load(open(p))['accuracy']:9.3f}")
    else:
        print(f"  {name:16s} {k:>4d}   MISSING")
PYEOF

# ---- AC2: Bunny scored during generation ------------------------------------
d=discovery/results/vlm/gencond/bunny_grid
[ -f $d/report.json ] || { echo "=== AC2 Bunny generation-conditioned $(date +%H:%M) ==="
  .venv-bunny/bin/python discovery/run_vlm_gencond.py --model bunny-grid \
    --comics testbed/comics --n-items 50 --item-offset 100 --out $d --full-precision 2>&1 \
    | grep -E 'items,|separation_ratio|real_vs_shuffled|mean_panels|Error|Traceback'; }
if [ -f $d/report.json ]; then
  echo "=== AC2: does the ranking change between prefill and generation? ==="
  .venv/bin/python - <<'PYEOF'
import json, numpy as np
pre = json.load(open('discovery/results/vlm/bunny_grid/report.json'))
gen = json.load(open('discovery/results/vlm/gencond/bunny_grid/report.json'))
print(f"  separation      prefill {pre['separation_ratio']:.3f} -> generation {gen['separation_ratio']:.3f}")
print(f"  real vs shuffled prefill {pre['real_vs_shuffled_top100']:.3f} -> generation {gen['real_vs_shuffled_top100']:.3f}")
a = np.load('discovery/results/vlm/bunny_grid/scores.npz')['scores'].ravel()
b = np.load('discovery/results/vlm/gencond/bunny_grid/scores.npz')['scores'].ravel()
top = lambda x,k=100: set(np.argsort(x)[::-1][:k].tolist())
print(f"  top-100 overlap between the two measurement points: {len(top(a)&top(b))}/100")
print("  A LOW overlap means head selection at prefill picks different heads than")
print("  selection during generation -- which would make measurement point a live")
print("  explanation for their 8.3%. A HIGH overlap rules that explanation out.")
PYEOF
fi
echo AC_DONE
