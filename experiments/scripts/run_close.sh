#!/bin/bash
# CLOSE-OUT — the last experiments before the research phase ends.
#   D4v : generation-conditioned scoring for Qwen2-VL. Recommendation #4 exists
#         only in audio so far. Ultravox reads 1.096 at prefill and 2.812 during
#         generation; Qwen2-VL sits at 1.240 at prefill, the same suspicious
#         range. PREDICTION, fixed here before the run: it rises substantially.
#   V20 : same model, same task, panels stacked VERTICALLY so each panel is a
#         CONTIGUOUS token run instead of 24 interleaved ones. Tests whether the
#         head set is a property of the layout or of the task — i.e. whether the
#         inheritance result is a shared MECHANISM or just a shared head list.
#         The decisive cell is CROSS-LAYOUT: heads discovered on horizontal
#         strips, used to steer vertical ones.
#   GATE: logit-difference readout for prosody. A structurally different
#         measurement from V15's free generation; if it is also at chance the
#         capability claim is safe, and if it has signal V16 becomes possible.
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until grep -q R2_DONE ~/reliability.log 2>/dev/null; do sleep 60; done
echo "=== CLOSE-OUT starting $(date +%H:%M) ==="

# ---- D4-vision -------------------------------------------------------------
d=discovery/results/vlm/gencond/qwen2_vl
[ -f $d/report.json ] || { echo "=== D4v Qwen2-VL generation-conditioned $(date +%H:%M) ==="
  .venv/bin/python discovery/run_vlm_gencond.py --model qwen2-vl --comics testbed/comics \
    --n-items 50 --item-offset 100 --out $d 2>&1 \
    | grep -E 'items,|separation_ratio|real_vs_shuffled|mean_panels|Error|Traceback'; }

# ---- V20 vertical layout ---------------------------------------------------
v=discovery/results/vlm/qwen2_vl_vertical
[ -f $v/report.json ] || { echo "=== V20 E1 vertical $(date +%H:%M) ==="
  .venv/bin/python discovery/run_vlm_discovery.py --model qwen2-vl-vertical \
    --comics testbed/comics --n-items 50 --item-offset 100 --out $v 2>&1 \
    | grep -E 'tokens per panel|separation_ratio|real_vs_shuffled|Error|Traceback'; }
if [ -f $v/report.json ]; then
  echo "=== V20 head-set overlap, vertical vs horizontal $(date +%H:%M) ==="
  .venv/bin/python - <<'PYEOF'
import numpy as np
h = np.load('discovery/results/vlm/qwen2_vl/scores.npz')['scores'].ravel()
v = np.load('discovery/results/vlm/qwen2_vl_vertical/scores.npz')['scores'].ravel()
top = lambda x,k: set(np.argsort(x)[::-1][:k].tolist())
n = h.size
for k in (25,50,100,200):
    print('  K=%-4d overlap %3d / %d   (naive chance %5.1f)' % (k, len(top(h,k)&top(v,k)), k, k*k/n))
PYEOF
  # ours-vertical, CROSS-LAYOUT (horizontal heads on vertical strips), random
  for spec in "top $v/scores.npz vtop100" \
              "top discovery/results/vlm/qwen2_vl/scores.npz crosslayout100" \
              "random $v/scores.npz vrandom100"; do
    set -- $spec
    o=discovery/results/vlm/e2/qwen2_vl_vertical_$3
    [ -f $o/steer_report.json ] || { echo "=== V20 E2 $3 $(date +%H:%M) ==="
      .venv/bin/python discovery/run_vlm_steer.py --model qwen2-vl-vertical \
        --comics testbed/comics --scores $2 --condition $1 --k 100 --n-items 20 \
        --item-offset 100 --out $o 2>&1 \
        | grep -E 'hook check|items:|"accuracy"|"unattributed"|Error|Traceback'; }
  done
fi

# ---- V16 logit-readout gate ------------------------------------------------
for axis in speaker emotion; do
  S=testbed/prosody_$axis
  for m in qwen2-audio salmonn ultravox; do
    mm=$(echo $m | tr '-' '_')
    g=discovery/results/v16/gate/${mm}_$axis
    [ -f $g/gate_report.json ] || { echo "=== V16 LOGIT GATE $m $axis $(date +%H:%M) ==="
      .venv/bin/python discovery/run_prosody_logit_gate.py --model $m --strips $S \
        --axis $axis --n-items 20 --item-offset 100 --out $g 2>&1 \
        | grep -E 'items:|"accuracy"|"chance"|Error|Traceback'; }
  done
done
echo "=== V16 gate summary $(date +%H:%M) ==="
.venv/bin/python - <<'PYEOF'
import json, glob, os
rows = sorted(glob.glob('discovery/results/v16/gate/*/gate_report.json'))
if not rows: print('  none'); raise SystemExit
print(f"  {'arm/axis':26s} {'logit acc':>10s} {'chance':>8s}   verdict")
for f in rows:
    d = json.load(open(f))
    tag = f"{d['model']}/{d['axis']}"
    v = 'SIGNAL -> V16 viable' if d['accuracy'] >= d['chance'] + 0.15 else 'at chance'
    print(f"  {tag:26s} {d['accuracy']:10.3f} {d['chance']:8.3f}   {v}")
print("\nBoth readouts at chance = the capability claim is safe on two very")
print("different measurements, and V16 is formally closed for that arm/axis.")
PYEOF
echo CLOSE_DONE
