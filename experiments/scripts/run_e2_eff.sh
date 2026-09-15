#!/bin/bash
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until grep -q E2_DISJOINT_DONE ~/e2_disjoint.log 2>/dev/null; do sleep 60; done
echo "=== disjoint done; mass efficiency curve $(date +%H:%M) ==="
# where do the gaze and mass rankings separate? gaze has 5/10/25/50/100/250/500
for m in qwen2-audio salmonn ultravox; do
  mm=$(echo $m | tr '-' '_')
  for k in 25 50; do
    d=discovery/results/e2/${mm}_masstop$k
    [ -f $d/steer_report.json ] && continue
    echo "=== $m MASS-top K=$k $(date +%H:%M) ==="
    .venv/bin/python discovery/run_steer.py --model $m --scores discovery/results/mass/$mm/scores.npz --condition top --k $k --n-items 20 --strips testbed/strips --out $d 2>&1 | grep -v "Warning\|warn\|Loading weights\|Fetching\|deprecated\|generation flags\|Setting\|WeightNorm\|meta device" | grep -E '"accuracy"|"unattributed_rate"|Error|Traceback'
  done
done
echo E2_EFF_DONE
