#!/bin/bash
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until grep -q E2_MASS_DONE ~/e2_mass.log; do sleep 60; done
echo "=== mass chain done; controlled head-set comparison $(date +%H:%M) ==="
for m in qwen2-audio salmonn ultravox; do
  mm=$(echo $m | tr '-' '_')
  G=discovery/results/$( [ $mm = qwen2_audio ] && echo qwen2_audio_v2 || echo $mm )/scores.npz
  .venv/bin/python discovery/make_disjoint_sets.py --gaze $G --mass discovery/results/mass/$mm/scores.npz --k 100 --out discovery/results/sets/$mm > discovery/results/sets/${mm}_build.log 2>&1
  for st in gaze_lowmass gaze_highmass mass_bandmatched; do
    d=discovery/results/e2/${mm}_$st
    [ -f $d/steer_report.json ] && continue
    echo "=== $m $st K=50 $(date +%H:%M) ==="
    .venv/bin/python discovery/run_steer.py --model $m --scores discovery/results/sets/$mm/$st.npz --condition top --k 50 --n-items 20 --strips testbed/strips --out $d 2>&1 | grep -v "Warning\|warn\|Loading weights\|Fetching\|deprecated\|generation flags\|Setting\|WeightNorm\|meta device" | grep -E 'hook check|"accuracy"|"unattributed_rate"|Error|Traceback'
  done
done
echo "=== mass efficiency curve at small K $(date +%H:%M) ==="
for m in qwen2-audio salmonn ultravox; do
  mm=$(echo $m | tr '-' '_')
  for k in 25 50; do
    d=discovery/results/e2/${mm}_masstop$k
    [ -f $d/steer_report.json ] && continue
    echo "=== $m MASS-top K=$k $(date +%H:%M) ==="
    .venv/bin/python discovery/run_steer.py --model $m --scores discovery/results/mass/$mm/scores.npz --condition top --k $k --n-items 20 --strips testbed/strips --out $d 2>&1 | grep -v "Warning\|warn\|Loading weights\|Fetching\|deprecated\|generation flags\|Setting\|WeightNorm\|meta device" | grep -E '"accuracy"|"unattributed_rate"|Error|Traceback'
  done
done
echo E2_SETS_DONE
