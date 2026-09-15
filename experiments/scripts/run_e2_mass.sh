#!/bin/bash
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until grep -q E2_KSWEEP_DONE ~/e2_ksweep.log; do sleep 60; done
echo "=== ksweep done, starting audio-mass control $(date +%H:%M) ==="
for m in qwen2-audio salmonn ultravox; do
  mm=$(echo $m | tr '-' '_')
  echo "--- rank $m by audio mass"
  .venv/bin/python discovery/rank_audio_mass.py --model $m --n-items 20 --strips testbed/strips --out discovery/results/mass/$mm 2>&1 | grep -E '"top1"|"top100_mean"|"median"|Error|Traceback'
  for k in 100 250; do
    echo "=== $m MASS-top K=$k $(date +%H:%M) ==="
    .venv/bin/python discovery/run_steer.py --model $m --scores discovery/results/mass/$mm/scores.npz --condition top --k $k --n-items 20 --strips testbed/strips --out discovery/results/e2/${mm}_masstop$k 2>&1 | grep -v "Warning\|warn\|Loading weights\|Fetching\|deprecated\|generation flags\|Setting\|WeightNorm\|meta device" | grep -E 'hook check|"accuracy"|"unattributed_rate"|Error|Traceback'
  done
done
echo E2_MASS_DONE
