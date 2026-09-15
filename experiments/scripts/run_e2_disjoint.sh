#!/bin/bash
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until grep -q E2_MASS_DONE ~/e2_mass.log; do sleep 60; done
echo "=== mass chain done; disjoint-set control $(date +%H:%M) ==="
for m in qwen2-audio salmonn ultravox; do
  mm=$(echo $m | tr '-' '_')
  G=discovery/results/$( [ $mm = qwen2_audio ] && echo qwen2_audio_v2 || echo $mm )/scores.npz
  .venv/bin/python discovery/make_disjoint_sets.py --gaze $G --mass discovery/results/mass/$mm/scores.npz --k 100 --out discovery/results/sets/$mm 2>&1 | grep -E '"size"|"mean_gaze"|"mean_mass"|overlap|Error'
  for set in gaze_not_mass mass_not_gaze; do
    n=$(.venv/bin/python -c "import numpy as np;print(int((np.load('discovery/results/sets/$mm/$set.npz')['scores']>0).sum()))")
    echo "=== $m $set K=$n $(date +%H:%M) ==="
    .venv/bin/python discovery/run_steer.py --model $m --scores discovery/results/sets/$mm/$set.npz --condition top --k $n --n-items 20 --strips testbed/strips --out discovery/results/e2/${mm}_$set 2>&1 | grep -v "Warning\|warn\|Loading weights\|Fetching\|deprecated\|generation flags\|Setting\|WeightNorm\|meta device" | grep -E 'hook check|"accuracy"|"unattributed_rate"|Error|Traceback'
  done
done
echo E2_DISJOINT_DONE
