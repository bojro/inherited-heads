#!/bin/bash
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() { model=$1; scores=$2; cond=$3; k=$4; seed=${5:-0}; d=discovery/results/e2/$(echo $model | tr '-' '_')_${cond}${k}$( [ "$seed" != "0" ] && echo _s$seed )
  [ -f $d/steer_report.json ] && { echo "=== skip $model $cond K=$k seed=$seed (exists)"; return; }
  echo "=== $model $cond K=$k seed=$seed $(date +%H:%M) ==="
  .venv/bin/python discovery/run_steer.py --model $model --scores $scores --condition $cond --k $k --seed $seed --n-items 20 --strips testbed/strips --out $d 2>&1 | grep -v "Warning\|warn\|Loading weights\|Fetching\|deprecated\|generation flags\|Setting\|WeightNorm\|meta device" | grep -E '"accuracy"|"unattributed_rate"|Error|Traceback|assert'; }
U=discovery/results/ultravox/scores.npz; S=discovery/results/salmonn/scores.npz; Q=discovery/results/qwen2_audio_v2/scores.npz
# small-K sweep: how few heads suffice? (the concentration claim, causally)
for k in 5 10 25 50; do run ultravox $U top $k; done
for k in 5 10 25 50; do run salmonn  $S top $k; done
# matched random controls at the K where each arm first works well
run ultravox $U random 50 0
run salmonn  $S random 50 0
run qwen2-audio $Q random 50 0
echo E2_KSWEEP_DONE
