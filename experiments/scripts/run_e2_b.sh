#!/bin/bash
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() { model=$1; scores=$2; cond=$3; k=$4; seed=${5:-0}; d=discovery/results/e2/$(echo $model | tr '-' '_')_${cond}${k}$( [ "$seed" != "0" ] && echo _s$seed )
  echo "=== $model $cond K=$k seed=$seed $(date +%H:%M) ==="
  .venv/bin/python discovery/run_steer.py --model $model --scores $scores --condition $cond --k $k --seed $seed --n-items 20 --strips testbed/strips --out $d 2>&1 | grep -v "Warning\|warn\|Loading weights\|Fetching\|deprecated\|generation flags\|Setting\|WeightNorm\|meta device" | grep -E '"accuracy"|"unattributed_rate"|Error|Traceback'; }
Q=discovery/results/qwen2_audio_v2/scores.npz; U=discovery/results/ultravox/scores.npz; S=discovery/results/salmonn/scores.npz
run qwen2-audio $Q none 0
run qwen2-audio $Q random 250 0
run qwen2-audio $Q random 250 1
run ultravox $U none 0
run ultravox $U top 100
run ultravox $U top 250
run ultravox $U random 250 0
run salmonn $S none 0
run salmonn $S top 100
run salmonn $S top 250
run salmonn $S random 250 0
echo E2_B_DONE
