#!/bin/bash
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() { model=$1; scores=$2; cond=$3; k=$4; seed=${5:-0}; d=discovery/results/e2/$(echo $model | tr '-' '_')_${cond}${k}$( [ "$seed" != "0" ] && echo _s$seed )
  echo "=== $model $cond K=$k seed=$seed $(date +%H:%M) ==="
  .venv/bin/python discovery/run_steer.py --model $model --scores $scores --condition $cond --k $k --seed $seed --n-items 20 --strips testbed/strips --out $d 2>&1 | grep -v "Warning\|warn\|Loading weights\|Fetching\|deprecated\|generation flags\|Setting\|WeightNorm\|meta device" | grep -E 'hook check|"accuracy"|"unattributed_rate"|Error|Traceback|assert'; }
Q=discovery/results/qwen2_audio_v2/scores.npz; U=discovery/results/ultravox/scores.npz; S=discovery/results/salmonn/scores.npz
run qwen2-audio $Q none 0
for k in 100 250 50 500 25 10 5; do run qwen2-audio $Q top $k; done
run qwen2-audio $Q random 100 0; run qwen2-audio $Q random 100 1; run qwen2-audio $Q random 250 0; run qwen2-audio $Q random 250 1
run qwen2-audio $Q all 0
run salmonn $S none 0
for k in 100 250 500; do run salmonn $S top $k; done
run salmonn $S random 100 0; run salmonn $S random 100 1; run salmonn $S random 250 0; run salmonn $S random 250 1
run salmonn $S all 0
run ultravox $U none 0
for k in 100 250 500; do run ultravox $U top $k; done
run ultravox $U random 100 0; run ultravox $U random 100 1; run ultravox $U random 250 0; run ultravox $U random 250 1
run ultravox $U all 0
echo E2_FULL_DONE
