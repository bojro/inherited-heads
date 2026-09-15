#!/bin/bash
# S2b -- give the SMALL matched controls the same seed treatment the K=100 one got.
#
# Ledger section 19 reads each unshared subset against a single-draw control at
# K=29/34/26 and reports intervals that account for item resampling only. Section
# 21 has now measured three draws of 29 heads from Ultravox's top-100 at .250 /
# .417 / .667. A -0.267 difference quoted against ONE draw from a population that
# wide is not a measurement, and the project already learned this at K=100 (range
# .147-.663 over five seeds).
#
# Four more seeds per arm, same construction, so each control has five.
# Gated on S2_DONE so it never shares the GPU.
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until grep -q S2_DONE ~/s2.log 2>/dev/null; do sleep 60; done
echo "=== S2b starting $(date +%H:%M:%S) ==="
G=discovery/results/sets/mseeds

run () {  # run <model> <set-name> <k>
  local o=discovery/results/mseeds/$2
  if [ -f $o/steer_report.json ]; then echo "  skip $2 (done)"; return; fi
  echo "=== $2 (K=$3) start $(date +%H:%M:%S) ==="
  .venv/bin/python discovery/run_steer.py --model $1 --strips testbed/strips \
    --scores $G/$2.npz --condition top --k $3 --n-items 20 --item-offset 100 \
    --out $o 2>&1 | grep -E 'hook check|items:|"accuracy"|"unattributed"|Error|Traceback'
  echo "=== $2 end $(date +%H:%M:%S) ==="
}

for sd in 1 2 3 4; do run ultravox    ultravox_randmatched29_s$sd 29; done
for sd in 1 2 3 4; do run qwen2-audio qwen2_audio_randmatched34_s$sd 34; done
for sd in 1 2 3 4; do run salmonn     salmonn_randmatched26_s$sd 26; done
echo S2B_DONE
