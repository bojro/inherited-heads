#!/bin/bash
# RISK-FIRST BLOCK. The two experiments that can invalidate the framing run
# BEFORE the 12-hour confirmatory suite, not after it.
#
# Rationale: the confirmatory suite measures how much the exploratory effects
# shrink on held-out items — important, but it refines a claim. S7 and R1 test
# whether the claim is the right one at all:
#   S7  non-monotone tasks — if steering only works on describe-prompts, the
#       result is consistent with rediscovering ASR alignment heads
#   R1  mid-generation retargeting — a playhead or an input-masking channel
#       cannot follow a target switch mid-decode; a high follow-rate is
#       independent evidence of a real lever
# Neither is in the frozen condition list of PREREGISTRATION.md, so running them
# earlier changes no pre-registered analysis. Head rankings are already frozen.
#
# Idempotent: run_overnight2.sh skips any step whose report.json exists, so it
# will simply not repeat these.
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until grep -q TEXT_DONE ~/text.log 2>/dev/null; do sleep 60; done
echo "=== risk-first block starting $(date +%H:%M) ==="

# S7 — non-monotone tasks (P0: the 2.2 monotonicity trap)
for m in qwen2-audio salmonn ultravox; do
  mm=$(echo $m | tr '-' '_')
  G=discovery/results/$( [ $mm = qwen2_audio ] && echo qwen2_audio_v2 || echo $mm )/scores.npz
  d=discovery/results/nonmono/$mm
  [ -f $d/report.json ] || { echo "=== S7 $m non-monotone $(date +%H:%M) ==="
    .venv/bin/python discovery/run_nonmonotone.py --model $m --scores $G --k 100 \
      --n-items 20 --item-offset 100 --strips testbed/strips --out $d 2>&1 \
      | grep -E 'hook check|acc |unsteered|bias_wins|cue_wins|Error|Traceback|AssertionError'; }
done

# R1 — E3 retargeting
for m in qwen2-audio salmonn ultravox; do
  mm=$(echo $m | tr '-' '_')
  G=discovery/results/$( [ $mm = qwen2_audio ] && echo qwen2_audio_v2 || echo $mm )/scores.npz
  d=discovery/results/e3/$mm
  [ -f $d/report.json ] || { echo "=== R1 $m retargeting $(date +%H:%M) ==="
    .venv/bin/python discovery/run_retarget.py --model $m --scores $G --k 100 \
      --n-items 20 --item-offset 100 --strips testbed/strips --out $d 2>&1 \
      | grep -E 'hook check|stay|follow|latency|Error|Traceback|AssertionError'; }
done
echo RISK_DONE
