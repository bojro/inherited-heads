#!/bin/bash
# Phase 2 of the overnight queue: the priority items that the
# confirmatory suite does not cover. Runs on held-out strips (offset 100).
# Resumable: every step skips if its report already exists.
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until grep -q CONFIRM_DONE ~/confirm.log 2>/dev/null; do sleep 300; done
echo "=== phase 2 starting $(date +%H:%M) ==="

# S7 — non-monotone tasks (P0: the §2.2 monotonicity trap)
for m in qwen2-audio salmonn ultravox; do
  mm=$(echo $m | tr '-' '_')
  G=discovery/results/$( [ $mm = qwen2_audio ] && echo qwen2_audio_v2 || echo $mm )/scores.npz
  d=discovery/results/nonmono/$mm
  [ -f $d/report.json ] || { echo "=== S7 $m non-monotone $(date +%H:%M) ==="
    .venv/bin/python discovery/run_nonmonotone.py --model $m --scores $G --k 100 \
      --n-items 20 --item-offset 100 --strips testbed/strips --out $d 2>&1 \
      | grep -E 'acc |Error|Traceback'; }
done

# R1 — E3 retargeting
for m in qwen2-audio salmonn ultravox; do
  mm=$(echo $m | tr '-' '_')
  G=discovery/results/$( [ $mm = qwen2_audio ] && echo qwen2_audio_v2 || echo $mm )/scores.npz
  d=discovery/results/e3/$mm
  [ -f $d/report.json ] || { echo "=== R1 $m retargeting $(date +%H:%M) ==="
    .venv/bin/python discovery/run_retarget.py --model $m --scores $G --k 100 \
      --n-items 20 --item-offset 100 --strips testbed/strips --out $d 2>&1 \
      | grep -E 'stay|follow|latency|Error|Traceback'; }
done

# D4 — generation-conditioned discovery (closes the comprehension loophole)
for m in qwen2-audio salmonn ultravox; do
  mm=$(echo $m | tr '-' '_')
  d=discovery/results/gencond/$mm
  [ -f $d/report.json ] || { echo "=== D4 $m generation-conditioned $(date +%H:%M) ==="
    .venv/bin/python discovery/run_gencond.py --model $m --n-items 30 --item-offset 100 \
      --strips testbed/strips --out $d 2>&1 \
      | grep -E 'separation_ratio|real_vs_shuffled|mean_segments|Error|Traceback'; }
done

# S8 — unsteered "segment k" accuracy for the arms we lack it for
for m in salmonn ultravox; do
  mm=$(echo $m | tr '-' '_')
  d=discovery/results/gen/$mm
  [ -f $d/gen_report.json ] || { echo "=== S8 $m unsteered accuracy $(date +%H:%M) ==="
    .venv/bin/python discovery/run_generate.py --model $m --n-items 20 --item-offset 100 \
      --strips testbed/strips --out $d 2>&1 | grep -E 'accuracy|unattributed|Error'; }
done
echo PHASE2_DONE
