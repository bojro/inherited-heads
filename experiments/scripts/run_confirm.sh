#!/bin/bash
# Confirmatory phase — conditions fixed by experiments/PREREGISTRATION.md (frozen 2026-08-17).
# Held-out strips ONLY: --item-offset 100. Resumable: completed runs are skipped,
# so a reboot costs at most the run in flight.
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OFF=100; N=50; OUT=discovery/results/confirm
mkdir -p $OUT
steer() { model=$1; scores=$2; cond=$3; k=$4; seed=${5:-0}; tag=$6
  d=$OUT/$(echo $model | tr '-' '_')_${tag}
  [ -f $d/steer_report.json ] && { echo "skip $tag"; return; }
  echo "=== $model $tag $(date +%H:%M) ==="
  .venv/bin/python discovery/run_steer.py --model $model --scores $scores --condition $cond \
    --k $k --seed $seed --n-items $N --item-offset $OFF --strips testbed/strips --out $d 2>&1 \
    | grep -v "Warning\|warn\|Loading weights\|Fetching\|deprecated\|generation flags\|Setting\|WeightNorm\|meta device" \
    | grep -E 'hook check|"accuracy"|"unattributed_rate"|Error|Traceback|AssertionError'; }
for m in qwen2-audio salmonn ultravox; do
  mm=$(echo $m | tr '-' '_')
  G=discovery/results/$( [ $mm = qwen2_audio ] && echo qwen2_audio_v2 || echo $mm )/scores.npz
  M=discovery/results/mass/$mm/scores.npz
  S=discovery/results/sets/$mm
  steer $m $G none 0 0 none0                      # H1 baseline
  for k in 25 50 100 250; do steer $m $G top $k 0 top$k; done     # H1, H2, H5
  for k in 50 100 250; do steer $m $G random $k 0 random$k; steer $m $G random $k 1 random${k}_s1; done  # H2, H6
  for k in 50 100; do steer $m $M top $k 0 masstop$k; done        # H3
  steer $m $G all 0 0 all0
  for st in gaze_lowmass gaze_highmass mass_bandmatched; do       # H4
    [ -f $S/$st.npz ] && steer $m $S/$st.npz top 50 0 $st; done
done
echo "=== robustness: prompts 1 and 2 on the key cells $(date +%H:%M) ==="
for m in qwen2-audio salmonn ultravox; do
  mm=$(echo $m | tr '-' '_')
  G=discovery/results/$( [ $mm = qwen2_audio ] && echo qwen2_audio_v2 || echo $mm )/scores.npz
  for p in 1 2; do
    d=$OUT/${mm}_top100_p$p
    [ -f $d/steer_report.json ] || { echo "=== $m top100 prompt$p $(date +%H:%M) ==="
      .venv/bin/python discovery/run_steer.py --model $m --scores $G --condition top --k 100 \
        --n-items $N --item-offset $OFF --strips testbed/strips --out $d 2>&1 | grep -E '"accuracy"|Error'; }
  done
done
# --- DISCLOSED ADDITION (not in the frozen condition list; see PREREGISTRATION.md
# Deviations 2026-08-17b). Depth-matched mass split: layer histogram identical by
# construction, gaze score matched, audio mass differs ~2x. Placed LAST so it can
# never delay a pre-registered cell; safe to interrupt.
echo "=== depth-matched mass split $(date +%H:%M) ==="
for m in qwen2-audio salmonn ultravox; do
  mm=$(echo $m | tr '-' '_')
  S=discovery/results/sets/$mm
  [ -f $S/sets_depthmatched.json ] || continue
  KSZ=$(.venv/bin/python -c "import json;print(json.load(open('$S/sets_depthmatched.json'))['size'])")
  for st in gaze_lowmass_dm gaze_highmass_dm; do
    steer $m $S/$st.npz top $KSZ 0 $st
  done
done
echo CONFIRM_DONE
