#!/bin/bash
# V15 — content-matched prosody strips (PREREGISTRATION 17x, corpora built 17:55).
# Every segment of a strip speaks the IDENTICAL sentence, so segment identity is
# carried only by speaker or by emotional prosody. If targeting works here,
# something routes acoustic information; transcript content cannot help.
#
# STAGE 1 is a CAPABILITY GATE and its thresholds are fixed here, before the data:
#   emotion axis (chance .167): pass at accuracy >= .333   (2x chance)
#   speaker axis (chance .500): pass at accuracy >= .700
# A model below its threshold CANNOT DO THE TASK when the segment is named for it,
# so its steering number would measure the model's inability, not the heads. Such
# an arm is reported as CAPABILITY-BLOCKED, never as a null. Stage 2 runs only for
# arm/axis pairs that pass.
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until grep -q V19_DONE ~/v19.log 2>/dev/null; do sleep 300; done
echo "=== V15 starting $(date +%H:%M) ==="

declare -A THRESH=( [emotion]=0.333 [speaker]=0.700 )
declare -A JUDGE=(  [emotion]=emotion [speaker]=gender )
declare -A PROMPT_IDX=( [speaker]=3 [emotion]=4 )
NEUTRAL_speaker="The audio contains six segments, separated by silences. Each is a different speaker reading the same sentence. Describe the voice of one of the speakers: is it a man or a woman, and what does the voice sound like?"
NEUTRAL_emotion="The audio contains six segments, separated by silences. Each is the same sentence spoken with a different emotion. What emotion is being expressed?"

for axis in emotion speaker; do
  for m in qwen2-audio ultravox salmonn; do
    mm=$(echo $m | tr '-' '_')
    S=testbed/prosody_$axis
    G=discovery/results/v15/gate/${mm}_$axis
    [ -f $G/gate_report.json ] || { echo "=== V15 GATE $m $axis $(date +%H:%M) ==="
      .venv/bin/python discovery/run_prosody_gate.py --model $m --strips $S \
        --axis $axis --n-items 10 --item-offset 100 --out $G 2>&1 \
        | grep -E 'items:|"accuracy"|"unattributed_rate"|Error|Traceback'; }
    [ -f $G/gate_report.json ] || { echo "V15 $m $axis: gate produced no report — skipping"; continue; }

    ACC=$(.venv/bin/python -c "import json;print(json.load(open('$G/gate_report.json'))['accuracy'])")
    PASS=$(.venv/bin/python -c "print('yes' if $ACC >= ${THRESH[$axis]} else 'no')")
    echo "=== V15 gate $m $axis: acc $ACC vs threshold ${THRESH[$axis]} -> $PASS ==="
    [ "$PASS" = yes ] || { echo "V15 $m $axis: CAPABILITY-BLOCKED — not a null, do not report as one"; continue; }

    D=discovery/results/v15/e1/${mm}_$axis
    [ -f $D/report.json ] || { echo "=== V15 E1 $m $axis $(date +%H:%M) ==="
      .venv/bin/python discovery/run_discovery.py --model $m --strips $S \
        --prompt ${PROMPT_IDX[$axis]} --n-items 50 --item-offset 100 --out $D 2>&1 \
        | grep -E 'items$|separation_ratio|real_vs_shuffled|n_heads_gt|Error|Traceback'; }
    [ -f $D/report.json ] || { echo "V15 $m $axis: E1 produced no report — skipping steering"; continue; }

    eval "NEUTRAL=\$NEUTRAL_$axis"
    for cond in top random none; do
      K=100; [ $cond = none ] && K=0
      O=discovery/results/v15/e2/${mm}_${axis}_${cond}${K}
      [ -f $O/steer_report.json ] || { echo "=== V15 E2 $m $axis $cond $(date +%H:%M) ==="
        .venv/bin/python discovery/run_steer.py --model $m --strips $S \
          --scores $D/scores.npz --condition $cond --k $K --n-items 20 --item-offset 100 \
          --judge ${JUDGE[$axis]} --prompt-text "$NEUTRAL" --out $O 2>&1 \
          | grep -E 'hook check|items:|"accuracy"|"unattributed_rate"|Error|Traceback'; }
    done
  done
done
echo V15_DONE
