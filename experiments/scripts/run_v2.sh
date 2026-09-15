#!/bin/bash
# V2 — the seed paper's OWN modality: their data, their models.
#   Qwen2-VL-7B   CONTROL, their Table 2: 66.2%. If our pipeline cannot find a
#                 working head set here, any null on Bunny is uninterpretable.
#   Bunny-3B      TARGET, their Table 2: 8.3% — their strongest null.
#                 Two presentations per H13 (grid ~72 tok/panel, strip ~20).
#                 BOTH are reported whichever way they fall.
# Gated on FOLLOWUPS_DONE: one 8GB card, everything serial.
cd "$(dirname "$0")/.."
# NO expandable_segments here. It is set in every other chain and is fine for
# the audio arms, but with Bunny it makes the vendored Phi die with
# "CUDA error: device-side assert triggered" partway through a run --
# deterministically, verified in both directions 2026-08-17 18:45 (with it:
# assert inside layer_norm; without it: 50/50 items clean). That is what
# killed both Bunny arms at 18:22 and 18:26. See PREREGISTRATION 17ac.
until grep -q FOLLOWUPS_DONE ~/followups.log 2>/dev/null; do sleep 300; done
grep -q DL_VLM_DONE ~/dl_vlm.log 2>/dev/null || { echo "V2 ABORT: VLM weights missing"; echo V2_DONE; exit 0; }
echo "=== V2 starting $(date +%H:%M) ==="

# ---- self-test on the control: a HARD GATE for the control arm only ----
echo "=== V2 self-test (qwen2-vl) $(date +%H:%M) ==="
.venv/bin/python discovery/run_vlm_discovery.py --self-test --model qwen2-vl \
  --comics testbed/comics --item-offset 100 2>&1 | tee /tmp/v2_selftest.log \
  | grep -E 'loader|spans|panel|self-test|PASSED|Error|Traceback|Assertion'
SELFTEST=fail
grep -q "SELF-TEST PASSED" /tmp/v2_selftest.log && SELFTEST=pass
echo "=== self-test: $SELFTEST $(date +%H:%M) ==="

run_arm () {                       # $1 = adapter name, $2 = results subdir
  local arm=$1 aa=$2
  # Bunny's vendored Phi is transformers-4.3x code. Under transformers 5 it can
  # be made to RUN (six shims) but it COMPUTES WRONG: "The capital of France is"
  # -> " a", captions "_,_,_,_". Under 4.37 the same adapter gives "The capital
  # of France is Paris." and a correct caption. So the Bunny arms run in
  # .venv-bunny (transformers 4.37, torch shared with the main venv) and every
  # Bunny number produced on transformers 5 is void. See PREREGISTRATION 17ad.
  local PY=.venv/bin/python
  case $arm in bunny-*) PY=.venv-bunny/bin/python ;; esac
  local d=discovery/results/vlm/$aa
  [ -f $d/report.json ] || { echo "=== V2 E1 $arm $(date +%H:%M) ==="
    $PY discovery/run_vlm_discovery.py --model $arm --comics testbed/comics \
      --n-items 50 --item-offset 100 --out $d 2>&1 \
      | grep -E 'separation_ratio|real_vs_shuffled|top100_mean|Error|Traceback'; }
  [ -f $d/report.json ] || { echo "V2 $arm: E1 produced no report — skipping its steering"; return; }

  local R=discovery/results/vlm/${aa}_raw_ranked.npz
  [ -f $R ] || $PY -c "
import numpy as np; x=np.load('$d/scores.npz')
np.savez_compressed('$R', scores=x['raw_scores']); print('built $R', x['raw_scores'].shape)"

  # ours vs THEIRS vs random, at matched K — the comparison V2 exists for
  for spec in "top $d/scores.npz top100" "top $R rawtop100" "random $d/scores.npz random100"; do
    set -- $spec
    local o=discovery/results/vlm/e2/${aa}_$3
    [ -f $o/steer_report.json ] || { echo "=== V2 E2 $arm $3 $(date +%H:%M) ==="
      $PY discovery/run_vlm_steer.py --model $arm --comics testbed/comics \
        --scores $2 --condition $1 --k 100 --n-items 20 --item-offset 100 --out $o 2>&1 \
        | grep -E 'hook check|items:|"accuracy"|"unattributed_rate"|Error|Traceback'; }
  done
}

if [ "$SELFTEST" = pass ]; then
  run_arm qwen2-vl qwen2_vl
else
  echo "V2: skipping the Qwen2-VL control — self-test did not pass"
fi

# Bunny runs regardless of the Qwen2-VL self-test: it is a different adapter with
# its own assertions (LLaVA-style splice, 27x27 grid, pad-to-square) and its own
# steering mechanism. A control failure makes Bunny harder to INTERPRET, not
# invalid to collect — and that interpretation is recorded, not decided later.
run_arm bunny-grid  bunny_grid
run_arm bunny-strip bunny_strip
echo V2_DONE
