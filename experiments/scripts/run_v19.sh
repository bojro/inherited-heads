#!/bin/bash
# V19 — is the seed paper's Bunny null a REGION-MAPPING artefact?
#
# Their released code is Qwen-only (zero mentions of Bunny/LLaVA/SigLIP/Ovis/
# InternVL), and its region map is `assign_panels_to_tokens`: split the token
# grid's COLUMNS proportionally to panel widths, then tile that split down every
# row. Exact for Qwen, whose native-resolution grid has no padding.
#
# Bunny is fixed 384x384 with image_aspect_ratio=pad. Their 6.12:1 strip becomes
# a band filling 16.3% of a padded square: 120 of 729 tokens carry image. Under
# their column-band map each panel gets 108-135 tokens of which 81-85% are BLANK
# PADDING, so most of the +/-10,000 bias lands on empty canvas. Under a
# content-aware map each panel is 20 real tokens.
#
# We measure .508 (their ranking, forced-choice) with the content-aware map
# against their published .083. PREDICTION, fixed here before the run: the
# column-band map drops steering toward chance (.167). If it does, their null is
# explained by a presentation/region choice rather than by Bunny's frozen
# encoder — which is the paper's main interpretive claim. If it does NOT drop,
# this hypothesis is wrong and gets reported as refuted.
cd "$(dirname "$0")/.."
until grep -q V15_DONE ~/v15_2.log 2>/dev/null; do sleep 120; done; sleep 60
echo "=== V19 starting $(date +%H:%M) ==="
PY=.venv-bunny/bin/python
d=discovery/results/vlm/bunny_strip_cols
[ -f $d/report.json ] || { echo "=== V19 E1 bunny-strip-cols $(date +%H:%M) ==="
  $PY discovery/run_vlm_discovery.py --model bunny-strip-cols --comics testbed/comics \
    --n-items 50 --item-offset 100 --out $d --full-precision 2>&1 \
    | grep -E 'separation_ratio|real_vs_shuffled|Error|Traceback'; }
[ -f $d/report.json ] || { echo "V19 ABORT: E1 produced no report"; echo V19_DONE; exit 0; }
R=discovery/results/vlm/bunny_strip_cols_raw_ranked.npz
[ -f $R ] || $PY -c "
import numpy as np; x=np.load('$d/scores.npz')
np.savez_compressed('$R', scores=x['raw_scores']); print('built $R', x['raw_scores'].shape)"
for spec in "top $d/scores.npz top100" "top $R rawtop100" "random $d/scores.npz random100"; do
  set -- $spec
  o=discovery/results/vlm/e2/bunny_strip_cols_$3
  [ -f $o/steer_report.json ] || { echo "=== V19 E2 $3 $(date +%H:%M) ==="
    $PY discovery/run_vlm_steer.py --model bunny-strip-cols --comics testbed/comics \
      --scores $2 --condition $1 --k 100 --n-items 20 --item-offset 100 --out $o \
      --full-precision 2>&1 | grep -E 'hook check|items:|"accuracy"|"unattributed"|Error|Traceback'; }
done
echo V19_DONE
