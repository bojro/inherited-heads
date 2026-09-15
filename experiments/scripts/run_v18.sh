#!/bin/bash
# V18 — Bunny's IMAGE gaze heads vs Bunny's OWN LM text gaze heads.
# Second vision point for the inheritance claim, which is currently 3 audio to 1
# vision. H12 taught us to use the ACTUAL backbone rather than a lookalike, so
# this runs the text strips through Bunny's own fine-tuned Phi-2 with no image
# at all -- not stock microsoft/phi-2. Runs in .venv-bunny for the same reason
# every other Bunny run does (17ad).
cd "$(dirname "$0")/.."
:
echo "=== V18 starting $(date +%H:%M) ==="
PY=.venv-bunny/bin/python
d=discovery/results/text/bunny_phi2
[ -f $d/report.json ] || { echo "=== V18 Bunny-LM text discovery $(date +%H:%M) ==="
  $PY discovery/run_text_discovery.py --model BAAI/Bunny-v1_0-3B \
    --text testbed/text_strips --n-items 50 --item-offset 100 --out $d --full-precision --trust-remote-code 2>&1 \
    | grep -E 'items$|separation_ratio|real_vs_shuffled|n_heads_gt|Error|Traceback'; }
[ -f $d/report.json ] || { echo "V18 ABORT: text discovery produced no report"; echo V18_DONE; exit 0; }
echo "=== V18 overlap $(date +%H:%M) ==="
$PY - <<'PYEOF'
import numpy as np
from math import comb
txt = np.load('discovery/results/text/bunny_phi2/scores.npz')['scores']
for arm in ('bunny_grid', 'bunny_strip'):
    img = np.load(f'discovery/results/vlm/{arm}/scores.npz')['scores']
    assert img.shape == txt.shape, (arm, img.shape, txt.shape)
    L, H = img.shape; n = img.size
    top = lambda x, k: set(np.argsort(x.ravel())[::-1][:k].tolist())
    ov = len(top(img, 100) & top(txt, 100)); ch = 100*100/n
    p = sum(comb(100,i)*comb(n-100,100-i) for i in range(ov,101))/comb(n,100)
    # layer-matched null: the naive baseline is wrong when both rankings prefer
    # the same depth band (PREREGISTRATION 17al)
    rng = np.random.default_rng(0)
    a_top = np.argsort(img.ravel())[::-1][:100]; t_top = top(txt, 100)
    hist = np.bincount(a_top // H, minlength=L)
    null = []
    for _ in range(5000):
        pick = []
        for layer, cnt in enumerate(hist):
            if cnt: pick.extend(layer*H + rng.choice(H, size=cnt, replace=False))
        null.append(len(set(pick) & t_top))
    null = np.array(null)
    print(f'V18 {arm}: image vs Bunny-LM text  overlap {ov}/100 | naive chance {ch:.1f} | '
          f'layer-matched null {null.mean():.1f} (95th {np.percentile(null,95):.0f}) | '
          f'hypergeom p={p:.3g} | p_layer={(null>=ov).mean():.4f}')
PYEOF
echo V18_DONE
