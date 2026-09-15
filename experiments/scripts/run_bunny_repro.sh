#!/bin/bash
# INDEPENDENT BUNNY RE-RUN. The Bunny arm carries this paper's empirical
# correction and it was debugged and measured inside a single session, after six
# shims and a venv swap. This re-runs it from a FRESH CLONE of the committed
# repository into a FRESH results tree, so nothing uncommitted and no cached
# score can contribute. Same data and same .venv-bunny (both are documented and
# gitignored, and re-deriving them would change what is being compared); what is
# being tested is the code path, which is where the risk is.
set -u
SRC="$(cd "$(dirname "$0")/../.." && pwd)"
DST="${DST:-$HOME/bunny_repro}"
export SRC DST
rm -rf $DST
git clone -q $SRC $DST || { echo "REPRO ABORT: clone failed"; echo BUNNY_REPRO_DONE; exit 0; }
cd $DST/experiments
# venvs and the comic images are gitignored; link the originals in
ln -s $SRC/experiments/.venv .venv
ln -s $SRC/experiments/.venv-bunny .venv-bunny
ln -s $SRC/experiments/testbed/comics/images testbed/comics/images
echo "=== BUNNY REPRO from $(git -C $DST rev-parse --short HEAD) $(date +%H:%M) ==="

PY=.venv-bunny/bin/python
for arm in bunny-grid bunny-strip; do
  aa=$(echo $arm | tr '-' '_')
  d=discovery/results/vlm/$aa
  echo "=== REPRO E1 $arm $(date +%H:%M) ==="
  $PY discovery/run_vlm_discovery.py --model $arm --comics testbed/comics \
    --n-items 50 --item-offset 100 --out $d --full-precision 2>&1 \
    | grep -E 'separation_ratio|real_vs_shuffled|Error|Traceback'
  [ -f $d/report.json ] || { echo "REPRO $arm: E1 produced no report"; continue; }
  R=discovery/results/vlm/${aa}_raw_ranked.npz
  $PY -c "
import numpy as np; x=np.load('$d/scores.npz')
np.savez_compressed('$R', scores=x['raw_scores'])"
  for spec in "top $d/scores.npz top100" "top $R rawtop100" "random $d/scores.npz random100"; do
    set -- $spec
    o=discovery/results/vlm/e2/${aa}_$3
    echo "=== REPRO E2 $arm $3 $(date +%H:%M) ==="
    $PY discovery/run_vlm_steer.py --model $arm --comics testbed/comics \
      --scores $2 --condition $1 --k 100 --n-items 20 --item-offset 100 --out $o \
      --full-precision 2>&1 | grep -E 'hook check|"accuracy"|"unattributed"|Error|Traceback'
  done
done

echo "=== REPRO vs ORIGINAL $(date +%H:%M) ==="
.venv/bin/python - <<'PYEOF'
import json, os
orig = os.path.join(os.environ["SRC"], "experiments/discovery/results")
new  = os.path.join(os.environ["DST"], "experiments/discovery/results")
print(f"{'cell':34s} {'original':>9s} {'repro':>9s} {'delta':>8s}")
rows = [(f"vlm/{a}/report.json", "separation_ratio") for a in ("bunny_grid","bunny_strip")]
rows += [(f"vlm/e2/{a}_{c}/steer_report.json", "accuracy")
         for a in ("bunny_grid","bunny_strip") for c in ("top100","rawtop100","random100")]
for rel, key in rows:
    def g(root):
        p = os.path.join(root, rel)
        return json.load(open(p))[key] if os.path.exists(p) else None
    o, n = g(orig), g(new)
    tag = rel.replace("/report.json","").replace("/steer_report.json","")
    if o is None or n is None:
        print(f"{tag:34s} {'--' if o is None else f'{o:9.3f}'} {'--' if n is None else f'{n:9.3f}'}   MISSING")
    else:
        print(f"{tag:34s} {o:9.3f} {n:9.3f} {n-o:+8.3f}")
print("\nSteering cells are 20 items x 6 targets of GREEDY decoding, so an exact match")
print("is expected; any non-zero delta means the code path is not deterministic and")
print("must be explained before the arm is cited.")
PYEOF
echo BUNNY_REPRO_DONE
