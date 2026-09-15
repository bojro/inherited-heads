#!/bin/bash
# V21 - the norm-weighted (Kobayashi 2020) variant of BOTH scores.
# PREREGISTRATION 17ar registers this as a threat to OUR result, not just theirs.
# Pre-committed reading is in discovery/run_normweighted.py's docstring, written
# before the run: our ranking surviving + raw still failing closes the hole; our
# ranking NOT surviving is a limitation; raw being RESCUED is a finding against us.
# Discovery on items 0-49 (exploratory, matching the frozen alpha rankings).
# Steering on items 100-149 at K=100 - identical protocol to run_confirm.sh, so
# the new cells are directly comparable to ours .990 and theirs .017.
cd "$(dirname "$0")/.."
D=discovery/results/normweighted/ultravox

[ -f $D/report.json ] || { echo "=== V21 discovery ultravox $(date +%H:%M) ==="
  .venv/bin/python discovery/run_normweighted.py --model ultravox \
    --strips testbed/strips --n-items 50 --item-offset 0 --out $D 2>&1 \
    | grep -vE "Loading weights|it/s\]$" ; }

echo "=== V21 steering $(date +%H:%M) ==="
for key in alpha_vnorm_norm alpha_ovnorm_norm alpha_vnorm_raw alpha_ovnorm_raw; do
  o=discovery/results/confirm/ultravox_$key
  [ -f $o/steer_report.json ] || { echo "--- $key ---"
    .venv/bin/python discovery/run_steer.py --model ultravox --strips testbed/strips \
      --scores $D/$key.npz --condition top --k 100 --n-items 50 --item-offset 100 \
      --out $o 2>&1 | grep -E '"accuracy"|"unattributed"|Error|Traceback'; }
done

echo "=== V21 SUMMARY $(date +%H:%M) ==="
.venv/bin/python - <<'PYEOF'
import json, os
b = "discovery/results/confirm/ultravox_"
rows = [("ours  (alpha, normalised)", "top100"), ("theirs(alpha, raw mass)", "rawtop100"),
        ("ours  x ||v||",  "alpha_vnorm_norm"),  ("ours  x ||Wo v||", "alpha_ovnorm_norm"),
        ("theirs x ||v||", "alpha_vnorm_raw"),   ("theirs x ||Wo v||","alpha_ovnorm_raw")]
print(f"  {'ranking':26s} {'steer acc':>10s} {'unattrib':>10s}")
for label, k in rows:
    p = f"{b}{k}/steer_report.json"
    if os.path.exists(p):
        d = json.load(open(p))
        print(f"  {label:26s} {d['accuracy']:10.3f} {d['unattributed_rate']:10.3f}")
    else:
        print(f"  {label:26s} {'MISSING':>10s}")
print("\n  layer-matched random 100 reference: 0.453   chance: 0.167")
PYEOF
echo V21_DONE
