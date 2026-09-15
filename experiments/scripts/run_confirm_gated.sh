#!/bin/bash
# wait for the RISK-FIRST block (S7 + R1), then run the pre-registered suite.
# Gate moved from TEXT_DONE to RISK_DONE on 2026-08-17 so the two experiments
# that can invalidate the framing resolve before the 12-hour confirmatory block
# rather than after it.
until grep -q RISK_DONE ~/riskfirst.log 2>/dev/null; do sleep 120; done
bash ~/run_confirm.sh
