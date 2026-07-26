#!/usr/bin/env bash
# Run ONE posthoc attribution checkpoint. Args: STEP EXPECTED_SHA CKPT_ROOT OUT_JSON
set -uo pipefail
STEP="$1"; SHA="$2"; ROOT="$3"; OUT="$4"
PY=/home/oseasy/miniconda3/envs/dicode310/bin/python3
SRC=/home/oseasy/experiments/exploratory_delayed_onset_20260724/posthoc_attribution/src
cd "$SRC"
echo "===== RUN step=$STEP start $(date -Is) root=$ROOT ====="
$PY posthoc_attribution.py --ckpt_root "$ROOT" --step "$STEP" --expected_sha "$SHA" --out "$OUT"
RC=$?
echo "===== RUN step=$STEP exit=$RC end $(date -Is) ====="
exit $RC
