#!/usr/bin/env bash
# ============================================================================
# Phase4A-direct-98304 (§三 / §五 / §六 / §九) — DIRECT 98304 LONG-RUN PAIR launcher.
# ============================================================================
# Launched by the director IMMEDIATELY after BOTH smoke arms pass ALL §四 correctness gates (§五):
# do NOT wait for a new director reply, do NOT wait for the CC4 state bank, do NOT decide on
# smoke PERFORMANCE. Launches BOTH 98304-step arms IN PARALLEL — persistent on GPU2, reset128 on
# GPU3 — captures each arm's REAL exit code via `wait $PID` (NEVER inferred from logs), writes a
# per-arm launch_status.json + a pair status file, and exits nonzero if EITHER arm is nonzero.
#
# §六 run rules: NO performance early-stop. A checkpoint is saved every 8192 env steps
# (save_every=4). 24576 env steps (update 12) is ONLY an ordinary intermediate checkpoint — NOT
# an independent experiment, NOT a stop gate, NOT a model-selection gate. Allowed stop reasons
# ONLY: certificate FAIL; checkpoint identity FAIL; NaN/Inf; nonzero exit; training definitively
# stuck; memory carry/reset invariant failure; Hindsight/AWR firewall failure; two-arm config
# mismatch except carry_mode.
#
# §七 interruption: Exact Resume is NOT verified; any interruption -> RESTART_FROM_STEP0 (never
# resume from an intermediate checkpoint as the same formal trajectory).
#
# Phase4A-direct-98304 hotfix (§三): OUTPUT-DIRECTORY FRESHNESS GATE. Before creating anything the
# launcher checks PERS_OUT / RESET_OUT / PAIR_STATUS; if ANY exists and is NON-EMPTY it exits
# immediately with RUN_OUTPUT_DIRECTORY_NOT_FRESH. It does NOT rm -rf / auto-clean / overwrite old
# checkpoints / append to old logs / auto-rename — the director moves old directories explicitly
# before a re-run. launch_status.json now ALSO records git_sha / config_file_sha256 /
# config_scientific_sha256 / driver_file_sha256 / run_class / the literal return code.
#
# Phase4A-direct-98304 hotfix (§一): the driver imports the frozen modules from
# runtime/frozen_modules + runtime/experiment_src, so PYTHONPATH is exported here (the previous
# deploy-only patch, now committed).
#
# SECURITY: GPU0/GPU1 are STRICTLY FORBIDDEN; this launcher only touches GPU2/GPU3.
# Monitoring (§九): tail -F <OUT>/launcher.stdout.log   (per arm)
# ============================================================================
set -u
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SNAPSHOT_ROOT="$(cd "$HERE/../.." && pwd)"
RUN_ROOT="${RUN_ROOT:-$SNAPSHOT_ROOT}"
PYTHON="${PYTHON:-/home/oseasy/miniconda3/envs/dicode310/bin/python}"
# Orbax base checkpoint (manager_dir/17500); inner-params SHA == d4e85af5... (verified on GPU).
CKPT17500="${CKPT17500:-/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/base_ckpt_v7fix55_armA_s0/rl_checkpoints/17500}"
DRIVER="$HERE/train_rmt16_p2replay.py"

# Phase4A-direct-98304 hotfix (§一): make the frozen modules importable by the driver.
export PYTHONPATH="$SNAPSHOT_ROOT/runtime/frozen_modules:$SNAPSHOT_ROOT/runtime/experiment_src${PYTHONPATH:+:$PYTHONPATH}"

RUN_CLASS="long_run_98304"
TOTAL_UPDATES=48         # 48 * 2048 = 98304 resolved env steps
SAVE_EVERY=4             # checkpoint every 4 outer updates (every 8192 env steps)
SEED=42
SEQLEN=129

GPU_PERSISTENT="GPU-8df11537-ab79-722d-606f-411966196c4c"   # GPU2
GPU_RESET128="GPU-f56a59b4-99f3-f2e5-11c6-d01685de8abd"     # GPU3

PERS_OUT="$RUN_ROOT/runs/RMT16-LONG98304-PERSISTENT"
RESET_OUT="$RUN_ROOT/runs/RMT16-LONG98304-RESET128"
PERS_CFG="$SNAPSHOT_ROOT/configs/rmt16_phase4a_long98304_persistent.yaml"
RESET_CFG="$SNAPSHOT_ROOT/configs/rmt16_phase4a_long98304_reset128.yaml"

PAIR_STATUS="$RUN_ROOT/runs/launch_long98304_pair_status.json"

# ---- Phase4A-direct-98304 hotfix (§三): provenance helpers (best-effort; never block the run) ----
GIT_SHA="$(cd "$SNAPSHOT_ROOT" && git rev-parse HEAD 2>/dev/null || echo UNKNOWN)"
DRIVER_SHA="$(sha256sum "$DRIVER" 2>/dev/null | cut -d' ' -f1)"
[ -n "${DRIVER_SHA:-}" ] || DRIVER_SHA="UNKNOWN"

cfg_file_sha() {  # $1 = config path
  local s; s="$(sha256sum "$1" 2>/dev/null | cut -d' ' -f1)"; echo "${s:-UNKNOWN}";
}
cfg_sci_sha() {  # $1 = config path  (uses the driver's own RTC canonicalization)
  local s; s="$(RMT_CFG="$1" "$PYTHON" - <<'PY' 2>/dev/null
import os
import phase4a_v2_runtime_config as RTC
_rec = RTC.load_formal_config(os.environ["RMT_CFG"])
print(RTC.scientific_config_sha256(_rec["config"]["scientific_config"]))
PY
)"; echo "${s:-UNKNOWN}";
}

# ---- Phase4A-direct-98304 hotfix (§三): output-directory freshness gate ----
freshness_gate() {
  # Refuse to run if any output dir exists and is NON-EMPTY, or PAIR_STATUS exists and is
  # NON-EMPTY. NO rm / clean / overwrite / append / auto-rename: the director moves old dirs.
  local bad=0 d
  for d in "$PERS_OUT" "$RESET_OUT"; do
    if [ -d "$d" ] && [ -n "$(ls -A "$d" 2>/dev/null)" ]; then
      echo "RUN_OUTPUT_DIRECTORY_NOT_FRESH: directory exists and is non-empty: $d"
      echo "  (move it explicitly before re-running; the launcher will not clean/overwrite it)"
      bad=1
    fi
  done
  if [ -s "$PAIR_STATUS" ]; then
    echo "RUN_OUTPUT_DIRECTORY_NOT_FRESH: pair status exists and is non-empty: $PAIR_STATUS"
    echo "  (move it explicitly before re-running; the launcher will not overwrite it)"
    bad=1
  fi
  return "$bad"
}

# ---- Phase4A-direct-98304 hotfix (§三/§四): preflight-only mode (regression tests 9/10) ----
# Runs the freshness gate + prints provenance, then exits WITHOUT launching python/training.
if [ "${1:-}" = "--preflight-only" ]; then
  echo "LONG98304_PREFLIGHT_ONLY run_class=$RUN_CLASS"
  echo "  git_sha=$GIT_SHA"
  echo "  driver_file_sha256=$DRIVER_SHA"
  echo "  persistent_config_file_sha256=$(cfg_file_sha "$PERS_CFG")"
  echo "  persistent_config_scientific_sha256=$(cfg_sci_sha "$PERS_CFG")"
  echo "  reset128_config_file_sha256=$(cfg_file_sha "$RESET_CFG")"
  echo "  reset128_config_scientific_sha256=$(cfg_sci_sha "$RESET_CFG")"
  echo "  pers_out=$PERS_OUT"
  echo "  reset_out=$RESET_OUT"
  echo "  pair_status=$PAIR_STATUS"
  if freshness_gate; then
    echo "OUTPUT_DIRECTORY_FRESHNESS_GATE=PASS"
    exit 0
  else
    echo "OUTPUT_DIRECTORY_FRESHNESS_GATE=FAIL"
    exit 1
  fi
fi

# ---- Normal launch path: freshness gate FIRST, then create out dirs ----
if ! freshness_gate; then
  echo "LONG98304_PAIR_ABORTED reason=RUN_OUTPUT_DIRECTORY_NOT_FRESH"
  exit 1
fi
mkdir -p "$PERS_OUT" "$RESET_OUT"

START_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
START_EPOCH="$(date +%s)"

echo "LONG98304_PAIR_LAUNCH snapshot_root=$SNAPSHOT_ROOT run_root=$RUN_ROOT run_class=$RUN_CLASS"
echo "  python=$PYTHON"
echo "  ckpt17500=$CKPT17500"
echo "  git_sha=$GIT_SHA"
echo "  driver_file_sha256=$DRIVER_SHA"

run_arm() {
  # $1=out  $2=cfg  $3=carry_mode  $4=gpu_uuid
  local OUT="$1" CFG="$2" CARRY="$3" GPU="$4"
  local LOGF="$OUT/launcher.stdout.log"
  local STATF="$OUT/launch_status.json"
  local s_utc s_ep CFG_FILE_SHA CFG_SCI_SHA
  CFG_FILE_SHA="$(cfg_file_sha "$CFG")"
  CFG_SCI_SHA="$(cfg_sci_sha "$CFG")"
  s_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"; s_ep="$(date +%s)"
  "$PYTHON" "$DRIVER" \
    --carry_mode "$CARRY" --replay_mode original_vtrace --run_class "$RUN_CLASS" \
    --sequence_length "$SEQLEN" --ckpt17500 "$CKPT17500" \
    --out "$OUT" --gpu_uuid "$GPU" \
    --formal_config "$CFG" --snapshot_root "$SNAPSHOT_ROOT" --run_root "$RUN_ROOT" \
    --total_updates "$TOTAL_UPDATES" --seed "$SEED" --save_every "$SAVE_EVERY" \
    >"$LOGF" 2>&1 &
  local PID=$!
  echo "LONG_ARM_LAUNCH arm=$CARRY pid=$PID gpu=$GPU out=$OUT cfg=$CFG log=$LOGF started_at=$s_utc"
  wait "$PID"
  local RC=$?
  local e_utc e_ep el
  e_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"; e_ep="$(date +%s)"; el=$(( e_ep - s_ep ))
  cat >"$STATF" <<EOF
{
  "arm": "$CARRY",
  "run_class": "$RUN_CLASS",
  "pid": $PID,
  "gpu_uuid": "$GPU",
  "out_dir": "$OUT",
  "formal_config": "$CFG",
  "snapshot_root": "$SNAPSHOT_ROOT",
  "run_root": "$RUN_ROOT",
  "ckpt17500": "$CKPT17500",
  "log_file": "$LOGF",
  "git_sha": "$GIT_SHA",
  "config_file_sha256": "$CFG_FILE_SHA",
  "config_scientific_sha256": "$CFG_SCI_SHA",
  "driver_file_sha256": "$DRIVER_SHA",
  "start_utc": "$s_utc",
  "end_utc": "$e_utc",
  "elapsed_seconds": $el,
  "real_return_code": $RC,
  "killed_by_signal": $([ "$RC" -gt 128 ] && echo $(( RC - 128 )) || echo 0),
  "exit_source": "wait_pid",
  "inferred_from_log": false
}
EOF
  echo "LONG_ARM_DONE arm=$CARRY pid=$PID real_rc=$RC elapsed_s=$el status=$STATF"
  return "$RC"
}

# Launch BOTH arms IN PARALLEL (each backgrounded); wait on BOTH and capture real RCs.
run_arm "$PERS_OUT" "$PERS_CFG" persistent "$GPU_PERSISTENT" &
PAIR_PERS=$!
run_arm "$RESET_OUT" "$RESET_CFG" reset128 "$GPU_RESET128" &
PAIR_RESET=$!

wait "$PAIR_PERS"; PERS_RC=$?
wait "$PAIR_RESET"; RESET_RC=$?

END_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
END_EPOCH="$(date +%s)"
PAIR_PASS=false
if [ "$PERS_RC" -eq 0 ] && [ "$RESET_RC" -eq 0 ]; then PAIR_PASS=true; fi

cat >"$PAIR_STATUS" <<EOF
{
  "run_class": "$RUN_CLASS",
  "total_updates": $TOTAL_UPDATES,
  "total_env_steps": $(( TOTAL_UPDATES * 2048 )),
  "git_sha": "$GIT_SHA",
  "driver_file_sha256": "$DRIVER_SHA",
  "start_utc": "$START_UTC",
  "end_utc": "$END_UTC",
  "elapsed_seconds": $(( END_EPOCH - START_EPOCH )),
  "persistent_rc": $PERS_RC,
  "reset128_rc": $RESET_RC,
  "persistent_out": "$PERS_OUT",
  "reset128_out": "$RESET_OUT",
  "pair_pass": $PAIR_PASS
}
EOF
echo "LONG98304_PAIR_DONE persistent_rc=$PERS_RC reset128_rc=$RESET_RC pair_pass=$PAIR_PASS status=$PAIR_STATUS"
[ "$PAIR_PASS" = true ]
