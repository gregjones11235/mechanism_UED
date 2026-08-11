#!/usr/bin/env bash
# ============================================================================
# CC2 §二 BASE_GTRXL_ORIGINAL_VTRACE_98304 — single-arm GPU2 launcher (smoke -> long98304).
# ============================================================================
# Performance-first high-potential Student candidate. base_gtrxl == the SAME RMT16 network module
# + SAME ckpt17500, but the RMT16 persistent-token READ path is SKIPPED (pure GTrXL backbone; RMT
# params get no gradient). PPO / Replay / seed / task / cadence / certificate / Hindsight-AWR
# firewall are UNCHANGED from the frozen protocol; the ONLY scientific difference is
# carry_mode=base_gtrxl.
#
# §四/§五 run strategy: run the 4096-step ENGINEERING SMOKE first; gate ONLY on engineering
# correctness (literal rc=0, certificate PASS, checkpoints readable, params finite, no NaN/Inf,
# real network + real replay path executed, Hindsight/AWR=0, output dir fresh, source/config/
# checkpoint identity complete). The smoke is NOT gated on reward / success rate / Tier3 /
# short-horizon performance / replay_update_count. On smoke rc=0 -> launch the 98304 long run
# DIRECTLY (no wait for CC4, no performance early-stop, no wall-clock kill watchdog; §六/§九).
#
# §六: GPU2 ONLY (GPU-8df11537-...). GPU0/GPU1 STRICTLY FORBIDDEN. Independent RUN_ROOT (env).
# §七 interruption: Exact Resume NOT verified; any interruption -> RESTART_FROM_STEP0.
#
# Output-directory FRESHNESS GATE (mirrors launch_long98304_pair.sh): before creating anything the
# launcher checks SMOKE_OUT / LONG_OUT / the two status files; if ANY exists and is NON-EMPTY it
# exits immediately with RUN_OUTPUT_DIRECTORY_NOT_FRESH. It does NOT rm -rf / auto-clean / overwrite
# old checkpoints / append to old logs / auto-rename — the director moves old dirs explicitly.
#
# The driver imports frozen modules from runtime/frozen_modules + runtime/experiment_src, so
# PYTHONPATH is exported here. Each run's REAL exit code is captured via `wait $PID` (NEVER inferred
# from logs) and written to <OUT>/launch_status.json (+ git_sha / config SHAs / driver SHA).
#
# SECURITY: GPU0/GPU1 are STRICTLY FORBIDDEN; this launcher only touches GPU2.
# Monitoring (§九): tail -F <OUT>/launcher.stdout.log
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

# Make the frozen modules importable by the driver.
export PYTHONPATH="$SNAPSHOT_ROOT/runtime/frozen_modules:$SNAPSHOT_ROOT/runtime/experiment_src${PYTHONPATH:+:$PYTHONPATH}"

CARRY="base_gtrxl"
SEED=42
SEQLEN=129
GPU_BASE_GTRXL="GPU-8df11537-ab79-722d-606f-411966196c4c"   # GPU2 (ONLY)

SMOKE_OUT="$RUN_ROOT/runs/BASEGTRXL-SMOKE-4096"
LONG_OUT="$RUN_ROOT/runs/BASEGTRXL-LONG98304"
SMOKE_CFG="$SNAPSHOT_ROOT/configs/rmt16_phase4a_smoke_base_gtrxl.yaml"
LONG_CFG="$SNAPSHOT_ROOT/configs/rmt16_phase4a_long98304_base_gtrxl.yaml"
SMOKE_STATUS="$RUN_ROOT/runs/launch_base_gtrxl_smoke_status.json"
LONG_STATUS="$RUN_ROOT/runs/launch_base_gtrxl_long_status.json"

# ---- provenance helpers (best-effort; never block the run) ----
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

# ---- output-directory freshness gate ----
freshness_gate() {
  # Refuse to run if any output dir exists and is NON-EMPTY, or a status file exists and is
  # NON-EMPTY. NO rm / clean / overwrite / append / auto-rename: the director moves old dirs.
  local bad=0 d s
  for d in "$SMOKE_OUT" "$LONG_OUT"; do
    if [ -d "$d" ] && [ -n "$(ls -A "$d" 2>/dev/null)" ]; then
      echo "RUN_OUTPUT_DIRECTORY_NOT_FRESH: directory exists and is non-empty: $d"
      echo "  (move it explicitly before re-running; the launcher will not clean/overwrite it)"
      bad=1
    fi
  done
  for s in "$SMOKE_STATUS" "$LONG_STATUS"; do
    if [ -s "$s" ]; then
      echo "RUN_OUTPUT_DIRECTORY_NOT_FRESH: status file exists and is non-empty: $s"
      echo "  (move it explicitly before re-running; the launcher will not overwrite it)"
      bad=1
    fi
  done
  return "$bad"
}

# ---- preflight-only mode (regression freshness tests) ----
# Runs the freshness gate + prints provenance for BOTH configs, then exits WITHOUT launching.
if [ "${1:-}" = "--preflight-only" ]; then
  echo "BASE_GTRXL_PREFLIGHT_ONLY carry_mode=$CARRY"
  echo "  git_sha=$GIT_SHA"
  echo "  driver_file_sha256=$DRIVER_SHA"
  echo "  smoke_config_file_sha256=$(cfg_file_sha "$SMOKE_CFG")"
  echo "  smoke_config_scientific_sha256=$(cfg_sci_sha "$SMOKE_CFG")"
  echo "  long_config_file_sha256=$(cfg_file_sha "$LONG_CFG")"
  echo "  long_config_scientific_sha256=$(cfg_sci_sha "$LONG_CFG")"
  echo "  smoke_out=$SMOKE_OUT"
  echo "  long_out=$LONG_OUT"
  echo "  gpu_uuid=$GPU_BASE_GTRXL"
  if freshness_gate; then
    echo "OUTPUT_DIRECTORY_FRESHNESS_GATE=PASS"
    exit 0
  else
    echo "OUTPUT_DIRECTORY_FRESHNESS_GATE=FAIL"
    exit 1
  fi
fi

# ---- Normal launch path: freshness gate FIRST ----
if ! freshness_gate; then
  echo "BASE_GTRXL_LAUNCH_ABORTED reason=RUN_OUTPUT_DIRECTORY_NOT_FRESH"
  exit 1
fi

echo "BASE_GTRXL_LAUNCH snapshot_root=$SNAPSHOT_ROOT run_root=$RUN_ROOT carry_mode=$CARRY"
echo "  python=$PYTHON"
echo "  ckpt17500=$CKPT17500"
echo "  git_sha=$GIT_SHA"
echo "  driver_file_sha256=$DRIVER_SHA"
echo "  gpu_uuid=$GPU_BASE_GTRXL (GPU2 only; GPU0/GPU1 forbidden)"

run_one() {
  # $1=out  $2=cfg  $3=run_class  $4=total_updates  $5=save_every  $6=status_file
  local OUT="$1" CFG="$2" RUN_CLASS="$3" TU="$4" SE="$5" STATF="$6"
  local LOGF="$OUT/launcher.stdout.log"
  local ARM_STATF="$OUT/launch_status.json"
  local s_utc s_ep e_utc e_ep el CFG_FILE_SHA CFG_SCI_SHA RC
  mkdir -p "$OUT"
  CFG_FILE_SHA="$(cfg_file_sha "$CFG")"
  CFG_SCI_SHA="$(cfg_sci_sha "$CFG")"
  s_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"; s_ep="$(date +%s)"
  "$PYTHON" "$DRIVER" \
    --carry_mode "$CARRY" --replay_mode original_vtrace --run_class "$RUN_CLASS" \
    --sequence_length "$SEQLEN" --ckpt17500 "$CKPT17500" \
    --out "$OUT" --gpu_uuid "$GPU_BASE_GTRXL" \
    --formal_config "$CFG" --snapshot_root "$SNAPSHOT_ROOT" --run_root "$RUN_ROOT" \
    --total_updates "$TU" --seed "$SEED" --save_every "$SE" \
    >"$LOGF" 2>&1 &
  local PID=$!
  echo "BASE_GTRXL_ARM_LAUNCH run_class=$RUN_CLASS pid=$PID gpu=$GPU_BASE_GTRXL out=$OUT cfg=$CFG log=$LOGF started_at=$s_utc"
  wait "$PID"
  RC=$?
  e_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"; e_ep="$(date +%s)"; el=$(( e_ep - s_ep ))
  cat >"$ARM_STATF" <<EOF
{
  "arm": "$CARRY",
  "network_family": "base_gtrxl",
  "memory_mode": "none",
  "run_class": "$RUN_CLASS",
  "pid": $PID,
  "gpu_uuid": "$GPU_BASE_GTRXL",
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
  cat >"$STATF" <<EOF
{
  "candidate_id": "BASE_GTRXL_ORIGINAL_VTRACE_98304",
  "arm": "$CARRY",
  "run_class": "$RUN_CLASS",
  "total_updates": $TU,
  "total_env_steps": $(( TU * 2048 )),
  "pid": $PID,
  "gpu_uuid": "$GPU_BASE_GTRXL",
  "out_dir": "$OUT",
  "formal_config": "$CFG",
  "git_sha": "$GIT_SHA",
  "config_file_sha256": "$CFG_FILE_SHA",
  "config_scientific_sha256": "$CFG_SCI_SHA",
  "driver_file_sha256": "$DRIVER_SHA",
  "start_utc": "$s_utc",
  "end_utc": "$e_utc",
  "elapsed_seconds": $el,
  "real_return_code": $RC,
  "exit_source": "wait_pid",
  "inferred_from_log": false
}
EOF
  echo "BASE_GTRXL_ARM_DONE run_class=$RUN_CLASS pid=$PID real_rc=$RC elapsed_s=$el status=$ARM_STATF"
  return "$RC"
}

# ---- §四/§五: SMOKE first (engineering correctness only) -> rc=0 -> LONG98304 directly ----
run_one "$SMOKE_OUT" "$SMOKE_CFG" engineering_smoke 2 2 "$SMOKE_STATUS"
SMOKE_RC=$?
if [ "$SMOKE_RC" -ne 0 ]; then
  echo "BASE_GTRXL_LONG98304_NOT_LAUNCHED reason=SMOKE_RC_NONZERO smoke_rc=$SMOKE_RC"
  exit "$SMOKE_RC"
fi
echo "BASE_GTRXL_SMOKE_PASS rc=0 -> launching 98304 long run directly (§五)"

run_one "$LONG_OUT" "$LONG_CFG" long_run_98304 48 4 "$LONG_STATUS"
LONG_RC=$?
echo "BASE_GTRXL_LAUNCH_DONE smoke_rc=$SMOKE_RC long_rc=$LONG_RC"
exit "$LONG_RC"
