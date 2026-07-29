#!/usr/bin/env bash
# ============================================================================
# Phase4A-direct-98304 (§二 / §五 / §九) — ENGINEERING SMOKE PAIR launcher.
# ============================================================================
# Launches BOTH 4096-step smoke arms IN PARALLEL — persistent on GPU2, reset128 on GPU3 —
# captures each arm's REAL exit code via `wait $PID` (NEVER inferred from log contents), writes
# a per-arm launch_status.json + a pair status file, and exits nonzero if EITHER arm is nonzero.
#
# This smoke GATES the conditional 98k launch (§五): only if BOTH arms pass ALL §四 correctness
# gates does the director launch launch_long98304_pair.sh. replay_update_count==0 /
# REPLAY_HORIZON_NOT_REACHED does NOT block — at 4096 steps the 512-anchor replay horizon is
# structurally unreachable, so a correct arm exits PASS_REPLAY_HORIZON_NOT_REACHED (real_rc=0).
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

RUN_CLASS="engineering_smoke"
TOTAL_UPDATES=2          # 2 * 2048 = 4096 resolved env steps
SAVE_EVERY=2             # step0 + step4096
SEED=42
SEQLEN=129

GPU_PERSISTENT="GPU-8df11537-ab79-722d-606f-411966196c4c"   # GPU2
GPU_RESET128="GPU-f56a59b4-99f3-f2e5-11c6-d01685de8abd"     # GPU3

PERS_OUT="$RUN_ROOT/runs/RMT16-SMOKE-PERSISTENT-4096"
RESET_OUT="$RUN_ROOT/runs/RMT16-SMOKE-RESET128-4096"
PERS_CFG="$SNAPSHOT_ROOT/configs/rmt16_phase4a_smoke_persistent.yaml"
RESET_CFG="$SNAPSHOT_ROOT/configs/rmt16_phase4a_smoke_reset128.yaml"

mkdir -p "$PERS_OUT" "$RESET_OUT"
PAIR_STATUS="$RUN_ROOT/runs/launch_smoke_pair_status.json"
START_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
START_EPOCH="$(date +%s)"

echo "SMOKE_PAIR_LAUNCH snapshot_root=$SNAPSHOT_ROOT run_root=$RUN_ROOT run_class=$RUN_CLASS"
echo "  python=$PYTHON"
echo "  ckpt17500=$CKPT17500"
echo "  git_sha=$(cd "$SNAPSHOT_ROOT" && git rev-parse HEAD 2>/dev/null || echo UNKNOWN)"

run_arm() {
  # $1=out  $2=cfg  $3=carry_mode  $4=gpu_uuid
  local OUT="$1" CFG="$2" CARRY="$3" GPU="$4"
  local LOGF="$OUT/launcher.stdout.log"
  local STATF="$OUT/launch_status.json"
  local s_utc s_ep
  s_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"; s_ep="$(date +%s)"
  "$PYTHON" "$DRIVER" \
    --carry_mode "$CARRY" --replay_mode original_vtrace --run_class "$RUN_CLASS" \
    --sequence_length "$SEQLEN" --ckpt17500 "$CKPT17500" \
    --out "$OUT" --gpu_uuid "$GPU" \
    --formal_config "$CFG" --snapshot_root "$SNAPSHOT_ROOT" --run_root "$RUN_ROOT" \
    --total_updates "$TOTAL_UPDATES" --seed "$SEED" --save_every "$SAVE_EVERY" \
    >"$LOGF" 2>&1 &
  local PID=$!
  echo "SMOKE_ARM_LAUNCH arm=$CARRY pid=$PID gpu=$GPU out=$OUT cfg=$CFG log=$LOGF started_at=$s_utc"
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
  "start_utc": "$s_utc",
  "end_utc": "$e_utc",
  "elapsed_seconds": $el,
  "real_return_code": $RC,
  "killed_by_signal": $([ "$RC" -gt 128 ] && echo $(( RC - 128 )) || echo 0),
  "exit_source": "wait_pid",
  "inferred_from_log": false
}
EOF
  echo "SMOKE_ARM_DONE arm=$CARRY pid=$PID real_rc=$RC elapsed_s=$el status=$STATF"
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
echo "SMOKE_PAIR_DONE persistent_rc=$PERS_RC reset128_rc=$RESET_RC pair_pass=$PAIR_PASS status=$PAIR_STATUS"
[ "$PAIR_PASS" = true ]
