#!/usr/bin/env bash
# ============================================================================
# Phase4A-v2 training launcher with REAL exit-code capture (CC2 directive §二).
# ============================================================================
# WHY THIS EXISTS
# ---------------
# The Phase4A probe was launched with `nohup ... &` and its success was read back by
# grepping the log for a summary line. That conflates "the log contains a summary" with
# "the process exited 0". A process can print a summary and then crash in teardown, or be
# OOM-killed / SIGKILL'd after flushing, and the log-grepping launcher would still report
# success. This launcher instead captures the REAL exit status of each background process
# via `wait $PID` and records it, with PID + start/completion wall-clock, to a status file.
# It NEVER infers exit=0 from log contents.
#
# GUARANTEES
#   * each training command is started in the background and its PID is captured ($!),
#   * the script `wait`s on that exact PID and records `$?` (the REAL return code, including
#     128+signal for kills, e.g. 137 = 128+9 SIGKILL/OOM),
#   * start time, completion time, elapsed seconds and the real return code are written to a
#     per-run status file (<out>/launch_status.json) AND echoed as a single parseable line,
#   * the launcher's OWN exit code is the training process's real return code, so a wrapping
#     scheduler / CI can rely on it directly (no log parsing).
#
# USAGE
#   launch_phase4a_v2.sh <out_dir> <log_basename> -- <python command...>
#   e.g.
#   launch_phase4a_v2.sh runs/FORMAL-PERSISTENT RMT16-Persistent-OrigVtrace -- \
#       /home/oseasy/miniconda3/envs/dicode310/bin/python train_rmt16_p2replay.py \
#       --carry_mode persistent --replay_mode original_vtrace --sequence_length 129 \
#       --ckpt17500 <path> --out runs/FORMAL-PERSISTENT --gpu_uuid GPU-8df1... --seed 42
#
# This script is a DELIVERABLE for FUTURE formal runs; it is NOT executed by the Phase4A-v2
# implementation round (NEW_TRAINING_RUNS=0 this round).
# ============================================================================
set -u
set -o pipefail

if [ "$#" -lt 3 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
  echo "usage: $0 <out_dir> <log_basename> -- <python command...>" >&2
  exit 2
fi

OUT_DIR="$1"; shift
LOG_BASE="$1"; shift
if [ "$1" != "--" ]; then
  echo "error: expected '--' separator before the command" >&2
  exit 2
fi
shift  # drop the '--'

if [ "$#" -lt 1 ]; then
  echo "error: no command given after '--'" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
LOG_FILE="$OUT_DIR/${LOG_BASE}.log"
STATUS_FILE="$OUT_DIR/launch_status.json"

# ---- timestamps are produced by `date`, NOT inferred from logs ----
START_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
START_EPOCH="$(date +%s)"

# Launch in background; capture the REAL PID. stdout+stderr -> log file.
"$@" >"$LOG_FILE" 2>&1 &
PID=$!

echo "LAUNCH pid=$PID start_utc=$START_UTC log=$LOG_FILE cmd=$*"

# Block until THAT pid exits and capture its REAL return code. `wait` returns the exit status
# of the waited-for process; if it was killed by a signal the code is 128+signum.
wait "$PID"
RC=$?

END_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
END_EPOCH="$(date +%s)"
ELAPSED=$(( END_EPOCH - START_EPOCH ))

# Record provenance. The return code is authoritative; the log is only human-readable context.
cat >"$STATUS_FILE" <<EOF
{
  "pid": $PID,
  "command": "$*",
  "log_file": "$LOG_FILE",
  "start_utc": "$START_UTC",
  "end_utc": "$END_UTC",
  "elapsed_seconds": $ELAPSED,
  "real_return_code": $RC,
  "killed_by_signal": $([ "$RC" -gt 128 ] && echo $(( RC - 128 )) || echo 0),
  "exit_source": "wait_pid",
  "inferred_from_log": false
}
EOF

# Single parseable line for schedulers.
echo "LAUNCH_DONE pid=$PID real_rc=$RC elapsed_s=$ELAPSED start=$START_UTC end=$END_UTC status=$STATUS_FILE"

# Propagate the REAL return code as this launcher's own exit code.
exit "$RC"
