#!/bin/bash
# Quick monitor for aggregation sweep experiments
LOG="${1:-/root/experiments/dicode_runs/aggregation/logs/sweep/s1_robust_weighted_t0.15.log}"

if [ ! -f "$LOG" ]; then
    echo "Log not found: $LOG"
    exit 1
fi

SESSIONS=$(grep -c "Session finished" "$LOG")
AGG_CALLS=$(grep -c "\[Aggregation\] Mode:" "$LOG")
ERRORS=$(grep -ci "error\|traceback" "$LOG" | head -1)
RUNNING=$(ps aux | grep "run_dicode" | grep -vc grep || echo 0)

echo "=== Aggregation Experiment Monitor ==="
echo "Log: $LOG"
echo "Sessions completed: $SESSIONS"
echo "Aggregation calls: $AGG_CALLS"
echo "Error lines: $ERRORS"
echo "Process running: $([ "$RUNNING" -gt 0 ] && echo 'YES' || echo 'NO')"
echo ""
echo "--- Last 5 Aggregation lines ---"
grep "\[Aggregation\]" "$LOG" | tail -5
echo ""
echo "--- Last 3 Session lines ---"
grep "Session finished\|Starting Session" "$LOG" | tail -3
