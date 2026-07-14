#!/bin/bash
# Check C1/C2 progress. If complete, run analysis and optionally launch L5 B2/B3.
C1_LOG="/root/experiments/dicode_runs/aggregation/logs/c1_c2/C1_soft_copeland.log"
C2_LOG="/root/experiments/dicode_runs/aggregation/logs/c1_c2/C2_budgeted_soft_copeland.log"
LOCK="/tmp/c1c2_analysis_done.lock"

C1_SESSIONS=$(grep -c "Session finished" "$C1_LOG" 2>/dev/null || echo 0)
C2_SESSIONS=$(grep -c "Session finished" "$C2_LOG" 2>/dev/null || echo 0)
TMUX_ALIVE=$(/usr/bin/tmux has-session -t c1c2 2>/dev/null && echo YES || echo NO)

echo "C1 soft_copeland: $C1_SESSIONS sessions"
echo "C2 budgeted_soft_copeland: $C2_SESSIONS sessions"
echo "tmux c1c2: $TMUX_ALIVE"

if [ "$C1_SESSIONS" -gt 600 ] && [ "$C2_SESSIONS" -gt 600 ]; then
    if [ -f "$LOCK" ]; then
        echo "Analysis already done."
        exit 0
    fi
    echo ""
    echo "=== Both experiments complete! Running C1/C2 comparison... ==="
    cd /root/experiments/dreaming-in-code-coop
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate dicode310
    PYTHONPATH=src:$PYTHONPATH python scripts/compare_c1_c2.py
    PYTHONPATH=src:$PYTHONPATH python scripts/summarize_aggregation_runs.py \
        --log-dir /root/experiments/dicode_runs/aggregation/logs \
        --output-dir /root/experiments/dicode_runs/aggregation
    PYTHONPATH=src:$PYTHONPATH python scripts/summarize_llm_collaboration_runs.py
    echo "C1/C2 analysis complete."
    touch "$LOCK"

    echo ""
    echo "=== Launching L5 B2/B3 (role ablation) ==="
    bash /root/experiments/dreaming-in-code-coop/scripts/_run_l5_b2_b3.sh

elif [ "$TMUX_ALIVE" = "NO" ] && [ "$C1_SESSIONS" -lt 600 ]; then
    echo "WARNING: tmux dead but experiments incomplete. Check logs for errors."
    grep -i "error\|traceback" "$C1_LOG" 2>/dev/null | grep -v "Failed to get flag\|FutureWarning\|Delay kernel\|invalid value\|Could not save" | tail -5
fi
