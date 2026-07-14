#!/bin/bash
cd /root/experiments/dreaming-in-code-coop
chmod +x scripts/run_l5_comparison.sh
tmux new-session -d -s "l5_comparison" "bash scripts/run_l5_comparison.sh"
echo "L5 launched at $(date)"
