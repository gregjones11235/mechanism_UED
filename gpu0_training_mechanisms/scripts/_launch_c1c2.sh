#!/bin/bash
cd /root/experiments/dreaming-in-code-coop
chmod +x scripts/run_c1_c2.sh
tmux new-session -d -s c1c2 "bash scripts/run_c1_c2.sh"
echo "C1/C2 launched at $(date)"
