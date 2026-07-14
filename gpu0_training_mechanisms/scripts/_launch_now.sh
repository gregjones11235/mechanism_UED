#!/bin/bash
/usr/bin/tmux new-session -d -s c1c2 bash /root/experiments/dreaming-in-code-coop/scripts/_c1c2_tmux.sh
echo "launched at $(date)"
