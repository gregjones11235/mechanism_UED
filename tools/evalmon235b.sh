#!/bin/bash
LOG=/workspace/eval_235b_15300.log
N=$(grep -cE "^  step [0-9]+: mean_return" $LOG 2>/dev/null)
tmux has-session -t eval235b 2>/dev/null && S=RUNNING || S=EXITED
echo "235B eval: $N/8 点 | 会话:$S"
grep -E "^  step [0-9]+: mean_return" $LOG 2>/dev/null
if [ "$S" = "EXITED" ] && [ "$N" -ge 8 ]; then echo ">>> 跑完了,find JSON 然后接补充eval(evalsup)"
elif [ "$S" = "EXITED" ]; then echo ">>> !!! 提前退出($N/8),tail -30 $LOG 贴 Claude"
else echo ">>> 在跑,别动"
fi
