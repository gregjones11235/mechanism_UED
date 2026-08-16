#!/bin/bash
LOG=/workspace/run_scratchStack.log
ALIVE=$(ps aux | grep -c '[r]un_dicode')
SESS=$(grep "Starting Session" $LOG | tail -1 | grep -oE "[0-9]+"); SESS=${SESS:--1}
HALLU=$(grep -c "AttributeError: type object" $LOG)
CUTOFF=172
echo "=== $(date '+%m-%d %H:%M') | alive:$ALIVE | session:$SESS/收线点$CUTOFF | 幻觉:$HALLU | GPU: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader) ==="
grep -E "\[SkillGraph\]|\[ScaffoldGate\]|\[Preflight\] kept" $LOG | tail -6
grep -E "Error|Traceback" $LOG | grep -vE "gate inactive|AttributeError: type object" | tail -3
if [ "$ALIVE" -eq 0 ] && grep -q "Run complete" $LOG; then echo ">>> 跑完了,STOP POD,收数"
elif [ "$ALIVE" -eq 0 ]; then echo ">>> !!! 中途死了(session $SESS),resume=新dir+拷0剧本(军规:严禁原地续跑),贴日志给 Claude"
elif [ "$SESS" -ge "$CUTOFF" ]; then echo ">>> !!! 到达 2e9 等效收线点,等最后一个Checkpointing的目录落盘后再 kill 收数!"
else echo ">>> 正在跑(session $SESS / 收线点 $CUTOFF),别停"
fi