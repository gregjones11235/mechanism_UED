#!/bin/bash
TARGET=/home/oseasy/student_pool_v1/common/COMMON_EVALUATOR_READY.json
OUTDIR=/home/oseasy/student_pool_v1/cc3/common_binding_wait
for i in $(seq 1 360); do
  if [ -f "$TARGET" ]; then
    date -u +"%Y-%m-%dT%H:%M:%SZ" > "$OUTDIR/FOUND_AT.txt"
    cp "$TARGET" "$OUTDIR/COMMON_EVALUATOR_READY.snapshot.json"
    ls -la /home/oseasy/student_pool_v1/common/ > "$OUTDIR/common_dir_listing.txt" 2>&1
    exit 0
  fi
  sleep 30
done
echo TIMEOUT_3H > "$OUTDIR/FOUND_AT.txt"
