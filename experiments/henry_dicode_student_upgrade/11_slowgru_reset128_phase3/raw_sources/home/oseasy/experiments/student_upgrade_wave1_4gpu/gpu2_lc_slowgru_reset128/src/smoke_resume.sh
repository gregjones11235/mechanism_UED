#!/usr/bin/env bash
# EventMemory32 smoke (gate 7) + roundtrip (gate 5) + exact-resume (gate 6) + no-NaN/entropy (gate 9).
# GPU2 only. Deterministic ops set inside the trainer.
set -e
PY=/home/oseasy/miniconda3/envs/dicode310/bin/python
BASE=/home/oseasy/experiments/student_upgrade_wave1_4gpu/gpu2_lc_slowgru_reset128
SRC=$BASE/src
RUN="$PY -u $SRC/run_slowgru_reset128_24576.py"
LOG=$BASE/smoke_resume.log
cd $SRC
{
echo "=================== SLOWGRU-RESET128 SMOKE + EXACT RESUME ==================="
date -u
echo ">>> Run A: continuous 0 -> 8192 (num_chunks=2, save 0/4096/8192)"
$RUN --out_dir $BASE/smoke_A --ckpt_root $BASE/smoke_A/ckpt --num_chunks 2 --save_steps 0,4096,8192
echo ">>> Run B1: fresh 0 -> 4096 (num_chunks=1, save 0/4096)"
$RUN --out_dir $BASE/smoke_B --ckpt_root $BASE/smoke_B/ckpt --num_chunks 1 --save_steps 0,4096
echo ">>> Run B2: resume from B1/4096 -> 8192 (num_chunks=1, save 8192)"
$RUN --out_dir $BASE/smoke_B2 --ckpt_root $BASE/smoke_B2/ckpt --resume_from $BASE/smoke_B/ckpt/4096/full_state.pkl --num_chunks 1 --save_steps 8192
echo ">>> COMPARE"
$PY -u $SRC/compare_resume.py $BASE
echo ">>> Gate 9 (entropy / NaN) from Run A per_update:"
$PY -u -c "import json,sys; ms=[json.loads(l) for l in open('$BASE/smoke_A/LC_SLOWGRU_RESET128_per_update.jsonl')]; ent=[m['entropy'] for m in ms]; print('updates=%d entropy_first=%.4f entropy_last=%.4f entropy_min=%.4f'%(len(ms),ent[0],ent[-1],min(ent))); print('ENTROPY_OK' if min(ent)>0.05 else 'ENTROPY_COLLAPSE')"
echo ">>> Gate 5 (rollout-boundary clear) from Run A summary:"
$PY -u -c "import json; s=json.load(open('$BASE/smoke_A/LC_SLOWGRU_RESET128_train_summary.json')); g=s['reset128_gates']; print('n_rollouts=%d boundary_clear_pass=%s clear_nontrivial_pass=%s'%(g['n_rollouts'],g['boundary_clear_pass'],g['clear_nontrivial_pass'])); print('init_ls_hash=%s'%g['init_ls_hash'][:16]); print('GATE5_BOUNDARY_CLEAR_OK' if (g['boundary_clear_pass'] and g['clear_nontrivial_pass']) else 'GATE5_FAIL')"
date -u
echo "=================== DONE ==================="
} 2>&1 | tee $LOG
