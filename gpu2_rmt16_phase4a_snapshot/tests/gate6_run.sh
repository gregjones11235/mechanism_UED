#!/bin/bash
# RMT16 Phase4A Gate-5 (step0 SHA) + Gate-6 (A/B training-no-perturbation).
# A = original --replay off ; B = --replay off --probe. Same step0/seed/GPU, 2 updates each.
# Runs SEQUENTIALLY on GPU2 (no concurrent GPU use). Full logs to files; no piping.
cd /home/oseasy/experiments/rmt16_replay_phase4a || exit 90
PY=/home/oseasy/miniconda3/envs/dicode310/bin/python
CKPT=/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/base_ckpt_v7fix55_armA_s0/rl_checkpoints/17500
GPU=GPU-8df11537-ab79-722d-606f-411966196c4c
mkdir -p logs runs reports

echo "GATE6_START $(date -u +%Y-%m-%dT%H:%M:%SZ)" > reports/gate6_run.marker

echo "=== ARM A (probe OFF) ===" >> reports/gate6_run.marker
$PY src/train_rmt16_p2replay.py \
  --carry_mode persistent --replay off \
  --ckpt17500 "$CKPT" --out runs/RMT16-GATE6-A --gpu_uuid "$GPU" \
  --total_updates 2 --seed 42 --equiv_dump \
  > logs/RMT16-GATE6-A.log 2>&1
echo "GATE6_A_EXIT=$?" >> reports/gate6_run.marker

echo "=== ARM B (probe ON) ===" >> reports/gate6_run.marker
$PY src/train_rmt16_p2replay.py \
  --carry_mode persistent --replay off --probe \
  --ckpt17500 "$CKPT" --out runs/RMT16-GATE6-B --gpu_uuid "$GPU" \
  --total_updates 2 --seed 42 --equiv_dump \
  > logs/RMT16-GATE6-B.log 2>&1
echo "GATE6_B_EXIT=$?" >> reports/gate6_run.marker

echo "=== COMPARE ===" >> reports/gate6_run.marker
$PY gate6_compare.py > reports/gate6_equiv_compare.log 2>&1
echo "GATE6_COMPARE_EXIT=$?" >> reports/gate6_run.marker
echo "GATE6_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> reports/gate6_run.marker
