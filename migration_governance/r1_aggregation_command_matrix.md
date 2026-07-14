# R1 Aggregation Command Matrix
Date: 2026-07-14 | Commit: 460ccb6 | Branch: exp/siege-aggregation-sota

## Shared Configuration (All Mechanisms)
- GPU: Physical GPU1 (UUID: GPU-f4d0f435-b393-6405-cb6d-7b4e787335de)
- CUDA_VISIBLE_DEVICES=1
- candidate_count=32, selected_count=8
- Frozen pool: hash TBD (generated per-comparison)
- Frozen cache: /root/experiments/dicode_runs/siege_aggregation/frozen_immutable_cache (96 entries, SHA256 TBD)
- PPO: 256 envs, 64 steps, 1 update (ENGINEERING_ONLY preflight)
- Seed: 0

## Mechanism Commands
All run from /root/experiments/dicode-siege-aggregation with:
  source activate dicode310
  PYTHONPATH=src:/root/experiments/dicode-aggregation-v2/src:$PYTHONPATH

### M0: Original DiCode (aggregation disabled, zero cache reads)
python scripts/run_data_plane_preflight_v3.py --mechanism original

### M1: Soft Copeland
python scripts/run_data_plane_preflight_v3.py --mechanism soft_copeland

### M2: Budgeted Soft Copeland
python scripts/run_data_plane_preflight_v3.py --mechanism budgeted_copeland

### M3: Auction raw
python scripts/run_data_plane_preflight_v3.py --mechanism auction_raw

### M4: Budgeted Auction
python scripts/run_data_plane_preflight_v3.py --mechanism auction_budgeted

## Prerequisites
- [ ] Gate R0 fully passing in dicode310 (Test 4: make_train+checkpoint)
- [ ] Frozen pool hash recorded
- [ ] Frozen cache SHA256 recorded
- [ ] All 5 mechanisms added to preflight v3 CLI choices
- [ ] Fresh unique output directories
- [ ] GPU0 protection verified

## NOT AUTHORIZED
- Long-horizon training (>1 PPO update)
- Multi-seed runs
- Performance claims from preflight data
