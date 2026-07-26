# D052 canonical_v2 — Architecture

The canonical D052 mechanism framework: deterministic, auditable, and
authorization-gated. It lives at `gpu1_aggregation_siege/d052/` (top-level,
deliberately separate from `src/dicode/` to avoid the four-package `import dicode`
collision). Baseline: Henry-branch HEAD `a2726e3`.

## Frozen canonical_v2 config (not overridable)

| Key | Value | Source-of-truth |
|---|---|---|
| achievement_schema | `craftax_67_v1` | `dicode_src/auction/craftax_achievements.py` (blob `5bb881a6`) |
| num_achievements | 67 (ids 0..66) | same; `assert ==67` |
| conditioning_type | `achievement_multi_hot` | `task_utils.py` `embedding[ach.value]=1.0` |
| conditioning_dimension | 67 | canonical_id == goal_vector_index == enum value |
| student_obs_dim | 8335 = base 8268 + 67 | `multitask.py` + `run_p9` `assert obs_dim==8335 & EMB==67` |
| candidate_pool_mode | `shared_frozen` | one pool, all selectors |
| unknown/empty/fallback policy | `error` | never silent |
| score_normalization | `rank_percentile_v1` | per-role, [0,1], deterministic ties |

The legacy `one_hot` / `obs_dim 8300` / `32-slot` interface is NOT in current HEAD
and is **banned** from canonical_v2.

## Data flow

```
candidate pool (shared_frozen)
   └─ generation/      validate + canonicalize + content-hash candidates; build pool
   └─ achievements/    official 67 registry + explicit alias allow-list
roles (Tutor/Critic/Explorer score; Modeler advises)
   └─ roles/           reconciled registry, judgment-batch validation, critic veto
   └─ normalization/   rank_percentile_v1, per-role independent
   └─ profiling/       deterministic Student profile + Modeler judgment firewall
selection
   └─ selectors/       ONE interface select(); S0/S1/S2 + Soft/Budgeted Copeland
                       + Auction raw/budgeted; critic policy; deterministic replay
execution mapping
   └─ execution/       candidate -> canonical name -> id -> goal-vector index ->
                       67 multi-hot -> obs 8335 -> training task id (certificate)
cells (per-cell authorization & launch)
   └─ cells/           DRAFT->VALIDATED->READY->AUTHORIZED->RUNNING->COMPLETE;
                       content-addressed identity; authorization-gated launch
   └─ training/ eval/  authorization-gated adapters (NO-OP this phase)
```

## Determinism

Every stage is deterministic and content-addressed: candidates (`chash`), pools
(`pool_hash`), selections (`selection_hash` binds selector/policy/k/seed/
selected_ids), cells (`cell_identity_hash` over ~20 fields), authorizations
(`authorization_hash`). Identical inputs reproduce identical outputs bit-for-bit
regardless of candidate ordering. Tie-break is always `(score DESC, candidate_id
ASC)`.

## Discipline (enforced in code + tests)

- **NO_LEGACY_ARTIFACT_OVERWRITE** — pool store & cell registry create with
  no-overwrite; `output_dir` checked against `DENY_LEGACY_OUTPUT_PREFIXES`.
- **NO_SILENT_FALLBACK** — unknown target / missing protocol_version / selector
  pool mismatch / execution deviation are hard errors.
- **NO_UNAUTHORIZED_TRAINING** — launch requires AUTHORIZED + valid authorization;
  a no-training authorization can only ever run 0 timesteps.
- **NO_RAW_DATA_NO_STRONG_CLAIM** — execution certificate forbids
  `executed_as_intended=True` with any failed gate; evaluation forbids a strong
  claim without raw results.

## Test gates

GATE 1 protocol_version · 2 schemas · 3 achievement registry (drift detector) ·
4 pool · 5 candidate validator · 6 roles+profiling · 7 normalization ·
8 critic policy · 9 selector determinism/unified interface · 10 execution mapping ·
11 cell lifecycle · 12 no-training guarantee · 13 training/eval adapters.

## Legacy reuse

See `reports/d052_canonical_legacy_reuse_map.md` (85 modules classified) and its
selector addendum. Legacy selectors are re-implemented self-contained; exact
numeric parity is NOT claimed. Frozen source evidence:
`audit_outputs/d052_legacy_source_freeze_*` (director workspace) +
`reports/d052_legacy_source_freeze.md` (repo).
