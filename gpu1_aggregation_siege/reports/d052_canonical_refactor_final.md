# D052 Canonical In-Place Refactor — Final Report

**Status: D052_CANONICAL_REFACTOR_IMPLEMENTED = true**
Branch `henry/d052-canonical-refactor` · baseline Henry-branch HEAD `a2726e3` ·
9 commits (`d9bcbbd` … this commit) · **221 tests passing (GATE 1–13)** ·
**long-training runs this phase: 0** · **legacy artifacts frozen & unchanged (48/48)**.

## 1. Goal

Make D052 a canonical, deterministic, auditable mechanism framework that: uses the
official Craftax-67 achievements as the only legal targets; consumes ONE shared
frozen candidate pool; runs four roles (Tutor/Critic/Explorer scoring + Modeler
advisory); normalizes scores per-role; exposes a unified selector interface
(Soft/Budgeted Copeland, Auction, baseline rungs); maps candidates to real Student
goals deterministically; and gates everything behind per-cell authorization &
launch — with full hash/manifest/evidence-chain/audit. **This phase does no long
training.**

## 2. Frozen canonical_v2 config (verified, not overridable)

| Key | Value | Evidence |
|---|---|---|
| achievement_schema | craftax_67_v1 | `dicode_src/auction/craftax_achievements.py` blob `5bb881a6` |
| num_achievements | 67 (ids 0..66) | same; `assert ==67` |
| conditioning | achievement_multi_hot, dim 67 | `task_utils.py` `embedding[ach.value]=1.0` |
| student_obs_dim | 8335 = 8268 + 67 | `multitask.py` + `run_p9` assert |
| candidate_pool_mode | shared_frozen | one pool, all selectors |
| policies | unknown/empty/fallback = error | never silent |
| score_normalization | rank_percentile_v1 | per-role, [0,1], deterministic ties |

`canonical_id == goal_vector_index == craftax enum value` (CONFIRMED identical).
Value 38 = `defeat_orc_solider` (upstream misspelling kept verbatim → canonical);
`defeat_orc_soldier` is the single explicit audited alias.

## 3. What was built (`gpu1_aggregation_siege/d052/`, 70 files)

| Subpackage | Responsibility | Gate |
|---|---|---|
| `legacy/` | frozen constants + protocol_version resolution (fail-closed) | 1 |
| `schemas/` | Pydantic `extra=forbid` schemas (candidate/roles/selector/execution/run_config) | 2 |
| `achievements/` | official-67 registry + alias allow-list + drift detector | 3 |
| `generation/` | candidate validator + shared frozen pool (no-overwrite store) | 4,5 |
| `roles/`, `profiling/`, `normalization/` | 4-role protocol, Student profile + Modeler firewall, rank_percentile_v1 | 6,7 |
| `selectors/` | unified `select()`; S0/S1/S2 + Soft/Budgeted Copeland + Auction; critic policy; deterministic replay | 8,9 |
| `execution/` | candidate→real-goal mapping certificate (6 gates) | 10 |
| `cells/` | lifecycle state machine, content-addressed identity, authorization-gated launch, CLI | 11,12 |
| `training/`, `evaluation/` | authorization-gated adapters (no-op this phase) | 13 |
| `docs/`, `examples/` | 6 docs + runnable training-free pipeline + DRAFT/BLOCKED cell templates | — |

## 4. Discipline enforcement (code + tests)

- **NO_LEGACY_ARTIFACT_OVERWRITE** — pool store & cell registry create no-overwrite;
  `output_dir` checked vs `DENY_LEGACY_OUTPUT_PREFIXES`. 48/48 frozen source files
  re-hashed AFTER the refactor == before (`d052_legacy_artifact_hashes_before_after.json`).
- **NO_SILENT_FALLBACK** — missing/unknown protocol_version, unknown target, selector
  pool mismatch, execution deviation → hard errors.
- **NO_UNAUTHORIZED_TRAINING** — launch requires AUTHORIZED + valid authorization; a
  no-training authorization is structurally incapable of timesteps (forced no-op;
  non-zero steps FAIL the cell). Absolute zero-timesteps invariant asserted across
  the registry (GATE 12).
- **NO_RAW_DATA_NO_STRONG_CLAIM** — certificate forbids `executed_as_intended=True`
  with any failed gate; evaluation forbids a strong claim without raw results.

## 5. Stop-conditions (15) — status

#1 67 single source CONFIRMED · #2 conditioning order CONFIRMED · #3 obs_dim 8335
CONFIRMED → none triggered. #4–#13, #15 not triggered (no hidden fallback found;
selectors deterministic; candidate→task mapping by canonical name, not salted hash;
no long training required; no legacy artifact modified; no CC2/CC4 write conflict —
work isolated under `gpu1_aggregation_siege/d052/`; no new API key; no provider
auto-switch; no gate lowered; legacy SHA unchanged; source tree at current HEAD).
**#14 PARTIAL/DOCUMENTED**: `git fetch` of origin is sandbox-blocked; proceeded on
the cached `origin/Henry-branch` == HEAD `a2726e3` (recorded in
`reports/d052_legacy_source_freeze.md`).

## 6. Premise corrections (recorded, not silenced)

- The legacy `one_hot` / `obs_dim 8300` / `32-slot` interface is NOT in current HEAD
  (only `gpu1/conf/training/default.yaml` L59 `conditioning_type:'one_hot'`); it is
  banned from canonical_v2 and the 8335/67 gate is still enforced.
- Legacy selectors re-implemented self-contained (import collision + jax/craftax +
  `/root` paths); exact numeric parity NOT claimed (no parity oracle exists). See
  `reports/d052_canonical_legacy_reuse_map.md` addendum.

## 7. Frozen target labels (§二十八)

See `d052_canonical_frozen_labels.json`. Headline:
`D052_CANONICAL_REFACTOR_IMPLEMENTED=true`, `D052_TEAMMATE_DOCUMENTATION=PASS`,
`D052_LEGACY_ARTIFACTS_FROZEN_UNCHANGED=true`, **`D052_LONG_TRAINING_RUNS=0`**,
`D052_4096_SMOKE_AUTHORIZED=false`, `D052_24576_AUTHORIZED=false`,
`D052_98304_AUTHORIZED=false`.

### 7b. Critic-policy label disambiguation (D052_PREMERGE_SEMANTIC_CLEANUP_V3)

The original label `D052_CRITIC_POLICY=PASS` was ambiguous: it conflated the
**synthetic fixture engineering test** (the canonical selector machinery works
end-to-end over deterministic offline fixtures) with a **frozen real canonical
scientific policy**. It is now `DEPRECATED_AMBIGUOUS_DO_NOT_USE`
(`deprecated=true`, `replacement_fields` listed in the JSON) and no auto-gate
consumes it. The split, scoped labels are:

* `D052_SYNTHETIC_CRITIC_SELECTOR_ENGINEERING=PASS` — engineering layer only;
  must NEVER be promoted to a frozen real policy.
* `REAL_CANONICAL_CRITIC_REJECT_DERIVATION_RULE=UNDECIDED` — dimension A: how
  the canonical `critic_reject` boolean is derived from a judgment (candidates
  `decision_reject` / `flags_too_hard`; owned by `reconciliation/judgment_adapter.py`).
* `REAL_CANONICAL_CRITIC_SELECTION_POLICY=UNDECIDED` — dimension B: how the
  selector consumes the critic signal (candidates `hard_veto` / `soft_penalty` /
  `score_only`; owned by `schemas/selector.py` + `selectors/` — the "critic policy"
  in the `selectors/` row above refers to THIS dimension only).
* `DEFAULT_CRITIC_REJECT_DERIVATION_RULE=NONE`, `DEFAULT_CRITIC_SELECTION_POLICY=NONE`,
  `REAL_CANONICAL_CONVERSION_WITHOUT_CRITIC_RULE=BLOCKED`,
  `REAL_CANONICAL_SELECTION_WITHOUT_CRITIC_POLICY=BLOCKED`.

The two dimensions are independent and must not substitute for each other;
`reconciliation/tier_c_gate.py` fails closed unless BOTH are explicitly frozen.

## 8. Audit artifacts (this directory)

- `d052_canonical_legacy_reuse_map.md/.json` — 85 modules classified (+ selector addendum)
- `d052_legacy_source_freeze.md` — freeze report, baseline, premise corrections
- `d052_legacy_artifact_hashes_before_after.json` — 48 frozen files, all unchanged
- `d052_source_provenance.json` — canonical source provenance (48 files + anchors)
- `d052_artifact_inventory.json` — 70 canonical `d052/` artifacts
- `d052_canonical_artifacts_SHA256SUMS` — verifiable manifest (`sha256sum -c` from repo root)
- `d052_canonical_frozen_labels.json` — final frozen labels + gate statuses
- Frozen raw evidence (director workspace, never committed):
  `audit_outputs/d052_legacy_source_freeze_20260726T060626Z/` (source_inventory.json + SHA256SUMS)

## 9. Guarantees & handoff

- No push to Henry-branch; independent branch only; no `git reset`/`clean`/force push;
  tag `d052-legacy-frozen-20260726` not pushed unreported.
- Zero new dependencies (pydantic/jsonschema/numpy/pytest/pyyaml already present).
- **Next phase** (requires explicit authorization): wire a real training runner into
  `d052/training/adapter.py` behind a `single_cell_training` authorization, then a
  4096 smoke could be authorized — until then every path runs 0 timesteps.
