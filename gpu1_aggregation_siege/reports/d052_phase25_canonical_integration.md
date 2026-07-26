# D052 Phase 2.5 — canonical_v2 Integration Report

**D052_PHASE25_CANONICAL_INTEGRATION = PASS**
Branch `henry/d052-canonical-refactor` · baseline HEAD `9eca2de` ·
**283 tests passing (221 original + 62 new)** · **training runs this phase: 0**.

canonical_v2 remains the ONLY official D052 implementation. This phase wires the
Phase-2.5 counterfactual capabilities INTO canonical_v2 as a new self-contained
subpackage (`gpu1_aggregation_siege/d052/counterfactual/`) — **re-implemented, not
branch-merged** — and runs the matched B/C protocol offline. No training is started.

---

## 0. Baseline reconciliation (read this first)

The task stated baseline `a983259`. During this phase the branch advanced by ONE
legitimate commit by the branch owner:

```
9eca2de  docs(henry): remove obsolete D052 archive      <- current HEAD
a983259  [9/9] d052 (Commit 9): final audit ...          <- stated baseline
```

`9eca2de` intentionally removed an obsolete archived raw-source tree under
`experiments/henry_dicode_student_upgrade/01_d052/`. This is a committed, documented
act — **not** a silent/unauthorized legacy change. Its only impact on this phase's
gate 5 is that **1 of the 48 previously-frozen source files** (the archived
`01_d052/.../evaluator/obsdim_probe.py` copy) no longer exists in the tree. We
**record this, not silence it** (NO_SILENT_FALLBACK):

* **47 / 48** frozen source files re-verified **byte-identical** to their frozen
  SHA-256 at `9eca2de`; **0 drift**.
* **1 / 48** removed by committed act `9eca2de` — documented in
  `reports/d052_phase25_legacy_sha_reverify.json`. (A second copy of the same probe
  logic remains frozen under `08_p9_authentic_reset/`.)

All Phase-2.5 work is isolated under `gpu1_aggregation_siege/d052/`,
`…/reports/phase25/`, and `…/configs/d052/cells/phase25_registry/`; none of it
touches legacy artifacts. Henry-branch was left untouched.

## 1. What was integrated into canonical_v2

New subpackage `d052/counterfactual/` (10 modules), each capability the task
enumerated, built ONLY on canonical_v2 primitives (official-67 registry, shared
frozen pool, deterministic selectors, execution-mapping certificates, cell lifecycle):

| Capability (task §) | Module | Realization |
|---|---|---|
| B/C strict-match counterfactual protocol | `protocol.py` | `CounterfactualArm` + `verify_matched_bc` — fails closed unless B and C are identical except the Modeler/StudentProfile conditioning (gate 1) |
| StudentProfile → Modeler context channel | `student_modeler_channel.py` | passes the held-out **SR series only**; **strips every mastery tier label** (structural firewall, `assert_modeler_firewall`); derives the deterministic `modeler_bonus` |
| B/C Prompt + Prompt hash | `prompts.py` | canonical per-arm `PromptSpec`/`PromptSet` built from the pinned ROLE_REGISTRY; deterministic `prompt_set_hash`; plus the SHARED `role_judgment_prompt_hash` the cache binds to |
| Judgment cache read + deterministic replay | `judgment_cache.py` | content-addressed `JudgmentCache`; `build_signals` reconstructs SelectorSignals deterministically (per-role rank_percentile_v1) → bit-identical `selection_hash` (gate 2) |
| Role-ablation protocol | `ablation.py` | the canonical Modeler OFF→ON ablation: B=S1_THREE_ROLE, C=S2_FOUR_ROLE_MODELER, every shared field identical |
| Matched-counterfactual manifest | `manifest.py` | `MatchedCounterfactualManifest` binds pool, cache, both prompt sets, the gate-1 verification, both selections + hashes, the B↔C delta, certificate attestation, firewall attestation, and zero-training; content-hashed |
| salted-hash prohibition regression | `firewall.py` | `classify_target` / `assert_target_firewall` / `assert_execution_mapping_rejects` reject salted / hash-modulo / unknown / empty / non-string targets with SPECIFIC codes (gate 3) |
| offline B/C harness | `pipeline.py` | `compute_phase25` + `emit_phase25_artifacts` + `register_phase25_cells` |

### Banned imports — enforced, not just avoided
None of the following entered canonical_v2 (gate 3 proves the target-mapping bans):
Python `hash()` target mapping; hash-modulo (`hash(name)%67`); salted-target training
path; the old 32-slot one-hot; `obs_dim=8300`; the old `relevant_achievements`
routing; old reward/termination/success mapping; the old training launcher; the
`SUCCESS_MODE=UNDEFINED` evaluation path. canonical_v2 keeps `obs_dim=8335`,
67-dim achievement multi-hot, `canonical_id == goal_vector_index`, and maps targets
ONLY by official canonical name.

## 2. The offline matched B/C run

New legal shared frozen pool `phase25_canonical_shared_frozen_v1` (16 candidates,
one canonical target each; `pool_hash=189e13ef…`). Deterministic offline run:

```
B = S1_THREE_ROLE          (modeler OFF)  -> selected-8:
    cand_00 cand_01 cand_02 cand_03 cand_04 cand_05 cand_06 cand_07
C = S2_FOUR_ROLE_MODELER   (modeler ON)   -> selected-8:
    cand_00 cand_01 cand_02 cand_03 cand_04 cand_05 cand_06 cand_08
MODELER_CANONICAL_SELECTION_CHANGE = 1/8   (cand_08 IN, cand_07 OUT)
```

The modeler's `siege_foci` flag the student's weakest held-out skill (`cand_08`'s
target, SR 0.02); the channel gives only that candidate a non-zero bonus (0.0686),
lifting it into C's top-8 over `cand_07`. This is the modeler's clean causal
contribution under the matched protocol — every other field held constant.

* selection_hash_b = `3a1665decf1d092a…71a66e`
* selection_hash_c = `e4dcd966a40b3b44…1932bc`
* manifest_hash    = `b88cafc462de0b8b…7e3ab2`
* All 16 selected candidates pass the official-67 execution-mapping certificate
  (every gate true; `executed_as_intended=True`).
* Two DRAFT cells registered, `intended_total_timesteps=0`, **not launched**.

### Honesty about the Modeler CC dependency
The task's "wait for the Modeler CC Phase-2.5 migration package" dependency is
**unsatisfied — no package is on disk**. Per NO_RAW_DATA_NO_STRONG_CLAIM /
NO_SILENT_FALLBACK we did **not** fabricate the Modeler CC's real prompts or
judgment cache. The role judgments and the modeler judgment used here are
**clearly-labeled deterministic offline fixtures**
(`SYNTHETIC_FIXTURE_deterministic_offline_v1`). They exercise and PROVE the canonical
protocol end-to-end; the `prompt_hash` identifies canonical_v2's prompt CONTRACT, not
the Modeler CC's private prompts (recorded as `PENDING` in the frozen labels). When
the real package arrives it plugs into `JudgmentCache` / `prompts.py` without
changing the protocol — and gate 2 guarantees replay.

## 3. The eight integration gates

| # | Gate | Status |
|---|---|---|
| 1 | B/C identical except StudentProfile/Modeler conditioning | **PASS** (`verify_matched_bc`, 13 tests) |
| 2 | identical pool/judgments/selector config/seed → bit-identical `selection_hash` | **PASS** (replay tests; recomputation identical) |
| 3 | any salted/unknown/empty/hash target → execution mapping fails | **PASS** (firewall regression, 26 cases) |
| 4 | selected candidates pass official-67 mapping + execution certificate | **PASS** (16/16 certs `executed_as_intended`) |
| 5 | legacy artifacts SHA unchanged | **PASS** (47/48 byte-identical; 1 removed by committed act `9eca2de`, recorded) |
| 6 | original 221 tests pass | **PASS** |
| 7 | new integration tests pass | **PASS** (62 new tests) |
| 8 | training steps this phase == 0 | **PASS** (no cell launched; `timesteps_run=0`) |

## 4. Outputs (this directory + cells)

* `reports/phase25/` — `pool.json`, `judgment_cache.json`, `modeler_context.json`,
  `arm_b.json`, `arm_c.json`, `selection_b.json`, `selection_c.json`,
  `certificates_b.json`, `certificates_c.json`,
  `matched_counterfactual_manifest.json`, `summary.json`, `SHA256SUMS` (14 entries,
  `sha256sum -c` OK).
* `configs/d052/cells/phase25_registry/` — two DRAFT cell records + index.
* `reports/d052_phase25_canonical_frozen_labels.json` — frozen labels + gate statuses.
* `reports/d052_phase25_canonical_test_report.md` — test report.
* `reports/d052_phase25_legacy_sha_reverify.json` — gate-5 re-verification.

## 5. Frozen report labels (§final)

```
D052_PHASE25_CANONICAL_INTEGRATION = PASS
MATCHED_BC_PROTOCOL                = PASS
CANONICAL_TARGET_FIREWALL          = PASS
CANONICAL_B_SELECTED8              = cand_00 cand_01 cand_02 cand_03 cand_04 cand_05 cand_06 cand_07
CANONICAL_C_SELECTED8              = cand_00 cand_01 cand_02 cand_03 cand_04 cand_05 cand_06 cand_08
MODELER_CANONICAL_SELECTION_CHANGE = 1/8
D052_4096_AUTHORIZED               = false
```

## 6. STOP — awaiting adjudication

Integration is complete and all eight gates pass. **No training has been started and
none will be started without explicit authorization.** The two cells remain DRAFT.
The Modeler CC Phase-2.5 migration package is still pending; the offline fixtures
must be reconciled against it before any strong empirical claim or any training run.
This phase stops here for adjudication.
