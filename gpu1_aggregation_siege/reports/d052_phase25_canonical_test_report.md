# D052 Phase 2.5 — canonical_v2 Test Report

**Result: 283 passed, 0 failed, 0 errors.**
221 original canonical_v2 tests (all still green — gate 6) + 62 new Phase-2.5
counterfactual tests (gate 7). Runner: pytest 9.1.1, Python 3.12.4.
Branch `henry/d052-canonical-refactor` · baseline HEAD `9eca2de`.

```
============================= 283 passed =============================
```

---

## 1. Coverage by integration gate

### Gate 1 — B/C identical except StudentProfile/Modeler conditioning
`d052/tests/test_counterfactual_matched_bc.py` (13 tests)

Drives `verify_matched_bc` against the canonical ablation arms
(`modeler_ablation_arms` → B=S1_THREE_ROLE, C=S2_FOUR_ROLE_MODELER):

* a correctly-matched pair PASSES; `verification_hash` is content-bound;
* the six PERMITTED deltas (`prompt_set_hash`, `modeler_enabled`,
  `student_profile_hash`, `modeler_context_hash`, selector S1→S2, arm label) are
  confirmed to actually differ while every shared binding stays identical;
* each non-permitted difference fails closed with a SPECIFIC code:
  `POOL_MISMATCH`, `CACHE_MISMATCH`, `WEIGHT_MISMATCH`, `K_MISMATCH`,
  `SEED_MISMATCH`, `CRITIC_POLICY_MISMATCH`, `ROLES_MISMATCH`,
  `B_NOT_MODELER_OFF`, `C_NOT_MODELER_ON`, `WRONG_ABLATION_SELECTORS`,
  `NOT_ARM_B` (swapped arms).

### Gate 2 — bit-identical `selection_hash` replay
`d052/tests/test_counterfactual_replay.py` (5 tests)

* `cache_hash` is insertion-order-insensitive (forward vs reversed judgments);
* bit-identical `selection_hash` under judgment-order shuffle AND under
  signal-order shuffle;
* the hash binds the seed (seed 7 ≠ seed 8);
* exact 3× replay from a freshly-rebuilt cache yields ONE unique `selection_hash`.

(Reinforced at integration level by `test_recomputation_is_bit_identical`.)

### Gate 3 — canonical-target firewall (salted-hash prohibition regression)
`d052/tests/test_counterfactual_firewall.py` (26 tests)

* parametrized classification of every banned scheme: **salted** targets
  (`::salt=`, `#hex`, `sha256:`, trailing 32-hex, `::hex`), **hash-modulo**
  outputs (raw integer id, `target_N`, `id_N`, `ach_N`, `goal_N`, `hash(name)%67`,
  `_mod67`, `0xhex`), plus **empty** / **non-string** / **unknown**;
* `assert_target_firewall` raises each SPECIFIC code (salted wins over unknown in a
  mixed set — never a silent generic pass);
* legal canonical names AND the single audited alias
  (`defeat_orc_soldier` → `defeat_orc_soliter`, id 38) resolve;
* the execution-mapping boundary rejects each banned class
  (`assert_execution_mapping_rejects`);
* the shared-frozen pool build itself rejects illegal raw targets;
* the regression guard is proven NON-silent (raises, does not quietly pass).

### Gate 4 — selected candidates pass the official-67 mapping + certificate
`d052/tests/test_counterfactual_integration.py`

* `test_all_certificates_pass_official_67_mapping`: all 16 certificates (8 B + 8 C)
  are `executed_as_intended=True`, `goal_vector_dim==67`, `student_obs_dim==8335`,
  `conditioning_type=="achievement_multi_hot"`, and all 6 required sub-gates true
  (`target_is_canonical`, `goal_vector_dim_67`, `goal_vector_index_aligned`,
  `student_obs_dim_8335`, `no_silent_fallback`, `task_compiled`).

### Gates 7 + 8 — integration suite (the rest of the integration file)
`d052/tests/test_counterfactual_integration.py` (12 tests)

* matched protocol holds end-to-end (B modeler-off, C modeler-on; B has no
  student_profile_hash, C carries it);
* selected-8 sizes and uniqueness for both arms;
* the modeler contrast is LIVE and accountable: change ≥ 1, `changed_in==["cand_08"]`,
  and the modeler bonus is non-zero ONLY on the flagged candidate;
* recomputation is bit-identical (manifest + both selection hashes + selected ids);
* modeler firewall: the modeler context carries NO tier labels
  (`assert_modeler_firewall` + attestation fields);
* firewall attestation recorded on the manifest (`canonical_target_firewall==PASS`);
* two DRAFT cells register legally (`state==DRAFT`, `intended_total_timesteps==0`,
  64-char identity hash; on-cell selection hashes match the live selections);
* zero training: `training_timesteps==0`, `timesteps_run==0`,
  `D052_LONG_TRAINING_RUNS==0`;
* no-overwrite discipline: re-registering cells and re-emitting artifacts both
  refuse rather than clobber.

## 2. Per-file tally

| Test file | Tests | Gate(s) |
|---|---|---|
| `test_counterfactual_firewall.py` | 26 | 3 |
| `test_counterfactual_matched_bc.py` | 13 | 1 |
| `test_counterfactual_replay.py` | 5 | 2 |
| `test_counterfactual_integration.py` | 18 | 4, 7, 8 (+1,2,3) |
| **Phase-2.5 subtotal** | **62** | |
| original canonical_v2 suite | 221 | 6 |
| **TOTAL** | **283** | **all** |

## 3. Gate status (test evidence)

| # | Gate | Evidence | Status |
|---|---|---|---|
| 1 | B/C identical except conditioning | matched_bc suite | **PASS** |
| 2 | bit-identical replay | replay suite + recomputation test | **PASS** |
| 3 | salted/unknown/empty/hash target → mapping fails | firewall suite (26) | **PASS** |
| 4 | selected pass official-67 + certificate | certificates test (16/16) | **PASS** |
| 5 | legacy artifacts SHA unchanged | reverify JSON (47/48 + 1 recorded removal) | **PASS** |
| 6 | original 221 tests pass | full suite run | **PASS** |
| 7 | new integration tests pass | 62 new tests | **PASS** |
| 8 | training steps == 0 | zero-training tests + no cell launched | **PASS** |

**No test lowers any hard gate. No silent fallback, no silent schema coercion, no
legacy-artifact overwrite. Training runs this phase: 0.**
