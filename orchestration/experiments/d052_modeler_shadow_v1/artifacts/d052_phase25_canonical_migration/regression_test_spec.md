# D052 Phase 2.5 — canonical_v2 Regression Test Spec

Purpose: guardrails canonical_v2 must pass before any curriculum-selection or training use.
These are RECOMMENDATIONS delivered with the migration bundle; canonical_v2 owns implementation.

## R1. Forbidden target encodings (hard fail)
- **No hash() targets**: reject any candidate whose `target_achievements`/goals are produced by Python
  `hash(...)` (salted or not). Assert goal provenance != "hash_mod_projection".
- **No salted targets**: reject targets that depend on `PYTHONHASHSEED` or any per-process salt.
  NOTE: fixing PYTHONHASHSEED is NOT a fix — it only freezes one arbitrary assignment (see salted_hash_audit.json).
- **No unknown/empty/default goals**: every task goal must resolve to a known canonical Achievement enum;
  reject empty goal lists, "UNKNOWN"/"DEFAULT" sentinels, or indices outside the canonical enum range.

## R2. B/C matched-field check
Given a frozen pool + frozen profile, assert that arms B and C differ in the rendered prompt ONLY by the
appended StudentProfile block:
- `prompt_registry.prompts.B_<role>` and `C_<role>` share an identical prefix (role instruction + raw summary)
  and identical suffix (candidate block + schema); the only inserted text is the profile JSON.
- Assert identical: candidate_order (sorted task_id), anonymized IDs (C001..C032), model/provider per role,
  temperature (=0), timeout, output schema, normalization config, selector config, seed (none), tie-break.

## R3. Judgment replay check
- Re-parse `judgments_B.jsonl` / `judgments_C.jsonl`: 96 records each, full {tutor,critic,x32} coverage per arm.
- Assert each record `parse_status=="ok"`, required score keys numeric, decision in {accept,hold,reject}.
- Assert `judgment_hash_sha256` matches the canonical-JSON hash of the stored judgment (tamper-evidence).
- Assert `role_label_normalized_to` equals the file role; any `role_label_in_raw` deviation must be logged
  (decorative glm echo) and must NOT affect scoring.

## R4. Selector determinism check
- Recompute Soft Copeland from `judgments_{B,C}.jsonl` using `selector_config.json` twice; assert identical
  scores and identical `selection_hash` (byte-for-byte) each run.
- Assert recomputed B selection_hash == `expected_behavior.B_selection_hash` and C == `C_selection_hash`
  (anchors; validates the bundle is self-consistent and replayable).
- Assert no RNG is invoked during scoring/selection (seed is None; aggregation is pure).

## R5. Canonical target semantics (post-repair)
- After canonical_v2 regenerates targets canonically: assert `is_terminal`/`is_success`/`get_reward` act on the
  REAL goals (round-trip: spec target_achievements -> canonical enum -> relevant_achievements is reversible & stable
  across processes WITHOUT relying on PYTHONHASHSEED).
- Assert eval `SUCCESS_MODE` is computable (not "UNDEFINED").

## R6. Profile integrity
- Assert `student_profile.profile_hash_sha256` matches; machine_facts equal the deterministic calculator output
  (recompute from `calculator_source_sha256` script); llm_interpretation is verbatim and alters no machine fact.
