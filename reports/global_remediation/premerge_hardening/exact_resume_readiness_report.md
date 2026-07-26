# Exact Resume readiness matrix (CC4 premerge hardening -- seven)

- UTC: `2026-07-26T13:31:49Z`
- **GLOBAL_EXACT_RESUME_HARNESS = READY**
- **REAL_EXACT_RESUME_EXECUTION = NOT_RUN**
- **EXACT_RESUME_BITEXACT = NOT_CLAIMED (self-test PASS != real bit-exact pass)**
- Archive census: manifest.json=95, *.pkl=0, full_state=0, train_state binary=0 => NO restorable checkpoint binary state locally.
- **Key rule**: manifest existence does NOT imply READY; with 0 binaries, no READY is assigned on manifest presence.

## Coverage legend
- `DECLARED`: present in checkpoint manifest/schema (declared); binary NOT verified
- `METADATA_ONLY`: only a hash / metadata field, not the actual state
- `MISSING`: required component not saved / not evidenced
- `UNCERTAIN`: not explicitly declared; cannot confirm from synced manifests
- `NA`: not applicable to this line
- `BINARY_NONE`: no restorable binary in local archive (0 pkl / 0 full_state)
- `SAVED_ON_SERVER_NOT_LOCAL`: full_state.pkl saved on server (roundtrip_ok) but absent locally

## Matrix (15 state dimensions)

| experiment | overall | params | optimizer | global_step | update_index | rng | env_state | replay_buffer | replay_rng | memory_state | pending_episodes | scheduler | scaler | iterator_state | metadata | full_state_binary_availability |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P2 | **PARTIAL** | DECLARED | DECLARED | DECLARED | DECLARED | DECLARED | UNCERTAIN | DECLARED | UNCERTAIN | UNCERTAIN | DECLARED | DECLARED | MISSING | UNCERTAIN | DECLARED | BINARY_NONE |
| W512 | **BLOCKED_MISSING_STATE** | METADATA_ONLY | MISSING | DECLARED | MISSING | MISSING | MISSING | NA | NA | MISSING | NA | MISSING | MISSING | MISSING | DECLARED | BINARY_NONE |
| RMT16 | **BLOCKED_MISSING_STATE** | METADATA_ONLY | MISSING | DECLARED | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | DECLARED | BINARY_NONE |
| D052 | **NOT_APPLICABLE** | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| REFERENCE_LC_family | **PARTIAL** | DECLARED | UNCERTAIN | DECLARED | DECLARED | UNCERTAIN | UNCERTAIN | NA | NA | UNCERTAIN | NA | UNCERTAIN | UNCERTAIN | UNCERTAIN | DECLARED | SAVED_ON_SERVER_NOT_LOCAL |

## Per-line notes
- **P2 (PARTIAL)**: Richest declared schema (p2_full_a_pure_pickle_v1): params/optimizer(EMA target)/action_rng/collector/pending_episodes(178 anchors/22115 transitions)/replay_buffer(size64)/global_step=98304/update_count=47. BUT 0 .pkl synced locally and no A/B continuation comparator log => cannot verify bit-exact continuation.
- **W512 (BLOCKED_MISSING_STATE)**: Disk checkpoint dir holds ONLY manifest.json (319B): params_sha256/step/seed/lr/update_epochs/num_minibatches/w512_long_size=384. NO params binary, optimizer, env_state, jax_rng, or w512_extra_state (long_buf/long_mask/delay line). GATE14 => nearly all required components missing.
- **RMT16 (BLOCKED_MISSING_STATE)**: Manifest-only (params_sha256/step/seed/lr/rmt_num_tokens=16). Schema records train_state.pkl MISSING env_state, no restore path, gate7/gate11 NOT_FOUND; rmt_extra_state (mem_tokens/seg_buf/seg_count) not demonstrably carried.
- **D052 (NOT_APPLICABLE)**: D052 is CC3's active canonical refactor (READ-ONLY for CC4); NO D052 checkpoint/run in this archive. The harness + exact_resume_schema.json are forward-applicable to D052 (same S4_dark PPO+GTrXL family).
- **REFERENCE_LC_family (PARTIAL)**: Strongest REAL on-server evidence: A/B continuation (A 0->8192; B1 0->4096 save; B2 resume 4096->8192), RESULT=EXACT_RESUME_PASS for all four LC variants. BUT the archived comparator (compare_resume.py) asserts params_sha equality + update_count only; it does NOT leaf-by-leaf assert optimizer_state/jax_rng/env_state/gtrxl_memory. full_state.pkl saved (roundtrip_ok) on server but NOT in local archive (0 pkl). => params-level bit-exact demonstrated; canonical FULL-state bit-exact NOT asserted.

## Two-level distinction
- CHECKPOINT_SAVE_VALID: save->load roundtrip yields usable state (necessary NOT sufficient). LC full_state.pkl roundtrip_ok=True demonstrates this level.
- EXACT_RESUME_BITEXACT: continuation A@4096==B1 & A@8192==B2 over FULL state. Only params-level demonstrated for LC; full-state NOT asserted; NOT_RUN for P2/W512/RMT16/P8/P9.

## Conclusions
- P2 = PARTIAL (declared near-complete schema, but binary missing + no A/B continuation log).
- W512 = BLOCKED_MISSING_STATE (manifest-only; no params/optimizer/env/rng/w512_extra_state).
- RMT16 = BLOCKED_MISSING_STATE (manifest-only; env_state + rmt_extra_state + replay/rng missing).
- D052 = NOT_APPLICABLE (CC3 read-only; no checkpoint in archive; harness/schema forward-applicable).
- REFERENCE_LC_family = PARTIAL (params-level bit-exact demonstrated on server; full-state NOT asserted; full_state.pkl not local).
- No experiment is READY; EXACT_RESUME_BITEXACT stays NOT_CLAIMED.

## Discipline
- no real resume run this round
- self-test PASS != real bit-exact PASS
- MISSING/BLOCKED never relabeled FAIL
- manifest presence not upgraded to READY
