# Legacy compatibility & isolation

canonical_v2 coexists with frozen legacy D052 artifacts under strict versioned
isolation. New runs default to `canonical_v2`; legacy is read-only and must be
explicitly enabled; nothing new ever writes into legacy territory.

## protocol_version resolution (`d052/legacy/protocol_version.py`)

- A config with **no** `protocol_version` → hard error
  (`MISSING_PROTOCOL_VERSION`). It is never inferred.
- A non-string or any value other than exactly `canonical_v2` / `legacy` → hard
  error (`UNKNOWN_PROTOCOL_VERSION`). Matching is exact, **no case-folding**.
- `legacy` is accepted ONLY with `allow_legacy_d052=True`, and emits a warning;
  legacy is deprecated, read-only, and is NEVER upgraded to canonical_v2.
- `assert_training_permitted(context)` forbids training under legacy
  (`LEGACY_TRAINING_FORBIDDEN`). canonical_v2 attaches a fresh copy of the frozen
  fixed config; legacy attaches none.

## What is frozen (read-only; SHA must not change)

Old candidate pools, judgment caches, selection manifests, training logs,
checkpoints, audit reports (`audit_outputs/d052_readonly_audit_*`,
`reports/d052_readonly_causal_audit.md/.json`), old Modeler shadow Phase 0–2
artifacts, and archived `shared_r0` raw files. Source freeze evidence:
`audit_outputs/d052_legacy_source_freeze_*` (director workspace, SHA256SUMS) +
`reports/d052_legacy_source_freeze.md` (repo). Module-level classification:
`reports/d052_canonical_legacy_reuse_map.md` (85 modules).

## Banned legacy interfaces (hard errors, never silent)

- `obs_dim 8300` / `32-slot` / legacy `one_hot` conditioning — not in current HEAD;
  banned from canonical_v2 (enforced by the execution-mapping gates).
- Salted-hash → achievement mapping and hash-modulo goals — forbidden; targets map
  to the canonical 67 by NAME (case-sensitive), with one explicit audited alias
  (`defeat_orc_soldier` → `defeat_orc_solider`, id 38).
- `unknown` / `default` / `empty` goal — forbidden; these are errors, and training
  on them is impossible.
- Silent provider fallback, silent candidate-drop, silent schema coercion —
  forbidden (`NO_SILENT_FALLBACK`, `NO_SILENT_SCHEMA_COERCION`).

## Reuse vs re-implementation

Legacy selector modules (`aggregation.py` blob `92a7e8b6`, `auction.py` blob
`ec351728`) are re-implemented self-contained in `d052/selectors/` (import
collision + jax/craftax deps + `/root` paths); exact numeric parity is NOT claimed
(see the reuse-map addendum). Everything reused is catalogued with its blob SHA in
the reuse map; everything refactored/deprecated/do-not-use is named there too.

## Branch & artifact discipline

Work happens only on the independent branch `henry/d052-canonical-refactor`; no
push to Henry-branch without explicit authorization; no `git reset` / `clean` /
force push; the `d052-legacy-frozen-20260726` tag is not pushed unreported. New
outputs go to new directories only; existing output directories are never
overwritten (`NO_LEGACY_ARTIFACT_OVERWRITE`).
