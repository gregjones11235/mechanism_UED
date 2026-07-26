# D052 Legacy Source Freeze

**Purpose.** Anchor the *before-state* of the D052 source so the in-place canonical
refactor (`D052_CANONICAL_IN_PLACE_REFACTOR`) is fully auditable, and record the
read-only freeze of legacy experiment artifacts. This is deliverable **[1/9]**
(task §三 / §二十七).

## Baseline

| Field | Value |
|---|---|
| Repo | `gregjones11235/mechanism_UED` (local worktree) |
| Branch | `henry/d052-canonical-refactor` (independent branch; **not** Henry-branch) |
| Baseline commit | `a2726e3ea75feff2b475b1e3408c30ef3f9acd7a` |
| Baseline == cached `origin/Henry-branch` HEAD | yes (fetch blocked by sandbox network; pushed disallowed by task anyway) |
| Freeze timestamp (UTC) | `20260726T060626Z` |
| Working tree at freeze | clean (`git status --porcelain` empty) |

## What is frozen, and where

### 1. Legacy SOURCE provenance (this deliverable)

Source code **may be refactored** (explicit task permission). To keep that
refactor auditable, the D052-relevant source files are snapshotted at the
baseline:

- Inventory: `audit_outputs/d052_legacy_source_freeze_20260726T060626Z/source_inventory.json`
  — 48 files, each with `git_blob_sha1` (`git rev-parse HEAD:<path>`) AND an
  independent `sha256` over the raw bytes, plus role / reuse / evidence notes.
- Independent check file: `audit_outputs/d052_legacy_source_freeze_20260726T060626Z/SHA256SUMS`.

**How a teammate verifies it (no git needed):**

```bash
cd <repo_root>          # gpu1_aggregation_siege's parent (the mechanism_UED checkout)
sha256sum -c audit_outputs/d052_legacy_source_freeze_20260726T060626Z/SHA256SUMS
# -> all 48 lines ": OK"
```

> The freeze inventory lives in the **director workspace** (`D:/Projects/dicode-codex-director/audit_outputs/…`),
> co-located with the Phase-1 read-only audit, on purpose: it is *frozen evidence*,
> not a mutable committed file. The human-readable reports and the `d052/` code
> live in the repo branch (committed). This split is the explicit location choice
> required by the task.

### 2. Legacy experiment ARTIFACTS (read-only discipline)

The following are **read-only frozen** and must never be overwritten, re-derived,
or silently "fixed" (iron rule `NO_LEGACY_ARTIFACT_OVERWRITE`):

- old candidate pools (`frozen_candidate_pool.json`)
- old judgment caches (`judgment_cache.jsonl`)
- old selected manifests (`d052_manifest.json`, `round_N_manifest.json`, cell `manifest.json`)
- old training logs and checkpoints
- `audit_outputs/d052_readonly_audit_20260726T043613Z/` (Phase-1 audit)
- `reports/d052_readonly_causal_audit.md` / `.json` (Phase-1 reports)
- old Modeler-shadow Phase 0–2 raw artifacts (none carry that naming in-repo — see finding 5 below)
- any archived `shared_r0` raw files

**Where these actually are.** Per the Phase-1 audit
(`d052_missing_raw_data.json`) and the in-repo
`experiments/henry_dicode_student_upgrade/inventory/d052_data_removed_by_request.txt`
(blob `5dcde66`, 527 paths: 338 `.json`, 181 `.jsonl`, 5 `.log`, 3 `.csv`), the
D052 **data** was purged by request and the training logs/checkpoints are
server-only (not pulled). What remains in this repo is **code** (`01_d052/raw_sources/…`,
classified `DO_NOT_USE`) plus `MANIFEST.sha256` rows that now reference deleted
files (ghost entries). Consequences recorded honestly:

- There are no local artifact bytes to hash *yet*. The `[9/9]` final audit will
  emit `legacy_artifact_hashes_before.json` / `legacy_artifact_hashes_after.json`
  over whatever legacy artifact files ARE present (code + manifests + inventory),
  proving they are byte-stable across the refactor.
- The `MANIFEST.sha256` ghost rows are documented as a known data-integrity flag,
  not silently repaired.

## Iron rules enforced by this refactor

| Rule | Value | Where enforced |
|---|---|---|
| `NO_RAW_DATA_NO_STRONG_CLAIM` | true | reports cite raw artifacts only; no summary-as-primary-evidence |
| `NO_SILENT_FALLBACK` | true | `fallback_policy=error`; selectors never backfill / reduce k / re-LLM |
| `NO_SILENT_SCHEMA_COERCION` | true | `protocol_version` gate rejects wrong case/type; candidate validator errors on illegal targets |
| `NO_LEGACY_ARTIFACT_OVERWRITE` | true | new runs write to new dirs; `[9/9]` hashes artifacts before/after |
| `NO_UNAUTHORIZED_TRAINING` | true | per-cell authorization gate; this phase runs **zero** training |

## Stop-condition status at freeze (task §二十六)

| # | Condition | Status |
|---|---|---|
| 1 | Cannot determine canonical-67 single source | **NOT triggered** — `dicode_src/auction/craftax_achievements.py` (blob `5bb881a6`), zero-dep, `assert ==67`, verified vs craftax `main`+`v1.4.5` |
| 2 | Cannot confirm goal-conditioning order | **NOT triggered** — canonical_id == goal_vector_index == enum `.value` (`task_utils.py:29,33`) |
| 3 | Cannot confirm obs_dim=8335 composition | **NOT triggered** — `multitask.py` `obs_dim=base+embedding_size`; in-repo `assert obs_dim==8335 and EMB==67`. See premise correction below. |
| 14 | Source tree: old D052 not at current HEAD / origin unverifiable | **PARTIAL** — code present at HEAD; origin fetch blocked by sandbox (documented). Proceeding on cached `origin/Henry-branch`==HEAD. |

All other stop-conditions are guarded by the gates built in Commits 2–9.

## Premise correction (NO_SILENT_ASSUMPTION)

The task brief (§一.6) references a legacy "32-slot task encoding / obs_dim=8300"
training path. **This is not present anywhere in current HEAD** (the only
conditioning field is `gpu1/conf/training/default.yaml` L59 `conditioning_type:"one_hot"`,
and no `8300` literal exists — prior-audit `8300` hits were hash false positives /
the voided `d052_unified_eval` conditioning). The canonical target
**obs_dim=8335 with a 67-dim achievement multi-hot IS confirmed** and is the right
target. We therefore (a) record this correction rather than assume it, and (b)
**still enforce** the `obs_dim=8335 / dim=67 / no-32-slot` hard gate to prevent
reintroduction. The exact +9 bonus scalars inside the 8268 base are not fully
verifiable in-repo (`craftax` not installed) — known limitation.

## Package-topology decision

Four packages all import as `dicode` (collision): `dicode_src`, `dicode_v6`,
`gpu0_training_mechanisms`, `gpu1_aggregation_siege`. The canonical framework is
built at **`gpu1_aggregation_siege/d052/`** (top-level, separate from
`src/dicode/`) because `gpu1_aggregation_siege` is the strict superset (carries
`mechanisms/auction.py`, `siege/production_dispatcher.py`, `scripts/`, `tests/`).
Full per-module disposition: `reports/d052_canonical_legacy_reuse_map.md` /
`.json` (85 modules classified).

## Scope guarantee for this commit

- No legacy artifact modified (source freeze is a separate read-only snapshot in
  the director workspace; no tracked legacy file edited).
- No training launched. No push. No `git reset/clean/force push`.
- New code only under `gpu1_aggregation_siege/d052/` and `gpu1_aggregation_siege/reports/`.
