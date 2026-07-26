# How to register a cell

A cell is the unit of per-cell authorization & launch. Registering a cell only
creates a DRAFT record — it never validates, prepares, or launches.

## Lifecycle

```
DRAFT -> VALIDATED -> READY -> AUTHORIZED -> RUNNING -> COMPLETE
                         (BLOCKED / FAILED on gate failure; FAILED & COMPLETE terminal)
```

## 1. Write a CellSpec

```json
{
  "cell_id": "c_2026_07_26_001",
  "protocol_version": "canonical_v2",
  "hypothesis": "Soft-Copeland top-2 improves held-out SR on collect_wood/eat_cow",
  "pool_id": "shared_pool_r0",
  "pool_hash": "<64-hex pool_hash of the shared frozen pool>",
  "selector": {
    "selector": "SOFT_COPELAND",
    "critic_policy": "hard_veto",
    "k": 2,
    "seed": 1234
  },
  "candidate_ids": ["t_a", "t_b"],
  "selection_hash": "<64-hex selection_hash from the SelectionResult>",
  "intended_total_timesteps": 4096,
  "output_dir": "runs/c_2026_07_26_001",
  "created_by": "CC3"
}
```

`protocol_version` MUST be `canonical_v2` (missing/unknown/legacy → hard error).
The canonical pins (`achievement_schema`, `conditioning_type`,
`conditioning_dimension=67`, `student_obs_dim=8335`, `candidate_pool_mode`,
`score_normalization`) default to the frozen values and are enforced as Literals.
`candidate_ids` MUST be the exact `selected_ids` of the bound `SelectionResult`.

## 2. Register / validate / prepare

CLI (`prepare`/`validate`/`status` NEVER launch):

```bash
export PYTHONPATH=gpu1_aggregation_siege
ROOT=gpu1_aggregation_siege/configs/d052/cells
python -m d052.cells.cli --root $ROOT register  --spec spec.json --actor CC3
python -m d052.cells.cli --root $ROOT validate  --cell-id c_2026_07_26_001 --actor CC3
python -m d052.cells.cli --root $ROOT prepare   --cell-id c_2026_07_26_001 --actor CC3
python -m d052.cells.cli --root $ROOT status    --cell-id c_2026_07_26_001
```

Or the Python API:

```python
from d052.cells import CellRegistry, CellSpec
reg = CellRegistry("gpu1_aggregation_siege/configs/d052/cells")
spec = CellSpec.model_validate_json(open("spec.json").read())
reg.register(spec, actor="CC3")          # -> DRAFT (refuses an existing cell_id)
reg.validate_cell(spec.cell_id, actor="CC3")  # -> VALIDATED, or BLOCKED with reason
reg.prepare(spec.cell_id, actor="CC3")        # -> READY (launch bundle, not launched)
```

## Validation checks (validate_cell)

- candidate_ids bound (non-empty, unique);
- `environment_version == craftax==1.4.5`;
- `output_dir` has no path traversal and does NOT write into a legacy/frozen area
  (`DENY_LEGACY_OUTPUT_PREFIXES`: `/root/`, `audit_outputs/`,
  `reports/d052_readonly`, `experiments/`, `checkpoints_legacy/`).

Any problem → state `BLOCKED` with a `block_reason`. Fix by registering a NEW cell
(a FAILED/BLOCKED record is preserved as evidence and never mutated).

## Identity

`spec.identity_hash()` is a sha256 over ~20 canonical fields (state excluded).
Authorization binds to THIS hash; editing the spec after authorization voids the
authorization (see `run_authorized_cell.md`).
