# D052 canonical cell registry (data location)

This directory is the **committed data root** for the canonical_v2 cell registry.
It is deliberately SEPARATE from the code package `gpu1_aggregation_siege/d052/cells/`
(so the package stays importable with no data files present, and code/data don't
collide).

## Layout (created by `d052.cells.CellRegistry`)

```
configs/d052/cells/
  registry.json                 # index: cell_id -> {state, cell_identity_hash, seq}
  cells/<cell_id>/record.json   # full lifecycle record (spec + state + history
                                #   + authorization + prepared_bundle + launch_manifest)
```

## Usage

```bash
export PYTHONPATH=gpu1_aggregation_siege
python -m d052.cells.cli --root gpu1_aggregation_siege/configs/d052/cells \
    register --spec <spec.json> --actor <id>
# ... validate / prepare / authorize / launch / status
```

## Discipline

- `register` refuses an existing `cell_id` (NO_LEGACY_ARTIFACT_OVERWRITE).
- `validate`, `prepare`, `status` NEVER launch training.
- `launch` is authorization-gated: requires state AUTHORIZED + a valid, non-revoked
  authorization whose `cell_identity_hash` matches the current spec. A no-training
  authorization is structurally incapable of running timesteps
  (D052_LONG_TRAINING_RUNS=0 this phase).
- No legacy/frozen artifacts are written here; `output_dir` in a spec is checked
  against `DENY_LEGACY_OUTPUT_PREFIXES`.

This phase commits NO live cells here (training runs = 0). Cell templates
(DRAFT/BLOCKED only) are added under `d052/examples/` in the documentation commit.
