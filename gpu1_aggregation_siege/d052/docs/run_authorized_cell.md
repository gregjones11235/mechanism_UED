# How to run an authorized cell

A READY cell only runs after explicit per-cell authorization. **NO_UNAUTHORIZED_
TRAINING**: `launch` requires state `AUTHORIZED` plus a valid, non-revoked
authorization whose `cell_identity_hash` matches the current spec. This phase issues
only `single_cell_no_training` authorizations, so a launch records intent and runs
**0 timesteps** (`D052_LONG_TRAINING_RUNS=0`).

## 1. Authorize

The authorization binds to the cell's CURRENT identity hash and to its exact
`intended_total_timesteps`:

```python
from d052.cells import CellRegistry, make_authorization, SCOPE_NO_TRAINING
reg = CellRegistry("gpu1_aggregation_siege/configs/d052/cells")
st = reg.status("c_2026_07_26_001")          # must be READY
spec_hash = st["cell_identity_hash"]
auth = make_authorization(
    cell_id="c_2026_07_26_001",
    cell_identity_hash=spec_hash,
    authorized_by="human-owner",
    scope=SCOPE_NO_TRAINING,                 # this phase: no training
    granted_total_timesteps=4096)            # == spec.intended_total_timesteps
reg.authorize("c_2026_07_26_001", auth, actor="CC3")   # -> AUTHORIZED
```

CLI:

```bash
python -m d052.cells.cli --root $ROOT authorize \
    --cell-id c_2026_07_26_001 --authorization auth.json --actor CC3
```

Authorization is refused (`AUTHORIZATION_MISMATCH`) if the cell_id differs, the
identity hash no longer matches (spec edited after grant), or the granted timesteps
differ from `intended_total_timesteps`; a revoked authorization is refused
(`REVOKED_AUTHORIZATION`).

## 2. Launch (gated)

```bash
python -m d052.cells.cli --root $ROOT launch --cell-id c_2026_07_26_001 --actor CC3
```

`launch` transitions `AUTHORIZED -> RUNNING -> COMPLETE` and writes a
`launch_manifest` (`authorization_hash`, `scope`, `runner_artifact`,
`timesteps_run`).

## The no-training gate (structural)

Under a `single_cell_no_training` authorization the registry **forces** the no-op
runner regardless of any runner supplied, and any non-zero `timesteps_run` FAILs
the cell (`NO_TRAINING_VIOLATION`) instead of completing it. So a no-training
authorization is structurally incapable of training. A `single_cell_training`
authorization selects the supplied runner — but the canonical training adapter
REFUSES that scope this phase (`NOT_IMPLEMENTED`), so real training still cannot
start. Both paths keep `timesteps_run == 0` until a later, explicitly-authorized
phase wires in a real runner.

## 3. Verify

```bash
python -m d052.cells.cli --root $ROOT status --cell-id c_2026_07_26_001
# state=COMPLETE, launched=true, timesteps_run=0
```

A complete, audited cell carries: the immutable spec + identity hash, the full
state history (seq, from→to, actor, reason), the authorization, and the launch
manifest — an end-to-end evidence chain from candidate to (intended) execution.
