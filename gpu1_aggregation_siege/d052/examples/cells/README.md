# Cell templates (DRAFT / BLOCKED only)

- `cell_template_draft.json` — well-formed; `register` → DRAFT, `validate` →
  VALIDATED, `prepare` → READY.
- `cell_template_blocked.json` — well-formed EXCEPT `output_dir` points into
  `audit_outputs/` (a denied legacy/frozen area); `register` → DRAFT, then
  `validate` → BLOCKED with reason `NO_LEGACY_ARTIFACT_OVERWRITE`.

No authorized/running/complete templates are shipped: this phase performs no
training (`D052_LONG_TRAINING_RUNS=0`). To exercise a full lifecycle, use the
tests (`d052/tests/test_cells_lifecycle.py`) or the flow in
`d052/docs/run_authorized_cell.md` with a no-training authorization.

Note: the `pool_hash` / `selection_hash` placeholders above are valid 64-hex
format so the specs parse; bind a real cell to an actual shared frozen pool and an
actual `SelectionResult` before registering anything you intend to advance.
