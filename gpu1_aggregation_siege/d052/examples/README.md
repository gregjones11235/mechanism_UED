# D052 canonical_v2 examples

Runnable, deterministic, and training-free. Set `PYTHONPATH=gpu1_aggregation_siege`
and run from the repo root:

```bash
export PYTHONPATH=gpu1_aggregation_siege
python -m d052.examples.example_pipeline
```

- `example_pipeline.py` — end-to-end deterministic pipeline: build a shared frozen
  pool → attach normalized role signals → run the unified selector → build an
  execution-mapping certificate for each selected candidate. Runs **0 timesteps**.
- `cells/cell_template_draft.json` — a valid CellSpec that registers as DRAFT.
- `cells/cell_template_blocked.json` — a CellSpec whose `output_dir` points into a
  denied legacy area, so `validate` moves it to BLOCKED (illustrates the gate).

Cell templates are DRAFT/BLOCKED ONLY — no authorized/running/complete templates
are shipped, because this phase performs no training.
