# Artifact Manifest

The machine-readable inventory is in `inventory/henry_experiment_inventory.json`; SHA256 rows for files are in `MANIFEST.sha256`.

Each archived phase directory contains a `README.md` and `raw_sources/` subtree preserving the remote source path layout for small source/config/report/evaluation files.

Excluded by policy:

- checkpoint entities and Orbax checkpoint payloads
- replay buffer binaries
- large binary snapshots
- JAX/Python caches
- conda environments
- WandB directories
- SSH keys, API tokens, GitHub tokens, passwords, `known_hosts`
- large zip files
- raw long logs and core dumps
- temporary compiled Python files
- single files larger than 5MB

Additional cleanup:

- On 2026-07-26, D052 historical data files under `01_d052/` with extensions `.json`, `.jsonl`, `.csv`, and `.log` were removed by request.
- The retained D052 content is engineering code and lightweight documentation only.
- Removed D052 paths are listed in `inventory/d052_data_removed_by_request.txt`.
