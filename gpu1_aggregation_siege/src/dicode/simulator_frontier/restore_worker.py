"""Fresh-process restore worker (spawned EXACTLY ONCE by the parent driver).

This module is the child-process entry point of the R4c production gate
(``python -m dicode.simulator_frontier.restore_worker --request R --evidence E``).
Running it in the parent process is never admissible: the parent driver
(``fresh_process_restore.run_fresh_process_restore_production``) rejects any
evidence whose child PID equals the driver PID, and requires the evidence
parent PID to match either the driver PID or the PID this call launched
(direct spawn, or the Windows venv launcher as documented intermediate).

Loader/replay resolution is BUNDLE-DRIVEN ONLY: the child re-verifies that
the request is hash-bound to its ``ProductionRegistryBundle`` (and that the
bundle's Student-ABI/manifest hashes match exactly), then imports each entry
point fresh through ``importlib``.  There is NO parent-process loader registry
anywhere — parent globals never survive the spawn, and no import side effect
is ever consulted.
"""

from __future__ import annotations

import sys

from .fresh_process_restore import restore_worker_main

if __name__ == "__main__":
    raise SystemExit(restore_worker_main(sys.argv))
