"""B2: preflight task-reuse contract (pure logic, no JAX dependency).

`evaluate_new_tasks` loads the candidate Env classes a second time via
`load_tasks_from_env_codes`. run_dicode.py already loaded the SAME classes from
the SAME archive with the SAME ids in its preflight block. This module decides
whether that first load can be reused in place of the second one.

Safety proposition (audited, see commit message):
- The Env class produced by the *validation* path (gen_manager
  ``_check_compilation_uncached`` -> ``Task(temp_file).env``) is a DIFFERENT
  object from ``load_tasks_from_env_codes`` results: different module name
  (temp file module vs a task-id-named module), different lifecycle (the temp
  file is deleted and its module is removed from ``sys.modules`` after
  validation), and different JAX static signature (the module path participates
  in jit static signatures / cached-attribute keys). Reusing validation objects
  would therefore be unsafe and is never attempted.
- The run_dicode first load (``_pf_classes`` / ``_pf_ok_ids``) uses the exact
  same ``load_tasks_from_env_codes(archive, ids)`` call as the second load, so
  it has the same source and the same semantics. Reusing it is safe and saves
  re-`exec`ing every candidate code string.

Contract: ``preloaded_task_classes`` and ``preloaded_task_ids`` must be
provided together (all-or-nothing). When the flag is on and they are missing,
we raise instead of silently falling back to a second load.
"""

from __future__ import annotations

from typing import Any


def resolve_preloaded_tasks(
    config: Any,
    new_task_ids: list[str],
    preloaded_task_classes: list | None,
    preloaded_task_ids: list[str] | None,
) -> tuple[list | None, bool]:
    """Decide whether to reuse the caller's first load of candidate Env classes.

    Returns ``(task_classes_or_None, reuse: bool)``. When ``reuse`` is False the
    caller must perform its own load (historical path). When True the returned
    ``task_classes`` are the preloaded objects, already validated for id
    order/count. Raises ValueError for contract violations (flag on but
    preloaded objects missing/mismatched) -- never a silent fallback.
    """
    perf = config.get("performance", {}) if hasattr(config, "get") else {}
    reuse = bool(perf.get("preflight_reuse_loaded_tasks", False))
    if not reuse:
        return None, False
    if preloaded_task_classes is None or preloaded_task_ids is None:
        raise ValueError(
            "performance.preflight_reuse_loaded_tasks is on but preloaded "
            "task classes/ids were not provided (all-or-nothing; no silent "
            "fallback to a second load).")
    if [str(x) for x in preloaded_task_ids] != [str(x) for x in new_task_ids]:
        raise ValueError(
            "preloaded_task_ids must equal new_task_ids in order and count "
            "(got %r vs %r)." % (list(preloaded_task_ids), list(new_task_ids)))
    if len(preloaded_task_classes) != len(new_task_ids):
        raise ValueError(
            "preloaded_task_classes count (%d) must equal new_task_ids count (%d)."
            % (len(preloaded_task_classes), len(new_task_ids)))
    return preloaded_task_classes, True
