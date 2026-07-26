"""v7fix4.4: system-built relay levels are exempt from the random registration cull.

Root incident (2026-07-12, job 3930279 rung-stall root cause): ``_process_worker_results``
random.sample'd ALL compiled designs down to ``num_generation_tasks``=10 out of ~20-22 per
session. Every level — including the 2 system-built relay levels that are the rung ladder's
exact training dose and the ONLY legal source of rung readings (P2.8) — faced a ~50% random
death, and the losers were left as code-less ``desc_generated`` husk nodes (the code write
happens only for sampled tasks). Observed: 9 of 14 relay levels husked across s82-96, rung
readings dry at s89/s91, stall/patience counters fed by the artifact.

The fix keeps system levels unconditionally and samples only the REMAINDER of the budget
from FM designs, so the archive-growth bound is preserved and the baseline path (which never
produces ``_system_code``) consumes the identical RNG stream.

``run_dicode.py`` imports jax/hydra/wandb at module level, so the function under test is
AST-extracted from the source and exec'd with only the stdlib ``random`` it needs — these
tests run on the real shipped code without the heavy deps.
"""

import ast
import os
import random
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
_RD_PATH = os.path.join(_REPO, "experiments", "training", "run_dicode.py")


def _load_process_worker_results():
    src = open(_RD_PATH, encoding="utf-8").read()
    tree = ast.parse(src)
    fn = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_process_worker_results"
    )
    module = ast.Module(body=[fn], type_ignores=[])
    ns = {"random": random, "DictConfig": object}  # annotation name + the one real dep
    exec(compile(ast.fix_missing_locations(module), _RD_PATH, "exec"), ns)
    return ns["_process_worker_results"]


class _ArchiveRecorder:
    def __init__(self):
        self.coded = []      # task ids that got update_node_code (i.e. survived the cull)
        self.statuses = {}

    def update_node_status(self, task_id, status):
        self.statuses[task_id] = status

    def set_task_active_status(self, task_id, active):
        pass

    def update_node_priority_score(self, task_id, score):
        pass

    def update_node_reasoning(self, task_id, reasoning):
        pass

    def update_node_code(self, task_id, code):
        self.coded.append(task_id)


def _mk_results(n_fm, n_system, start=0):
    out = []
    for i in range(n_fm):
        out.append({
            "generated_task_id": f"fm_{start + i}",
            "code_string": "x = 1",
            "compiled": True,
        })
    for i in range(n_system):
        out.append({
            "generated_task_id": f"sys_{i}",
            "code_string": "relay code",
            "_system_code": "relay code",
            "compiled": True,
        })
    return out


def _run(process, results, limit, seed=42):
    gm = types.SimpleNamespace(archive=_ArchiveRecorder())
    config = types.SimpleNamespace(
        dicode_manager=types.SimpleNamespace(num_generation_tasks=limit)
    )
    random.seed(seed)
    new_task_ids, compiled_count = process(results, gm, config)
    return new_task_ids, compiled_count, gm.archive


def test_system_levels_survive_the_cull():
    """20 FM + 2 system compiled, limit 10: both system levels ALWAYS registered."""
    process = _load_process_worker_results()
    for seed in (0, 1, 7, 42, 1234):
        ids, compiled_count, archive = _run(process, _mk_results(20, 2), limit=10, seed=seed)
        assert compiled_count == 22
        assert "sys_0" in ids and "sys_1" in ids, (seed, ids)
        assert len(ids) == 10                      # archive-growth bound preserved
        assert set(ids) == set(archive.coded)      # survivors (and only they) got code


def test_no_cull_when_under_limit():
    """8 FM + 2 system == limit 10: everyone kept, nothing sampled away."""
    process = _load_process_worker_results()
    ids, _, _ = _run(process, _mk_results(8, 2), limit=10)
    assert len(ids) == 10
    assert {"sys_0", "sys_1"} <= set(ids)


def test_system_overflow_edge_keeps_all_system():
    """More system levels than the whole limit: keep them all (they cost no FM tokens and
    are the ladder's dose), sample zero FM — documented deliberate over-limit."""
    process = _load_process_worker_results()
    ids, _, _ = _run(process, _mk_results(5, 3), limit=1)
    assert set(ids) == {"sys_0", "sys_1", "sys_2"}


def test_baseline_path_identical():
    """No system levels (the baseline arm): selection must equal the ORIGINAL algorithm
    outcome under the same seed — same list order, same k, same single random.sample call,
    same RNG stream. Pins byte-equivalent baseline behaviour."""
    process = _load_process_worker_results()
    results = _mk_results(15, 0)

    ids_new, _, _ = _run(process, results, limit=10, seed=42)

    random.seed(42)  # the pre-fix4.4 algorithm, verbatim
    compiled_tasks = [r for r in results if r.get("compiled")]
    original = [r["generated_task_id"] for r in random.sample(compiled_tasks, 10)]

    assert ids_new == original
