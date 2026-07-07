"""v6fix7 P1b — siege ecosystem exemptions: training quota, CAS force-activation, coop partition,
drill-transfer gap hint.

Pins the fixes for audit hazard #4 ("the drill is learned so it is discarded"):
  - apply_siege_focus_quota guarantees focus-teaching levels keep training slots (swap-in, size
    unchanged; strict no-op when siege off / no focus / quota met);
  - attempt_to_activate_task force-activates siege-tagged levels past the CAS learnability bar;
  - _coop_select gives siege-tagged candidates guaranteed seats and top-k's only the rest;
  - _render_siege_gap_hint quantifies "drill won in sandbox but not transferring" (SCALAR-style
    train-eval gap) so the modeler knows when to add the pressure back.

Same stub-injection pattern as test_rehearsal.py (selection/evolution_efficient import gen_manager,
which pulls jax; we stub the module — the functions only duck-type it).
"""

import importlib.util
import os
import sys
import threading
import types

import networkx as nx

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for p in (os.path.join(_REPO, "src"), _REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

_stub = types.ModuleType("dicode.dreaming.gen_manager")
_stub.GenManager = object
_stub.TaskArchive = object
sys.modules.setdefault("dicode.dreaming.gen_manager", _stub)

from dicode.selection import apply_siege_focus_quota  # noqa: E402
from dicode.evolution_efficient import attempt_to_activate_task  # noqa: E402


class _NB:
    def __init__(self, foci):
        self._foci = foci

    def focus_skills(self):
        return list(self._foci)


class _Archive:
    def __init__(self):
        self.graph = nx.DiGraph()
        self._lock = threading.Lock()

    @property
    def active_task_count(self):
        return sum(1 for _, d in self.graph.nodes(data=True) if d.get("is_active"))

    def set_task_active_status(self, task_id, status):
        self.graph.nodes[task_id]["is_active"] = bool(status)


class _TG:
    def __init__(self, notebook):
        self._siege_notebook = notebook


class _GM:
    def __init__(self, archive, notebook):
        self.archive = archive
        self.task_generator = _TG(notebook) if notebook is not None else None
        if notebook is None:
            self.task_generator = None


class _DM:
    def __init__(self, **kw):
        self.siege_focus_quota = kw.get("quota", 4)
        self.active_task_capacity = kw.get("capacity", 3)
        self.min_entry_score_threshold = kw.get("min_entry", 0.1)


class _Cfg:
    def __init__(self, dm):
        self.dicode_manager = dm


WALL = "make_iron_pickaxe"


def _archive_with(levels):
    """levels: list of (node_id, attrs)."""
    ar = _Archive()
    for nid, attrs in levels:
        ar.graph.add_node(nid, **attrs)
    return ar


# ---------- apply_siege_focus_quota ----------


def test_quota_noop_when_siege_off():
    gm = _GM(_archive_with([]), None)
    gm.task_generator = None
    batch = ["a", "b"]
    assert apply_siege_focus_quota(gm, _Cfg(_DM()), list(batch)) == batch


def test_quota_noop_when_no_focus():
    gm = _GM(_archive_with([("a", {})]), _NB([]))
    batch = ["a"]
    assert apply_siege_focus_quota(gm, _Cfg(_DM()), list(batch)) == batch


def test_quota_swaps_in_focus_levels_keeping_size():
    levels = [
        # in-batch fillers, low priority, not teaching the wall
        ("f1", {"is_active": True, "priority_score": 0.01, "description": "Relevant Achievements: EAT_COW"}),
        ("f2", {"is_active": True, "priority_score": 0.02, "description": "Relevant Achievements: EAT_COW"}),
        ("f3", {"is_active": True, "priority_score": 0.30, "description": "Relevant Achievements: EAT_COW"}),
        # focus-teaching actives OUTSIDE the batch (drill lineage), newest first should win
        ("d_new", {"is_active": True, "priority_score": 0.0, "session_created": 9,
                   "drill_target": WALL, "description": "Relevant Achievements: MAKE_IRON_PICKAXE"}),
        ("d_old", {"is_active": True, "priority_score": 0.0, "session_created": 3,
                   "drill_target": WALL, "description": "Relevant Achievements: MAKE_IRON_PICKAXE"}),
        # already-in-batch focus teacher (via Relevant parse, no meta tags)
        ("d_in", {"is_active": True, "priority_score": 0.5,
                  "description": "Relevant Achievements: MAKE_IRON_PICKAXE, COLLECT_IRON"}),
    ]
    gm = _GM(_archive_with(levels), _NB([WALL]))
    batch = ["f1", "f2", "f3", "d_in"]
    out = apply_siege_focus_quota(gm, _Cfg(_DM(quota=3)), list(batch))
    assert len(out) == len(batch)                      # size unchanged
    assert "d_in" in out                               # focus teachers never swapped out
    assert "d_new" in out and "d_old" in out           # shortfall 2 -> both drills in, newest first
    assert "f3" in out                                 # highest-priority filler survives
    assert "f1" not in out and "f2" not in out         # lowest-priority fillers were the victims


def test_quota_noop_when_already_met():
    levels = [
        ("d1", {"is_active": True, "description": "Relevant Achievements: MAKE_IRON_PICKAXE"}),
        ("d2", {"is_active": True, "drill_target": WALL, "description": ""}),
    ]
    gm = _GM(_archive_with(levels), _NB([WALL]))
    batch = ["d1", "d2"]
    assert apply_siege_focus_quota(gm, _Cfg(_DM(quota=2)), list(batch)) == batch


# ---------- CAS force-activation ----------


def test_cas_force_activates_siege_level_past_score_bar():
    levels = [
        ("act1", {"is_active": True, "priority_score": 0.2}),
        ("act2", {"is_active": True, "priority_score": 0.21}),
        ("act3", {"is_active": True, "priority_score": 0.22}),
        ("drill", {"is_active": False, "priority_score": 0.0, "siege_wall": WALL}),
    ]
    gm = _GM(_archive_with(levels), _NB([WALL]))
    cfg = _Cfg(_DM(capacity=3))
    # score 0.0 would lose every comparison; the siege tag must force it in (evicting the worst).
    assert attempt_to_activate_task(gm, "drill", 0.0, cfg) is True
    assert gm.archive.graph.nodes["drill"]["is_active"] is True
    assert gm.archive.graph.nodes["act1"]["is_active"] is False  # worst evicted
    assert gm.archive.active_task_count == 3


def test_cas_untagged_level_still_rejected():
    levels = [
        ("act1", {"is_active": True, "priority_score": 0.2}),
        ("act2", {"is_active": True, "priority_score": 0.21}),
        ("act3", {"is_active": True, "priority_score": 0.22}),
        ("plain", {"is_active": False, "priority_score": 0.0}),
    ]
    gm = _GM(_archive_with(levels), _NB([WALL]))
    cfg = _Cfg(_DM(capacity=3))
    assert attempt_to_activate_task(gm, "plain", 0.0, cfg) is False
    assert gm.archive.graph.nodes["plain"].get("is_active") is False


# ---------- _coop_select siege partition + gap hint (need the real TaskGenerator) ----------

_GM_PATH = os.path.join(_REPO, "src", "dicode", "dreaming", "gen_manager.py")
_spec = importlib.util.spec_from_file_location("dicode_v6_gen_manager_eco_test", _GM_PATH)
_gmmod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gmmod)
TaskGenerator = _gmmod.TaskGenerator


class _CoopCfg:
    coop_w_cov = 0.0
    coop_w_amb = 0.0
    coop_w_lrn = 1.0


class _ArchiveNoNodes:
    def __init__(self):
        self.graph = nx.DiGraph()
        self._lock = threading.Lock()


def _parsed(desc, meta=None):
    return {"description": desc, "reasoning": "r", "level_meta": meta}


def test_coop_select_siege_candidates_bypass_cull():
    tg = object.__new__(TaskGenerator)
    tg._siege_notebook = _NB([WALL])
    tg.config = _CoopCfg()
    tg.archive = _ArchiveNoNodes()
    drill_meta = {"type": "CONSOLIDATE", "drill_target": WALL, "siege_wall": WALL}
    all_parsed = [
        _parsed("Relevant Achievements: EAT_COW", None),
        _parsed("Relevant Achievements: MAKE_IRON_PICKAXE", drill_meta),
        _parsed("Relevant Achievements: EAT_PLANT", None),
        _parsed("Relevant Achievements: MAKE_IRON_PICKAXE, COLLECT_IRON",
                {"type": "DEPTH", "drill_target": None, "siege_wall": WALL}),
    ]
    parents = [["p0"], ["p1"], ["p2"], ["p3"]]
    examples = [[], [], [], []]
    kept_p, kept_pa, _ = tg._coop_select(all_parsed, parents, examples, 3, 7, {})
    # both siege-tagged candidates are guaranteed in; exactly one non-siege fills the last seat.
    assert len(kept_p) == 3
    assert all_parsed[1] in kept_p and all_parsed[3] in kept_p
    assert kept_pa.count(["p1"]) == 1 and kept_pa.count(["p3"]) == 1


def test_coop_select_no_focus_runs_plain_topk():
    tg = object.__new__(TaskGenerator)
    tg._siege_notebook = _NB([])
    tg.config = _CoopCfg()
    tg.archive = _ArchiveNoNodes()
    all_parsed = [_parsed("Relevant Achievements: EAT_COW"), _parsed("Relevant Achievements: EAT_PLANT")]
    kept_p, _, _ = tg._coop_select(all_parsed, [["a"], ["b"]], [[], []], 1, 7, {})
    assert len(kept_p) == 1


def test_gap_hint_flags_overfit_drill():
    tg = object.__new__(TaskGenerator)
    tg._siege_notebook = _NB([WALL])
    ar = _ArchiveNoNodes()
    ar.graph.add_node(
        "drill_1",
        drill_target=WALL,
        performance_history=[{"sr": 0.5}, {"sr": 0.95}],  # trained SR 95% inside the drill
    )
    tg.archive = ar
    hint = tg._render_siege_gap_hint({WALL: 29.0})
    assert "DRILL-TRANSFER GAP" in hint
    assert "ADD THE PRESSURE BACK" in hint


def test_gap_hint_quiet_when_gap_small_or_siege_off():
    tg = object.__new__(TaskGenerator)
    tg._siege_notebook = _NB([WALL])
    ar = _ArchiveNoNodes()
    ar.graph.add_node("drill_1", drill_target=WALL, performance_history=[{"sr": 0.6}])
    tg.archive = ar
    hint = tg._render_siege_gap_hint({WALL: 55.0})
    assert "ADD THE PRESSURE BACK" not in hint  # gap small -> informational line only
    tg2 = object.__new__(TaskGenerator)
    tg2._siege_notebook = None
    assert tg2._render_siege_gap_hint({WALL: 10.0}) == ""
