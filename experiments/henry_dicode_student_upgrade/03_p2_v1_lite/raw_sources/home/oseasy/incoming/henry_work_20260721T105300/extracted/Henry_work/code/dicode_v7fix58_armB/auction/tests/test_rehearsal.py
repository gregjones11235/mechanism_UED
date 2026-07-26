"""Offline tests for v6 §3.6 REHEARSAL append logic (selection.append_rehearsal_tasks).

Pins the user's 2026-07-04 requirements:
  - the siege batch is kept IN FULL (rehearsal is EXTRA, never shrinks the siege allotment),
  - rehearsal fires ONLY when a protected skill is CURRENTLY FORGETTING (no proactive rehearsal),
  - a chain may consume MULTIPLE slots (per_forgetting_skill * #forgetting skills),
  - two ceilings bound it: fraction of the siege allotment, and an absolute total cap (e.g. 24),
  - strict no-op when siege off / no protected set / nothing forgetting.

selection.py's module top-level imports jax via gen_manager, which isn't installed here, so we inject
a stub `dicode.dreaming.gen_manager` into sys.modules before importing selection. The function under
test only uses gen_manager as a duck-typed holder (archive._lock, archive.graph, _siege_notebook,
_profile_log), so a fake object suffices.
"""

import sys
import types
import threading

import networkx as nx
import pytest


# ---- inject a stub gen_manager module so `import dicode.selection` doesn't pull in jax -------------
_stub = types.ModuleType("dicode.dreaming.gen_manager")
_stub.GenManager = object
_stub.TaskArchive = object
sys.modules.setdefault("dicode.dreaming.gen_manager", _stub)

from dicode.selection import append_rehearsal_tasks  # noqa: E402
from auction.siege_notebook import (  # noqa: E402
    MATURITY_MIN_SNAPSHOTS,
    SiegeNotebook,
)


class _FakeArchive:
    def __init__(self):
        self.graph = nx.DiGraph()
        self._lock = threading.Lock()

    def add_level(self, node_id, description, is_active=True):
        self.graph.add_node(node_id, description=description, is_active=is_active)


class _FakeProfileLog:
    """Minimal stand-in exposing forgetting_candidates()."""
    def __init__(self, forgetting_names):
        self._f = forgetting_names

    def forgetting_candidates(self, **kwargs):
        # accepts the family-split threshold kwargs selection.py now passes (§2⑥); the stub ignores
        # them and just returns its canned forgetting list.
        return [{"achievement": n} for n in self._f]


class _FakeTG:
    """Stand-in for gen_manager.task_generator, which is where the siege notebook + profile log
    actually live (see gen_manager.TaskGenerator)."""
    def __init__(self, notebook, profile_log):
        self._siege_notebook = notebook
        self._profile_log = profile_log


class _FakeGM:
    """Fake GenManager. By default it mirrors the REAL structure: the notebook + profile log hang off
    a nested ``task_generator`` (not off the GM itself). Pass ``hoisted=True`` to instead attach them
    directly to the GM (the pre-2026-07-05 forward-compat fallback path)."""
    def __init__(self, archive, notebook, profile_log, hoisted=False):
        self.archive = archive
        if hoisted:
            # legacy/forward-compat: attrs on the GM itself, no task_generator.
            self._siege_notebook = notebook
            self._profile_log = profile_log
        else:
            # real structure: attrs live on gen_manager.task_generator.
            self.task_generator = _FakeTG(notebook, profile_log)


class _DM:
    def __init__(self, **kw):
        self.rehearsal_per_forgetting_skill = kw.get("per", 2)
        self.rehearsal_max_frac = kw.get("frac", 0.5)
        self.rehearsal_total_cap = kw.get("cap", 24)


class _Cfg:
    def __init__(self, dm):
        self.dicode_manager = dm


def _notebook_with_protected(tmp_path, protected_skills, session=11):
    """Build a SiegeNotebook whose protected_set contains the given skills.

    v6fix7 P1a (#8): protection now requires a VERIFIED CONQUEST — the focus must HOLD at/above
    mastered_sr for conquest_consecutive consecutive snapshots (a one-shot +delta record stays
    'progress' and protects nothing). Session 10 opens the focus; sessions 11-12 hold it at
    mastered -> conquered -> target + links protected.
    """
    nb = SiegeNotebook(str(tmp_path / "siege.json"))
    from auction.siege_notebook import MATURITY_MIN_MASTERED, MATURITY_SKILL_SR
    prof = {f"solid_{i}": MATURITY_SKILL_SR + 10 for i in range(MATURITY_MIN_MASTERED)}
    focus = protected_skills[0]
    links = protected_skills[1:]
    prof[focus] = 20.0
    for l in links:
        prof[l] = 80.0
    nb.apply_llm_update(
        10, prof,
        {"focus": focus, "prereq_tree": [{"skill": l, "role": "r"} for l in links]},
        num_snapshots=MATURITY_MIN_SNAPSHOTS,
    )
    # hold the focus at mastered for two consecutive snapshots -> verified conquest -> protected.
    conquered = dict(prof)
    conquered[focus] = 75.0
    nb.apply_llm_update(session, dict(conquered), None, num_snapshots=MATURITY_MIN_SNAPSHOTS)
    conquered[focus] = 78.0
    nb.apply_llm_update(session + 1, dict(conquered), None, num_snapshots=MATURITY_MIN_SNAPSHOTS)
    assert set(protected_skills).issubset(set(nb.protected_set())), "conquest setup failed"
    return nb


SIEGE_BATCH = [f"siege_{i}" for i in range(16)]  # a full 16-level siege allotment


def test_noop_when_no_siege_notebook(tmp_path):
    gm = _FakeGM(_FakeArchive(), None, None)
    out = append_rehearsal_tasks(gm, _Cfg(_DM()), list(SIEGE_BATCH))
    assert out == SIEGE_BATCH


def test_noop_when_protected_empty(tmp_path):
    nb = SiegeNotebook(str(tmp_path / "s.json"))  # empty, nothing conquered
    gm = _FakeGM(_FakeArchive(), nb, _FakeProfileLog([]))
    out = append_rehearsal_tasks(gm, _Cfg(_DM()), list(SIEGE_BATCH))
    assert out == SIEGE_BATCH


def test_noop_when_nothing_forgetting(tmp_path):
    # protected set exists, but no protected skill is forgetting -> keep all fire-power on siege.
    nb = _notebook_with_protected(tmp_path, ["defeat_gnome_warrior", "place_torch"])
    arch = _FakeArchive()
    arch.add_level("lvl_torch", "Relevant Achievements: PLACE_TORCH")
    gm = _FakeGM(arch, nb, _FakeProfileLog([]))  # forgetting = none
    out = append_rehearsal_tasks(gm, _Cfg(_DM()), list(SIEGE_BATCH))
    assert out == SIEGE_BATCH


def test_appends_when_protected_skill_forgetting(tmp_path):
    nb = _notebook_with_protected(tmp_path, ["defeat_gnome_warrior", "place_torch"])
    arch = _FakeArchive()
    arch.add_level("lvl_torch_a", "Relevant Achievements: PLACE_TORCH")
    arch.add_level("lvl_torch_b", "Relevant Achievements: PLACE_TORCH, COLLECT_WOOD")
    arch.add_level("lvl_unrelated", "Relevant Achievements: DEFEAT_ZOMBIE")
    # place_torch is protected AND forgetting -> rehearsal fires.
    gm = _FakeGM(arch, nb, _FakeProfileLog(["place_torch"]))
    out = append_rehearsal_tasks(gm, _Cfg(_DM(per=2)), list(SIEGE_BATCH))
    # siege batch kept in full, in order, at the front
    assert out[:16] == SIEGE_BATCH
    extra = out[16:]
    # per_forgetting_skill=2, 1 forgetting skill -> quota 2; only 2 torch levels teach it
    assert set(extra) <= {"lvl_torch_a", "lvl_torch_b"}
    assert len(extra) == 2
    assert "lvl_unrelated" not in extra  # doesn't teach a forgetting skill


def test_siege_batch_never_shrinks(tmp_path):
    nb = _notebook_with_protected(tmp_path, ["defeat_gnome_warrior", "place_torch"])
    arch = _FakeArchive()
    for i in range(10):
        arch.add_level(f"lvl_torch_{i}", "Relevant Achievements: PLACE_TORCH")
    gm = _FakeGM(arch, nb, _FakeProfileLog(["place_torch"]))
    out = append_rehearsal_tasks(gm, _Cfg(_DM(per=99)), list(SIEGE_BATCH))
    # every original siege level is still present
    for s in SIEGE_BATCH:
        assert s in out
    assert out[:16] == SIEGE_BATCH


def test_frac_ceiling(tmp_path):
    # per*forgetting huge, but frac=0.25 of 16 = 4 caps it.
    nb = _notebook_with_protected(tmp_path, ["defeat_gnome_warrior", "place_torch"])
    arch = _FakeArchive()
    for i in range(20):
        arch.add_level(f"lvl_torch_{i}", "Relevant Achievements: PLACE_TORCH")
    gm = _FakeGM(arch, nb, _FakeProfileLog(["place_torch"]))
    out = append_rehearsal_tasks(gm, _Cfg(_DM(per=99, frac=0.25)), list(SIEGE_BATCH))
    assert len(out) - 16 == 4  # floor(16*0.25)


def test_absolute_total_cap(tmp_path):
    # total cap 20: with 16 siege, at most 4 rehearsal appended regardless of frac/per.
    nb = _notebook_with_protected(tmp_path, ["defeat_gnome_warrior", "place_torch"])
    arch = _FakeArchive()
    for i in range(20):
        arch.add_level(f"lvl_torch_{i}", "Relevant Achievements: PLACE_TORCH")
    gm = _FakeGM(arch, nb, _FakeProfileLog(["place_torch"]))
    out = append_rehearsal_tasks(gm, _Cfg(_DM(per=99, frac=1.0, cap=20)), list(SIEGE_BATCH))
    assert len(out) == 20  # capped, siege intact
    assert out[:16] == SIEGE_BATCH


def test_multiple_forgetting_skills_more_slots(tmp_path):
    # a chain with two forgetting skills gets 2*per slots (chain may use multiple rehearsal slots).
    nb = _notebook_with_protected(tmp_path, ["defeat_gnome_warrior", "place_torch", "make_torch"])
    arch = _FakeArchive()
    arch.add_level("lvl_place_a", "Relevant Achievements: PLACE_TORCH")
    arch.add_level("lvl_place_b", "Relevant Achievements: PLACE_TORCH")
    arch.add_level("lvl_make_a", "Relevant Achievements: MAKE_TORCH")
    arch.add_level("lvl_make_b", "Relevant Achievements: MAKE_TORCH")
    gm = _FakeGM(arch, nb, _FakeProfileLog(["place_torch", "make_torch"]))
    out = append_rehearsal_tasks(gm, _Cfg(_DM(per=2, frac=1.0, cap=24)), list(SIEGE_BATCH))
    # 2 forgetting skills * per 2 = quota 4
    assert len(out) - 16 == 4


def test_excludes_levels_already_in_siege_batch(tmp_path):
    nb = _notebook_with_protected(tmp_path, ["defeat_gnome_warrior", "place_torch"])
    arch = _FakeArchive()
    arch.add_level("already_in_batch", "Relevant Achievements: PLACE_TORCH")
    arch.add_level("fresh_torch", "Relevant Achievements: PLACE_TORCH")
    gm = _FakeGM(arch, nb, _FakeProfileLog(["place_torch"]))
    siege = list(SIEGE_BATCH) + ["already_in_batch"]  # a siege level already trains place_torch
    out = append_rehearsal_tasks(gm, _Cfg(_DM(per=2)), siege)
    extra = out[len(siege):]
    assert "already_in_batch" not in extra  # not re-added
    assert "fresh_torch" in extra


# ---- §2⑥ family-split FORGETTING prefilter: combat milestones (low peak) must be flaggable --------

def test_forgetting_family_split_flags_low_peak_combat(tmp_path):
    """A combat milestone that peaked at only 18% then slid must be flagged; a gear skill peaking at
    the SAME 18% must NOT (its bar is the high non-combat 40%). This is the rule that a forgetting
    combat milestone (exactly what §2.5 records) can trigger rehearsal, while noise on low-peak gear
    does not spam it."""
    from auction.student_profile_log import StudentProfileLog
    p = str(tmp_path / "sp.json")
    log = StudentProfileLog(path=p)
    # session 1: both at their peak; session 2: both slid by 12pp. (global_agent_profile keys are
    # ``skill_<name>`` in 0..100; _normalize strips the prefix.)
    log.record(1, {"skill_defeat_zombie": 18.0, "skill_make_iron_pickaxe": 18.0})
    log.record(2, {"skill_defeat_zombie": 6.0, "skill_make_iron_pickaxe": 6.0})
    flagged = {f["achievement"] for f in log.forgetting_candidates()}  # family-split defaults
    assert "defeat_zombie" in flagged        # combat: peak 18 >= 15, drop 12 >= 10 -> forgetting
    assert "make_iron_pickaxe" not in flagged  # gear: peak 18 < 40 -> below its bar, not flagged


# ---- BUGFIX regression (2026-07-05): notebook + profile log live on gen_manager.task_generator ----
# The v6-OLD run read them off gen_manager directly, so rehearsal silently no-op'd for the whole run.
# These pin the holder resolution so a future refactor that moves the attrs can't silently re-break it.

def test_reads_notebook_from_task_generator_not_gm(tmp_path):
    """REAL structure: notebook + plog hang off gen_manager.task_generator, and gen_manager itself has
    NO _siege_notebook attr. Rehearsal must still find them and fire. This is the exact shape that made
    the v6-OLD run produce zero rehearsal rescues."""
    nb = _notebook_with_protected(tmp_path, ["defeat_gnome_warrior", "place_torch"])
    arch = _FakeArchive()
    arch.add_level("lvl_torch_a", "Relevant Achievements: PLACE_TORCH")
    arch.add_level("lvl_torch_b", "Relevant Achievements: PLACE_TORCH")
    gm = _FakeGM(arch, nb, _FakeProfileLog(["place_torch"]))  # default: attrs on .task_generator
    assert not hasattr(gm, "_siege_notebook")  # guard: GM itself must NOT carry the notebook
    assert hasattr(gm.task_generator, "_siege_notebook")
    out = append_rehearsal_tasks(gm, _Cfg(_DM(per=2)), list(SIEGE_BATCH))
    assert out[:16] == SIEGE_BATCH
    assert len(out) - 16 == 2  # rehearsal actually fired (the bug: this was 0)


def test_hoisted_fallback_still_works(tmp_path):
    """Forward-compat: if a future version hoists the attrs onto the GM (no task_generator), the
    fallback branch must still resolve them."""
    nb = _notebook_with_protected(tmp_path, ["defeat_gnome_warrior", "place_torch"])
    arch = _FakeArchive()
    arch.add_level("lvl_torch_a", "Relevant Achievements: PLACE_TORCH")
    arch.add_level("lvl_torch_b", "Relevant Achievements: PLACE_TORCH")
    gm = _FakeGM(arch, nb, _FakeProfileLog(["place_torch"]), hoisted=True)
    assert not hasattr(gm, "task_generator")
    out = append_rehearsal_tasks(gm, _Cfg(_DM(per=2)), list(SIEGE_BATCH))
    assert len(out) - 16 == 2
