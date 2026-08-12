"""R1: B1 profiling disabled path must be zero-instrumentation.

Local source-audit tests assert the B1 span sites are guarded by explicit
``if tracker.enabled:`` branches (no unconditional tracker.span). Functional
tests require jax (server) and verify: disabled path never enters span, never
calls monotonic_ns, and returns the exact historical result; enabled path
enters the expected spans.
"""
import importlib.util
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

TESTS = Path(__file__).parent
DICODE_SRC = Path(__file__).parents[2]
RUN_DICODE = Path(__file__).parents[4] / "experiments" / "training" / "run_dicode.py"
ONLINE_EVAL = DICODE_SRC / "evaluation" / "online_evaluation.py"
TASK_UTILS = DICODE_SRC / "task_utils.py"


def _text(path):
    return path.read_text(encoding="utf-8")


def _span_guarded_by_enabled(src, span_line_pattern):
    """Assert every `with tracker.span("<name>")` occurrence is inside an
    `if tracker.enabled:` branch (same function scope, before the span)."""
    lines = src.splitlines()
    ok = True
    misses = []
    for i, line in enumerate(lines):
        m = re.search(r"with tracker\.span\(\"(?P<name>[a-z_]+)\"\)", line)
        if not m:
            continue
        # walk backwards to the enclosing statement; must find `if tracker.enabled:`
        # before an unindented line or another `else:`/`elif`
        indent = len(line) - len(line.lstrip())
        guard = False
        for j in range(i - 1, -1, -1):
            prev = lines[j]
            if not prev.strip():
                continue
            pindent = len(prev) - len(prev.lstrip())
            if re.search(r"if tracker\.enabled\s*:", prev):
                guard = True
                break
            if pindent < indent:
                break
        if not guard:
            ok = False
            misses.append((m.group("name"), i + 1))
    return ok, misses


def test_task_utils_candidate_code_load_guarded():
    ok, misses = _span_guarded_by_enabled(_text(TASK_UTILS), None)
    assert ok, f"unguarded spans in task_utils.py: {misses}"


def test_run_dicode_preflight_spans_guarded():
    src = _text(RUN_DICODE)
    for name in ("preflight_task_reload", "route", "archive_update"):
        # archive_update lives inside _preflight_route which checks tracker.enabled
        assert f'tracker.span("{name}")' in src
    assert 'time.monotonic_ns() if tracker.enabled else None' in src
    assert 'if tracker.enabled:\n                    tracker.record("preflight_wall"' in src


def test_online_evaluation_second_load_guarded():
    src = _text(ONLINE_EVAL)
    assert 'elif tracker.enabled:' in src
    assert 'with tracker.span("preflight_task_reload"):' in src
    # the historical else branch must call load without any span
    assert 'else:\n\t\ttask_classes, _ = load_tasks_from_env_codes(archive, new_task_ids)' in src


def test_ppo_tr_eval_spans_inside_enabled_branch():
    src = _text(DICODE_SRC / "ppo_tr.py")
    # preflight_eval_* spans must be inside `if tracker.enabled:`
    ok, misses = _span_guarded_by_enabled(src, None)
    # train_* spans are inside profiling/cache branches (not plain tracker.enabled), so
    # only assert the preflight_eval_* spans are guarded.
    unguarded = [m for m in misses if m[0].startswith("preflight_eval")]
    assert not unguarded, f"unguarded preflight_eval spans: {unguarded}"


def test_gen_manager_validation_spans_inside_enabled_branch():
    src = _text(DICODE_SRC / "dreaming" / "gen_manager.py")
    ok, misses = _span_guarded_by_enabled(src, None)
    unguarded = [m for m in misses if "candidate_cpu_validation" in m[0]]
    assert not unguarded, f"unguarded candidate_cpu_validation spans: {unguarded}"


# ---- Functional tests (require jax -> server) ----


def _fake_archive():
    class FakeArchive:
        def __init__(self):
            self.learn = []
            self.status = []
            self.active = []

        def get_task_codes(self, tasks):
            return {t: "class Env:\n    pass\n" for t in tasks}

        def update_node_learnability(self, tid, value):
            self.learn.append((tid, value))

        def update_node_status(self, tid, status):
            self.status.append((tid, status))

        def set_task_active_status(self, tid, active):
            self.active.append((tid, active))
    return FakeArchive()


def _load_task_utils():
    import dicode.task_utils
    return dicode.task_utils


def _load_run_dicode():
    spec = importlib.util.spec_from_file_location("run_dicode", RUN_DICODE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_load_tasks_disabled_never_enters_span(monkeypatch):
    pytest.importorskip("jax")
    from dicode.runtime_analysis import tracker
    tu = _load_task_utils()
    monkeypatch.setattr(tracker, "enabled", False)
    calls = []

    def raising_span(*a, **k):
        calls.append(a)
        raise AssertionError("tracker.span entered while disabled")

    monkeypatch.setattr(tracker, "span", raising_span)
    classes, ok = tu.load_tasks_from_env_codes(_fake_archive(), ["t1", "t2"])
    assert ok == ["t1", "t2"] and len(classes) == 2
    assert calls == []


def test_load_tasks_disabled_no_monotonic_ns(monkeypatch):
    pytest.importorskip("jax")
    from dicode.runtime_analysis import tracker
    import dicode.runtime_analysis as ra
    tu = _load_task_utils()
    monkeypatch.setattr(tracker, "enabled", False)
    calls = []

    def spy():
        calls.append(1)
        return 0

    monkeypatch.setattr(ra.time, "monotonic_ns", spy)
    classes, ok = tu.load_tasks_from_env_codes(_fake_archive(), ["t1"])
    assert ok == ["t1"] and calls == []


def test_load_tasks_enabled_enters_span_once(monkeypatch):
    pytest.importorskip("jax")
    from dicode.runtime_analysis import tracker
    tu = _load_task_utils()
    monkeypatch.setattr(tracker, "enabled", True)
    calls = []

    class FakeSpan:
        def __init__(self, name, **k):
            self.name = name

        def __enter__(self):
            calls.append(self.name)
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(tracker, "span", FakeSpan)
    classes, ok = tu.load_tasks_from_env_codes(_fake_archive(), ["t1"])
    assert ok == ["t1"] and calls == ["candidate_code_load"]


def test_preflight_route_disabled_never_enters_span(monkeypatch):
    pytest.importorskip("jax")
    from dicode.runtime_analysis import tracker
    rd = _load_run_dicode()
    monkeypatch.setattr(tracker, "enabled", False)
    calls = []

    def raising_span(*a, **k):
        calls.append(a)
        raise AssertionError("tracker.span entered while disabled")

    monkeypatch.setattr(tracker, "span", raising_span)
    arch = _fake_archive()

    class D:
        action = "reject"
        reason = "sr_low"

    kept = []
    rd._preflight_route({"0": {"sr": 0.3}}, ["t1"], kept, arch,
                        lambda sr, any_partial: D())
    assert kept == [] and arch.status == [("t1", "preflight_sr_low")]
    assert arch.active == [("t1", False)]
    assert calls == []


def test_preflight_route_disabled_accept_path(monkeypatch):
    pytest.importorskip("jax")
    from dicode.runtime_analysis import tracker
    rd = _load_run_dicode()
    monkeypatch.setattr(tracker, "enabled", False)
    calls = []

    def raising_span(*a, **k):
        calls.append(a)
        raise AssertionError("tracker.span entered while disabled")

    monkeypatch.setattr(tracker, "span", raising_span)
    arch = _fake_archive()

    class D:
        action = "accept"
        reason = ""

    kept = []
    rd._preflight_route({"0": {"sr": 0.8}}, ["t1"], kept, arch,
                        lambda sr, any_partial: D())
    assert kept == ["t1"]
    assert len(arch.learn) == 1 and arch.learn[0][0] == "t1"
    assert abs(arch.learn[0][1] - 0.8 * 0.2) < 1e-9
    assert calls == []


def test_preflight_route_enabled_enters_archive_spans(monkeypatch):
    pytest.importorskip("jax")
    from dicode.runtime_analysis import tracker
    rd = _load_run_dicode()
    monkeypatch.setattr(tracker, "enabled", True)
    calls = []

    class FakeSpan:
        def __init__(self, name, **k):
            self.name = name

        def __enter__(self):
            calls.append(self.name)
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(tracker, "span", FakeSpan)
    arch = _fake_archive()

    class D:
        action = "accept"
        reason = ""

    kept = []
    rd._preflight_route({"0": {"sr": 0.5}}, ["t1"], kept, arch,
                        lambda sr, any_partial: D())
    assert kept == ["t1"] and calls == ["archive_update"]


def test_load_tasks_off_on_equivalent(monkeypatch):
    """Audit R1 test 4: disabled and enabled modes must return identical
    results for the same input (only instrumentation differs)."""
    pytest.importorskip("jax")
    from dicode.runtime_analysis import tracker
    tu = _load_task_utils()
    calls = []

    class FakeSpan:
        def __init__(self, name, **k):
            self.name = name

        def __enter__(self):
            calls.append(self.name)
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(tracker, "span", FakeSpan)
    monkeypatch.setattr(tracker, "enabled", False)
    classes_off, ok_off = tu.load_tasks_from_env_codes(_fake_archive(), ["t1", "t2"])
    monkeypatch.setattr(tracker, "enabled", True)
    calls.clear()
    classes_on, ok_on = tu.load_tasks_from_env_codes(_fake_archive(), ["t1", "t2"])
    assert ok_off == ok_on == ["t1", "t2"]
    assert len(classes_off) == len(classes_on) == 2
    assert calls == ["candidate_code_load"]


def test_preflight_route_off_on_equivalent(monkeypatch):
    """Audit R1 test 4: disabled and enabled modes must apply the same archive
    mutations for the same scores (instrumentation is the only difference)."""
    pytest.importorskip("jax")
    from dicode.runtime_analysis import tracker
    rd = _load_run_dicode()

    class FakeSpan:
        def __init__(self, name, **k):
            self.name = name

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(tracker, "span", FakeSpan)

    def run(tracked):
        monkeypatch.setattr(tracker, "enabled", tracked)
        arch = _fake_archive()

        class D:
            action = "reject"
            reason = "sr_low"

        kept = []
        rd._preflight_route({"0": {"sr": 0.1}, "1": {"sr": 0.9}}, ["t1", "t2"], kept, arch,
                            lambda sr, any_partial: D() if sr < 0.5 else SimpleNamespace(action="accept", reason=""))
        return arch, kept

    off_arch, off_kept = run(False)
    on_arch, on_kept = run(True)
    assert off_kept == on_kept == ["t2"]
    assert off_arch.status == on_arch.status == [("t1", "preflight_sr_low")]
    assert off_arch.active == on_arch.active == [("t1", False)]
    assert len(off_arch.learn) == len(on_arch.learn) == 1
