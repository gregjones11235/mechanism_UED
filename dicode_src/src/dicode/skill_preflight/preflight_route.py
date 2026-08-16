"""Shared production preflight route/archive-update helper (R3 repair).

Extracted from ``experiments/training/run_dicode._preflight_route`` so the
production driver and the frozen preflight replay share one implementation
(off/on equivalence is guaranteed by construction). The route loop applies the
production accept/reject decisions and mutates the archive (learnability for
accept, status+active for reject); profiling spans are entered only when the
production tracker is enabled.
"""
from __future__ import annotations


def preflight_route(scores, ok_ids, kept, archive, route_fn, tracker=None):
    """Production preflight accept/reject loop + archive mutation.

    ``tracker`` is the production ``RuntimeTracker`` (or an object exposing
    ``enabled`` and ``span``); pass ``None`` to run without any profiling spans.
    """
    for _pf_i, _tid in enumerate(ok_ids):
        _sr = float(scores.get(str(_pf_i), {}).get("sr", -1.0))
        # sr < 0 => no episode finished => no partial progress
        _d = route_fn(max(_sr, 0.0), any_partial_progress=(_sr >= 0.0))
        if _d.action == "accept":
            kept.append(_tid)
            _clip = min(max(_sr, 0.0), 1.0)
            if tracker is not None and tracker.enabled:
                with tracker.span("archive_update"):
                    archive.update_node_learnability(_tid, _clip * (1.0 - _clip))
            else:
                archive.update_node_learnability(_tid, _clip * (1.0 - _clip))
        else:
            if tracker is not None and tracker.enabled:
                with tracker.span("archive_update"):
                    archive.update_node_status(_tid, f"preflight_{_d.reason}")
                    archive.set_task_active_status(_tid, False)
            else:
                archive.update_node_status(_tid, f"preflight_{_d.reason}")
                archive.set_task_active_status(_tid, False)
            print(f"  [Preflight] reject {_tid}: {_d.reason} (sr={_sr:.2f})")
