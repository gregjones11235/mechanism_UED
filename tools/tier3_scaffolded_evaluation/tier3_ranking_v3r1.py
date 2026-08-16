#!/usr/bin/env python3
"""CC4 Tier3 — V3R1 FRONT-first top-tie-only ranking machine (pure, JAX-free).

Implements the 总控 re-close ruling CC4_RECLOSE_FORMAL_GLOBAL_RANKING_WITH_
FRONT_FIRST_AND_TOP_TIE_ONLY. This module is the frozen comparison + grouping
contract; it is pure and side-effect free so the independent verifier and the
reclose driver can both import it.

Two semantic changes over the V3 (frozen V2DT) machine:

1. RANKING_PRIMARY_ORDER = FRONT_TRANSITION_FIRST. Lexicographic DESCENDING:
     (front_l2 transition_count,
      front_l2 mean graph_distance_progress,
      full success_count,
      back_l2 defeat_count)
   FULL success_count is NOT the first field. No reward/length/size/param/speed
   /age/complexity tie-breaks are permitted.

2. ONLY_TOP_TIE_BLOCKS_WINNER = true. A full-tuple tie only cancels the winner
   when it occurs in the TOP equivalence group:
     * top group size == 1  -> formal_winner = that candidate (unique top), even
       if lower positions tie  -> ranking_status ORDERED_WITH_LOWER_TIES;
     * top group size  > 1  -> formal_winner = None,
       ranking_status INCONCLUSIVE_TOP_TIE;
     * no ties at all       -> ranking_status ORDERED.
   Lower tie groups are disclosed (tie_group_rank + candidate_ids + scope
   LOWER_POSITION + winner_blocking=false) but never block the unique top, and
   no internal order is ever claimed for a tie group. candidate_id is used ONLY
   for deterministic output serialization, never as a scientific tie-break.

Usage:
  python tools/tier3_scaffolded_evaluation/tier3_ranking_v3r1.py --self-test
"""
from __future__ import annotations

import argparse
import math

# --- frozen protocol identifiers -------------------------------------------
RANKING_PROTOCOL = "TIER3_FRONT_FIRST_LEXICOGRAPHIC_V1"
FORMAL_RANKING_PROTOCOL = "V3R1_FRONT_FIRST_TOP_TIE_ONLY"
NEG20_PROTOCOL = "NEG20_V3R1_PRIMARY_NONREDUNDANT_SECONDARY"
SUMMARY_SCHEMA = "mechanism_UED.tier3_formal_ranking_summary/v3r1"
GATE_SCHEMA = "mechanism_UED.tier3_formal_evaluation_gate/v3r1"
READY_SCHEMA = "mechanism_UED.common_evaluator_v3r1_ranking_ready/v1"
ONLY_TOP_TIE_BLOCKS_WINNER = True
SELECTION_TIE_TOLERANCE = 1e-12

# New FRONT-first lexicographic order (frozen). FULL success_count is third.
RULE_ORDER = [
    "front_l2 transition_count",
    "front_l2 mean graph_distance_progress",
    "full success_count",
    "back_l2 defeat_count",
]
# Legacy V3/V2DT order, recorded for the old-vs-new disclosure only.
OLD_RULE_ORDER = [
    "full success_count",
    "front_l2 transition_count",
    "front_l2 mean graph_distance_progress",
    "back_l2 defeat_count",
]


class FailClosed(RuntimeError):
    """Fail-closed on malformed / non-finite ranking input."""


def _require(cond, msg):
    if not cond:
        raise FailClosed("FAIL CLOSED (RANK_V3R1): %s" % msg)


def validate_rule_tuple(rule_tuple):
    """Fail closed on NaN / Inf / missing / wrong-arity metric values."""
    _require(isinstance(rule_tuple, (list, tuple)) and len(rule_tuple) == 4,
             "rule_tuple must have exactly 4 levels, got %r" % (rule_tuple,))
    for i, v in enumerate(rule_tuple):
        _require(v is not None and isinstance(v, (int, float))
                 and not isinstance(v, bool),
                 "level %d (%s) missing or non-numeric: %r"
                 % (i, RULE_ORDER[i], v))
        fv = float(v)
        _require(math.isfinite(fv),
                 "level %d (%s) not finite: %r" % (i, RULE_ORDER[i], v))
    return tuple(float(v) for v in rule_tuple)


def compare_rule_tuples_v3r1(a, b, tol=SELECTION_TIE_TOLERANCE):
    """Lexicographic DESCENDING over the FRONT-first tuple with per-level tie
    tolerance. -1: a strictly ranks above b; 1: below; 0: full four-level tie."""
    a = validate_rule_tuple(a)
    b = validate_rule_tuple(b)
    for av, bv in zip(a, b):
        if av - bv > tol:
            return -1
        if bv - av > tol:
            return 1
    return 0


def rank_students_v3r1(entries):
    """entries: [{"candidate_id": str, "rule_tuple": (ft, fp, fs, bd)}].

    Returns a dict:
      ordered_groups       list of groups in strict descending order; each
                           {"candidate_ids" (sorted, stable output only),
                            "rule_tuple", "tie_group_rank", "size"}
      ranks                {cid: int|None} competition rank; None for any member
                           of a >=2 tie group (no internal order claimed)
      unique_top_candidate_id  the sole top candidate, or None if top ties
      formal_winner        == unique_top_candidate_id (top size 1) else None
      top_tie              True iff the top equivalence group has size > 1
      ranking_status       ORDERED | ORDERED_WITH_LOWER_TIES |
                           INCONCLUSIVE_TOP_TIE
      lower_tie_groups     [{"tie_group_rank","candidate_ids","tie_scope",
                            "winner_blocking"}] for ties BELOW the top group
      comparison_provenance  rule order / tolerance / protocol / tie policy
    """
    for e in entries:
        _require(isinstance(e, dict) and "candidate_id" in e
                 and "rule_tuple" in e, "bad entry: %r" % (e,))
        e["rule_tuple"] = validate_rule_tuple(e["rule_tuple"])
    # candidate_id is used ONLY to make the sort deterministic for equal tuples;
    # it never assigns a scientific rank (equal tuples land in the same group).
    ordered = sorted(entries,
                     key=lambda e: tuple(-v for v in e["rule_tuple"])
                     + (str(e["candidate_id"]),))
    groups = []
    for e in ordered:
        if (groups and compare_rule_tuples_v3r1(groups[-1][0]["rule_tuple"],
                                                e["rule_tuple"]) == 0):
            groups[-1].append(e)
        else:
            groups.append([e])

    ordered_groups = []
    ranks = {}
    lower_tie_groups = []
    pos = 1
    for gi, g in enumerate(groups):
        ids = sorted(str(e["candidate_id"]) for e in g)
        ordered_groups.append({
            "candidate_ids": ids,
            "rule_tuple": list(g[0]["rule_tuple"]),
            "tie_group_rank": pos,
            "size": len(g),
        })
        if len(g) > 1:
            for e in g:
                ranks[str(e["candidate_id"])] = None
            if gi > 0:  # a tie BELOW the top group: disclosed, non-blocking
                lower_tie_groups.append({
                    "tie_group_rank": pos,
                    "candidate_ids": ids,
                    "tie_scope": "LOWER_POSITION",
                    "winner_blocking": False,
                })
        else:
            ranks[str(g[0]["candidate_id"])] = pos
        pos += len(g)

    top_size = len(groups[0]) if groups else 0
    top_tie = top_size > 1
    unique_top_candidate_id = (
        str(groups[0][0]["candidate_id"]) if top_size == 1 else None)
    formal_winner = unique_top_candidate_id  # ONLY_TOP_TIE_BLOCKS_WINNER
    if not groups:
        ranking_status = "INCONCLUSIVE_TOP_TIE"
    elif top_tie:
        ranking_status = "INCONCLUSIVE_TOP_TIE"
    elif lower_tie_groups:
        ranking_status = "ORDERED_WITH_LOWER_TIES"
    else:
        ranking_status = "ORDERED"

    return {
        "ordered_groups": ordered_groups,
        "ranks": ranks,
        "unique_top_candidate_id": unique_top_candidate_id,
        "formal_winner": formal_winner,
        "top_tie": top_tie,
        "ranking_status": ranking_status,
        "lower_tie_groups": lower_tie_groups,
        "comparison_provenance": {
            "rule_order": list(RULE_ORDER),
            "old_rule_order": list(OLD_RULE_ORDER),
            "ranking_protocol": RANKING_PROTOCOL,
            "formal_ranking_protocol": FORMAL_RANKING_PROTOCOL,
            "tie_tolerance": SELECTION_TIE_TOLERANCE,
            "only_top_tie_blocks_winner": ONLY_TOP_TIE_BLOCKS_WINNER,
            "descending": True,
            "candidate_id_is_scientific_tiebreak": False,
        },
    }


# ---------------------------------------------------------------------------
# §九 self-tests A–K
# ---------------------------------------------------------------------------
def _mk(cid, ft, fp, fs, bd):
    return {"candidate_id": cid, "rule_tuple": (ft, fp, fs, bd)}


def run_self_test():
    checks = 0

    def ok(cond, msg):
        nonlocal checks
        checks += 1
        _require(cond, "SELF_TEST FAIL: %s" % msg)

    # A. 唯一第一，无其他并列 -> ORDERED + unique winner
    r = rank_students_v3r1([_mk("a", 3, 0.9, 9, 7),
                            _mk("b", 2, 0.5, 14, 8),
                            _mk("c", 1, 0.4, 17, 6)])
    ok(r["ranking_status"] == "ORDERED", "A status ORDERED")
    ok(r["formal_winner"] == "a" and r["top_tie"] is False, "A winner a")
    ok(r["lower_tie_groups"] == [], "A no lower ties")

    # B. 唯一第一，第三名并列 -> ORDERED_WITH_LOWER_TIES + winner kept
    r = rank_students_v3r1([_mk("a", 3, 0.9, 9, 7),
                            _mk("b", 2, 0.6, 14, 8),
                            _mk("x", 2, 0.5, 14, 8),
                            _mk("y", 2, 0.5, 14, 8)])
    ok(r["ranking_status"] == "ORDERED_WITH_LOWER_TIES", "B status lower ties")
    ok(r["formal_winner"] == "a", "B winner a kept despite lower tie")
    ok(len(r["lower_tie_groups"]) == 1
       and r["lower_tie_groups"][0]["candidate_ids"] == ["x", "y"]
       and r["lower_tie_groups"][0]["tie_group_rank"] == 3
       and r["lower_tie_groups"][0]["winner_blocking"] is False,
       "B lower tie group rank3 non-blocking")
    ok(r["ranks"]["x"] is None and r["ranks"]["y"] is None, "B tied rank None")
    ok(r["ranks"]["b"] == 2, "B rank b=2")

    # C. 第一名两人完全并列 -> INCONCLUSIVE_TOP_TIE + winner None
    r = rank_students_v3r1([_mk("p", 3, 0.9, 9, 7),
                            _mk("q", 3, 0.9, 9, 7),
                            _mk("c", 1, 0.4, 17, 6)])
    ok(r["ranking_status"] == "INCONCLUSIVE_TOP_TIE", "C status top tie")
    ok(r["formal_winner"] is None and r["top_tie"] is True, "C winner None")

    # D. FRONT transition 优先：A(ft=3,full=9) 先于 B(ft=2,full=17)
    r = rank_students_v3r1([_mk("B", 2, 0.5, 17, 8),
                            _mk("A", 3, 0.1, 9, 1)])
    ok(r["ordered_groups"][0]["candidate_ids"] == ["A"], "D FRONT-first A above B")
    ok(r["formal_winner"] == "A", "D winner A")

    # E. FRONT transition 相同 -> progress 决胜
    r = rank_students_v3r1([_mk("lo", 2, 0.3, 9, 7),
                            _mk("hi", 2, 0.8, 9, 7)])
    ok(r["formal_winner"] == "hi", "E progress higher wins")

    # F. FRONT 两项相同 -> FULL 决胜
    r = rank_students_v3r1([_mk("lo", 2, 0.5, 9, 7),
                            _mk("hi", 2, 0.5, 17, 7)])
    ok(r["formal_winner"] == "hi", "F full higher wins")

    # G. 前三项相同 -> BACK 决胜
    r = rank_students_v3r1([_mk("lo", 2, 0.5, 9, 3),
                            _mk("hi", 2, 0.5, 9, 8)])
    ok(r["formal_winner"] == "hi", "G back higher wins")

    # H. 四项完全相同 -> 同一 tie group
    r = rank_students_v3r1([_mk("g1", 2, 0.5, 9, 7),
                            _mk("g2", 2, 0.5, 9, 7)])
    ok(r["top_tie"] is True and r["ranking_status"] == "INCONCLUSIVE_TOP_TIE"
       and len(r["ordered_groups"]) == 1
       and r["ordered_groups"][0]["size"] == 2, "H same tie group")

    # I. 输入顺序置换 -> 结果/组/winner 完全一致
    base = [_mk("a", 3, 0.9, 9, 7), _mk("b", 2, 0.6, 14, 8),
            _mk("x", 2, 0.5, 14, 8), _mk("y", 2, 0.5, 14, 8),
            _mk("c", 0, 0.4, 0, 7)]
    import itertools
    ref = rank_students_v3r1([dict(e) for e in base])
    for perm in itertools.permutations(base):
        got = rank_students_v3r1([dict(e) for e in perm])
        ok(got["ordered_groups"] == ref["ordered_groups"]
           and got["formal_winner"] == ref["formal_winner"]
           and got["ranking_status"] == ref["ranking_status"]
           and got["lower_tie_groups"] == ref["lower_tie_groups"],
           "I permutation invariant")

    # J. NaN / Inf / missing -> fail closed
    for bad in [(float("nan"), 0.5, 9, 7), (2, float("inf"), 9, 7),
                (2, 0.5, None, 7), (2, 0.5, 9), ("2", 0.5, 9, 7)]:
        try:
            rank_students_v3r1([_mk("z", *bad)] if len(bad) == 4
                               else [{"candidate_id": "z", "rule_tuple": bad}])
            ok(False, "J accepted bad tuple %r" % (bad,))
        except FailClosed:
            ok(True, "J fail-closed on %r" % (bad,))

    # K. teacher 参考：_machine_ 只排学生；teacher 由驱动排除。此处验证若误把
    #    teacher 当学生传入它不会获得任何特殊豁免（与学生同等对待），真正的
    #    “teacher 不得成 winner / student_rank=null” 在驱动/复验器用真实数据断言。
    r = rank_students_v3r1([_mk("s1", 3, 0.9, 9, 7),
                            _mk("teacher", 5, 0.99, 19, 7)])
    ok("teacher" in r["ranks"], "K machine treats teacher as plain entry")

    print("RANKING_V3R1_SELF_TEST_PASS checks=%d" % checks)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return run_self_test()
    ap.error("this module is a pure ranking machine; use --self-test")


if __name__ == "__main__":
    raise SystemExit(main())
