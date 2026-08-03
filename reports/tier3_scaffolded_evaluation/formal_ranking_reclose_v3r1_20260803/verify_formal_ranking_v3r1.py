#!/usr/bin/env python3
"""CC4 Tier3 — INDEPENDENT verifier for the V3R1 ranking re-close.

Independently re-derives every claim of the V3R1 re-close from the IMMUTABLE
old-V3 evidence and cross-checks the published V3R1 artifacts. It trusts
NOTHING the reclose driver wrote: it re-hashes the old evidence, re-extracts
the primary metrics, re-classifies the episode records, and re-runs the frozen
FRONT-first top-tie-only ranking function itself, then compares to the
published summary / gate / READY.

Verified invariants (总控 §三–§十二):
  * old V3 + V2 evidence byte-immutable vs the pre-reclose snapshots;
  * old V3 summary/gate hashes == frozen pins == sidecars;
  * per-candidate PRIMARY metric parity old-V3 == V3R1 (4 frozen metrics);
  * FRONT-first order; FULL success_count is NOT the first field;
  * winner == ranking-function output (never a constant); permutation invariant;
  * top-tie-only winner policy; lower ties disclosed non-blocking;
  * teacher reference-only, never the winner, student_rank null;
  * secondary dedup removes BACK/FULL primary duplication only; FRONT untouched;
  * gate authorized with retained scientific-claim constraints;
  * COMMON_EVALUATOR_V3_READY.json NOT overwritten (V3R1 READY is separate);
  * NaN/Inf/missing metric values fail closed.

Usage:
  python verify_formal_ranking_v3r1.py --evidence-dir D --v2-evidence-dir D \
      --reclose-dir D --baseline-dir D
Exit 0 = ALL GREEN; any failure -> non-zero (fail closed).
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys

# Import the FROZEN pure machines under test (same functions the driver used).
_TOOLS = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
    "tools", "tier3_scaffolded_evaluation"))
sys.path.insert(0, _TOOLS)
import tier3_ranking_v3r1 as rankmod          # noqa: E402
import tier3_taxonomy_v3 as t3                # noqa: E402
import tier3_taxonomy_v3r1 as t3r1            # noqa: E402

FROZEN_V3_SUMMARY_SHA256 = ("dab522cf7bcc43ed74f0bc1e9cab20c01c98d972d7ed"
                            "ceb2717f9dc18445b659")
FROZEN_V3_GATE_SHA256 = ("c529ebf3ddbf37085b85b0a79018d9cc06ce5a096dc744d1"
                         "8a97a3e0c8b72528")

STUDENTS = [
    "BASE_GTRXL_ORIGINAL_VTRACE_98304",
    "CONTROL_CONTINUOUS_98304",
    "PERSISTENT_RMT16_ORIGINAL_VTRACE_98304",
    "RESET128_RMT16_ORIGINAL_VTRACE_98304",
    "SLOWGRU_PERSISTENT_CANONICAL_98304",
    "SLOWGRU_RESET128_CANONICAL_98304",
]
TEACHER = "BASELINE_TEACHER_CKPT17500"
ALL_CANDIDATES = sorted(STUDENTS + [TEACHER])
SCENARIOS = ("full", "front_l2", "back_l2")
EXPECTED_VALID_STARTS = {"full": 64, "front_l2": 8, "back_l2": 8}
EXPECTED_EPISODES = {"full": 64, "front_l2": 8, "back_l2": 8}


class VerifyFail(RuntimeError):
    pass


_CHECKS = [0]


def ok(cond, msg):
    _CHECKS[0] += 1
    if not cond:
        raise VerifyFail("CHECK #%d FAILED: %s" % (_CHECKS[0], msg))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Independent metric extraction (mirrors the frozen evaluator semantics)
# ---------------------------------------------------------------------------
def extract_metrics(evidence_dir, cid):
    fdir = os.path.join(evidence_dir, "cc4", cid, "formal_evaluation_v3")
    results = {}
    for sc in SCENARIOS:
        results[sc] = _load_json(os.path.join(
            fdir, "evaluation_result_v3.%s.json" % sc))
    full_m = results["full"]["evaluation"]["metrics"]
    front_m = results["front_l2"]["evaluation"]["metrics"]
    back_m = results["back_l2"]["evaluation"]["metrics"]
    ok(full_m["primary"]["valid_starts"] == EXPECTED_VALID_STARTS["full"],
       "%s full valid_starts" % cid)
    ok(front_m["primary"]["valid_starts"] == EXPECTED_VALID_STARTS["front_l2"],
       "%s front valid_starts" % cid)
    ok(back_m["primary"]["valid_starts"] == EXPECTED_VALID_STARTS["back_l2"],
       "%s back valid_starts" % cid)
    ft = front_m["primary"]["successes"]
    fp = front_m["dense"]["value"]
    fs = full_m["primary"]["successes"]
    bd = back_m["primary"]["successes"]
    ok(isinstance(fp, (int, float)) and fp == fp
       and fp not in (float("inf"), float("-inf")),
       "%s front progress finite" % cid)
    ok(back_m.get("diagnostics", {}).get("survival", {}).get("defeat_count") == bd,
       "%s back defeat == diagnostics" % cid)
    return {"front_l2 transition_count": ft,
            "front_l2 mean graph_distance_progress": fp,
            "full success_count": fs,
            "back_l2 defeat_count": bd}, results


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evidence-dir", required=True)
    ap.add_argument("--v2-evidence-dir", required=True)
    ap.add_argument("--reclose-dir", required=True)
    ap.add_argument("--baseline-dir", required=True)
    ap.add_argument("--allow-missing-ready", action="store_true",
                    help="embedded driver mode: READY / SHA256SUMS_V3R1 / report "
                         "are not written yet when the driver invokes this "
                         "verifier; a FINAL full run without this flag must "
                         "verify them")
    args = ap.parse_args(argv)

    evidence_dir = os.path.abspath(args.evidence_dir)
    v2_evidence_dir = os.path.abspath(args.v2_evidence_dir)
    reclose_dir = os.path.abspath(args.reclose_dir)
    baseline_dir = os.path.abspath(args.baseline_dir)

    # ---- Section 1: old V3 / V2 byte-immutability vs pre-reclose snapshots ----
    for tag, ev_dir, snap in (("old_v3", evidence_dir, "old_v3.sha"),
                              ("old_v2", v2_evidence_dir, "old_v2.sha")):
        snap_path = os.path.join(baseline_dir, snap)
        ok(os.path.isfile(snap_path), "%s baseline snapshot present" % tag)
        entries = {}
        with open(snap_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if not line:
                    continue
                sha, rel = line.split(None, 1)
                rel = rel.strip().lstrip("*")
                while rel.startswith("./"):
                    rel = rel[2:]
                entries[rel] = sha
        live = {}
        for root, _d, files in os.walk(ev_dir):
            for f in files:
                p = os.path.join(root, f)
                rel = os.path.relpath(p, ev_dir).replace(os.sep, "/")
                live[rel] = sha256_file(p)
        ok(set(live) == set(entries),
           "%s file set identical to snapshot (no add/remove)" % tag)
        for rel, sha in entries.items():
            ok(live.get(rel) == sha, "%s byte-immutable: %s" % (tag, rel))

    # ---- Section 2: old V3 summary/gate hashes == pins == sidecars ----
    cc4 = os.path.join(evidence_dir, "cc4")
    summ_path = os.path.join(cc4, "FORMAL_RANKING_SUMMARY_V3.json")
    gate_path = os.path.join(cc4, "FORMAL_EVALUATION_GATE_V3.json")
    summ_sha = sha256_file(summ_path)
    gate_sha = sha256_file(gate_path)
    side_s = open(summ_path + ".sha256", encoding="utf-8").read().split()[0]
    side_g = open(gate_path + ".sha256", encoding="utf-8").read().split()[0]
    ok(summ_sha == side_s == FROZEN_V3_SUMMARY_SHA256, "old summary sha pinned")
    ok(gate_sha == side_g == FROZEN_V3_GATE_SHA256, "old gate sha pinned")
    old_summary = _load_json(summ_path)
    old_gate = _load_json(gate_path)
    ok(old_gate.get("formal_ranking_summary_sha256") == summ_sha,
       "old gate links old summary")
    ok(old_summary["ranking_status"] == "INCONCLUSIVE_FULL_TIE"
       and old_summary["formal_winner"] is None,
       "old V3 close is INCONCLUSIVE_FULL_TIE / winner null (baseline)")
    ok(old_summary["schema"] == "mechanism_UED.tier3_formal_ranking_summary/v3",
       "old summary schema is v3 (not v3r1)")

    # ---- Section 3: independent primary metric extraction + parity ----
    old_participants = {p["candidate_id"]: p
                        for p in old_summary["participants"]}
    ok(sorted(old_participants) == ALL_CANDIDATES, "participant set")
    indep = {}
    for cid in ALL_CANDIDATES:
        m, _res = extract_metrics(evidence_dir, cid)
        indep[cid] = m
        # Parity with the OLD V3 published rule_tuple (the 4 frozen metrics).
        for name in ("front_l2 transition_count",
                     "front_l2 mean graph_distance_progress",
                     "full success_count", "back_l2 defeat_count"):
            ok(old_participants[cid]["rule_tuple"][name] == m[name],
               "PRIMARY PARITY %s %s: old==indep" % (cid, name))

    # Highest FRONT transition_count among students must be the eventual winner
    # ONLY if the ranking function says so; we do NOT hardcode the winner.

    # ---- Section 4: independent FRONT-first recompute ----
    entries = [{"candidate_id": c,
                "rule_tuple": (indep[c]["front_l2 transition_count"],
                               indep[c]["front_l2 mean graph_distance_progress"],
                               indep[c]["full success_count"],
                               indep[c]["back_l2 defeat_count"])}
               for c in STUDENTS]
    ranking = rankmod.rank_students_v3r1([dict(e) for e in entries])
    ok(ranking["comparison_provenance"]["rule_order"][0]
       == "front_l2 transition_count", "FRONT-first is field 1")
    ok(ranking["comparison_provenance"]["rule_order"]
       != ["full success_count"] + ranking["comparison_provenance"]
       ["rule_order"][1:], "FULL success_count is NOT the first field")
    ok(TEACHER not in ranking["ranks"], "teacher not in student ranking")
    winner = ranking["formal_winner"]
    ok(winner is None or winner in STUDENTS, "winner is a student or null")
    ok(winner != TEACHER, "teacher is never the winner")

    # Permutation invariance on the REAL metric set.
    ref_groups = ranking["ordered_groups"]
    ref_winner = ranking["formal_winner"]
    ref_status = ranking["ranking_status"]
    for perm in itertools.permutations(entries):
        got = rankmod.rank_students_v3r1([dict(e) for e in perm])
        ok(got["ordered_groups"] == ref_groups
           and got["formal_winner"] == ref_winner
           and got["ranking_status"] == ref_status
           and got["lower_tie_groups"] == ranking["lower_tie_groups"],
           "permutation invariance on real metrics")

    # Winner is COMPUTED, not a constant: perturbing one metric must move the
    # ranking output (proves the result tracks data, not a hardcoded string).
    base_tuple = dict(indep[STUDENTS[0]])
    moved = False
    for cid in STUDENTS:
        for key in base_tuple:
            pert = [dict(e) for e in entries]
            for e in pert:
                if e["candidate_id"] == cid:
                    t = list(e["rule_tuple"])
                    idx = ["front_l2 transition_count",
                           "front_l2 mean graph_distance_progress",
                           "full success_count",
                           "back_l2 defeat_count"].index(key)
                    t[idx] = t[idx] + 1 if idx != 1 else min(t[idx] + 0.1, 1.0)
                    e["rule_tuple"] = tuple(t)
            r2 = rankmod.rank_students_v3r1(pert)
            if (r2["ordered_groups"] != ref_groups
                    or r2["formal_winner"] != ref_winner):
                moved = True
                break
        if moved:
            break
    ok(moved, "ranking output changes when a metric changes (not hardcoded)")

    # NaN / Inf / missing fail closed (independent reproduction of test J).
    for bad in [(float("nan"), 0.5, 9, 7), (2, float("inf"), 9, 7),
                (2, 0.5, None, 7), (2, 0.5, 9)]:
        try:
            rankmod.rank_students_v3r1(
                [{"candidate_id": "x", "rule_tuple": bad}])
            ok(False, "accepted malformed tuple %r" % (bad,))
        except rankmod.FailClosed:
            ok(True, "fail-closed on %r" % (bad,))

    # ---- Section 5: secondary dedup reclassification (independent) ----
    for cid in ALL_CANDIDATES:
        fdir = os.path.join(evidence_dir, "cc4", cid, "formal_evaluation_v3")
        by_sc = {sc: [] for sc in SCENARIOS}
        with open(os.path.join(fdir, "episode_records.jsonl"),
                  encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    by_sc[rec["scenario"]].append(rec)
        for sc in SCENARIOS:
            ok(len(by_sc[sc]) == EXPECTED_EPISODES[sc],
               "%s/%s episode count" % (cid, sc))
            for rec in by_sc[sc]:
                v3c = t3.classify_episode_v3(sc, rec)
                r = t3r1.classify_episode_v3r1(sc, rec)
                ok(r["primary_outcome"] == v3c["primary_outcome"]
                   and r["taxonomy_status"] == v3c["taxonomy_status"]
                   and r["composite"] == v3c["composite"],
                   "%s/%s/%s primary classification unchanged"
                   % (cid, sc, rec["episode_id"]))
                if sc == "front_l2":
                    ok(r["secondary_events"] == list(v3c["secondary_events"]),
                       "%s FRONT secondary untouched" % cid)
                else:
                    ok("DEFEAT_KOBOLD" not in r["secondary_events"],
                       "%s/%s DEFEAT_KOBOLD removed from secondary" % (cid, sc))

    # ---- Section 6: published V3R1 artifacts ----
    rsumm_path = os.path.join(reclose_dir, "FORMAL_RANKING_SUMMARY_V3R1.json")
    rgate_path = os.path.join(reclose_dir, "FORMAL_EVALUATION_GATE_V3R1.json")
    ready_path = os.path.join(reclose_dir,
                              "COMMON_EVALUATOR_V3R1_RANKING_READY.json")
    sums_path = os.path.join(reclose_dir, "SHA256SUMS_V3R1")
    # summary + gate always exist at this point; READY + SHA256SUMS are written
    # AFTER the driver's embedded verifier invocation, so they are only required
    # on the final full run (no --allow-missing-ready).
    for p in (rsumm_path, rgate_path):
        ok(os.path.isfile(p), "published artifact present: %s"
           % os.path.basename(p))
    ready_exists = os.path.isfile(ready_path)
    sums_exists = os.path.isfile(sums_path)
    if not args.allow_missing_ready:
        ok(ready_exists, "published artifact present: COMMON_EVALUATOR_V3R1_"
                         "RANKING_READY.json (full run)")
        ok(sums_exists, "published artifact present: SHA256SUMS_V3R1 (full run)")
    rsumm = _load_json(rsumm_path)
    rgate = _load_json(rgate_path)
    ready = _load_json(ready_path) if ready_exists else None

    # sidecars + SHA256SUMS_V3R1 must verify byte-for-byte.
    for p in (rsumm_path, rgate_path):
        side = open(p + ".sha256", encoding="utf-8").read().split()[0]
        ok(side == sha256_file(p), "sidecar matches: %s" % os.path.basename(p))
    if sums_exists:
        with open(sums_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if not line:
                    continue
                sha, rel = line.split(None, 1)
                ok(sha256_file(os.path.join(reclose_dir, rel.strip())) == sha,
                   "SHA256SUMS_V3R1 verifies: %s" % rel.strip())

    ok(rsumm["schema"] == rankmod.SUMMARY_SCHEMA, "summary schema v3r1")
    ok(rgate["schema"] == rankmod.GATE_SCHEMA, "gate schema v3r1")
    ok(rsumm["RANKING_PROTOCOL"] == "TIER3_FRONT_FIRST_LEXICOGRAPHIC_V1",
       "RANKING_PROTOCOL frozen id")
    ok(rsumm["FORMAL_RANKING_PROTOCOL"] == "V3R1_FRONT_FIRST_TOP_TIE_ONLY",
       "FORMAL_RANKING_PROTOCOL frozen id")
    ok(rsumm["RANKING_PRIMARY_ORDER"] == "FRONT_TRANSITION_FIRST",
       "RANKING_PRIMARY_ORDER")
    ok(rsumm["ONLY_TOP_TIE_BLOCKS_WINNER"] is True, "ONLY_TOP_TIE_BLOCKS_WINNER")
    ok(rsumm["NEW_RULE_ORDER"][0] == "front_l2 transition_count",
       "NEW_RULE_ORDER field 1 = FRONT transition")
    ok(rsumm["OLD_RULE_ORDER"][0] == "full success_count",
       "OLD_RULE_ORDER disclosed (FULL first)")
    ok(rsumm["NEG20_PROTOCOL"] == "NEG20_V3R1_PRIMARY_NONREDUNDANT_SECONDARY",
       "NEG20 v3r1 protocol")
    ok(rsumm["SECONDARY_EVENT_PRIMARY_DUPLICATION_REMOVED"] is True,
       "dedup flag published")
    ok(rsumm["source_v3_evidence"]["FORMAL_RANKING_SUMMARY_V3_sha256"]
       == summ_sha, "reclose cites real old summary sha")
    ok(rsumm["source_v3_evidence"]["FORMAL_EVALUATION_GATE_V3_sha256"]
       == gate_sha, "reclose cites real old gate sha")
    ok(rsumm["old_v3_reference"]["OLD_V3_RANKING_STATUS"]
       == "INCONCLUSIVE_FULL_TIE", "old status disclosed")
    ok(rsumm["old_v3_reference"]["OLD_V3_FORMAL_WINNER"] is None,
       "old winner null disclosed")

    # Published ranking must equal the independent recompute EXACTLY.
    ok(rsumm["formal_winner"] == winner, "published winner == recomputed")
    ok(rsumm["ranking_status"] == ranking["ranking_status"],
       "published ranking_status == recomputed")
    ok(rsumm["ordered_groups"] == ranking["ordered_groups"],
       "published ordered_groups == recomputed")
    ok(rsumm["lower_tie_groups"] == ranking["lower_tie_groups"],
       "published lower_tie_groups == recomputed")
    ok(rsumm["TOP_TIE"] == ranking["top_tie"], "TOP_TIE matches")
    if ranking["ranking_status"] == "ORDERED_WITH_LOWER_TIES":
        ok(rsumm["formal_winner"] is not None,
           "unique top winner kept despite lower ties")
        ok(all(g["winner_blocking"] is False
               for g in rsumm["lower_tie_groups"]),
           "lower ties are winner_blocking=false")
        ok(all(g["tie_scope"] == "LOWER_POSITION"
               for g in rsumm["lower_tie_groups"]),
           "lower ties scoped LOWER_POSITION")
    if ranking["ranking_status"] == "INCONCLUSIVE_TOP_TIE":
        ok(rsumm["formal_winner"] is None, "top tie -> winner null")

    # Per-participant parity + tie bookkeeping + teacher exclusion.
    rparts = {p["candidate_id"]: p for p in rsumm["participants"]}
    ok(sorted(rparts) == ALL_CANDIDATES, "reclose participant set")
    for cid in ALL_CANDIDATES:
        p = rparts[cid]
        rt = p["front_first_rule_tuple"]
        ok(rt["front_l2 transition_count"]
           == indep[cid]["front_l2 transition_count"]
           and rt["front_l2 mean graph_distance_progress"]
           == indep[cid]["front_l2 mean graph_distance_progress"]
           and rt["full success_count"] == indep[cid]["full success_count"]
           and rt["back_l2 defeat_count"] == indep[cid]["back_l2 defeat_count"],
           "published tuple == independent metrics (%s)" % cid)
        # old-vs-new metric parity per candidate
        for name, key in (("front_l2 transition_count",
                           "front_l2 transition_count"),
                          ("full success_count", "full success_count"),
                          ("back_l2 defeat_count", "back_l2 defeat_count")):
            ok(old_participants[cid]["rule_tuple"][name] == rt[key],
               "%s parity %s" % (cid, name))
    tp = rparts[TEACHER]
    ok(tp["excluded_from_student_ranking"] is True
       and tp["student_rank"] is None
       and tp["candidate_class"] == "TEACHER_REFERENCE"
       and tp["tie_status"] == "TEACHER_REFERENCE_ONLY",
       "teacher reference-only / rank null")
    for cid in STUDENTS:
        sp = rparts[cid]
        ok(sp["student_rank"] == ranking["ranks"][cid],
           "%s student_rank matches machine" % cid)
        ok(sp["excluded_from_student_ranking"] is False,
           "%s not excluded" % cid)
    ok(rsumm["teacher_included_in_student_ranking"] is False,
       "teacher not in ranking (summary flag)")
    ok(rsumm["scientific_claim_authorized"] is False
       and rsumm["single_training_seed"] is True
       and rsumm["multi_seed_confirmation_skipped_by_director"] is True
       and rsumm["scaffolded_results_can_replace_full_task"] is False,
       "retained scientific constraints in summary")
    ok(rsumm["ENVIRONMENT_RERUNS"] == 0 and rsumm["STUDENTS_RETRAINED"] == 0
       and rsumm["CHECKPOINTS_MODIFIED"] is False
       and rsumm["CANDIDATE_EXCEPTIONS_USED"] == 0,
       "no reruns / retraining / checkpoint mods / exemptions")

    # Published V3R1 secondary counts must equal independent reclassification.
    for cid in ALL_CANDIDATES:
        fdir = os.path.join(evidence_dir, "cc4", cid, "formal_evaluation_v3")
        v3r1_counts = {sc: {} for sc in SCENARIOS}
        with open(os.path.join(fdir, "episode_records.jsonl"),
                  encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                r = t3r1.classify_episode_v3r1(rec["scenario"], rec)
                for ev in r["secondary_events"]:
                    v3r1_counts[rec["scenario"]][ev] = \
                        v3r1_counts[rec["scenario"]].get(ev, 0) + 1
        published = rparts[cid]["v3r1_secondary_event_counts_by_scenario"]
        for sc in SCENARIOS:
            ok(published.get(sc, {}) == dict(sorted(v3r1_counts[sc].items())),
               "%s/%s v3r1 secondary counts match reclassification" % (cid, sc))
            if sc != "front_l2":
                ok("DEFEAT_KOBOLD" not in published.get(sc, {}),
                   "%s/%s no DEFEAT_KOBOLD in published secondary" % (cid, sc))

    # ---- Section 7: gate + READY ----
    ok(rgate["FORMAL_RANKING_AUTHORIZED_V3R1"]
       == all(rgate["gates"].values()),
       "gate authorized iff all gates true")
    ok(all(rgate["gates"].values()), "all reclose gates green")
    ok(rgate["formal_ranking_summary_sha256"] == sha256_file(rsumm_path),
       "gate links published summary sha")
    ok(rgate["formal_winner"] == winner, "gate winner == recomputed")
    rc = rgate["retained_constraints"]
    ok(rc["scientific_claim_authorized"] is False
       and rc["single_training_seed"] is True
       and rc["multi_seed_confirmation_skipped_by_director"] is True
       and rc["scaffolded_results_can_replace_full_task"] is False,
       "retained constraints in gate")
    ok(rgate["prohibitions_honored"]["rollouts_rerun"] is False
       and rgate["prohibitions_honored"]["retraining_performed"] is False
       and rgate["prohibitions_honored"]["checkpoints_modified"] is False
       and rgate["prohibitions_honored"]["full_first_ordering_used"] is False
       and rgate["prohibitions_honored"]["v2_v3_evidence_overwritten"] is False,
       "prohibitions honored in gate")

    if ready is not None:
        ok(ready["FORMAL_RANKING_PROTOCOL"] == "V3R1_FRONT_FIRST_TOP_TIE_ONLY",
           "READY protocol")
        ok(ready["SUMMARY_SHA256"] == sha256_file(rsumm_path),
           "READY summary sha")
        ok(ready["GATE_SHA256"] == sha256_file(rgate_path), "READY gate sha")
        ok(ready["SOURCE_V3_SUMMARY_SHA256"] == summ_sha,
           "READY source summary sha")
        ok(ready["SOURCE_V3_GATE_SHA256"] == gate_sha, "READY source gate sha")
        ok(ready["OLD_V3_IMMUTABLE"] is True, "READY OLD_V3_IMMUTABLE")
        ok(ready["PRIMARY_METRIC_PARITY"] is True, "READY PRIMARY_METRIC_PARITY")
        ok(ready["RANKING_STATUS"] == ranking["ranking_status"], "READY status")
        ok(ready["FORMAL_WINNER"] == winner, "READY winner")
        ok(ready["TOP_TIE"] == ranking["top_tie"], "READY top tie")
        ok(ready["LOWER_TIE_GROUPS"] == ranking["lower_tie_groups"],
           "READY lower ties")
        ok(ready["pending_gates"] == [], "READY pending_gates empty")
        ok(ready["schema"] == rankmod.READY_SCHEMA, "READY schema v3r1")
        ok(ready["COMMON_EVALUATOR_V3R1_RANKING_READY"]
           == ready["FORMAL_RANKING_AUTHORIZED_V3R1"]
           == rgate["FORMAL_RANKING_AUTHORIZED_V3R1"],
           "READY authorization consistent with gate")

    # ---- Section 8: old V3 READY NOT overwritten; new artifacts are separate ----
    old_ready = os.path.join(cc4, "COMMON_EVALUATOR_V3_READY.json")
    ok(os.path.isfile(old_ready), "old COMMON_EVALUATOR_V3_READY.json present")
    orj = _load_json(old_ready)
    ok(orj.get("schema", "").startswith("mechanism_UED.tier3_common_evaluator"),
       "old READY schema untouched")
    ok("V3R1" not in orj.get("schema", ""), "old READY not masqueraded as v3r1")
    # The V3R1 READY must be a SEPARATE file, not the old one.
    ok(os.path.basename(ready_path) == "COMMON_EVALUATOR_V3R1_RANKING_READY.json",
       "V3R1 READY is a distinct standalone file")

    # No V3R1 artifact may masquerade with a v3/v2 schema string.
    masq_targets = [rsumm_path, rgate_path] + ([ready_path] if ready_exists
                                               else [])
    for p in masq_targets:
        body = open(p, encoding="utf-8").read()
        ok('"mechanism_UED.tier3_formal_ranking_summary/v3"' not in body,
           "%s does not claim the old v3 summary schema" % os.path.basename(p))
    # The new directory must not contain any file pretending to be the old READY.
    ok(not os.path.isfile(os.path.join(reclose_dir,
                                       "COMMON_EVALUATOR_V3_READY.json")),
       "new dir does not replicate the old V3 READY filename")

    print("VERIFY_FORMAL_RANKING_V3R1_PASS checks=%d winner=%s status=%s"
          % (_CHECKS[0], winner, ranking["ranking_status"]))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerifyFail as exc:
        print("VERIFY_FORMAL_RANKING_V3R1_FAIL: %s" % exc, file=sys.stderr)
        raise SystemExit(2)
