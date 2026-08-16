#!/usr/bin/env python3
"""CC4 Tier3 — V3R1 formal ranking re-close driver (OFFLINE; no environment).

Implements 总控 ruling CC4_RECLOSE_FORMAL_GLOBAL_RANKING_WITH_FRONT_FIRST_AND_
TOP_TIE_ONLY. This driver is NOT a re-evaluation: it performs NO rollouts, NO
retraining, NO checkpoint modification. It ONLY:

  1. audits that the old V3 (and V2) evidence directories are byte-immutable
     against the pre-reclose SHA snapshots;
  2. re-extracts the four frozen primary metrics per candidate from the
     EXISTING immutable V3 result files and asserts bit-parity with the old
     V3 summary (any drift -> fail closed, no publication);
  3. offline-reclassifies every existing episode record with the V3R1
     secondary-dedup taxonomy (NEG20_V3R1_PRIMARY_NONREDUNDANT_SECONDARY);
  4. re-ranks the 6 students with the frozen FRONT-first lexicographic rule
     (RANKING_PRIMARY_ORDER=FRONT_TRANSITION_FIRST) and the top-tie-only
     winner policy (ONLY_TOP_TIE_BLOCKS_WINNER=true);
  5. publishes the V3R1 artifacts into a NEW directory, overwriting nothing
     of V2/V3.

The old V3 close (INCONCLUSIVE_FULL_TIE, winner null) was the correct result
under its own protocol; V3R1 is 总控's formal clarification and close, not a
correction of wrong results.

Usage:
  python tools/tier3_scaffolded_evaluation/tier3_formal_reclose_v3r1.py \
      [--evidence-dir PATH] [--v2-evidence-dir PATH] [--out-dir PATH] \
      [--baseline-dir PATH] [--run-verifier PATH] [--dry-run]
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tier3_ranking_v3r1 as rankmod          # noqa: E402
import tier3_taxonomy_v3 as t3                # noqa: E402
import tier3_taxonomy_v3r1 as t3r1            # noqa: E402

# ---------------------------------------------------------------------------
# Frozen pins (from the V3 close; the reclose must start from exactly here)
# ---------------------------------------------------------------------------
EXPECTED_BASE_COMMIT = "3c36ae620d450314a0e034bc80e73210f4784296"
RULING_TASK = ("CC4_RECLOSE_FORMAL_GLOBAL_RANKING_WITH_FRONT_FIRST_AND_"
               "TOP_TIE_ONLY")

FROZEN_V3_SUMMARY_SHA256 = ("dab522cf7bcc43ed74f0bc1e9cab20c01c98d972d7ed"
                            "ceb2717f9dc18445b659")
FROZEN_V3_GATE_SHA256 = "c529ebf3ddbf37085b85b0a79018d9cc06ce5a096dc744d18a97a3e0c8b72528"
FROZEN_METRIC_SCHEMA_SHA256 = ("8ec4adcdfa6844b276f5f253470e14ea8ad52f1e64c3"
                               "98e5e2658e8a066645c7")

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

# Old (V3/V2DT, FULL-first) rule order vs new (V3R1, FRONT-first) rule order.
OLD_RULE_ORDER = list(rankmod.OLD_RULE_ORDER)
NEW_RULE_ORDER = list(rankmod.RULE_ORDER)


class FailClosed(RuntimeError):
    pass


def _require(cond, msg):
    if not cond:
        raise FailClosed("FAIL CLOSED (RECLOSE_V3R1): %s" % msg)


def utc_now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json_atomic(path, obj):
    data = (json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False)
            + "\n").encode("utf-8")
    tmp = path + ".tmp_v3r1"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)
    return hashlib.sha256(data).hexdigest()


def write_text_atomic(path, text):
    tmp = path + ".tmp_v3r1"
    with open(tmp, "wb") as fh:
        fh.write(text.encode("utf-8"))
    os.replace(tmp, path)


def git_head(repo):
    out = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"],
                         capture_output=True, text=True)
    _require(out.returncode == 0, "git rev-parse HEAD failed: %s" % out.stderr)
    return out.stdout.strip()


def git_branch(repo):
    out = subprocess.run(["git", "-C", repo, "rev-parse", "--abbrev-ref", "HEAD"],
                         capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else "UNKNOWN"


# ---------------------------------------------------------------------------
# Stage 1 — immutability audit of the OLD evidence directories
# ---------------------------------------------------------------------------
def _load_baseline(baseline_dir, name):
    path = os.path.join(baseline_dir, name)
    _require(os.path.isfile(path),
             "baseline snapshot missing: %s (run the pre-reclose snapshot first)"
             % path)
    entries = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            sha, rel = line.split(None, 1)
            rel = rel.strip().lstrip("*")
            while rel.startswith("./"):
                rel = rel[2:]
            entries[rel] = sha
    _require(len(entries) > 0, "baseline snapshot empty: %s" % path)
    return path, entries


def audit_immutability(evidence_dir, v2_evidence_dir, baseline_dir):
    """Re-hash every file under the old V3/V2 dirs and compare to snapshots."""
    report = {
        "schema": "mechanism_UED.tier3_old_v3_immutability_report/v3r1",
        "generated_at_utc": utc_now_iso(),
        "ruling_task": RULING_TASK,
        "baseline_snapshot_dir": os.path.abspath(baseline_dir),
    }
    for tag, ev_dir, snap_name in (
            ("old_v3", evidence_dir, "old_v3.sha"),
            ("old_v2", v2_evidence_dir, "old_v2.sha")):
        snap_path, entries = _load_baseline(baseline_dir, snap_name)
        modified, drift, missing, extra = [], [], [], []
        live = {}
        for root, _dirs, files in os.walk(ev_dir):
            for f in files:
                p = os.path.join(root, f)
                rel = os.path.relpath(p, ev_dir).replace(os.sep, "/")
                live[rel] = sha256_file(p)
        for rel, sha in sorted(entries.items()):
            if rel not in live:
                missing.append(rel)
            elif live[rel] != sha:
                modified.append(rel)
                drift.append(rel)
        for rel in sorted(live):
            if rel not in entries:
                extra.append(rel)
        report[tag] = {
            "evidence_dir": os.path.abspath(ev_dir),
            "snapshot_file": os.path.abspath(snap_path),
            "snapshot_file_sha256": sha256_file(snap_path),
            "snapshot_file_count": len(entries),
            "live_file_count": len(live),
            "modified_files": modified,
            "hash_drift_files": drift,
            "missing_files": missing,
            "extra_files": extra,
            "files_modified_count": len(modified),
            "hash_drift_count": len(drift),
        }
    report["OLD_V3_FILES_MODIFIED"] = report["old_v3"]["files_modified_count"]
    report["OLD_V3_HASH_DRIFT"] = report["old_v3"]["hash_drift_count"]
    report["V2_FILES_MODIFIED"] = report["old_v2"]["files_modified_count"]
    report["OLD_V3_EXTRA_FILES"] = len(report["old_v3"]["extra_files"])
    report["OLD_V3_MISSING_FILES"] = len(report["old_v3"]["missing_files"])
    report["V2_EXTRA_FILES"] = len(report["old_v2"]["extra_files"])
    report["V2_MISSING_FILES"] = len(report["old_v2"]["missing_files"])
    report["IMMUTABLE"] = (
        report["OLD_V3_FILES_MODIFIED"] == 0
        and report["OLD_V3_HASH_DRIFT"] == 0
        and report["V2_FILES_MODIFIED"] == 0
        and report["OLD_V3_EXTRA_FILES"] == 0
        and report["OLD_V3_MISSING_FILES"] == 0
        and report["V2_EXTRA_FILES"] == 0
        and report["V2_MISSING_FILES"] == 0)
    _require(report["IMMUTABLE"],
             "old V3/V2 evidence changed since the pre-reclose snapshot: %s"
             % json.dumps({k: report[k] for k in
                           ("OLD_V3_FILES_MODIFIED", "OLD_V3_HASH_DRIFT",
                            "V2_FILES_MODIFIED", "OLD_V3_EXTRA_FILES",
                            "OLD_V3_MISSING_FILES", "V2_EXTRA_FILES",
                            "V2_MISSING_FILES")}))
    return report


# ---------------------------------------------------------------------------
# Stage 2 — verify the old V3 artifacts we recompute from
# ---------------------------------------------------------------------------
def verify_old_v3_sources(evidence_dir):
    cc4 = os.path.join(evidence_dir, "cc4")
    sources = {"candidate_certificate_sha256": {}, "candidate_file_sha256": {}}

    # Summary + gate: file hash == sidecar == frozen pins; gate links summary.
    summ_path = os.path.join(cc4, "FORMAL_RANKING_SUMMARY_V3.json")
    gate_path = os.path.join(cc4, "FORMAL_EVALUATION_GATE_V3.json")
    for p in (summ_path, gate_path):
        _require(os.path.isfile(p) and os.path.isfile(p + ".sha256"),
                 "missing old V3 artifact or sidecar: %s" % p)
    summ_sha = sha256_file(summ_path)
    gate_sha = sha256_file(gate_path)
    side_summ = open(summ_path + ".sha256", encoding="utf-8").read().split()[0]
    side_gate = open(gate_path + ".sha256", encoding="utf-8").read().split()[0]
    _require(summ_sha == side_summ == FROZEN_V3_SUMMARY_SHA256,
             "old V3 summary sha mismatch: %s vs sidecar %s vs pin %s"
             % (summ_sha, side_summ, FROZEN_V3_SUMMARY_SHA256))
    _require(gate_sha == side_gate == FROZEN_V3_GATE_SHA256,
             "old V3 gate sha mismatch: %s vs sidecar %s vs pin %s"
             % (gate_sha, side_gate, FROZEN_V3_GATE_SHA256))
    old_summary = json.load(open(summ_path, encoding="utf-8"))
    old_gate = json.load(open(gate_path, encoding="utf-8"))
    _require(old_gate.get("formal_ranking_summary_sha256") == summ_sha,
             "old V3 gate does not link the old V3 summary")
    _require(old_summary["schema"] == "mechanism_UED.tier3_formal_ranking_summary/v3",
             "unexpected old summary schema")
    sources["FORMAL_RANKING_SUMMARY_V3_sha256"] = summ_sha
    sources["FORMAL_EVALUATION_GATE_V3_sha256"] = gate_sha
    sources["old_ranking_status"] = old_summary["ranking_status"]
    sources["old_formal_winner"] = old_summary["formal_winner"]
    _require(old_summary["ranking_status"] == "INCONCLUSIVE_FULL_TIE"
             and old_summary["formal_winner"] is None,
             "old V3 summary is not the expected INCONCLUSIVE_FULL_TIE close")

    # Per-candidate SHA256SUMS_FORMAL_V3 must verify byte-for-byte.
    old_participants = {p["candidate_id"]: p
                        for p in old_summary["participants"]}
    _require(sorted(old_participants) == ALL_CANDIDATES,
             "old summary participant set mismatch: %s" % sorted(old_participants))
    for cid in ALL_CANDIDATES:
        fdir = os.path.join(cc4, cid, "formal_evaluation_v3")
        sums_path = os.path.join(fdir, "SHA256SUMS_FORMAL_V3")
        _require(os.path.isfile(sums_path), "missing %s" % sums_path)
        recorded = {}
        with open(sums_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if not line:
                    continue
                sha, rel = line.split(None, 1)
                recorded[rel.strip().lstrip("*")] = sha
        expected_files = {
            "episode_records.jsonl", "provenance_v3.json",
            "evaluation_certificate_v3.json", "evaluation_result_v3.full.json",
            "evaluation_result_v3.front_l2.json",
            "evaluation_result_v3.back_l2.json"}
        _require(set(recorded) == expected_files,
                 "%s: SHA256SUMS_FORMAL_V3 file list mismatch" % cid)
        for rel, sha in sorted(recorded.items()):
            actual = sha256_file(os.path.join(fdir, rel))
            _require(actual == sha,
                     "%s/%s: sha mismatch recorded=%s actual=%s"
                     % (cid, rel, sha, actual))
        cert_sha = recorded["evaluation_certificate_v3.json"]
        _require(cert_sha == old_participants[cid]["certificate_sha256"],
                 "%s: certificate sha differs from old summary" % cid)
        sources["candidate_certificate_sha256"][cid] = cert_sha
        sources["candidate_file_sha256"][cid] = dict(recorded)
    return sources, old_summary, old_gate, old_participants


# ---------------------------------------------------------------------------
# Stage 3 — primary metric extraction + parity (frozen V3 values)
# ---------------------------------------------------------------------------
def extract_primary_metrics(evidence_dir, cid):
    fdir = os.path.join(evidence_dir, "cc4", cid, "formal_evaluation_v3")
    out = {}
    results = {}
    for sc in SCENARIOS:
        p = os.path.join(fdir, "evaluation_result_v3.%s.json" % sc)
        res = json.load(open(p, encoding="utf-8"))
        results[sc] = res
        _require(res.get("candidate_id") == cid,
                 "%s/%s: candidate_id mismatch" % (cid, sc))
        _require(res.get("rehearsal") in (False, None)
                 or res.get("rehearsal") is False,
                 "%s/%s: rehearsal flag present in formal pool" % (cid, sc))
        _require(res.get("episodes_executed") == res.get("episodes_planned")
                 == EXPECTED_EPISODES[sc],
                 "%s/%s: episode count mismatch" % (cid, sc))
        _require(not res.get("aborted_in_scenario"),
                 "%s/%s: aborted result used for reclose" % (cid, sc))
    full_m = results["full"]["evaluation"]["metrics"]
    front_m = results["front_l2"]["evaluation"]["metrics"]
    back_m = results["back_l2"]["evaluation"]["metrics"]
    out["full_success_count"] = full_m["primary"]["successes"]
    _require(full_m["primary"]["valid_starts"] == EXPECTED_VALID_STARTS["full"],
             "%s/full: valid_starts != 64" % cid)
    out["front_transition_count"] = front_m["primary"]["successes"]
    _require(front_m["primary"]["valid_starts"] == EXPECTED_VALID_STARTS["front_l2"],
             "%s/front_l2: valid_starts != 8" % cid)
    fp = front_m["dense"]["value"]
    _require(isinstance(fp, (int, float)) and not isinstance(fp, bool)
             and fp == fp and fp not in (float("inf"), float("-inf"))
             and 0.0 <= fp <= 1.0,
             "%s/front_l2: dense progress non-finite/out-of-range: %r" % (cid, fp))
    out["front_mean_progress"] = fp
    out["back_defeat_count"] = back_m["primary"]["successes"]
    _require(back_m["primary"]["valid_starts"] == EXPECTED_VALID_STARTS["back_l2"],
             "%s/back_l2: valid_starts != 8" % cid)
    surv = back_m.get("diagnostics", {}).get("survival", {})
    _require(surv.get("defeat_count") == out["back_defeat_count"],
             "%s/back_l2: primary successes != diagnostics defeat_count" % cid)
    out["valid_start_counts"] = {sc: results[sc]["evaluation"]["valid_start_count"]
                                 for sc in SCENARIOS}
    out["episode_counts"] = {sc: results[sc]["evaluation"]["episode_count"]
                             for sc in SCENARIOS}
    # FRONT-first frozen comparison tuple (§一/§八): (ft, fp, full, back).
    out["front_first_rule_tuple"] = (
        out["front_transition_count"], out["front_mean_progress"],
        out["full_success_count"], out["back_defeat_count"])
    return out, results


def assert_primary_parity(cid, metrics, old_participant):
    old_tuple = old_participant["rule_tuple"]
    pairs = [
        ("full success_count", metrics["full_success_count"]),
        ("front_l2 transition_count", metrics["front_transition_count"]),
        ("front_l2 mean graph_distance_progress", metrics["front_mean_progress"]),
        ("back_l2 defeat_count", metrics["back_defeat_count"]),
    ]
    for name, v3r1_value in pairs:
        _require(name in old_tuple,
                 "%s: old summary rule_tuple lacks %r" % (cid, name))
        _require(old_tuple[name] == v3r1_value,
                 "%s: PRIMARY METRIC PARITY VIOLATION on %r: old V3=%r v3r1=%r "
                 "(fail closed; V3R1 must not be published)"
                 % (cid, name, old_tuple[name], v3r1_value))


# ---------------------------------------------------------------------------
# Stage 4 — offline V3R1 reclassification (secondary dedup; no env reruns)
# ---------------------------------------------------------------------------
def reclassify_offline(evidence_dir, cid, metrics, old_participant):
    fdir = os.path.join(evidence_dir, "cc4", cid, "formal_evaluation_v3")
    path = os.path.join(fdir, "episode_records.jsonl")
    records_by_sc = {sc: [] for sc in SCENARIOS}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            _require(rec.get("scenario") in SCENARIOS,
                     "%s: unregistered scenario in jsonl: %r"
                     % (cid, rec.get("scenario")))
            records_by_sc[rec["scenario"]].append(rec)
    for sc in SCENARIOS:
        _require(len(records_by_sc[sc]) == EXPECTED_EPISODES[sc],
                 "%s/%s: expected %d episode records, found %d"
                 % (cid, sc, EXPECTED_EPISODES[sc], len(records_by_sc[sc])))

    v3_counts = {sc: {} for sc in SCENARIOS}
    v3r1_counts = {sc: {} for sc in SCENARIOS}
    primary_counts = {sc: {} for sc in SCENARIOS}
    dedup_applied = {sc: 0 for sc in SCENARIOS}
    composite_counts = {sc: 0 for sc in SCENARIOS}
    for sc in SCENARIOS:
        for rec in records_by_sc[sc]:
            v3c = t3.classify_episode_v3(sc, rec)
            r = t3r1.classify_episode_v3r1(sc, rec)
            # §七 hard parity: primary classification must not move.
            _require(r["primary_outcome"] == v3c["primary_outcome"]
                     and r["taxonomy_status"] == v3c["taxonomy_status"]
                     and r["composite"] == v3c["composite"],
                     "%s/%s/%s: V3R1 primary classification drifted from V3"
                     % (cid, sc, rec["episode_id"]))
            for ev in v3c["secondary_events"]:
                v3_counts[sc][ev] = v3_counts[sc].get(ev, 0) + 1
            for ev in r["secondary_events"]:
                v3r1_counts[sc][ev] = v3r1_counts[sc].get(ev, 0) + 1
            po = r["primary_outcome"]
            primary_counts[sc][po] = primary_counts[sc].get(po, 0) + 1
            if r["composite"]:
                composite_counts[sc] += 1
            if r["secondary_dedup"]["applied"]:
                dedup_applied[sc] += 1

    # §七 parity vs the frozen published primary metrics.
    _require(primary_counts["front_l2"].get("FRONT_TRANSITION_SUCCESS", 0)
             == metrics["front_transition_count"],
             "%s: FRONT transition_count parity violation" % cid)
    _require(primary_counts["full"].get("FULL_DEFEAT_KOBOLD_SUCCESS", 0)
             == metrics["full_success_count"],
             "%s: FULL success_count parity violation" % cid)
    _require(primary_counts["back_l2"].get("BACK_DEFEAT_KOBOLD_SUCCESS", 0)
             == metrics["back_defeat_count"],
             "%s: BACK defeat_count parity violation" % cid)

    # The recomputed V3 secondary counts must reproduce the old summary exactly
    # (proves we re-read the same records the old close published).
    old_sec = old_participant["secondary_event_counts_by_scenario"]
    for sc in SCENARIOS:
        _require(v3_counts[sc] == old_sec.get(sc, {}),
                 "%s/%s: recomputed V3 secondary counts != old summary "
                 "(%r vs %r)" % (cid, sc, v3_counts[sc], old_sec.get(sc, {})))

    # §六: the ONLY permitted diff vs V3 is removing DEFEAT_KOBOLD where it
    # duplicated a BACK/FULL defeat primary.
    for sc in SCENARIOS:
        keys = set(v3_counts[sc]) | set(v3r1_counts[sc])
        for ev in keys:
            old_n = v3_counts[sc].get(ev, 0)
            new_n = v3r1_counts[sc].get(ev, 0)
            if sc == "front_l2":
                _require(old_n == new_n,
                         "%s/front_l2: secondary %r changed (%d -> %d) but FRONT "
                         "must be untouched" % (cid, ev, old_n, new_n))
            elif ev == "DEFEAT_KOBOLD":
                _require(old_n - new_n == dedup_applied[sc],
                         "%s/%s: DEFEAT_KOBOLD removal count %d != dedup-applied "
                         "episodes %d" % (cid, sc, old_n - new_n, dedup_applied[sc]))
            else:
                _require(old_n == new_n,
                         "%s/%s: non-defeat secondary %r changed (%d -> %d)"
                         % (cid, sc, ev, old_n, new_n))
        _require(v3r1_counts[sc].get("DEFEAT_KOBOLD", 0) == 0
                 or sc == "front_l2",
                 "%s/%s: DEFEAT_KOBOLD still present in BACK/FULL secondary"
                 % (cid, sc))

    return {
        "records_reclassified": {sc: len(records_by_sc[sc]) for sc in SCENARIOS},
        "v3_secondary_event_counts": v3_counts,
        "v3r1_secondary_event_counts": {sc: dict(sorted(v3r1_counts[sc].items()))
                                        for sc in SCENARIOS},
        "primary_outcome_counts": primary_counts,
        "composite_episode_count_by_scenario": composite_counts,
        "dedup_applied_episodes_by_scenario": dedup_applied,
        "environment_rerun": False,
        "classification_only": True,
        "neg20_protocol": t3r1.NEG20_PROTOCOL,
        "taxonomy_v3r1_lf_sha256": t3r1.module_lf_sha256(),
        "taxonomy_v3_lf_sha256": t3.module_lf_sha256(),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run(args):
    repo_root = os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    evidence_dir = os.path.abspath(args.evidence_dir or os.path.join(
        repo_root, "reports", "tier3_scaffolded_evaluation",
        "formal_evaluation_evidence_v3_20260801"))
    v2_evidence_dir = os.path.abspath(args.v2_evidence_dir or os.path.join(
        repo_root, "reports", "tier3_scaffolded_evaluation",
        "formal_evaluation_evidence_20260801"))
    out_dir = os.path.abspath(args.out_dir or os.path.join(
        repo_root, "reports", "tier3_scaffolded_evaluation",
        "formal_ranking_reclose_v3r1_20260803"))
    baseline_dir = os.path.abspath(args.baseline_dir or os.path.join(
        repo_root, "tmp_v3r1_baseline"))

    # Hard guard: the new artifacts NEVER land inside the old evidence dirs.
    for old in (evidence_dir, v2_evidence_dir):
        _require(not (out_dir == old or out_dir.startswith(old + os.sep)),
                 "output dir must not be inside the old evidence dir")
    _require(os.path.basename(out_dir) == "formal_ranking_reclose_v3r1_20260803",
             "output dir must be formal_ranking_reclose_v3r1_20260803")

    started = utc_now_iso()
    head = git_head(repo_root)
    branch = git_branch(repo_root)
    _require(head == EXPECTED_BASE_COMMIT,
             "HEAD %s != frozen reclose baseline %s (commit first / stop)"
             % (head, EXPECTED_BASE_COMMIT))

    print("[v3r1] stage1: old-evidence immutability audit ...")
    immutability = audit_immutability(evidence_dir, v2_evidence_dir, baseline_dir)

    print("[v3r1] stage2: verifying old V3 sources (sums/sidecars/pins) ...")
    sources, old_summary, old_gate, old_participants = \
        verify_old_v3_sources(evidence_dir)

    print("[v3r1] stage3+4: per-candidate extraction, parity, reclassification ...")
    per_candidate = {}
    for cid in ALL_CANDIDATES:
        metrics, _results = extract_primary_metrics(evidence_dir, cid)
        assert_primary_parity(cid, metrics, old_participants[cid])
        recl = reclassify_offline(evidence_dir, cid, metrics,
                                  old_participants[cid])
        per_candidate[cid] = {"metrics": metrics, "reclassification": recl}
        print("[v3r1]   %s tuple=%s dedup=%s"
              % (cid, metrics["front_first_rule_tuple"],
                 recl["dedup_applied_episodes_by_scenario"]))

    print("[v3r1] stage5: FRONT-first ranking (students only) ...")
    entries = [{"candidate_id": cid,
                "rule_tuple": per_candidate[cid]["metrics"]["front_first_rule_tuple"]}
               for cid in STUDENTS]
    ranking = rankmod.rank_students_v3r1(entries)
    _require(ranking["comparison_provenance"]["rule_order"] == NEW_RULE_ORDER,
             "ranking machine rule order != frozen FRONT-first order")
    _require(ranking["comparison_provenance"]["candidate_id_is_scientific_tiebreak"]
             is False, "candidate_id used as scientific tie-break")
    _require(TEACHER not in ranking["ranks"],
             "teacher leaked into the student ranking")
    _require(ranking["formal_winner"] != TEACHER,
             "teacher became formal_winner")

    # Per-candidate tie status disclosure (§二/§十).
    tie_status = {}
    for g in ranking["ordered_groups"]:
        if g["size"] > 1:
            if g["tie_group_rank"] == 1:
                st = "TOP_TIE_INCONCLUSIVE"
            else:
                st = "LOWER_TIE_GROUP_MEMBER"
        else:
            st = "UNIQUE_TOP" if g["tie_group_rank"] == 1 else "ORDERED_NO_TIE"
        for c in g["candidate_ids"]:
            tie_status[c] = st

    print("[v3r1] stage6: gate evaluation ...")
    # R7 equivalent: the frozen self-tests must pass (tests A–K + taxonomy §六/§七).
    tests = {}
    for name, fn in (("ranking_machine_self_test_ABCDEFGHIJK",
                      rankmod.run_self_test),
                     ("taxonomy_v3r1_self_test", t3r1.run_self_test)):
        try:
            fn()
            tests[name] = "PASS"
        except Exception as exc:            # noqa: BLE001 — fail closed
            tests[name] = "FAIL: %s" % exc
    tests_green = all(v == "PASS" for v in tests.values())

    gates = {
        "R1_ALL_6_STUDENTS_V3_EVIDENCE_COMPLETE": all(
            per_candidate[c]["metrics"]["episode_counts"] == {
                sc: EXPECTED_EPISODES[sc] for sc in SCENARIOS}
            and per_candidate[c]["metrics"]["valid_start_counts"] == {
                sc: EXPECTED_VALID_STARTS[sc] for sc in SCENARIOS}
            for c in STUDENTS),
        "R2_TEACHER_REFERENCE_COMPLETE": (
            per_candidate[TEACHER]["metrics"]["episode_counts"] == {
                sc: EXPECTED_EPISODES[sc] for sc in SCENARIOS}),
        "R3_ALL_V3_CERTIFICATE_SHAS_VERIFY": True,   # enforced in stage2
        "R4_OLD_V3_V2_FILES_UNMODIFIED": immutability["IMMUTABLE"],
        "R5_PRIMARY_METRIC_PARITY": True,            # enforced in stage3/4
        "R6_FRONT_FIRST_RULE_CODE_PROTOCOL_CONSISTENT": (
            NEW_RULE_ORDER[0] == "front_l2 transition_count"
            and rankmod.RANKING_PROTOCOL == "TIER3_FRONT_FIRST_LEXICOGRAPHIC_V1"
            and rankmod.FORMAL_RANKING_PROTOCOL == "V3R1_FRONT_FIRST_TOP_TIE_ONLY"
            and rankmod.ONLY_TOP_TIE_BLOCKS_WINNER is True
            and NEW_RULE_ORDER != OLD_RULE_ORDER),
        "R7_TOP_TIE_ONLY_TESTS_ALL_PASS": tests_green,
        "R8_TEACHER_EXCLUDED_FROM_STUDENT_RANKING": (
            TEACHER not in ranking["ranks"]
            and ranking["formal_winner"] != TEACHER),
        "R9_NO_REROLLOUT": True,     # structural: this driver runs no environment
        "R10_NO_CHECKPOINT_MOD_NO_RETRAINING_NO_EXEMPTION": True,  # structural
        "R11_NO_FULL_FIRST_ORDERING": NEW_RULE_ORDER[0] != "full success_count",
        "R12_NO_CANDIDATE_ID_SCIENTIFIC_TIEBREAK": (
            ranking["comparison_provenance"][
                "candidate_id_is_scientific_tiebreak"] is False),
    }
    authorized = all(gates.values())

    # ----- participants block (students + teacher reference) -----
    participants = []
    for cid in STUDENTS + [TEACHER]:
        m = per_candidate[cid]["metrics"]
        old_p = old_participants[cid]
        is_teacher = cid == TEACHER
        participants.append({
            "candidate_id": cid,
            "candidate_class": ("TEACHER_REFERENCE" if is_teacher else "STUDENT"),
            "excluded_from_student_ranking": is_teacher,
            "student_rank": (None if is_teacher else ranking["ranks"][cid]),
            "tie_group_rank": (None if is_teacher else next(
                g["tie_group_rank"] for g in ranking["ordered_groups"]
                if cid in g["candidate_ids"])),
            "tie_status": ("TEACHER_REFERENCE_ONLY" if is_teacher
                           else tie_status[cid]),
            "front_first_rule_tuple": {
                "front_l2 transition_count": m["front_transition_count"],
                "front_l2 mean graph_distance_progress": m["front_mean_progress"],
                "full success_count": m["full_success_count"],
                "back_l2 defeat_count": m["back_defeat_count"],
            },
            "valid_start_counts": m["valid_start_counts"],
            "v3_certificate_sha256": sources["candidate_certificate_sha256"][cid],
            "reuse_status_by_scenario": old_p["reuse_status_by_scenario"],
            "runtime_family": old_p["runtime_family"],
            "v3r1_secondary_event_counts_by_scenario":
                per_candidate[cid]["reclassification"][
                    "v3r1_secondary_event_counts"],
            "secondary_dedup_applied_episodes_by_scenario":
                per_candidate[cid]["reclassification"][
                    "dedup_applied_episodes_by_scenario"],
            "composite_episode_count_by_scenario":
                per_candidate[cid]["reclassification"][
                    "composite_episode_count_by_scenario"],
        })

    total_dedup_episodes = sum(
        sum(per_candidate[c]["reclassification"]
            ["dedup_applied_episodes_by_scenario"].values())
        for c in ALL_CANDIDATES)

    summary = {
        "schema": rankmod.SUMMARY_SCHEMA,
        "FORMAL_RANKING_PROTOCOL": rankmod.FORMAL_RANKING_PROTOCOL,
        "RANKING_PROTOCOL": rankmod.RANKING_PROTOCOL,
        "NEG20_PROTOCOL": t3r1.NEG20_PROTOCOL,
        "RANKING_PRIMARY_ORDER": "FRONT_TRANSITION_FIRST",
        "ONLY_TOP_TIE_BLOCKS_WINNER": rankmod.ONLY_TOP_TIE_BLOCKS_WINNER,
        "NEW_RULE_ORDER": NEW_RULE_ORDER,
        "OLD_RULE_ORDER": OLD_RULE_ORDER,
        "generated_at_utc": utc_now_iso(),
        "ruling_task": RULING_TASK,
        "git": {
            "base_commit": EXPECTED_BASE_COMMIT,
            "reclose_head_at_generation": head,
            "branch": branch,
        },
        "source_v3_evidence": {
            "evidence_dir": evidence_dir.replace(os.sep, "/"),
            "FORMAL_RANKING_SUMMARY_V3_sha256":
                sources["FORMAL_RANKING_SUMMARY_V3_sha256"],
            "FORMAL_EVALUATION_GATE_V3_sha256":
                sources["FORMAL_EVALUATION_GATE_V3_sha256"],
            "metric_schema_sha256": FROZEN_METRIC_SCHEMA_SHA256,
        },
        "old_v3_reference": {
            "note": ("the old V3 close was the correct result under its own "
                     "FULL-first full-tie protocol; it is NOT wrong or faked. "
                     "V3R1 is 总控's formal clarification and re-close under the "
                     "FRONT-first top-tie-only policy, recomputed from the same "
                     "immutable V3 episode records and metrics."),
            "OLD_V3_FORMAL_WINNER": sources["old_formal_winner"],
            "OLD_V3_RANKING_STATUS": sources["old_ranking_status"],
            "OLD_V3_ORDERED_BY_STRICT_RULE":
                old_summary["ordered_by_strict_rule"],
            "OLD_V3_INCONCLUSIVE_GROUPS": old_summary["inconclusive_groups"],
        },
        "participants": participants,
        "ordered_groups": ranking["ordered_groups"],
        "lower_tie_groups": ranking["lower_tie_groups"],
        "TOP_GROUP": (ranking["ordered_groups"][0]["candidate_ids"]
                      if ranking["ordered_groups"] else []),
        "TOP_GROUP_SIZE": (ranking["ordered_groups"][0]["size"]
                           if ranking["ordered_groups"] else 0),
        "TOP_TIE": ranking["top_tie"],
        "unique_top_candidate_id": ranking["unique_top_candidate_id"],
        "formal_winner": ranking["formal_winner"],
        "ranking_status": ranking["ranking_status"],
        "comparison_provenance": ranking["comparison_provenance"],
        "primary_metric_parity": {
            "V3_PRIMARY_METRICS_EQ_V3R1_PRIMARY_METRICS": True,
            "parity_checked_metrics": [
                "front_l2 transition_count",
                "front_l2 mean graph_distance_progress",
                "full success_count",
                "back_l2 defeat_count",
            ],
            "candidates_checked": ALL_CANDIDATES,
        },
        "SECONDARY_EVENT_PRIMARY_DUPLICATION_REMOVED": True,
        "secondary_dedup": {
            "protocol": t3r1.NEG20_PROTOCOL,
            "rule": t3r1.SECONDARY_DEDUP_RULE,
            "total_deduplicated_episodes": total_dedup_episodes,
            "front_secondary_events_untouched": True,
            "environment_reruns": 0,
            "classification_only": True,
        },
        "student_count_eligible": "6/6",
        "teacher_included_in_student_ranking": False,
        "scientific_claim_authorized": False,
        "scientific_claim_status":
            "FORMAL_SCIENTIFIC_CLAIM: NOT_AUTHORIZED_SINGLE_TRAINING_SEED",
        "single_training_seed": True,
        "multi_seed_confirmation_skipped_by_director": True,
        "scaffolded_results_can_replace_full_task": False,
        "usage_scope": ("工程阶段 strongest Student selection ONLY; no "
                        "statistical SOTA claim"),
        "ENVIRONMENT_RERUNS": 0,
        "STUDENTS_RETRAINED": 0,
        "CHECKPOINTS_MODIFIED": False,
        "CANDIDATE_EXCEPTIONS_USED": 0,
        "taxonomies": {
            "taxonomy_v3_lf_sha256": t3.module_lf_sha256(),
            "taxonomy_v3r1_lf_sha256": t3r1.module_lf_sha256(),
            "ranking_machine_module": "tier3_ranking_v3r1.py",
            "selection_tie_tolerance": rankmod.SELECTION_TIE_TOLERANCE,
        },
    }

    gate = {
        "schema": rankmod.GATE_SCHEMA,
        "FORMAL_RANKING_PROTOCOL": rankmod.FORMAL_RANKING_PROTOCOL,
        "RANKING_PROTOCOL": rankmod.RANKING_PROTOCOL,
        "FORMAL_RANKING_AUTHORIZED_V3R1": authorized,
        "gates": gates,
        "gate_failures": [k for k, v in gates.items() if not v],
        "self_tests": tests,
        "ranking_status": ranking["ranking_status"],
        "formal_winner": ranking["formal_winner"],
        "formal_ranking_summary_sha256": None,   # filled after summary write
        "independent_verification": {
            "required": True,
            "verifier_script": "verify_formal_ranking_v3r1.py",
            "note": ("the gate is not final until the independent verifier "
                     "re-derives every claim from the immutable V3 evidence; "
                     "see COMMON_EVALUATOR_V3R1_RANKING_READY.json"),
        },
        "retained_constraints": {
            "scientific_claim_authorized": False,
            "single_training_seed": True,
            "multi_seed_confirmation_skipped_by_director": True,
            "scaffolded_results_can_replace_full_task": False,
            "usage_scope": "工程阶段 strongest Student selection ONLY",
        },
        "prohibitions_honored": {
            "retraining_performed": False,
            "rollouts_rerun": False,
            "checkpoints_modified": False,
            "state_banks_modified": False,
            "seeds_changed": False,
            "horizon_changed": False,
            "episode_counts_changed": False,
            "greedy_argmax_changed": False,
            "candidates_deleted": [],
            "candidate_exemptions": [],
            "full_first_ordering_used": False,
            "candidate_id_scientific_tiebreak_used": False,
            "v2_v3_evidence_overwritten": False,
            "original_v3_certificates_modified": False,
            "old_v3_summary_rewritten": False,
        },
        "immutability": {
            "OLD_V3_FILES_MODIFIED": immutability["OLD_V3_FILES_MODIFIED"],
            "OLD_V3_HASH_DRIFT": immutability["OLD_V3_HASH_DRIFT"],
            "V2_FILES_MODIFIED": immutability["V2_FILES_MODIFIED"],
        },
        "generated_at_utc": utc_now_iso(),
        "ruling_task": RULING_TASK,
    }

    protocol_decision = {
        "schema": "mechanism_UED.tier3_ranking_protocol_decision/v3r1",
        "ruling_task": RULING_TASK,
        "decided_by": "总控 (director ruling)",
        "recorded_by": "CC4",
        "generated_at_utc": utc_now_iso(),
        "decision": {
            "RANKING_PRIMARY_ORDER": "FRONT_TRANSITION_FIRST",
            "RANKING_PROTOCOL": rankmod.RANKING_PROTOCOL,
            "FORMAL_RANKING_PROTOCOL": rankmod.FORMAL_RANKING_PROTOCOL,
            "ONLY_TOP_TIE_BLOCKS_WINNER": True,
            "NEW_RULE_ORDER": NEW_RULE_ORDER,
            "OLD_RULE_ORDER": OLD_RULE_ORDER,
            "tie_tolerance": rankmod.SELECTION_TIE_TOLERANCE,
            "candidate_id_is_scientific_tiebreak": False,
        },
        "rationale": [
            "FULL success_count must not be the first ranking field; the FRONT "
            "transition capability is the director-specified primary ordering "
            "criterion.",
            "A full-tuple tie BELOW the top position must not cancel a unique "
            "top winner; only a tie within the top equivalence group blocks the "
            "winner.",
            "Tied candidates keep student_rank=null but disclose tie_group_rank; "
            "no order is ever invented via candidate_id.",
        ],
        "scope_and_limits": {
            "applies_to": "Tier3 scaffolded-evaluation formal global ranking",
            "retroactive_effect": ("recomputed from existing immutable V3 "
                                   "records; no reruns, no retraining"),
            "old_v3_status": ("kept immutable and cited; the old V3 close was "
                              "correct under its own protocol"),
            "scientific_claim_authorized": False,
        },
    }

    if args.dry_run:
        print("[v3r1] DRY RUN: all stages passed; nothing written.")
        print(json.dumps({"ranking_status": ranking["ranking_status"],
                          "formal_winner": ranking["formal_winner"],
                          "authorized": authorized}, indent=2))
        return 0

    os.makedirs(out_dir, exist_ok=True)
    print("[v3r1] stage7: writing artifacts to %s" % out_dir)

    # Immutability report + protocol decision + audit first (no cross-hashes).
    audit = {
        "schema": "mechanism_UED.tier3_ranking_recomputation_audit/v3r1",
        "ruling_task": RULING_TASK,
        "generated_at_utc": utc_now_iso(),
        "recomputation_kind": ("offline recompute from immutable V3 records; "
                               "no environment reruns"),
        "ENVIRONMENT_RERUNS": 0,
        "STUDENTS_RETRAINED": 0,
        "CHECKPOINTS_MODIFIED": False,
        "source_evidence_dir": evidence_dir.replace(os.sep, "/"),
        "source_v3_summary_sha256": sources["FORMAL_RANKING_SUMMARY_V3_sha256"],
        "source_v3_gate_sha256": sources["FORMAL_EVALUATION_GATE_V3_sha256"],
        "rule_order_change": {
            "OLD_RULE_ORDER": OLD_RULE_ORDER,
            "NEW_RULE_ORDER": NEW_RULE_ORDER,
            "changed_field_positions": {
                "full success_count": {"old_position": 1, "new_position": 3},
                "front_l2 transition_count": {"old_position": 2,
                                              "new_position": 1},
                "front_l2 mean graph_distance_progress": {"old_position": 3,
                                                          "new_position": 2},
                "back_l2 defeat_count": {"old_position": 4, "new_position": 4},
            },
        },
        "tie_policy_change": {
            "old": ("ANY >=2 full-tuple tie group voided the entire ranking "
                    "(INCONCLUSIVE_FULL_TIE, winner null)"),
            "new": ("ONLY a tie inside the TOP equivalence group blocks the "
                    "winner; lower ties are disclosed and non-blocking"),
            "ONLY_TOP_TIE_BLOCKS_WINNER": True,
        },
        "candidates": {},
    }
    for cid in ALL_CANDIDATES:
        m = per_candidate[cid]["metrics"]
        old_p = old_participants[cid]
        audit["candidates"][cid] = {
            "source_files": {sc: ("cc4/%s/formal_evaluation_v3/"
                                  "evaluation_result_v3.%s.json" % (cid, sc))
                             for sc in SCENARIOS},
            "source_file_sha256": sources["candidate_file_sha256"][cid],
            "old_v3_rule_tuple_full_first": old_p["rule_tuple"],
            "v3r1_front_first_rule_tuple": list(m["front_first_rule_tuple"]),
            "primary_metric_parity": "EXACT",
            "reclassification": {
                "records_reclassified":
                    per_candidate[cid]["reclassification"]["records_reclassified"],
                "dedup_applied_episodes_by_scenario":
                    per_candidate[cid]["reclassification"][
                        "dedup_applied_episodes_by_scenario"],
                "v3_secondary_event_counts":
                    per_candidate[cid]["reclassification"][
                        "v3_secondary_event_counts"],
                "v3r1_secondary_event_counts":
                    per_candidate[cid]["reclassification"][
                        "v3r1_secondary_event_counts"],
                "primary_outcome_counts":
                    per_candidate[cid]["reclassification"]["primary_outcome_counts"],
            },
            "old_v3_student_rank": old_p["student_rank"],
        }
    write_json_atomic(os.path.join(out_dir, "RANKING_RECOMPUTATION_AUDIT_V3R1.json"),
                      audit)
    write_json_atomic(os.path.join(out_dir, "OLD_V3_IMMUTABILITY_REPORT.json"),
                      immutability)
    write_json_atomic(os.path.join(out_dir,
                                   "RANKING_PROTOCOL_DECISION_V3R1.json"),
                      protocol_decision)

    # Summary, then gate (gate links the summary hash), then sidecars/sums.
    summary_sha = write_json_atomic(
        os.path.join(out_dir, "FORMAL_RANKING_SUMMARY_V3R1.json"), summary)
    gate["formal_ranking_summary_sha256"] = summary_sha
    gate_sha = write_json_atomic(
        os.path.join(out_dir, "FORMAL_EVALUATION_GATE_V3R1.json"), gate)
    write_text_atomic(os.path.join(out_dir, "FORMAL_RANKING_SUMMARY_V3R1.json.sha256"),
                      "%s  FORMAL_RANKING_SUMMARY_V3R1.json\n" % summary_sha)
    write_text_atomic(os.path.join(out_dir, "FORMAL_EVALUATION_GATE_V3R1.json.sha256"),
                      "%s  FORMAL_EVALUATION_GATE_V3R1.json\n" % gate_sha)

    # ----- independent verifier (must pass before READY is written) -----
    verifier_status = "NOT_RUN"
    if args.run_verifier:
        vpath = os.path.abspath(args.run_verifier)
        _require(os.path.isfile(vpath), "verifier not found: %s" % vpath)
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        # Embedded mode: READY / SHA256SUMS_V3R1 are written AFTER this
        # verification passes; the final full run (no flag) checks them too.
        vrun = subprocess.run([sys.executable, vpath,
                               "--evidence-dir", evidence_dir,
                               "--v2-evidence-dir", v2_evidence_dir,
                               "--reclose-dir", out_dir,
                               "--baseline-dir", baseline_dir,
                               "--allow-missing-ready"],
                              capture_output=True, text=True, env=env)
        sys.stdout.write(vrun.stdout)
        sys.stderr.write(vrun.stderr)
        verifier_status = "PASS" if vrun.returncode == 0 else \
            "FAIL_EXIT_%d" % vrun.returncode
    verifier_green = verifier_status == "PASS"
    final_authorized = authorized and (verifier_green
                                       if args.run_verifier else False)

    ready = {
        "schema": rankmod.READY_SCHEMA,
        "written_by": "tier3_formal_reclose_v3r1.py (sole writer)",
        "COMMON_EVALUATOR_V3R1_RANKING_READY": final_authorized,
        "FORMAL_RANKING_PROTOCOL": rankmod.FORMAL_RANKING_PROTOCOL,
        "FORMAL_RANKING_AUTHORIZED_V3R1": final_authorized,
        "RANKING_STATUS": ranking["ranking_status"],
        "FORMAL_WINNER": ranking["formal_winner"],
        "TOP_TIE": ranking["top_tie"],
        "LOWER_TIE_GROUPS": ranking["lower_tie_groups"],
        "SUMMARY_SHA256": summary_sha,
        "GATE_SHA256": gate_sha,
        "SOURCE_V3_SUMMARY_SHA256": sources["FORMAL_RANKING_SUMMARY_V3_sha256"],
        "SOURCE_V3_GATE_SHA256": sources["FORMAL_EVALUATION_GATE_V3_sha256"],
        "OLD_V3_IMMUTABLE": immutability["IMMUTABLE"],
        "PRIMARY_METRIC_PARITY": True,
        "SECONDARY_EVENT_PRIMARY_DUPLICATION_REMOVED": True,
        "ENVIRONMENT_RERUNS": 0,
        "STUDENTS_RETRAINED": 0,
        "CHECKPOINTS_MODIFIED": False,
        "teacher_included_in_student_ranking": False,
        "scientific_claim_authorized": False,
        "single_training_seed": True,
        "scaffolded_results_can_replace_full_task": False,
        "usage_scope": "工程阶段 strongest Student selection ONLY",
        "independent_verifier": {
            "status": verifier_status,
            "required": True,
        },
        "pending_gates": [] if final_authorized else (
            (["INDEPENDENT_VERIFIER_NOT_PASS"] if not verifier_green else [])
            + [k for k, v in gates.items() if not v]),
        "old_v3_ready_file_untouched": True,
        "generated_at_utc": utc_now_iso(),
        "ruling_task": RULING_TASK,
    }
    write_json_atomic(os.path.join(out_dir,
                                   "COMMON_EVALUATOR_V3R1_RANKING_READY.json"),
                      ready)

    # SHA256SUMS over every artifact in the new directory (exclude the sums
    # file itself and sidecars' own lines are included by sha256sum -c design).
    sums_lines = []
    for f in sorted(os.listdir(out_dir)):
        p = os.path.join(out_dir, f)
        if os.path.isfile(p) and f != "SHA256SUMS_V3R1":
            sums_lines.append("%s  %s" % (sha256_file(p), f))
    write_text_atomic(os.path.join(out_dir, "SHA256SUMS_V3R1"),
                      "\n".join(sums_lines) + "\n")

    print("[v3r1] DONE ranking_status=%s formal_winner=%s authorized=%s "
          "verifier=%s" % (ranking["ranking_status"],
                           ranking["formal_winner"], final_authorized,
                           verifier_status))
    print("[v3r1] summary_sha256=%s" % summary_sha)
    print("[v3r1] gate_sha256=%s" % gate_sha)
    if not final_authorized:
        print("[v3r1] FAIL CLOSED: FORMAL_RANKING_AUTHORIZED_V3R1=false; "
              "the new artifacts record the refusal.")
        return 2
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--evidence-dir", default=None)
    ap.add_argument("--v2-evidence-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--baseline-dir", default=None)
    ap.add_argument("--run-verifier", default=None,
                    help="path to verify_formal_ranking_v3r1.py; run after the "
                         "artifacts are written and before READY is finalized")
    ap.add_argument("--dry-run", action="store_true",
                    help="compute + verify everything, write nothing")
    args = ap.parse_args(argv)
    try:
        return run(args)
    except FailClosed as exc:
        print("FAIL CLOSED: %s" % exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
