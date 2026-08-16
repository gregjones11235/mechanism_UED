#!/usr/bin/env python3
"""CC4 Tier3 — COMMON_EVALUATOR_READY.json FIELD SUPPLEMENT (closing contract 2 §一).

The READY marker written by tier3_pool_common_assembly.finalize_ready() already
carries the eight gates and the core top-level SHAs. The follow-up pool contract
additionally requires these fields to appear IN the marker itself, each bound to
a real recomputed value (never a bare claim):

  negative_test_report_sha256     whole-file SHA256 of common/negative_test_report.json
  assembly_manifest_sha256        whole-file SHA256 of common/assembly_manifest.json
  full_profile_sha256             canonical-JSON SHA of profile["scenarios"]["full"]
  FULL_PROFILE_STATUS             FROZEN iff scenarios.full.FULL_PROFILE_READY is true
  common_sha256sums_self_check    PASS — re-run over every SHA256SUMS entry, live
  FORMAL_RANKING_AUTHORIZED       false while the 6-student binding count is < 6/6

SAFETY: this tool ONLY ADDS fields to the existing marker. It verifies every
pre-existing top-level SHA against the live files first and FAILS CLOSED on any
drift; it never touches any other file; COMMON_EVALUATOR_READY.json is excluded
from SHA256SUMS by construction (asserted here), so the sums stay valid. Pure
except for reading/writing that one JSON document. --self-test is fully pure.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SCHEMA = "mechanism_UED.common_evaluator_ready/v1"


class FailClosed(Exception):
    """Hard stop on any supplement violation."""


def require(cond, msg):
    if not cond:
        raise FailClosed(msg)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_sha256(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def _utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: str, doc: dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)


def verify_sha256sums_live(common_root: str) -> dict:
    """Re-run the SHA256SUMS check over the live tree. Returns a status doc;
    the marker field is PASS iff mismatches/missing are both empty."""
    sums_path = os.path.join(common_root, "SHA256SUMS")
    require(os.path.isfile(sums_path), "FAIL CLOSED: missing %s" % sums_path)
    mismatches, missing, listed = [], [], 0
    with open(sums_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            sha, rel = line.split("  ", 1)
            listed += 1
            p = os.path.join(common_root, *rel.split("/"))
            if not os.path.isfile(p):
                missing.append(rel)
            elif _sha256_file(p) != sha:
                mismatches.append(rel)
    return {"status": "PASS" if not mismatches and not missing and listed > 0
                     else "FAIL",
            "listed": listed, "mismatches": mismatches, "missing": missing}


def supplement(common_root: str) -> dict:
    common_root = str(common_root)
    ready_path = os.path.join(common_root, "COMMON_EVALUATOR_READY.json")
    require(os.path.isfile(ready_path),
            "FAIL CLOSED: %s does not exist — run finalize_ready first" % ready_path)
    with open(ready_path, encoding="utf-8") as fh:
        doc = json.load(fh)
    require(doc.get("schema") == SCHEMA,
            "FAIL CLOSED: unexpected READY schema %r" % doc.get("schema"))

    # 1. Every pre-existing top-level SHA must still reproduce from the live
    #    files — refuse to write anything on drift.
    whole = {"common_evaluator_sha256": "common_evaluator.py",
             "common_runner_sha256": "common_runner.py",
             "evaluation_profile_sha256": "evaluation_profile.json",
             "metric_schema_sha256": "metric_schema.json",
             "environment_lock_sha256": "environment_lock.json"}
    for key, rel in whole.items():
        live = _sha256_file(os.path.join(common_root, rel))
        require(doc.get(key) == live,
                "FAIL CLOSED: READY.%s drifted from live %s (%s... vs %s...)"
                % (key, rel, str(doc.get(key))[:12], live[:12]))

    # 2. The marker must NOT be listed in SHA256SUMS (it is excluded by the
    #    assembler's construction) — otherwise adding fields would break sums.
    with open(os.path.join(common_root, "SHA256SUMS"), encoding="utf-8") as fh:
        sums_text = fh.read()
    require("COMMON_EVALUATOR_READY.json" not in sums_text,
            "FAIL CLOSED: marker is listed in SHA256SUMS — cannot supplement safely")

    # 3. Recompute the required supplement fields from the real files.
    neg_sha = _sha256_file(os.path.join(common_root, "negative_test_report.json"))
    man_sha = _sha256_file(os.path.join(common_root, "assembly_manifest.json"))
    with open(os.path.join(common_root, "evaluation_profile.json"),
              encoding="utf-8") as fh:
        profile = json.load(fh)
    full_block = profile.get("scenarios", {}).get("full")
    require(isinstance(full_block, dict) and full_block,
            "FAIL CLOSED: profile scenarios.full missing")
    full_sha = _canonical_sha256(full_block)
    full_status = "FROZEN" if full_block.get("FULL_PROFILE_READY") is True else "PENDING"
    sums_status = verify_sha256sums_live(common_root)

    # 4. Additive merge only. Existing keys are never overwritten (verified
    #    equal above anyway); record provenance of the supplement itself.
    additions = {
        "negative_test_report_sha256": neg_sha,
        "assembly_manifest_sha256": man_sha,
        "full_profile_sha256": full_sha,
        "FULL_PROFILE_STATUS": full_status,
        "common_sha256sums_self_check": sums_status["status"],
        "common_sha256sums_evidence": sums_status,
        "FORMAL_RANKING_AUTHORIZED": False,
        "formal_ranking_authorization_basis":
            "false while STUDENT_COMMON_BINDING_PASS_COUNT < 6/6 "
            "(only rmt16_gtrxl_cc2 is registered in the common ABI; the other "
            "four students + teacher reference await owner-registered runtimes)",
        "ready_marker_supplemented_utc": _utc_now(),
        "supplement_discipline":
            "additive fields only; pre-existing SHAs re-verified live; no other "
            "file touched; SHA256SUMS unaffected (marker excluded by construction)",
    }
    for k, v in additions.items():
        if k in doc and doc[k] != v and not k.endswith("_utc"):
            require(False, "FAIL CLOSED: refusing to change existing READY.%s" % k)
        doc[k] = v

    _atomic_json(ready_path, doc)
    print("READY_MARKER_SUPPLEMENTED root=%s FULL_PROFILE_STATUS=%s "
          "common_sha256sums_self_check=%s FORMAL_RANKING_AUTHORIZED=false"
          % (common_root, full_status, sums_status["status"]))
    return doc


def self_test() -> int:
    import tempfile

    problems = []

    def check(name, cond):
        if not cond:
            problems.append(name)

    def make_root(base):
        common = os.path.join(base, "common")
        os.makedirs(common)
        files = {"common_evaluator.py": b"evaluator engine\n",
                 "common_runner.py": b"runner engine\n",
                 "negative_test_report.json": b'{"fail": 0}\n'}
        profile = {"scenarios": {"full": {"FULL_PROFILE_READY": True,
                                          "seeds": [200000, 200001]}}}
        files["evaluation_profile.json"] = json.dumps(
            profile, sort_keys=True, indent=2).encode() + b"\n"
        files["metric_schema.json"] = b'{"schema": "x"}\n'
        files["environment_lock.json"] = b'{"pinned": true}\n'
        files["assembly_manifest.json"] = b'{"assembler": "x"}\n'
        for rel, data in files.items():
            with open(os.path.join(common, rel), "wb") as fh:
                fh.write(data)
        sums = "".join(
            "%s  %s\n" % (hashlib.sha256(files[r]).hexdigest(), r)
            for r in sorted(files))
        with open(os.path.join(common, "SHA256SUMS"), "w", encoding="utf-8") as fh:
            fh.write(sums)
        ready = {"schema": SCHEMA, "COMMON_EVALUATOR_READY": True,
                 "common_evaluator_sha256": hashlib.sha256(
                     files["common_evaluator.py"]).hexdigest(),
                 "common_runner_sha256": hashlib.sha256(
                     files["common_runner.py"]).hexdigest(),
                 "evaluation_profile_sha256": hashlib.sha256(
                     files["evaluation_profile.json"]).hexdigest(),
                 "metric_schema_sha256": hashlib.sha256(
                     files["metric_schema.json"]).hexdigest(),
                 "environment_lock_sha256": hashlib.sha256(
                     files["environment_lock.json"]).hexdigest()}
        _atomic_json(os.path.join(common, "COMMON_EVALUATOR_READY.json"), ready)
        return common, files, profile

    with tempfile.TemporaryDirectory() as base:
        common, files, profile = make_root(base)
        doc = supplement(common)
        check("adds_negative_report_sha",
              doc["negative_test_report_sha256"]
              == hashlib.sha256(files["negative_test_report.json"]).hexdigest())
        check("adds_assembly_manifest_sha",
              doc["assembly_manifest_sha256"]
              == hashlib.sha256(files["assembly_manifest.json"]).hexdigest())
        check("adds_full_profile_canonical_sha",
              doc["full_profile_sha256"]
              == _canonical_sha256(profile["scenarios"]["full"]))
        check("full_status_frozen", doc["FULL_PROFILE_STATUS"] == "FROZEN")
        check("sums_self_check_pass", doc["common_sha256sums_self_check"] == "PASS")
        check("ranking_not_authorized", doc["FORMAL_RANKING_AUTHORIZED"] is False)
        check("pre_existing_sha_preserved",
              doc["common_evaluator_sha256"]
              == hashlib.sha256(files["common_evaluator.py"]).hexdigest())
        # The marker itself must still not be covered by sums; sums still valid.
        st = verify_sha256sums_live(common)
        check("sums_still_pass_after_supplement", st["status"] == "PASS")
        # Idempotent: a second supplement does not fail.
        doc2 = supplement(common)
        check("idempotent", doc2["full_profile_sha256"] == doc["full_profile_sha256"])

    with tempfile.TemporaryDirectory() as base:
        common, files, _ = make_root(base)
        # Tamper a covered file -> drift on a pre-existing SHA -> fail closed.
        ready_path = os.path.join(common, "COMMON_EVALUATOR_READY.json")
        d = json.load(open(ready_path, encoding="utf-8"))
        d["metric_schema_sha256"] = "0" * 64
        _atomic_json(ready_path, d)
        try:
            supplement(common)
            check("drift_fail_closed", False)
        except FailClosed:
            check("drift_fail_closed", True)

    with tempfile.TemporaryDirectory() as base:
        common, files, _ = make_root(base)
        # Tamper a sums-covered file without touching the marker -> the live
        # sums re-check must report FAIL inside the evidence.
        with open(os.path.join(common, "metric_schema.json"), "ab") as fh:
            fh.write(b"tamper\n")
        st = verify_sha256sums_live(common)
        check("live_sums_detects_tamper",
              st["status"] == "FAIL" and st["mismatches"] == ["metric_schema.json"])

    check("canonical_sha_stable",
          _canonical_sha256({"b": 1, "a": [1, 2]})
          == _canonical_sha256({"a": [1, 2], "b": 1}))

    if problems:
        print("TIER3_READY_MARKER_SUPPLEMENT_SELF_TEST_FAIL")
        for p in problems:
            print("  -", p)
        return 1
    print("TIER3_READY_MARKER_SUPPLEMENT_SELF_TEST_PASS "
          "(additive only; drift fail-closed; live sums re-check)")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--self-test" in argv:
        return self_test()
    if "--supplement" in argv:
        i = argv.index("--common-root") if "--common-root" in argv else None
        require(i is not None,
                "usage: tier3_ready_marker_supplement.py --supplement "
                "--common-root DIR | --self-test")
        supplement(argv[i + 1])
        return 0
    print("usage: tier3_ready_marker_supplement.py --supplement "
          "--common-root DIR | --self-test")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
