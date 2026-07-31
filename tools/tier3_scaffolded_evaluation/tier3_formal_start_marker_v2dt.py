#!/usr/bin/env python3
"""CC4 Tier3 — SECONDARY_AUDIT_PASS marker (start-authorization record).

Writes the immutable pre-run audit record required by 总控 constraint 8
("启动前记录 SECONDARY_AUDIT_PASS marker 或等价审计记录") to
<pool>/cc4/SECONDARY_AUDIT_PASS.json (+ .sha256 sidecar).

DESIGN NOTE (deliberate deviation from the original plan, documented in the
runbook and final report): this tool writes ONLY the marker. The original
plan had the marker tool also flip COMMON_EVALUATOR_V2_READY's
FORMAL_RANKING_STARTED flag — but the frozen V2 common verifier
(smokev2.verify_engine_and_common_v2, step 1e) requires
FORMAL_RANKING_STARTED to be false, so a pre-run flip would fail-close EVERY
formal run. Resolution: the marker alone authorizes start (the formal driver
verifies it in stage 1b); the READY ranking flags are flipped exactly once,
AT CLOSING, by the ranking tool (tier3_formal_ranking_v2dt.py) — single
READY writer, zero races, and after the flip the driver refuses reruns.

The marker records: the verbatim audit verdict, the 8 verbatim audit
grounds, evidence references (binding gate SHA cec16711…, READY file live
SHA, offline verifier 164/164), recorded_at_utc, git HEAD, and a SHA over
the frozen V2 pin snapshot. An existing marker is NEVER overwritten (audit
records are append-only in spirit; a new audit requires a new marker name).

JAX-free; safe on any host. Usage:
  python tools/tier3_scaffolded_evaluation/tier3_formal_start_marker_v2dt.py \
      --pool-cc4-dir /home/oseasy/student_pool_v1/cc4 \
      --common-dir /home/oseasy/student_pool_v1/common_v2
  --self-test   structural checks (any host)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import tier3_projection_runtime as proj                            # noqa: E402
import tier3_projection_binding_smoke_v2 as smokev2                # noqa: E402
import tier3_evaluation_certificate_v2dt as certmod                # noqa: E402

MARKER_SCHEMA = "mechanism_UED.tier3_secondary_audit_marker/v2dt"
MARKER_NAME = "SECONDARY_AUDIT_PASS.json"
SECONDARY_AUDIT_VERDICT = certmod.SECONDARY_AUDIT_VERDICT
POOL_BINDING_GATE_V2DT_SHA256 = \
    "cec167117a7aa8e67a3d5eb60839e711e72d950135553e4035a87e6c9859a352"

# The 8 audit grounds, VERBATIM from the secondary audit conclusion.
AUDIT_GROUNDS = [
    "COMMON_EVALUATOR_V2_READY=true",
    "STUDENT_COMMON_BINDING_PASS_COUNT=6/6",
    "TEACHER_REFERENCE_BINDING=PASS",
    "gate_failures=[]",
    "verify_v2dt_closing_evidence.py 164/164 PASS",
    "CHECKPOINTS_MODIFIED=false",
    "CONTROL_RETRAINED=false",
    "CANDIDATE_EXCEPTION_USED=false",
]
START_PRECONDITION = "FORMAL_RANKING_STARTED=false"


def _git_commit_head():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(HERE),
            capture_output=True, text=True, timeout=30)
        head = out.stdout.strip()
        if out.returncode == 0 and len(head) == 40:
            return head
    except Exception as exc:
        return "unavailable: %r" % exc
    return "unavailable"


def verify_start_preconditions(common_dir, pool_cc4_dir,
                               expected_gate_sha=POOL_BINDING_GATE_V2DT_SHA256):
    """Live cross-checks performed BEFORE writing the marker. Fail-closed."""
    ready_path = os.path.join(common_dir, "COMMON_EVALUATOR_V2_READY.json")
    proj.require(os.path.isfile(ready_path),
                 "FAIL CLOSED (MARKER_PRECOND): %s missing" % ready_path)
    ready = proj.read_json(ready_path)
    proj.require(ready.get("COMMON_EVALUATOR_V2_READY") is True,
                 "FAIL CLOSED (MARKER_PRECOND): COMMON_EVALUATOR_V2_READY is "
                 "not true")
    proj.require(ready.get("FORMAL_RANKING_STARTED") is False,
                 "FAIL CLOSED (MARKER_PRECOND): FORMAL_RANKING_STARTED is %r"
                 % ready.get("FORMAL_RANKING_STARTED"))
    proj.require(ready.get("STUDENT_COMMON_BINDING_PASS_COUNT") == "6/6",
                 "FAIL CLOSED (MARKER_PRECOND): binding count %r"
                 % ready.get("STUDENT_COMMON_BINDING_PASS_COUNT"))
    gate_path = os.path.join(pool_cc4_dir, "POOL_BINDING_GATE_V2DT.json")
    proj.require(os.path.isfile(gate_path),
                 "FAIL CLOSED (MARKER_PRECOND): %s missing" % gate_path)
    gate_sha = proj.sha256_file(gate_path)
    proj.require(gate_sha == expected_gate_sha,
                 "FAIL CLOSED (MARKER_PRECOND): gate sha %s != frozen %s"
                 % (gate_sha, expected_gate_sha))
    return {"ready_sha256": proj.sha256_file(ready_path),
            "gate_sha256": gate_sha}


def write_marker(pool_cc4_dir, common_dir, recorded_at_utc=None,
                 expected_gate_sha=POOL_BINDING_GATE_V2DT_SHA256):
    marker_path = os.path.join(pool_cc4_dir, MARKER_NAME)
    proj.require(not os.path.exists(marker_path),
                 "FAIL CLOSED (AUDIT_MARKER_EXISTS_IMMUTABLE): %s already "
                 "exists — audit records are never overwritten" % marker_path)
    proj.require(os.path.isdir(pool_cc4_dir),
                 "FAIL CLOSED: pool cc4 dir missing: %s" % pool_cc4_dir)
    refs = verify_start_preconditions(common_dir, pool_cc4_dir,
                                      expected_gate_sha)
    pins = certmod.pins_snapshot()
    marker = {
        "schema": MARKER_SCHEMA,
        "verdict": SECONDARY_AUDIT_VERDICT,
        "audit_authority": "INDEPENDENT_SECONDARY_AUDIT (authorized by 总控)",
        "grounds": AUDIT_GROUNDS,
        "start_precondition": START_PRECONDITION,
        "evidence": {
            "binding_gate_file": "POOL_BINDING_GATE_V2DT.json",
            "binding_gate_sha256": refs["gate_sha256"],
            "common_evaluator_v2_ready_sha256": refs["ready_sha256"],
            "offline_verifier": "verify_v2dt_closing_evidence.py",
            "offline_verifier_result": "164/164 PASS",
            "closing_evidence_commit": "b736c8c",
        },
        "recorded_at_utc": recorded_at_utc or smokev2.utc_now_iso(),
        "git_commit_head": _git_commit_head(),
        "pins_snapshot_sha256": proj.sha256_bytes(
            proj.canonical_json_bytes(pins)),
        "pins_snapshot": pins,
        "pool_cc4_dir": pool_cc4_dir,
        "pool_root": os.path.dirname(os.path.normpath(common_dir)),
        "common_dir": common_dir,
        "note": "pre-run authorization record; the READY ranking-flag flip "
                "happens once at closing via tier3_formal_ranking_v2dt.py "
                "(single READY writer)",
    }
    smokev2.write_json(marker_path, marker)
    sha = proj.sha256_file(marker_path)
    with open(marker_path + ".sha256", "w", encoding="utf-8",
              newline="\n") as fh:
        fh.write("%s  %s\n" % (sha, MARKER_NAME))
    return marker_path, sha


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool-cc4-dir", default=None,
                    help="default: parent of --common-dir + /cc4")
    ap.add_argument("--common-dir", default="/home/oseasy/student_pool_v1/common_v2")
    ap.add_argument("--recorded-at-utc", default=None,
                    help="override timestamp (ISO8601); default = now")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return run_self_test()
    pool_cc4_dir = args.pool_cc4_dir or os.path.join(
        os.path.dirname(os.path.normpath(args.common_dir)), "cc4")
    path, sha = write_marker(pool_cc4_dir, args.common_dir,
                             args.recorded_at_utc)
    print("[marker] SECONDARY_AUDIT_PASS written: %s" % path, flush=True)
    print("[marker] sha256=%s (sidecar written)" % sha, flush=True)
    print("[marker] verdict=%s" % SECONDARY_AUDIT_VERDICT, flush=True)
    print("[marker] READY flip is NOT performed here — it happens once at "
          "ranking close (tier3_formal_ranking_v2dt.py)", flush=True)
    return 0


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------
def run_self_test():
    import tempfile
    checks = 0

    def ok(cond, msg):
        nonlocal checks
        checks += 1
        proj.require(cond, "MARKER_SELF_TEST FAIL: %s" % msg)

    with tempfile.TemporaryDirectory() as td:
        cc4 = os.path.join(td, "cc4")
        common = os.path.join(td, "common_v2")
        os.makedirs(cc4)
        os.makedirs(common)

        def ready(started=False, count="6/6", v2=True):
            smokev2.write_json(os.path.join(
                common, "COMMON_EVALUATOR_V2_READY.json"),
                {"COMMON_EVALUATOR_V2_READY": v2,
                 "FORMAL_RANKING_STARTED": started,
                 "STUDENT_COMMON_BINDING_PASS_COUNT": count,
                 "pending_gates": ["INDEPENDENT_SECONDARY_AUDIT"]})

        def gate(content=b'{"synthetic_gate": true}\n'):
            p = os.path.join(cc4, "POOL_BINDING_GATE_V2DT.json")
            with open(p, "wb") as fh:
                fh.write(content)
            return proj.sha256_file(p)

        # missing READY -> precondition fail
        try:
            verify_start_preconditions(common, cc4, "0" * 64)
            ok(False, "missing READY accepted")
        except proj.FailClosed:
            checks += 1
        ready(v2=False)
        try:
            verify_start_preconditions(common, cc4, "0" * 64)
            ok(False, "V2_READY=false accepted")
        except proj.FailClosed:
            checks += 1
        ready(started=True)
        try:
            verify_start_preconditions(common, cc4, "0" * 64)
            ok(False, "FORMAL_RANKING_STARTED=true accepted")
        except proj.FailClosed:
            checks += 1
        ready(count="5/6")
        try:
            verify_start_preconditions(common, cc4, "0" * 64)
            ok(False, "binding count 5/6 accepted")
        except proj.FailClosed:
            checks += 1
        ready()
        try:
            verify_start_preconditions(common, cc4, "0" * 64)
            ok(False, "missing gate accepted")
        except proj.FailClosed:
            checks += 1
        gsha = gate()
        try:
            verify_start_preconditions(common, cc4, "0" * 64)
            ok(False, "gate sha mismatch accepted")
        except proj.FailClosed:
            checks += 1
        ok(verify_start_preconditions(common, cc4, gsha)["gate_sha256"]
           == gsha, "preconditions pass with matching gate")

        # write + sidecar + fields
        path, sha = write_marker(cc4, common,
                                 recorded_at_utc="1970-01-01T00:00:00+00:00",
                                 expected_gate_sha=gsha)
        ok(os.path.isfile(path) and os.path.isfile(path + ".sha256"),
           "marker + sidecar written")
        side = open(path + ".sha256", encoding="utf-8").read().split()[0]
        ok(side == sha == proj.sha256_file(path), "sidecar matches")
        m = proj.read_json(path)
        ok(m["verdict"] == SECONDARY_AUDIT_VERDICT, "verdict verbatim")
        ok(m["grounds"] == AUDIT_GROUNDS and len(m["grounds"]) == 8,
           "8 grounds verbatim")
        ok(m["evidence"]["binding_gate_sha256"] == gsha, "gate sha recorded")
        ok(m["pins_snapshot_sha256"] == proj.sha256_bytes(
            proj.canonical_json_bytes(certmod.pins_snapshot())), "pins sha")
        ok(m["pool_cc4_dir"] == cc4, "pool dir recorded")
        ok(len(m["git_commit_head"]) >= 10, "git head recorded")

        # immutability: second write refused
        try:
            write_marker(cc4, common, expected_gate_sha=gsha)
            ok(False, "overwrite accepted")
        except proj.FailClosed as exc:
            ok("AUDIT_MARKER_EXISTS_IMMUTABLE" in str(exc),
               "overwrite refused")

        # driver start gate accepts this exact marker shape
        import tier3_formal_evaluation_v2dt as driver
        # driver pins the REAL gate sha; verify_formal_start also checks
        # pool_cc4_dir equality + verdict — the only mismatch here is the
        # gate sha (synthetic), so expect fail-closed for the RIGHT reason
        try:
            driver.verify_formal_start(common, cc4)
            ok(False, "synthetic-gate marker passed the frozen gate pin")
        except proj.FailClosed as exc:
            ok("binding-gate sha" in str(exc),
               "driver rejects synthetic gate sha (frozen pin enforced)")

    print("MARKER_SELF_TEST_PASS checks=%d" % checks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
