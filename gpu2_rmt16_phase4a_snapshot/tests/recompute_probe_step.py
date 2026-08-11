#!/usr/bin/env python3
"""Phase4A-v2 §二 / GATE 1 — OFFLINE recompute of the L512 probe first_ge512 PRECISE resolved step.

Re-derives each probe episode's authoritative `completion_resolved_env_step` from the EXISTING
episode records (written by the frozen 16384-step probe), WITHOUT rerunning the probe:

    resolved = update_index * num_envs * rollout_steps
             + rollout_step * num_envs
             + env_id
             + 1

Constants for the frozen probe: num_envs=16, rollout_steps=128. Each record carries
(update_index, rollout_step, env_id, length, episode_id). The reachability conclusion is
re-checked (any length>=512 => reachable) and is expected to remain BOTH=PASS; this script
only CORRECTS the step provenance, it does NOT change the verdict.

Usage:
    python recompute_probe_step.py <episodes.jsonl> [label]
    python recompute_probe_step.py --selftest          # synthetic method self-test (no data needed)

Phase4A-v2.1 (§六) two-arm remote-recomputability mode (from the FROZEN in-repo raw probe JSONL):
    python recompute_probe_step.py \
        --persistent evidence/raw_probe/persistent_probe_episodes.jsonl \
        --reset128   evidence/raw_probe/reset128_probe_episodes.jsonl \
        [--sha256sums evidence/raw_probe/SHA256SUMS] \
        --out reports/rmt16_l512_probe_recomputed.json

The --out report is RECOMPUTED from the input records at run time — NOTHING is hardcoded; if the
frozen evidence is unavailable the mode must be reported as BLOCKED_SOURCE_UNAVAILABLE, never
faked. Prints a JSON report; exit 0 on success, 1 on fail-closed (hash mismatch / missing source).
"""
import hashlib
import json
import os
import sys

NUM_ENVS = 16
ROLLOUT_STEPS = 128
GE_LEN = 512


def completion_resolved_env_step(update_index, num_envs, rollout_steps, rollout_step, env_id):
    """Authoritative precise resolved env step (1-indexed). See phase4a_v2_counters."""
    return (int(update_index) * int(num_envs) * int(rollout_steps)
            + int(rollout_step) * int(num_envs) + int(env_id) + 1)


def completion_global_step_deprecated(update_index, num_envs, rollout_steps, rollout_step):
    """Old, non-precise formula (kept only to show the before/after delta)."""
    return int(update_index) * (int(num_envs) * int(rollout_steps)) + int(rollout_step)


def recompute_from_records(records, num_envs=NUM_ENVS, rollout_steps=ROLLOUT_STEPS, ge_len=GE_LEN):
    """records: chronological list of episode dicts (file order == collection order)."""
    total = len(records)
    ge = [r for r in records if int(r["length"]) >= ge_len]
    first = None
    for r in ge:  # first in chronological (file) order
        ui = int(r["update_index"]); rs = int(r["rollout_step"]); eid = int(r["env_id"])
        resolved = completion_resolved_env_step(ui, num_envs, rollout_steps, rs, eid)
        deprecated = completion_global_step_deprecated(ui, num_envs, rollout_steps, rs)
        first = dict(
            episode_id=int(r["episode_id"]), length=int(r["length"]),
            update_index=ui, rollout_step=rs, env_id=eid,
            first_ge512_resolved_env_step=resolved,
            first_ge512_global_step_deprecated=deprecated,
            step_correction_delta=resolved - deprecated,
        )
        break
    return dict(
        num_envs=num_envs, rollout_steps=rollout_steps, ge_len=ge_len,
        total_completed_episodes=total,
        count_ge512=len(ge),
        reachable=bool(len(ge) > 0),
        first_ge512=first,
    )


def _selftest():
    # Synthetic records: lengths [100, 600, 200]; the 600-step ep is first >=512.
    recs = [
        dict(episode_id=0, length=100, update_index=0, rollout_step=10, env_id=2),
        dict(episode_id=1, length=600, update_index=2, rollout_step=5, env_id=3),
        dict(episode_id=2, length=200, update_index=3, rollout_step=1, env_id=0),
    ]
    rep = recompute_from_records(recs)
    # resolved = 2*16*128 + 5*16 + 3 + 1 = 4096 + 80 + 4 = 4180
    expected = 2 * 16 * 128 + 5 * 16 + 3 + 1
    assert rep["count_ge512"] == 1, rep
    assert rep["reachable"] is True, rep
    assert rep["first_ge512"]["first_ge512_resolved_env_step"] == expected == 4180, rep
    # deprecated = 2*2048 + 5 = 4101; delta = 79
    assert rep["first_ge512"]["first_ge512_global_step_deprecated"] == 4101, rep
    assert rep["first_ge512"]["step_correction_delta"] == 79, rep
    # empty / not-reachable case
    rep2 = recompute_from_records([dict(episode_id=0, length=100, update_index=0,
                                        rollout_step=0, env_id=0)])
    assert rep2["reachable"] is False and rep2["first_ge512"] is None, rep2
    return rep


def _load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_sha256sums(files, sums_path):
    """files: {basename: path}. Fail closed: any missing manifest entry / mismatch => BLOCKED."""
    manifest = {}
    with open(sums_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            digest, name = line.split(None, 1)
            manifest[name.strip().lstrip("*")] = digest.lower()
    results = {}
    ok_all = True
    for name, path in files.items():
        actual = _sha256_file(path)
        expected = manifest.get(os.path.basename(path))
        ok = bool(expected is not None and actual == expected)
        ok_all = ok_all and ok
        results[os.path.basename(path)] = dict(sha256=actual, expected=expected, match=ok)
    return ok_all, results


def two_arm_recompute(persistent_path, reset128_path, sha256sums_path=None, out_path=None):
    """§六 remote-recomputability recompute from the FROZEN raw probe JSONL (no rerun).

    Everything in the report is DERIVED from the input records at run time. Fail-closed:
    missing source file -> RAW_PROBE_EVIDENCE_REMOTE_RECOMPUTABILITY=BLOCKED_SOURCE_UNAVAILABLE;
    hash mismatch -> RAW_PROBE_SOURCE_HASH_MISMATCH=BLOCKED."""
    report = dict(
        recomputed_by="tests/recompute_probe_step.py",
        method="offline recompute from frozen raw probe JSONL; NO probe rerun, NO hardcoded values",
        num_envs=NUM_ENVS, rollout_steps=ROLLOUT_STEPS, ge_len=GE_LEN,
        inputs=dict(persistent=persistent_path, reset128=reset128_path))

    for arm, path in (("persistent", persistent_path), ("reset128", reset128_path)):
        if not (path and os.path.isfile(path)):
            report["RAW_PROBE_EVIDENCE_REMOTE_RECOMPUTABILITY"] = "BLOCKED_SOURCE_UNAVAILABLE"
            report["blocked_arm"] = arm
            report["blocked_path"] = path
            return report

    if sha256sums_path:
        if not os.path.isfile(sha256sums_path):
            report["RAW_PROBE_EVIDENCE_REMOTE_RECOMPUTABILITY"] = "BLOCKED_SOURCE_UNAVAILABLE"
            report["blocked_path"] = sha256sums_path
            return report
        ok, results = _verify_sha256sums(
            {"persistent": persistent_path, "reset128": reset128_path}, sha256sums_path)
        report["hash_verification"] = dict(sha256sums=sha256sums_path, files=results, all_match=ok)
        if not ok:
            report["RAW_PROBE_SOURCE_HASH_MISMATCH"] = "BLOCKED"
            return report

    rep_p = recompute_from_records(_load_jsonl(persistent_path)); rep_p["source"] = persistent_path
    rep_r = recompute_from_records(_load_jsonl(reset128_path)); rep_r["source"] = reset128_path
    report["arms"] = dict(persistent=rep_p, reset128=rep_r)

    step_p = (rep_p["first_ge512"] or {}).get("first_ge512_resolved_env_step")
    step_r = (rep_r["first_ge512"] or {}).get("first_ge512_resolved_env_step")
    report["first_ge512_resolved_env_step_persistent"] = step_p
    report["first_ge512_resolved_env_step_reset128"] = step_r
    report["cross_arm_resolved_step_agree"] = bool(
        step_p is not None and step_p == step_r)
    if report["cross_arm_resolved_step_agree"]:
        report["first_ge512_resolved_env_step"] = step_p

    if rep_p["reachable"] and rep_r["reachable"]:
        verdict = "BOTH"
    elif rep_p["reachable"]:
        verdict = "PERSISTENT_ONLY"
    elif rep_r["reachable"]:
        verdict = "RESET128_ONLY"
    else:
        verdict = "NEITHER"
    report["L512_REACHABILITY"] = verdict
    report["RAW_PROBE_EVIDENCE_REMOTE_RECOMPUTABILITY"] = "PASS"

    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        report["report_written"] = out_path
    return report


def main(argv):
    # ---- legacy / self-test CLIs (unchanged) ----
    if len(argv) > 1 and argv[1] == "--selftest":
        rep = _selftest()
        print("RECOMPUTE_SELFTEST=PASS")
        print(json.dumps(rep, indent=2))
        return 0
    # ---- §六 two-arm remote-recomputability CLI ----
    if any(a in argv for a in ("--persistent", "--reset128", "--out")):
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("--persistent", required=True)
        ap.add_argument("--reset128", required=True)
        ap.add_argument("--sha256sums", default=None)
        ap.add_argument("--out", default=None)
        a = ap.parse_args(argv[1:])
        report = two_arm_recompute(a.persistent, a.reset128, a.sha256sums, a.out)
        print(json.dumps(report, indent=2))
        blocked = ("RAW_PROBE_SOURCE_HASH_MISMATCH" in report
                   or report.get("RAW_PROBE_EVIDENCE_REMOTE_RECOMPUTABILITY",
                                  "").startswith("BLOCKED"))
        return 1 if blocked else 0
    if len(argv) < 2:
        print("usage: recompute_probe_step.py <episodes.jsonl> [label] | --selftest\n"
              "       recompute_probe_step.py --persistent <jsonl> --reset128 <jsonl> "
              "[--sha256sums <SHA256SUMS>] [--out <report.json>]", file=sys.stderr)
        return 2
    path = argv[1]
    label = argv[2] if len(argv) > 2 else path
    rep = recompute_from_records(_load_jsonl(path))
    rep["source"] = label
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
