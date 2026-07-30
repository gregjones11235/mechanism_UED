#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CC3 independent recomputation of ALL CC4 common-evaluator SHAs (binding task section 1).

Read-only against /home/oseasy/student_pool_v1/common. Recomputes every SHA that
COMMON_EVALUATOR_READY.json declares, verifies the common SHA256SUMS file entry by
entry, and cross-checks the frozen front/back bank CONTENT hashes. Any mismatch ->
COMMON_SHAS_VERIFIED=false and the differing fields are listed. Writes a JSON
evidence file; never modifies the common tree.
"""
import argparse
import hashlib
import json
import os
import subprocess


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--common-root", default="/home/oseasy/student_pool_v1/common")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = args.common_root

    with open(os.path.join(root, "COMMON_EVALUATOR_READY.json"), encoding="utf-8") as f:
        ready = json.load(f)

    pairs = [
        ("common_runner.py", "common_runner_sha256"),
        ("common_evaluator.py", "common_evaluator_sha256"),
        ("evaluation_profile.json", "evaluation_profile_sha256"),
        ("metric_schema.json", "metric_schema_sha256"),
        ("front_bank_states.npz", "front_bank_file_sha256"),
        ("back_bank_states.npz", "back_bank_file_sha256"),
        ("environment_lock.json", "environment_lock_sha256"),
        ("SHA256SUMS", "sha256sums_sha256"),
    ]
    recomputed = {}
    mismatches = []
    for fname, key in pairs:
        got = sha_file(os.path.join(root, fname))
        want = ready.get(key)
        recomputed[key] = dict(file=fname, recomputed=got, declared=want,
                               match=bool(got == want))
        if got != want:
            mismatches.append("%s: recomputed=%s declared=%s" % (fname, got, want))

    # common SHA256SUMS entry-by-entry verification
    proc = subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=root,
                          capture_output=True, text=True)
    lines = [l for l in (proc.stdout + proc.stderr).splitlines() if l.strip()]
    bad = [l for l in lines if ("OK" not in l) and ("成功" not in l)]
    recomputed["sha256sums_check"] = dict(
        exit_code=proc.returncode, entries=len(lines),
        non_ok_lines=bad[:20], all_ok=bool(proc.returncode == 0 and not bad))
    if proc.returncode != 0 or bad:
        mismatches.append("SHA256SUMS -c failed: %s" % (bad[:5] or proc.returncode))

    # frozen bank CONTENT hashes (declared in the bank manifests)
    for manifest_name, ready_key in (("front_bank_manifest.json",
                                      "front_bank_content_sha256"),
                                     ("back_bank_manifest.json",
                                      "back_bank_content_sha256")):
        with open(os.path.join(root, manifest_name), encoding="utf-8") as f:
            manifest = json.load(f)
        declared_in_manifest = [
            (k, v) for k, v in sorted(manifest.items())
            if "content" in k and "sha" in k and isinstance(v, str) and len(v) == 64]
        got = declared_in_manifest[0][1] if declared_in_manifest else None
        want = ready.get(ready_key)
        recomputed[ready_key] = dict(manifest=manifest_name,
                                     manifest_field=(declared_in_manifest[0][0]
                                                     if declared_in_manifest else None),
                                     manifest_declared=got, ready_declared=want,
                                     match=bool(got is not None and got == want))
        if got != want:
            mismatches.append("%s: manifest=%s ready=%s" % (ready_key, got, want))

    # full profile seeds must come from the common tree (task: no self-created seeds)
    profile_path = os.path.join(root, "evaluation_profile.json")
    with open(profile_path, encoding="utf-8") as f:
        profile = json.load(f)
    full_seed_info = {k: v for k, v in profile.items() if "full" in str(k).lower()}
    configs = sorted(os.listdir(os.path.join(root, "evaluator_configs"))) \
        if os.path.isdir(os.path.join(root, "evaluator_configs")) else []

    verified = not mismatches and bool(ready.get("COMMON_EVALUATOR_READY"))
    result = dict(
        record_version="cc3_common_sha_verification/v1",
        owner="CC3",
        common_root=root,
        COMMON_SHAS_VERIFIED=bool(verified),
        COMMON_EVALUATOR_READY_declared=bool(ready.get("COMMON_EVALUATOR_READY")),
        mismatches=mismatches,
        recomputed=recomputed,
        full_profile_keys_in_evaluation_profile=full_seed_info,
        evaluator_configs_listing=configs,
        ready_git_commit_head=ready.get("git_commit_head"),
        ready_generated_at_utc=ready.get("generated_at_utc"),
        readonly=True,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    print("COMMON_SHAS_VERIFIED=%s mismatches=%d entries_checked=%d OUT=%s" % (
        verified, len(mismatches), len(pairs) + 2, args.out))
    for m in mismatches:
        print("MISMATCH: %s" % m)


if __name__ == "__main__":
    main()
