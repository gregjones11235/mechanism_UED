import os, sys, json, hashlib, subprocess

# Section 8 hard gate: prove the committed git snapshot bytes == the runtime
# source bytes the probe will actually execute, and both == the manifest.
# Runs on the SERVER (runtime + server clone coexist here).
# Identity fields per director: source_snapshot_commit / runtime_source_manifest_sha
# / runtime_file_sha256.  Deliberately NO runtime_git_head (runtime dir is NOT a git repo).

REPO = "/home/oseasy/git_work/mechanism_UED_rmt16_probe"
SNAP_PREFIX = "gpu2_rmt16_phase4a_snapshot"
MANIFEST = os.path.join(REPO, SNAP_PREFIX, "provenance/manifests/runtime_source_manifest.json")
REPORT = os.path.join(REPO, SNAP_PREFIX, "reports/prelaunch_runtime_git_parity.json")

MODIFIED3 = ["runtime/wrapper_src/wrappers_cl.py",
             "runtime/experiment_src/rmt_collect.py",
             "runtime/experiment_src/train_rmt16_p2replay.py"]
FROZEN8 = ["runtime/frozen_modules/%s.py" % m for m in
           ["network_rmt16", "rmt_replay_buffer", "rmt_memory_anchor", "rmt16_memory",
            "rmt_ppo", "rmt_hindsight", "replay_buffer", "rmt_replay_learner"]]


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(p):
    with open(p, "rb") as f:
        return sha256_bytes(f.read())


def line_ending(b):
    if b"\r\n" in b:
        return "CRLF"
    if b"\n" in b:
        return "LF"
    return "none"


def git_blob_bytes(snapshot_path):
    # `git show HEAD:<path>` emits the blob content verbatim (no added newline),
    # exactly as the director-specified `git show HEAD:<path> | sha256sum`.
    r = subprocess.run(["git", "-C", REPO, "show", "HEAD:%s/%s" % (SNAP_PREFIX, snapshot_path)],
                       capture_output=True)
    if r.returncode != 0:
        return None, r.stderr.decode("utf-8", "replace").strip()
    return r.stdout, None


def git_head():
    return subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"],
                          capture_output=True).stdout.decode().strip()


def git_branch():
    return subprocess.run(["git", "-C", REPO, "branch", "--show-current"],
                          capture_output=True).stdout.decode().strip()


with open(MANIFEST, "rb") as f:
    manifest_bytes = f.read()
manifest = json.loads(manifest_bytes.decode("utf-8"))
runtime_source_manifest_sha = sha256_bytes(manifest_bytes)
source_snapshot_commit = git_head()

per_file = []
hard_gate_files = []
all_match = True
hard_gate_pass = True

for e in manifest["files"]:
    sp = e["snapshot_path"]
    orig = e["original_absolute_path"]
    manifest_sha = e["sha256"]
    is_runtime = sp.startswith("runtime/")

    blob, err = git_blob_bytes(sp)
    git_blob_sha = sha256_bytes(blob) if blob is not None else None
    blob_line_ending = line_ending(blob) if blob is not None else None

    runtime_exists = os.path.exists(orig)
    if runtime_exists:
        with open(orig, "rb") as f:
            rb = f.read()
        runtime_sha = sha256_bytes(rb)
        runtime_line_ending = line_ending(rb)
    else:
        runtime_sha = None
        runtime_line_ending = None

    match_blob_manifest = (git_blob_sha == manifest_sha)
    match_blob_runtime = (git_blob_sha == runtime_sha) if runtime_exists else False
    match_all3 = match_blob_manifest and match_blob_runtime and runtime_exists

    rec = dict(snapshot_path=sp, original_absolute_path=orig, role=e.get("role"),
               modified_or_frozen=e.get("modified_or_frozen"),
               manifest_sha256=manifest_sha, git_blob_sha256=git_blob_sha,
               runtime_file_sha256=runtime_sha, runtime_exists=runtime_exists,
               git_blob_line_ending=blob_line_ending, runtime_line_ending=runtime_line_ending,
               match_blob_manifest=match_blob_manifest, match_blob_runtime=match_blob_runtime,
               match_all3=bool(match_all3), git_show_error=err)
    per_file.append(rec)

    if not match_all3:
        all_match = False
    if is_runtime:
        hard_gate_files.append(sp)
        if not match_all3:
            hard_gate_pass = False

# Explicit spotlight on the CRLF wrapper + 3 modified + 8 frozen.
def rec_for(sp):
    for r in per_file:
        if r["snapshot_path"] == sp:
            return r
    return None

spotlight = {}
for sp in MODIFIED3 + FROZEN8:
    r = rec_for(sp)
    spotlight[sp] = (None if r is None else dict(
        manifest=r["manifest_sha256"][:16], git_blob=(r["git_blob_sha256"] or "")[:16],
        runtime=(r["runtime_file_sha256"] or "")[:16], match_all3=r["match_all3"],
        line_ending=r["runtime_line_ending"]))

GIT_SNAPSHOT_BYTE_PARITY = "PASS" if (hard_gate_pass and all_match) else "FAIL"

report = dict(
    verdict="GIT_SNAPSHOT_BYTE_PARITY=%s" % GIT_SNAPSHOT_BYTE_PARITY,
    GIT_SNAPSHOT_BYTE_PARITY=GIT_SNAPSHOT_BYTE_PARITY,
    # ---- director-mandated identity fields ----
    source_snapshot_commit=source_snapshot_commit,
    probe_branch=git_branch(),
    runtime_source_manifest_sha256=runtime_source_manifest_sha,
    # per-file runtime_file_sha256 lives inside per_file[].runtime_file_sha256
    runtime_git_head=None,  # absent BY DESIGN: runtime dir is not a git repository
    runtime_git_head_note=("not emitted because the runtime source trees "
                           "(experiments/rmt16_replay_phase4a and incoming/.../dicode_v7fix58_armB) "
                           "are NOT git repositories; code identity is source_snapshot_commit + "
                           "runtime_source_manifest_sha + per-file runtime_file_sha256."),
    # ---- gate scope ----
    hard_gate_scope="all snapshot_path under runtime/ (3 modified + 8 frozen + 7 deps = 18 files)",
    hard_gate_file_count=len(hard_gate_files),
    hard_gate_pass=bool(hard_gate_pass),
    all_manifest_entries_match=bool(all_match),
    total_manifest_files=len(manifest["files"]),
    spotlight_modified3_frozen8=spotlight,
    wrapper_crlf=spotlight.get("runtime/wrapper_src/wrappers_cl.py"),
    per_file=per_file,
)

with open(REPORT, "w") as f:
    json.dump(report, f, indent=2)

print("source_snapshot_commit=%s" % source_snapshot_commit)
print("probe_branch=%s" % git_branch())
print("runtime_source_manifest_sha256=%s" % runtime_source_manifest_sha)
print("hard_gate_files=%d  hard_gate_pass=%s" % (len(hard_gate_files), hard_gate_pass))
print("all_manifest_entries_match=%s" % all_match)
print("--- spotlight (manifest / git_blob / runtime / match / lineending) ---")
for sp in MODIFIED3 + FROZEN8:
    s = spotlight[sp]
    if s is None:
        print("  MISSING %s" % sp)
    else:
        print("  %-52s %s %s %s %s %s" % (sp, s["manifest"], s["git_blob"], s["runtime"],
                                          s["match_all3"], s["line_ending"]))
# list any mismatch
bad = [r["snapshot_path"] for r in per_file if not r["match_all3"]]
if bad:
    print("MISMATCH_FILES:")
    for b in bad:
        print("  " + b)
print("GIT_SNAPSHOT_BYTE_PARITY=%s" % GIT_SNAPSHOT_BYTE_PARITY)
print("[saved] %s" % REPORT)
sys.exit(0 if GIT_SNAPSHOT_BYTE_PARITY == "PASS" else 1)
