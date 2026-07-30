#!/usr/bin/env python3
"""PUBLIC COMMON RUNNER ENTRY (closing contract S5).

The single public entry point of the common candidate-runtime runner. It never
hardcodes an architecture: family registration / dispatch lives in the bound
engine (tier3_candidate_runtime.py, ABI mechanism_UED.candidate_runtime_abi/v1);
this shim verifies the engine binding by full SHA256 and delegates. Base GTrXL /
Control / SlowGRU / Teacher runtimes are registered by their OWN owners; unknown
or missing families fail closed in the engine.
"""
import hashlib
import json
import os
import sys

ASSEMBLED_REPO_ROOT = "/home/oseasy/cc4_tier3_eval_20260730/repo"
RUNNER_REL = ("tools", "tier3_scaffolded_evaluation", "tier3_candidate_runtime.py")
COMMON_RUNNER_ENGINE_SHA256 = "6af09be4efdb3eef66ef68579177085ac7d410109b40e10359160fbec23f681f"
ABI_DOC_SHA256 = "61e52af6ff64a3071f8b64916c80906275dcb201d37feaa0382ed988d03d7f6a"
GIT_COMMIT_AT_ASSEMBLY = "7fe25a72fbe0e0e01dd74e25376a313698cee67b"
ASSEMBLED_AT_UTC = "2026-07-30T14:27:35.978819+00:00"


def _lf_sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read().replace(b"\r\n", b"\n")).hexdigest()


def runner_repo_root():
    return os.environ.get("CC4_EVALUATOR_REPO", ASSEMBLED_REPO_ROOT)


def runner_path():
    return os.path.join(runner_repo_root(), *RUNNER_REL)


def verify_runner_binding():
    p = runner_path()
    if not os.path.isfile(p):
        raise SystemExit("FAIL CLOSED: common runner engine missing at %s" % p)
    got = _lf_sha(p)
    if got != COMMON_RUNNER_ENGINE_SHA256:
        raise SystemExit("FAIL CLOSED: runner engine SHA drift %s != frozen %s"
                         % (got, COMMON_RUNNER_ENGINE_SHA256))
    return p


def _engine_module():
    p = verify_runner_binding()
    d = os.path.dirname(p)
    for entry in (d, os.path.join(runner_repo_root(), "dicode_src", "src")):
        if os.path.isdir(entry) and entry not in sys.path:
            sys.path.insert(0, entry)
    import tier3_candidate_runtime as engine
    return engine


def load_candidate(checkpoint_contract):
    """ABI entry: family dispatch happens inside the engine (fail closed)."""
    return _engine_module().load_candidate(checkpoint_contract)


def binding_identity():
    return {
        "schema": "mechanism_UED.common_runner_binding_identity/v1",
        "runner_path": runner_path(),
        "common_runner_engine_sha256": COMMON_RUNNER_ENGINE_SHA256,
        "abi_doc_sha256": ABI_DOC_SHA256,
        "git_commit_at_assembly": GIT_COMMIT_AT_ASSEMBLY,
        "assembled_at_utc": ASSEMBLED_AT_UTC,
    }


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--binding-identity" in argv:
        print(json.dumps(binding_identity(), indent=2, sort_keys=True))
        return 0
    if "--metadata" in argv:
        with open(argv[argv.index("--metadata") + 1], encoding="utf-8") as fh:
            contract = json.load(fh)
        rt = load_candidate(contract)
        print(json.dumps(rt.candidate_metadata(), indent=2, sort_keys=True))
        return 0
    print("usage: common_runner.py --binding-identity | --metadata CONTRACT.json")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
