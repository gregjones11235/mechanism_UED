#!/usr/bin/env python3
"""PUBLIC COMMON EVALUATOR ENTRY (closing contract S2/S3).

The single public entry point of the common evaluator. It owns NO evaluation
semantics of its own: it verifies that the deployed engine source is bit-identical
to the full SHA256 bound at assembly time, then delegates ALL behaviour (canonical
world/profile, frozen banks, episode loop, reset/step, terminal labels, FRONT
transition / graph-distance progress, BACK/FULL DEFEAT_KOBOLD, metric aggregation,
evaluation certificate) to tools/tier3_scaffolded_evaluation/tier3_evaluator.py.
Post-assembly tampering with the engine fails closed.
"""
import hashlib
import json
import os
import sys

ASSEMBLED_REPO_ROOT = "/home/oseasy/cc4_tier3_eval_20260730/repo"
ENGINE_REL = ("tools", "tier3_scaffolded_evaluation", "tier3_evaluator.py")
COMMON_EVALUATOR_ENGINE_SHA256 = "54ae18db24c6a826d91bfc7ea49dee39a777b800e24dcc9b8897398def8da715"
EVALUATION_PROFILE_SHA256 = "7147370115621bda0500d55d8fd506a119ef8d6467a08329aaf6e088fbf9ea73"
METRIC_SCHEMA_SHA256 = "3a1712c4074dcb8fe8043c5a67e3ad7c730f252c533ad148a7181ba28f953da0"
GIT_COMMIT_AT_ASSEMBLY = "7fe25a72fbe0e0e01dd74e25376a313698cee67b"
ASSEMBLED_AT_UTC = "2026-07-30T14:27:35.978819+00:00"


def _lf_sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read().replace(b"\r\n", b"\n")).hexdigest()


def engine_repo_root():
    return os.environ.get("CC4_EVALUATOR_REPO", ASSEMBLED_REPO_ROOT)


def engine_path():
    return os.path.join(engine_repo_root(), *ENGINE_REL)


def verify_engine_binding():
    p = engine_path()
    if not os.path.isfile(p):
        raise SystemExit("FAIL CLOSED: common evaluator engine missing at %s" % p)
    got = _lf_sha(p)
    if got != COMMON_EVALUATOR_ENGINE_SHA256:
        raise SystemExit("FAIL CLOSED: engine SHA drift %s != frozen %s"
                         % (got, COMMON_EVALUATOR_ENGINE_SHA256))
    return p


def binding_identity():
    return {
        "schema": "mechanism_UED.common_evaluator_binding_identity/v1",
        "engine_path": engine_path(),
        "common_evaluator_engine_sha256": COMMON_EVALUATOR_ENGINE_SHA256,
        "evaluation_profile_sha256": EVALUATION_PROFILE_SHA256,
        "metric_schema_sha256": METRIC_SCHEMA_SHA256,
        "git_commit_at_assembly": GIT_COMMIT_AT_ASSEMBLY,
        "assembled_at_utc": ASSEMBLED_AT_UTC,
    }


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--binding-identity" in argv:
        print(json.dumps(binding_identity(), indent=2, sort_keys=True))
        return 0
    p = verify_engine_binding()
    for entry in (os.path.dirname(p),
                  os.path.join(engine_repo_root(), "dicode_src", "src")):
        if os.path.isdir(entry) and entry not in sys.path:
            sys.path.insert(0, entry)
    import tier3_evaluator
    return tier3_evaluator.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
