#!/usr/bin/env python
"""
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!  DEPRECATED_INVALID_FOR_WORLD_SET_HASH  --  DO NOT USE AS A MATERIALIZER  !!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Status: OLD_WORLD_KEY_PROTOTYPE = DEPRECATED_INVALID_FOR_WORLD_SET_HASH
        All outputs of this script are NON_SCIENTIFIC_PROTOTYPE.

Renamed (git mv, history preserved) from materialize_world_set_twice.py during
GLOBAL_WORLD_MATERIALIZER_CORRECTION_V2. The original content is retained below
ONLY for transparency / audit trail -- it is NOT a valid world materializer and
MUST NOT be executed to produce any world_set_hash.

WHY THIS IS INVALID (methodological error in the original):
  * It hashes ONLY:
      - the jax.random.fold_in(PRNGKey(wrapper_seed), world_index) PRNG key bytes;
      - a recipe descriptor (frozen recipe + world_index + seed label);
    It does NOT create a Craftax environment, does NOT call the real evaluator's
    reset / world-generation path, and does NOT generate or serialize any actual
    EnvState / TaskParams / world params.
  * Therefore its hashes are PRNG-key / recipe-descriptor identity hashes, NOT
    hashes of real evaluation worlds. They MUST NOT be used for GLOBAL_WORLD_SET_HASH.
  * Two-run agreement here ONLY proves the key/descriptor code is deterministic;
    it does NOT prove anything about real Craftax worlds.
  * evaluation_seed was written ONLY into the descriptor text and did NOT enter
    the PRNGKey / world-generation RNG path. Changing the numeric evaluation_seed
    while keeping the label did NOT change the old "hash" -- proof it was not bound
    to real world generation.

Correct classification of anything this script produces:
    WORLD_KEY_MANIFEST_PROTOTYPE  (NOT  WORLD_SET_MATERIALIZER_READY)

Replacement (actual Craftax world materializer):
    tools/global_evaluation/materialize_craftax_world_set_twice.py

This entry point now FAILS CLOSED immediately (exit 2) so it can never be mistaken
for a working materializer. The archived prototype functions are left intact below
the guard for historical inspection only.
"""
import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys

_DEPRECATED = "DEPRECATED_INVALID_FOR_WORLD_SET_HASH"


def _fail_closed_deprecated():
    sys.stderr.write(
        "\n"
        "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
        "!!  %s\n"
        "!!  This is the OLD world-KEY-manifest prototype. It does NOT\n"
        "!!  generate Craftax worlds and MUST NOT produce a world_set_hash.\n"
        "!!  All of its outputs are NON_SCIENTIFIC_PROTOTYPE.\n"
        "!!  Use tools/global_evaluation/materialize_craftax_world_set_twice.py.\n"
        "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
        % _DEPRECATED
    )
    return 2
import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys

# --------------------------------------------------------------------------- #
# Frozen recipe (MUST match audit_outputs/global_world_set_v1/world_manifest.json)
# --------------------------------------------------------------------------- #
FROZEN_RECIPE = {
    "fold_in_rule": "jax.random.fold_in(PRNGKey(wrapper_seed), world_index)",
    "wrapper_seed": 0,
    "condition_on_task": True,
    "optimistic_reset_ratio": 16,
    "mode": "score",
    "bonus_type": "none",
    "max_timesteps": 4096,
    "num_worlds": 256,
    "world_index_order": "0..255 ascending",
}
ALLOWED_SEEDS = {"seed42": 42, "seed100000": 100000}
EXPECTED_CRAFTAX = "1.4.5"
EXIT_OK = 0
EXIT_FAIL_CLOSED = 2
EXIT_USAGE = 3


def eprint(*a):
    print(*a, file=sys.stderr)


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def probe_version(module_name):
    """Return the module __version__, or None if the module cannot be imported."""
    try:
        mod = importlib.import_module(module_name)
    except Exception:
        return None
    return getattr(mod, "__version__", None)


class FailClosed(Exception):
    """Raised when a hard requirement (dep / version / source identity / mismatch) is unmet."""


def require(cond, msg):
    if not cond:
        raise FailClosed(msg)


# --------------------------------------------------------------------------- #
# Dependency / identity gate (requirements 13, 14, 15, 16, 17)
# --------------------------------------------------------------------------- #
def collect_identity(args, this_script_path):
    """Probe runtime identity. For a FORMAL run every field must resolve (fail closed)."""
    ident = {}
    ident["jax_version"] = probe_version("jax")            # requirement 13
    ident["jaxlib_version"] = probe_version("jaxlib")
    ident["craftax_version"] = probe_version("craftax")    # requirement 14
    ident["generation_script_sha256"] = sha256_file(this_script_path)  # requirement 16
    ident["env_source_path"] = args.env_source
    ident["env_source_sha256"] = (
        sha256_file(args.env_source)
        if args.env_source and os.path.isfile(args.env_source) else None
    )                                                     # requirement 15
    ident["expected_craftax_version"] = EXPECTED_CRAFTAX
    return ident


def assert_formal_identity(ident):
    """Requirement 17: fail closed on ANY missing version / source identity."""
    require(ident["jax_version"] is not None,
            "FAIL CLOSED: jax not importable / no __version__")
    require(ident["jaxlib_version"] is not None,
            "FAIL CLOSED: jaxlib not importable / no __version__")
    require(ident["craftax_version"] is not None,
            "FAIL CLOSED: craftax not importable / no __version__")
    require(ident["craftax_version"] == EXPECTED_CRAFTAX,
            "FAIL CLOSED: craftax version %r != expected %r"
            % (ident["craftax_version"], EXPECTED_CRAFTAX))
    require(ident["env_source_sha256"] is not None,
            "FAIL CLOSED: environment wrapper source (--env-source) missing; cannot record source SHA")
    require(ident["generation_script_sha256"] is not None,
            "FAIL CLOSED: cannot hash generation script")


# --------------------------------------------------------------------------- #
# Stable per-world serialization + hashing (requirements 5, 6, 7, 8, 9)
# --------------------------------------------------------------------------- #
def materialize_world_set(seed_id, evaluation_seed):
    """Generate the 256-world set deterministically and return per-world hashes + total.

    Requirement 5/6: frozen fold_in rule, fixed ascending world order.
    Requirement 7: stable serialization (canonical JSON descriptor + folded-key bytes).
    Requirement 8: SHA256 per world.
    Requirement 9: ordered world_set_hash over the ascending per-world hashes.

    The per-world identity is fully determined by (frozen recipe, world_index,
    evaluation_seed) through jax.random.fold_in; threefry folding is deterministic and
    version-stable, so the folded-key bytes are a genuine JAX-materialized world identity.
    """
    import jax  # noqa: F401  (fail closed earlier if absent)
    import numpy as np

    per_world = {}
    ordered_hashes = []
    base_key = jax.random.PRNGKey(FROZEN_RECIPE["wrapper_seed"])
    for world_index in range(FROZEN_RECIPE["num_worlds"]):     # 0..255 ascending
        folded = jax.random.fold_in(base_key, world_index)     # frozen fold_in rule
        folded_bytes = np.asarray(folded).tobytes()
        descriptor = {
            "schema": "mechanism_UED.world/v1",
            "world_index": world_index,
            "evaluation_seed": int(evaluation_seed),
            "seed_id": seed_id,
            "recipe": FROZEN_RECIPE,
            "folded_key_sha256": sha256_bytes(folded_bytes),
        }
        blob = (json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode("utf-8")
                + b"\x00" + folded_bytes)
        world_sha = sha256_bytes(blob)                         # requirement 8
        per_world[str(world_index)] = world_sha
        ordered_hashes.append(world_sha)                       # ascending order preserved
    # requirement 9: ordered total hash
    world_set_hash = sha256_bytes(("".join(ordered_hashes)).encode("ascii"))
    return per_world, world_set_hash


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #
def do_dry_run(args, ident, this_script_path):
    """Validate dependencies, parameters, and output paths ONLY. No formal hash generated."""
    report = {
        "mode": "dry-run",
        "parameters": {
            "seed_id": args.seed_id,
            "evaluation_seed": ALLOWED_SEEDS.get(args.seed_id),
            "num_worlds": FROZEN_RECIPE["num_worlds"],
            "out": args.out,
        },
        "parameter_checks": {
            "seed_allowed": args.seed_id in ALLOWED_SEEDS,
            "num_worlds_is_256": FROZEN_RECIPE["num_worlds"] == 256,
        },
        "dependency_probe": {
            "jax": ident["jax_version"] or "ABSENT",
            "jaxlib": ident["jaxlib_version"] or "ABSENT",
            "craftax": ident["craftax_version"] or "ABSENT",
            "craftax_expected": EXPECTED_CRAFTAX,
        },
        "identity_probe": {
            "env_source_path": ident["env_source_path"],
            "env_source_sha256": ident["env_source_sha256"] or "MISSING",
            "generation_script_sha256": ident["generation_script_sha256"],
        },
        "output_path_writable": False,
        "formal_hash_generated": False,
    }
    # output path check (create dir, write a probe file, remove it)
    try:
        os.makedirs(args.out, exist_ok=True)
        probe_file = os.path.join(args.out, ".dryrun_probe")
        with open(probe_file, "w", encoding="utf-8") as f:
            f.write("probe")
        os.remove(probe_file)
        report["output_path_writable"] = True
    except Exception as e:  # noqa: BLE001
        report["output_path_error"] = str(e)

    formal_would_fail_closed = []
    if ident["jax_version"] is None:
        formal_would_fail_closed.append("jax ABSENT")
    if ident["jaxlib_version"] is None:
        formal_would_fail_closed.append("jaxlib ABSENT")
    if ident["craftax_version"] is None:
        formal_would_fail_closed.append("craftax ABSENT")
    elif ident["craftax_version"] != EXPECTED_CRAFTAX:
        formal_would_fail_closed.append("craftax version mismatch")
    if ident["env_source_sha256"] is None:
        formal_would_fail_closed.append("env source SHA missing")
    if args.seed_id not in ALLOWED_SEEDS:
        formal_would_fail_closed.append("seed not allowed")
    report["formal_run_would_fail_closed"] = formal_would_fail_closed
    report["dry_run_verdict"] = ("PASS_VALIDATIONS" if not formal_would_fail_closed
                                 and report["output_path_writable"]
                                 and report["parameter_checks"]["seed_allowed"]
                                 else "VALIDATIONS_INCOMPLETE")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    # dry-run exits 0: it reports findings; fail-closed (exit 2) is reserved for the FORMAL run.
    return EXIT_OK


def do_single_run(args, ident):
    """One materialization pass (requirement: run in an independent process)."""
    assert_formal_identity(ident)                       # requirement 17
    evaluation_seed = ALLOWED_SEEDS[args.seed_id]
    per_world, world_set_hash = materialize_world_set(args.seed_id, evaluation_seed)
    os.makedirs(args.out, exist_ok=True)
    result = {
        "schema": "mechanism_UED.world_hashes/v1",
        "seed_id": args.seed_id,
        "evaluation_seed": evaluation_seed,
        "frozen_recipe": FROZEN_RECIPE,
        "world_count": len(per_world),
        "world_index_order": FROZEN_RECIPE["world_index_order"],
        "per_world_hashes": per_world,                   # requirement 8
        "world_set_hash": world_set_hash,                # requirement 9
        "identity": ident,                               # requirements 13-16
    }
    out_path = os.path.join(args.out, "world_hashes.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps({"mode": "single-run", "out": out_path,
                      "world_count": len(per_world), "world_set_hash": world_set_hash},
                     ensure_ascii=False))
    return EXIT_OK


def do_orchestrate(args, ident, this_script_path):
    """Requirement 10/11/12: run twice in independent processes and compare."""
    assert_formal_identity(ident)                       # fail closed before spawning
    run_dirs = [os.path.join(args.out, "run_A"), os.path.join(args.out, "run_B")]
    results = []
    for rd in run_dirs:
        cmd = [sys.executable, this_script_path, "--single-run",
               "--seed", args.seed_id, "--out", rd, "--env-source", args.env_source or ""]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        require(proc.returncode == EXIT_OK,
                "FAIL CLOSED: independent run in %s exited %d\nstderr: %s"
                % (rd, proc.returncode, proc.stderr))
        with open(os.path.join(rd, "world_hashes.json"), encoding="utf-8") as f:
            results.append(json.load(f))
    a, b = results
    # requirement 11: compare per-world hashes
    require(a["per_world_hashes"] == b["per_world_hashes"],
            "FAIL CLOSED: per-world hashes differ between the two independent runs")
    # requirement 12: compare total hash
    require(a["world_set_hash"] == b["world_set_hash"],
            "FAIL CLOSED: world_set_hash differs between the two independent runs")
    os.makedirs(args.out, exist_ok=True)
    agreed = {
        "schema": "mechanism_UED.world_set_agreement/v1",
        "seed_id": args.seed_id,
        "two_independent_runs": True,
        "per_world_hash_agreement": True,
        "world_set_hash_agreement": True,
        "world_count": a["world_count"],
        "world_set_hash": a["world_set_hash"],
        "run_A": os.path.join(run_dirs[0], "world_hashes.json"),
        "run_B": os.path.join(run_dirs[1], "world_hashes.json"),
        "identity": a["identity"],
    }
    with open(os.path.join(args.out, "world_set_agreement.json"), "w", encoding="utf-8") as f:
        json.dump(agreed, f, indent=2, ensure_ascii=False)
    print(json.dumps({"mode": "orchestrate", "agreement": True,
                      "world_set_hash": a["world_set_hash"]}, ensure_ascii=False))
    return EXIT_OK


def main(argv=None):
    this_script_path = os.path.abspath(__file__)
    ap = argparse.ArgumentParser(description="CC4 frozen world-set materialization (fail-closed).")
    ap.add_argument("--seed", dest="seed_id", default="seed42",
                    choices=sorted(ALLOWED_SEEDS.keys()),
                    help="frozen world-set id (seed42 or seed100000)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--env-source", dest="env_source", default=None,
                    help="path to DistributedMultiTaskOptimisticLogWrapper source (for source SHA)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", dest="mode", action="store_const", const="dry-run")
    mode.add_argument("--single-run", dest="mode", action="store_const", const="single-run")
    mode.add_argument("--orchestrate", dest="mode", action="store_const", const="orchestrate")
    ap.set_defaults(mode="orchestrate")
    args = ap.parse_args(argv)

    ident = collect_identity(args, this_script_path)
    try:
        if args.mode == "dry-run":
            return do_dry_run(args, ident, this_script_path)
        if args.mode == "single-run":
            return do_single_run(args, ident)
        return do_orchestrate(args, ident, this_script_path)
    except FailClosed as e:
        eprint(str(e))
        return EXIT_FAIL_CLOSED


if __name__ == "__main__":
    # FAIL CLOSED: this prototype must never run as a materializer.
    sys.exit(_fail_closed_deprecated())
