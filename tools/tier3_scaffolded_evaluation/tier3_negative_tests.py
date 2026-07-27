#!/usr/bin/env python3
"""CC4 Tier3 — negative tests (§十七 NEG01–NEG26).

Each negative test constructs an INVALID input and asserts the corresponding guard
REJECTS it (fail-closed). A test PASSES when the rejection is correctly detected;
the suite requirement is FAIL=0 (no negative test silently accepts a violation).
BLOCKED is allowed only with a documented environment-capability absence — never a
fake PASS.

Commit-2 coverage (this file, runs now):
  NEG01–NEG18  boundary / builder / state-bank / predicate level
  NEG24        scaffold hash must never be the GLOBAL_WORLD_SET_HASH
  NEG26        state/sample selection must be blind to Student performance

Commit-3 coverage (added with the evaluator / checkpoint-adapter / certificate
modules; see the PENDING_COMMIT_3 registry):
  NEG19 episode missing valid_start; NEG20 ambiguous termination silently labelled;
  NEG21 checkpoint params SHA mismatch; NEG22 checkpoint observation shape mismatch;
  NEG23 evaluation tries to update params; NEG25 scaffold result claims full-task success.

These are NOT stubbed as passing: until the owning module exists they are reported as
PENDING and excluded from the FAIL count (never counted as PASS).
"""
from __future__ import annotations

import json
import os
import sys

# Runnable-as-script AND importable-as-package.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tier3_source_audit as audit                 # noqa: E402
import tier3_event_predicates as pred             # noqa: E402
import tier3_state_serializer as ser              # noqa: E402
import tier3_scaffold_builder as builder          # noqa: E402
import tier3_state_bank_materializer as mat       # noqa: E402

# Every guard may raise its own module's FailClosed (or the reused V3 one).
FAILCLOSED = (audit.FailClosed, pred.FailClosed, ser.FailClosed, builder.FailClosed,
              mat.FailClosed, ser.v3mat.FailClosed)

KOBOLD = mat.SYNTHETIC_KOBOLD_TYPE_ID   # synthetic test type_id (real one is BLOCKED)


def rejects(fn) -> bool:
    """True iff fn() raises a FailClosed (the guard correctly rejected the input)."""
    try:
        fn()
        return False
    except FAILCLOSED:
        return True


# ---------------------------------------------------------------------------
# Synthetic states (identical to the materializer's; clearly test-only)
# ---------------------------------------------------------------------------
def front_state(**over):
    s = mat.synthesize_front_start(0)
    s.update(over)
    return s


def back_state(**over):
    s = mat.synthesize_back_start(0, KOBOLD)
    s.update(over)
    return s


# ---------------------------------------------------------------------------
# NEG01–NEG18, NEG24, NEG26
# ---------------------------------------------------------------------------
def neg01():
    """Boundary source SHA mismatch vs the real on-disk source -> fail."""
    role = "world_builder"
    real_sha = audit.sha256_file(audit.resolve_source_path(role))

    def check(claimed):
        if claimed != real_sha:
            raise audit.FailClosed(
                "FAIL CLOSED (NEG01): boundary source_file_sha256 %s != on-disk %s"
                % (claimed[:16], real_sha[:16]))
    # correct SHA passes; tampered SHA rejected
    check(real_sha)
    return rejects(lambda: check("0" * 64))


def neg02():
    """Builder source realpath/SHA mismatch -> fail (V3 executed-source identity)."""
    other = str(audit.resolve_source_path("game_mechanics"))  # a DIFFERENT real file
    return rejects(lambda: builder.bind_builder_source_identity(imported_file=other))


def neg03():
    """Canonical task source SHA mismatch -> fail."""
    return rejects(lambda: builder.verify_canonical_task_source(expected_sha256="0" * 64))


def neg04():
    """Missing required EnvState field -> fail."""
    bad = front_state()
    del bad["player_level"]
    return rejects(lambda: ser.assert_required_envstate_fields(bad))


def neg05():
    """State payload hash tampered -> fail (hash compare; bytes are opaque)."""
    sha, payload = ser.normalized_payload_hash(front_state())
    return rejects(lambda: ser.verify_payload_hash("0" * 64, payload))


def neg06():
    """State bank order changed -> hash changes (order-sensitive)."""
    m = mat.process_a_materialize(mat.FRONT, 5)
    hashes = [e["state_payload_hash"] for e in m["entries"]]
    src = mat.source_shas_for_bank()
    h_fwd = mat.state_bank_hash(hashes, mat.FRONT, src)
    h_rev = mat.state_bank_hash(list(reversed(hashes)), mat.FRONT, src)
    return h_fwd != h_rev


def neg07():
    """Persistent vs Reset128 using different (per-arm) state banks -> fail."""
    m = mat.process_a_materialize(mat.FRONT, 3)
    m["per_arm"] = {"persistent": "bankA", "reset128": "bankB"}
    return rejects(lambda: mat.assert_no_arm_partition(m))


def neg08():
    """Arm-specific scaffold metadata -> fail."""
    spec = builder.build_front_spec()
    spec["arm_id"] = "persistent"
    return rejects(lambda: builder.assert_no_arm_specific_metadata(spec))


def neg09():
    """Extra observation field / changed observation schema -> fail."""
    spec = builder.build_front_spec()
    spec["observation_schema"] = "canonical_craftax_symbolic + exit_direction_arrow"
    return rejects(lambda: builder.validate_scaffold_legality(spec))


def neg10():
    """Action space changed -> fail."""
    spec = builder.build_back_spec()
    spec["action_space"] = "reduced_discrete_8"
    return rejects(lambda: builder.validate_scaffold_legality(spec))


def neg11():
    """Hidden boss direction injected into observation -> fail."""
    spec = builder.build_back_spec()
    spec["legality"]["no_hidden_boss_direction"] = False
    return rejects(lambda: builder.validate_scaffold_legality(spec))


def neg12():
    """Invalid inventory value (negative / non-int) -> fail."""
    return (rejects(lambda: builder.validate_inventory({"wood": -1}))
            and rejects(lambda: builder.validate_inventory({"wood": 1.5}))
            and rejects(lambda: builder.validate_inventory({"": 3})))


def neg13():
    """Invalid player position (negative / outside grid) -> fail."""
    return (rejects(lambda: builder.validate_player_position((-1, 3), 48, 48))
            and rejects(lambda: builder.validate_player_position((5, 99), 48, 48)))


def neg14():
    """Front start already beyond the corridor exit -> invalid scaffold."""
    beyond = front_state(player_level=audit.CORRIDOR_EXIT_FLOOR)
    return pred.valid_front_scaffold_start(beyond) is False


def neg15():
    """Back start already has DEFEAT_KOBOLD -> invalid scaffold."""
    solved = back_state(achieved={pred.DEFEAT_KOBOLD})
    return pred.valid_back_scaffold_start(solved, KOBOLD) is False


def neg16():
    """Back state with no live Kobold (kill task requires one) -> invalid scaffold."""
    no_kobold = back_state(mobs=[])
    return pred.valid_back_scaffold_start(no_kobold, KOBOLD) is False


def neg17():
    """Progress must always lie in [0,1] (sweep every reachable cell)."""
    walk = [[False] * 5 for _ in range(5)]
    for c in range(5):
        walk[2][c] = True
    walk[1][2] = True                      # a dead-end spur
    start, exit_pos = (2, 0), (2, 4)
    ok = True
    for r in range(5):
        for c in range(5):
            if not walk[r][c]:
                continue
            p = pred.normalized_corridor_progress(
                front_state(player_position=(r, c)), walk, start, exit_pos)
            ok = ok and (0.0 <= p <= 1.0)
    return ok


def neg18():
    """Unreachable exit without an explicit blocked label -> fail."""
    walk = [[False] * 5 for _ in range(5)]
    walk[2][0] = True
    walk[2][1] = True                       # exit (2,4) isolated
    return rejects(lambda: pred.normalized_corridor_progress(
        front_state(player_position=(2, 0)), walk, (2, 0), (2, 4)))


def neg24():
    """Scaffold bank hash labelled as GLOBAL_WORLD_SET_HASH -> fail."""
    m = mat.process_a_materialize(mat.FRONT, 3)
    m["hash_label"] = mat.GLOBAL_HASH_LABEL
    return rejects(lambda: mat.assert_not_global_world_set_hash(m))


def neg26():
    """State/sample selection based on Student performance -> fail."""
    m = mat.process_a_materialize(mat.FRONT, 3)
    m["seeds"] = [s + 1 for s in m["seeds"]]   # no longer reproducible from schedule
    return rejects(lambda: mat.assert_selection_is_result_blind(m))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
NEG_TESTS = [
    ("NEG01", "boundary source SHA mismatch", neg01),
    ("NEG02", "builder source realpath/SHA mismatch", neg02),
    ("NEG03", "canonical task source SHA mismatch", neg03),
    ("NEG04", "missing required EnvState field", neg04),
    ("NEG05", "state payload hash tampered", neg05),
    ("NEG06", "state bank order changed -> hash changes", neg06),
    ("NEG07", "per-arm (Persistent vs Reset128) state bank", neg07),
    ("NEG08", "arm-specific scaffold metadata", neg08),
    ("NEG09", "extra observation field / schema change", neg09),
    ("NEG10", "action space changed", neg10),
    ("NEG11", "hidden boss direction injected", neg11),
    ("NEG12", "invalid inventory value", neg12),
    ("NEG13", "invalid player position", neg13),
    ("NEG14", "front start already beyond exit", neg14),
    ("NEG15", "back start already DEFEAT_KOBOLD", neg15),
    ("NEG16", "back state has no live Kobold", neg16),
    ("NEG17", "progress always within [0,1]", neg17),
    ("NEG18", "unreachable exit without blocked label", neg18),
    ("NEG24", "scaffold hash used as GLOBAL_WORLD_SET_HASH", neg24),
    ("NEG26", "result-based state/sample selection", neg26),
]

# Owning modules land in Commit 3; NOT counted as PASS until then.
PENDING_COMMIT_3 = [
    ("NEG19", "episode missing valid_start", "tier3_evaluator / episode record"),
    ("NEG20", "ambiguous termination silently labelled", "tier3_failure_taxonomy"),
    ("NEG21", "checkpoint params SHA mismatch", "tier3_checkpoint_adapter"),
    ("NEG22", "checkpoint observation shape mismatch", "tier3_checkpoint_adapter"),
    ("NEG23", "evaluation tries to update params", "tier3_checkpoint_adapter"),
    ("NEG25", "scaffold result claims full-task success", "tier3_evaluation_certificate"),
]


def run_all():
    results = []
    n_fail = 0
    for neg_id, desc, fn in NEG_TESTS:
        try:
            ok = bool(fn())
        except Exception as exc:           # unexpected error == a real failure
            ok = False
            desc = "%s (unexpected exception: %r)" % (desc, exc)
        if not ok:
            n_fail += 1
        results.append({"id": neg_id, "description": desc, "rejected_correctly": ok,
                        "status": "PASS" if ok else "FAIL"})
    return results, n_fail


def self_test() -> int:
    results, n_fail = run_all()
    implemented = len(NEG_TESTS)
    pending = len(PENDING_COMMIT_3)
    for r in results:
        print("  [%s] %s - %s" % (r["status"], r["id"], r["description"]))
    for neg_id, desc, owner in PENDING_COMMIT_3:
        print("  [PENDING] %s - %s (lands with %s in Commit 3)" % (neg_id, desc, owner))
    if n_fail != 0:
        print("TIER3_NEGATIVE_TESTS_FAIL (FAIL=%d/%d implemented)" % (n_fail, implemented))
        return 1
    print("TIER3_NEGATIVE_TESTS_PASS (FAIL=0; implemented=%d/26, pending_commit3=%d)"
          % (implemented, pending))
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--json" in argv:
        results, n_fail = run_all()
        print(json.dumps({"fail": n_fail, "results": results,
                          "pending_commit3": [p[0] for p in PENDING_COMMIT_3]},
                         indent=2, ensure_ascii=False, sort_keys=True))
        return 0 if n_fail == 0 else 1
    return self_test()


if __name__ == "__main__":
    raise SystemExit(main())
