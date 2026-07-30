#!/usr/bin/env python3
"""CC4 Tier3 — evaluation certificate (honest labels; no over-claiming).

Turns an evaluation_result into a signed-off certificate that records EXACTLY what was
proven and what remains blocked. It is the last line of defence against scientific
over-claiming: it refuses to let a scaffold result masquerade as a full-task result.

Guards (negative tests):
  NEG24 scaffold hash presented as GLOBAL_WORLD_SET_HASH       -> fail closed
  NEG25 scaffold result claims full-task success / breakthrough -> fail closed
  NEG29 certificate provenance missing/invalid (child pid/argv/times/literal exit
          code/exit_source/driver SHA)                            -> fail closed
  NEG37 self-declared exit code (legacy run_exit_code / exit_source != wait_pid /
          engine stamping its own exit)                           -> fail closed

TWO-STAGE BINDING (task §二): the evaluator ENGINE writes certificates WITHOUT any
exit provenance — it cannot know its own literal exit code. Only the parent runner
(tier3_evaluation_runner.py) finalizes them after wait()-ing on the engine child:
  * engine stage  (finalized=False): assert_engine_binding_complete — every engine
    binding field, and NO exit-provenance / legacy self-declared field may be present;
  * final stage   (finalized=True):  assert_eval_binding_complete — additionally the
    RUNNER-SUPPLIED exit provenance (child_process_pid / child_process_argv /
    actual_started_at_utc / actual_finished_at_utc / literal_exit_code == 0 /
    exit_source == "wait_pid" / inferred_from_log == False /
    evaluation_runner_source_sha256). Self-declared exit codes are never accepted.

Student status split (task §三): the ambiguous has_student_data flag is gone. A
certificate carries the explicit student_state quad
(student_checkpoint_loaded / student_policy_rollout_executed /
 performance_evaluation_executed / scientific_claim_authorized) and the status labels
REAL_STUDENT_INTERFACE_SMOKE / REAL_STUDENT_PERFORMANCE_EVALUATION /
FORMAL_SCIENTIFIC_CLAIM=NOT_AUTHORIZED_SINGLE_TRAINING_SEED. A real Student rollout
is never labelled NOT_RUN again (the old REAL_STUDENT_EVALUATION key is removed).

Honest status discipline: this round can only ever claim IMPLEMENTED_STATIC /
TESTED_SYNTHETIC (plus TESTED_REAL_ENV_RESET on a JAX host). It NEVER emits
FRONT_SCAFFOLD_EVALUATION=PASS, TIER3_FRONT_HALF_BREAKTHROUGH,
FORMAL_SCIENTIFIC_PASS, PERSISTENT_PROVEN_BETTER or TIER3_SOLVED — single training
seed, provisional selection only, no scientific superiority claim.
"""
from __future__ import annotations

import os
import sys

# Runnable-as-script AND importable-as-package.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tier3_source_audit as audit            # noqa: E402
import tier3_state_serializer as ser          # noqa: E402
import tier3_metrics as metrics               # noqa: E402
import tier3_failure_taxonomy as taxonomy     # noqa: E402
import tier3_state_bank_materializer as mat   # noqa: E402

SCHEMA = "mechanism_UED.tier3_evaluation_certificate/v1"
CERT_VERSION = "tier3_evaluation_certificate/v1"

FULL = metrics.FULL
FRONT = metrics.FRONT
BACK = metrics.BACK

GLOBAL_HASH_LABEL = "GLOBAL_WORLD_SET_HASH"
SCAFFOLD_HASH_LABELS = {"FRONT_SCAFFOLD_STATE_BANK_HASH", "BACK_SCAFFOLD_STATE_BANK_HASH"}

# Forbidden over-claims for a scaffold certificate (NEG25).
FORBIDDEN_OVERCLAIMS = {
    "FRONT_SCAFFOLD_EVALUATION=PASS",
    "BACK_SCAFFOLD_EVALUATION=PASS",
    "TIER3_FRONT_HALF_BREAKTHROUGH",
    "TIER3_BACK_HALF_BREAKTHROUGH",
    "DEFEAT_KOBOLD_SOLVED",
    "TIER3_SOLVED",
    "SOTA",
    "PERSISTENT_BEATS_RESET128",
    "REPLAY_SCIENTIFIC_GAIN",
    # task §六: provisional single-seed selection may NEVER authorize these.
    "FORMAL_SCIENTIFIC_PASS",
    "PERSISTENT_PROVEN_BETTER",
    "RESET128_PROVEN_BETTER",
    "STRONG_STUDENT_PROVEN",
}

# REAL certificate bindings (task §五): a certificate that records an evaluation
# must bind ACTUAL VALUES for every field below — a hash LABEL or an omitted value
# fails closed (NEG27). These are the interface + provenance facts that make the
# certificate auditable against the frozen Tier3 contract.
#
# ENGINE stage fields — written by the evaluator engine itself.
ENGINE_BINDING_REQUIRED_FIELDS = (
    "state_bank_hash", "state_payload_hashes", "checkpoint_file_sha256",
    "cc2_params_sha256", "checkpoint_step", "carry_mode", "run_class",
    "episode_records_sha256", "cc2_policy_source_sha256", "evaluator_source_sha256",
    "predicate_code_sha256", "observation_shape", "action_dim", "params_unchanged",
    "performance_claim_authorized",
    "driver_source_sha256",
    # Frozen final-checkpoint contract binding (task §一/§五): the contract the loaded
    # checkpoint was verified against, and which arm of it.
    "checkpoint_contract_sha256", "checkpoint_contract_arm",
    # Frozen evaluation-contract identity (task §五): the actual run values.
    "action_mode", "max_timesteps", "evaluation_seed_schedule", "state_entry_ids",
    # Actual runtime environment versions (task §五).
    "python_version", "jax_version", "jaxlib_version", "numpy_version",
    "flax_version", "craftax_version",
    "evaluator_git_commit",
    # Scientific boundary (task §五/§六): single training seed, provisional only.
    "scientific_claim_authorized", "single_training_seed", "provisional_selection_only",
)
# RUNNER stage fields — supplied ONLY by tier3_evaluation_runner.py after it has
# wait()-ed on the engine child and read its literal exit code (task §二).
EXIT_PROVENANCE_REQUIRED_FIELDS = (
    "child_process_pid", "child_process_argv",
    "actual_started_at_utc", "actual_finished_at_utc",
    "literal_exit_code", "exit_source", "inferred_from_log",
    "evaluation_runner_source_sha256",
)
# The FULL finalized binding (engine + runner).
EVAL_BINDING_REQUIRED_FIELDS = ENGINE_BINDING_REQUIRED_FIELDS + EXIT_PROVENANCE_REQUIRED_FIELDS
# Legacy SELF-DECLARED provenance fields (pre-runner design). Their presence in ANY
# binding fails closed (NEG37): an engine writing its own exit code / pid / argv is
# exactly the provenance hole the parent/child runner closes.
FORBIDDEN_LEGACY_PROVENANCE_FIELDS = ("run_exit_code", "process_pid", "process_argv",
                                      "run_start_utc", "run_end_utc")
EVAL_BINDING_SHA_FIELDS = (
    "state_bank_hash", "checkpoint_file_sha256", "cc2_params_sha256",
    "episode_records_sha256", "cc2_policy_source_sha256", "evaluator_source_sha256",
    "predicate_code_sha256", "driver_source_sha256",
    "checkpoint_contract_sha256", "evaluation_runner_source_sha256",
)
ENGINE_BINDING_SHA_FIELDS = tuple(f for f in EVAL_BINDING_SHA_FIELDS
                                  if f != "evaluation_runner_source_sha256")
FROZEN_OBSERVATION_SHAPE = [8335]      # canonical S4 symbolic obs (unchanged)
FROZEN_ACTION_DIM = 43                 # canonical craftax action set (unchanged)
FROZEN_ACTION_MODE = "greedy_argmax"   # the one frozen action mode (task §五/§六)
RUN_CLASSES = ("INTERFACE_SMOKE", "FORMAL_EVALUATION",
               "PROVISIONAL_STRONG_STUDENT_SELECTION")
CONTRACT_ARMS = ("persistent", "reset128")
# Frozen held-out performance-evaluation start schedule (task §七).
PERF_FULL_SEED_BASE = 200_000
PERF_FULL_N = 64
VERSION_FIELDS = ("python_version", "jax_version", "jaxlib_version", "numpy_version",
                  "flax_version", "craftax_version")
CERT_MODES = ("synthetic", "interface_smoke", "performance_evaluation")
STUDENT_STATE_KEYS = ("student_checkpoint_loaded", "student_policy_rollout_executed",
                      "performance_evaluation_executed", "scientific_claim_authorized")


class FailClosed(Exception):
    """Hard stop on over-claiming / hash mislabelling / incomplete binding."""


def require(cond, msg):
    if not cond:
        raise FailClosed(msg)


def _is_sha256_hex(v) -> bool:
    return (isinstance(v, str) and len(v) == 64
            and all(c in "0123456789abcdef" for c in v))


def _require_iso_utc(value, field):
    """NEG29: a provenance timestamp must be a non-empty, parseable ISO-8601 string."""
    require(isinstance(value, str) and value,
            "FAIL CLOSED (NEG29): eval_binding.%s %r is not a non-empty ISO-8601 string "
            "(the actual run start/end time must be bound)" % (field, value))
    import datetime as _dt
    try:
        _dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise FailClosed(
            "FAIL CLOSED (NEG29): eval_binding.%s %r is not a parseable ISO-8601 "
            "timestamp" % (field, value))


# ---------------------------------------------------------------------------
# NEG27 / NEG37: REAL certificate value binding (never just hash labels; never a
# self-declared exit code)
# ---------------------------------------------------------------------------
def _assert_common_binding(binding: dict, sha_fields):
    """Value/type checks shared by the engine stage and the finalized stage."""
    for f in sha_fields:
        require(_is_sha256_hex(binding[f]),
                "FAIL CLOSED (NEG27): eval_binding.%s = %r is not a 64-hex sha256 VALUE "
                "(a label or truncated hash is forbidden)" % (f, binding[f]))
    require(isinstance(binding["state_payload_hashes"], list)
            and all(_is_sha256_hex(h) for h in binding["state_payload_hashes"]),
            "FAIL CLOSED (NEG27): state_payload_hashes must be an ordered list of "
            "64-hex sha256 values")
    require(list(binding["observation_shape"]) == FROZEN_OBSERVATION_SHAPE,
            "FAIL CLOSED (NEG27): eval_binding observation_shape %s != frozen %s "
            "(observation interface changed)"
            % (binding["observation_shape"], FROZEN_OBSERVATION_SHAPE))
    require(int(binding["action_dim"]) == FROZEN_ACTION_DIM,
            "FAIL CLOSED (NEG27): eval_binding action_dim %r != frozen %d "
            "(action interface changed)" % (binding["action_dim"], FROZEN_ACTION_DIM))
    require(binding["params_unchanged"] is True,
            "FAIL CLOSED (NEG27/NEG23): eval_binding params_unchanged must be exactly "
            "True (params SHA identical before/after evaluation)")
    require(binding["carry_mode"] in CONTRACT_ARMS,
            "FAIL CLOSED (NEG27): eval_binding carry_mode %r not in %s"
            % (binding["carry_mode"], CONTRACT_ARMS))
    require(binding["run_class"] in RUN_CLASSES,
            "FAIL CLOSED (NEG27): eval_binding run_class %r not in %s"
            % (binding["run_class"], RUN_CLASSES))
    require(binding["performance_claim_authorized"] is False
            or binding["run_class"] == "FORMAL_EVALUATION",
            "FAIL CLOSED (NEG27): run_class=INTERFACE_SMOKE / "
            "PROVISIONAL_STRONG_STUDENT_SELECTION can never authorize a performance claim")
    # ---- frozen final-checkpoint contract identity (task §一/§五) ----
    require(binding["checkpoint_contract_arm"] in CONTRACT_ARMS,
            "FAIL CLOSED (NEG27): checkpoint_contract_arm %r not in %s"
            % (binding["checkpoint_contract_arm"], CONTRACT_ARMS))
    require(binding["carry_mode"] == binding["checkpoint_contract_arm"],
            "FAIL CLOSED (NEG27): carry_mode %r != checkpoint_contract_arm %r (the "
            "evaluated checkpoint must be the contract arm)"
            % (binding["carry_mode"], binding["checkpoint_contract_arm"]))
    # ---- frozen evaluation-contract identity (task §五/§六) ----
    require(binding["action_mode"] == FROZEN_ACTION_MODE,
            "FAIL CLOSED (NEG27): action_mode %r != frozen %r"
            % (binding["action_mode"], FROZEN_ACTION_MODE))
    require(isinstance(binding["max_timesteps"], int)
            and not isinstance(binding["max_timesteps"], bool)
            and binding["max_timesteps"] > 0,
            "FAIL CLOSED (NEG27): max_timesteps %r is not a positive int"
            % binding["max_timesteps"])
    if binding["run_class"] == "PROVISIONAL_STRONG_STUDENT_SELECTION":
        require(binding["max_timesteps"] == 4096,
                "FAIL CLOSED (NEG27): PROVISIONAL_STRONG_STUDENT_SELECTION max_timesteps "
                "%r != frozen 4096" % binding["max_timesteps"])
    _assert_seed_schedule(binding["evaluation_seed_schedule"], binding["run_class"])
    _assert_state_entry_ids(binding["state_entry_ids"], binding["run_class"])
    # ---- actual runtime environment versions (task §五) ----
    for f in VERSION_FIELDS:
        require(isinstance(binding[f], str) and binding[f],
                "FAIL CLOSED (NEG27): eval_binding.%s %r is not a non-empty version "
                "string (the ACTUAL runtime versions must be bound)" % (f, binding[f]))
    require(isinstance(binding["evaluator_git_commit"], str)
            and len(binding["evaluator_git_commit"]) == 40
            and all(c in "0123456789abcdef" for c in binding["evaluator_git_commit"]),
            "FAIL CLOSED (NEG27): evaluator_git_commit %r is not a 40-hex git commit"
            % binding["evaluator_git_commit"])
    # ---- scientific boundary (task §五/§六): never authorized this round ----
    require(binding["scientific_claim_authorized"] is False,
            "FAIL CLOSED (NEG27): scientific_claim_authorized must be exactly False "
            "(single training seed; no scientific superiority claim)")
    require(binding["single_training_seed"] is True,
            "FAIL CLOSED (NEG27): single_training_seed must be exactly True")
    require(binding["provisional_selection_only"] is True,
            "FAIL CLOSED (NEG27): provisional_selection_only must be exactly True")


def _assert_seed_schedule(sched, run_class):
    """evaluation_seed_schedule must be a non-empty dict of {scenario: {kind,
    seeds[int...]}} entries. Structural validity (kind non-empty, seeds a non-empty
    list of ints) is checked for EVERY scenario key PRESENT — a single-scenario smoke
    certificate binds only the scenarios it executed. Under
    PROVISIONAL_STRONG_STUDENT_SELECTION all three scenarios must be present and the
    FROZEN held-out schedule (task §七) must reproduce exactly."""
    require(isinstance(sched, dict) and sched,
            "FAIL CLOSED (NEG27): evaluation_seed_schedule %r is not a non-empty dict"
            % (sched,))
    known = (FULL, FRONT, BACK)
    unknown = sorted(k for k in sched if k not in known)
    require(not unknown,
            "FAIL CLOSED (NEG27): evaluation_seed_schedule unknown scenario key(s) %s"
            % unknown)
    for sc in known:
        if sc not in sched:
            continue
        e = sched[sc]
        require(isinstance(e, dict),
                "FAIL CLOSED (NEG27): evaluation_seed_schedule.%s is not a dict" % sc)
        require(isinstance(e.get("kind"), str) and e["kind"],
                "FAIL CLOSED (NEG27): evaluation_seed_schedule.%s.kind missing" % sc)
        seeds = e.get("seeds")
        require(isinstance(seeds, list) and seeds
                and all(isinstance(s, int) and not isinstance(s, bool) for s in seeds),
                "FAIL CLOSED (NEG27): evaluation_seed_schedule.%s.seeds must be a "
                "non-empty list of ints" % sc)
    if run_class == "PROVISIONAL_STRONG_STUDENT_SELECTION":
        missing = [sc for sc in known if sc not in sched]
        require(not missing,
                "FAIL CLOSED (NEG27): PROVISIONAL evaluation_seed_schedule missing "
                "scenario(s) %s (both arms run all three scenarios on an identical "
                "schedule)" % missing)
        want_full = [PERF_FULL_SEED_BASE + i for i in range(PERF_FULL_N)]
        require(sched[FULL]["seeds"] == want_full,
                "FAIL CLOSED (NEG27): PROVISIONAL full seeds %s... != frozen held-out "
                "%d..%d (64 canonical reset seeds)"
                % (sched[FULL]["seeds"][:3], PERF_FULL_SEED_BASE,
                   PERF_FULL_SEED_BASE + PERF_FULL_N - 1))
        require(sched[FRONT]["seeds"] == mat.fixed_seed_schedule(
                    FRONT, mat.FROZEN_BANK_N, mat.FROZEN_SEED_BASE, mat.FROZEN_SEED_STRIDE),
                "FAIL CLOSED (NEG27): PROVISIONAL front seeds != frozen bank schedule "
                "(all %d FRONT states, each exactly once)" % mat.FROZEN_BANK_N)
        require(sched[BACK]["seeds"] == mat.fixed_seed_schedule(
                    BACK, mat.FROZEN_BANK_N, mat.FROZEN_SEED_BASE, mat.FROZEN_SEED_STRIDE),
                "FAIL CLOSED (NEG27): PROVISIONAL back seeds != frozen bank schedule "
                "(all %d BACK states, each exactly once)" % mat.FROZEN_BANK_N)


def _assert_state_entry_ids(sei, run_class):
    """state_entry_ids must be a non-empty dict of {scenario: [non-empty str...]}.
    Structural validity is checked for every scenario key PRESENT (a single-scenario
    smoke run binds only its scenarios); PROVISIONAL requires all three scenarios
    with the frozen 64/8/8 entry counts (task §七)."""
    require(isinstance(sei, dict) and sei,
            "FAIL CLOSED (NEG27): state_entry_ids %r is not a non-empty dict" % (sei,))
    known = (FULL, FRONT, BACK)
    unknown = sorted(k for k in sei if k not in known)
    require(not unknown,
            "FAIL CLOSED (NEG27): state_entry_ids unknown scenario key(s) %s" % unknown)
    for sc in known:
        if sc not in sei:
            continue
        ids = sei[sc]
        require(isinstance(ids, list) and ids
                and all(isinstance(x, str) and x for x in ids),
                "FAIL CLOSED (NEG27): state_entry_ids.%s must be a non-empty list of "
                "non-empty strings" % sc)
    if run_class == "PROVISIONAL_STRONG_STUDENT_SELECTION":
        missing = [sc for sc in known if sc not in sei]
        require(not missing,
                "FAIL CLOSED (NEG27): PROVISIONAL state_entry_ids missing scenario(s) %s"
                % missing)
        require(len(sei[FULL]) == PERF_FULL_N,
                "FAIL CLOSED (NEG27): PROVISIONAL state_entry_ids.full has %d entries "
                "!= %d" % (len(sei[FULL]), PERF_FULL_N))
        require(len(sei[FRONT]) == mat.FROZEN_BANK_N,
                "FAIL CLOSED (NEG27): PROVISIONAL state_entry_ids.front_l2 has %d "
                "entries != %d" % (len(sei[FRONT]), mat.FROZEN_BANK_N))
        require(len(sei[BACK]) == mat.FROZEN_BANK_N,
                "FAIL CLOSED (NEG27): PROVISIONAL state_entry_ids.back_l2 has %d "
                "entries != %d" % (len(sei[BACK]), mat.FROZEN_BANK_N))


def _assert_no_legacy_self_declared_provenance(binding: dict):
    """NEG37: legacy SELF-DECLARED provenance fields must never appear. An engine
    writing its own run_exit_code / pid / argv / times is the exact hole the
    parent/child runner closes."""
    legacy = sorted(f for f in FORBIDDEN_LEGACY_PROVENANCE_FIELDS if f in binding)
    require(not legacy,
            "FAIL CLOSED (NEG37): eval_binding carries legacy self-declared provenance "
            "field(s) %s — exit provenance may ONLY be bound by the parent runner "
            "(literal wait() exit code)" % legacy)


def assert_engine_binding_complete(cert: dict) -> dict:
    """ENGINE stage (finalized=False): every ENGINE_BINDING_REQUIRED_FIELDS field must
    bind an ACTUAL VALUE, and NO exit-provenance / legacy self-declared field may be
    present — the engine cannot know its own literal exit code (task §二)."""
    binding = cert.get("eval_binding")
    require(isinstance(binding, dict),
            "FAIL CLOSED (NEG27): certificate has no eval_binding dict — a real "
            "evaluation certificate must bind actual values, not labels")
    missing = [f for f in ENGINE_BINDING_REQUIRED_FIELDS
               if binding.get(f) in (None, "", [], {})]
    require(not missing,
            "FAIL CLOSED (NEG27): eval_binding missing / empty engine field(s) %s — "
            "actual values required, hash labels are not enough" % missing)
    _assert_no_legacy_self_declared_provenance(binding)
    early = sorted(f for f in EXIT_PROVENANCE_REQUIRED_FIELDS if f in binding)
    require(not early,
            "FAIL CLOSED (NEG37): engine-stage certificate already carries exit "
            "provenance field(s) %s — only the parent runner may bind them after "
            "wait()-ing on the engine child" % early)
    _assert_common_binding(binding, ENGINE_BINDING_SHA_FIELDS)
    return binding


def assert_eval_binding_complete(cert: dict) -> dict:
    """FINALIZED stage (finalized=True): the engine binding PLUS the RUNNER-SUPPLIED
    exit provenance (NEG29), with self-declared exit codes rejected (NEG37)."""
    binding = cert.get("eval_binding")
    require(isinstance(binding, dict),
            "FAIL CLOSED (NEG27): certificate has no eval_binding dict — a real "
            "evaluation certificate must bind actual values, not labels")
    missing = [f for f in EVAL_BINDING_REQUIRED_FIELDS
               if binding.get(f) in (None, "", [], {})]
    require(not missing,
            "FAIL CLOSED (NEG27): eval_binding missing / empty field(s) %s — actual "
            "values required, hash labels are not enough" % missing)
    _assert_no_legacy_self_declared_provenance(binding)
    _assert_common_binding(binding, EVAL_BINDING_SHA_FIELDS)
    # ---- NEG29: RUNNER-SUPPLIED exit provenance (literal wait() exit code) ----
    require(isinstance(binding["child_process_pid"], int)
            and not isinstance(binding["child_process_pid"], bool)
            and binding["child_process_pid"] > 0,
            "FAIL CLOSED (NEG29): eval_binding child_process_pid %r is not a positive "
            "int (the certificate must bind the ACTUAL evaluator child PID captured "
            "by the parent runner)" % binding["child_process_pid"])
    require(isinstance(binding["child_process_argv"], list)
            and binding["child_process_argv"]
            and all(isinstance(a, str) and a for a in binding["child_process_argv"]),
            "FAIL CLOSED (NEG29): eval_binding child_process_argv %r is not a "
            "non-empty list of non-empty strings (the ACTUAL child argv must be bound)"
            % binding["child_process_argv"])
    _require_iso_utc(binding["actual_started_at_utc"], "actual_started_at_utc")
    _require_iso_utc(binding["actual_finished_at_utc"], "actual_finished_at_utc")
    require(isinstance(binding["literal_exit_code"], int)
            and not isinstance(binding["literal_exit_code"], bool)
            and binding["literal_exit_code"] == 0,
            "FAIL CLOSED (NEG29): eval_binding literal_exit_code %r != 0 (a finalized "
            "certificate is only ever emitted when the parent runner's wait() returned "
            "literal exit code 0)" % binding["literal_exit_code"])
    require(binding["exit_source"] == "wait_pid",
            "FAIL CLOSED (NEG37): eval_binding exit_source %r != 'wait_pid' — a "
            "self-declared / log-inferred exit code is never accepted"
            % binding["exit_source"])
    require(binding["inferred_from_log"] is False,
            "FAIL CLOSED (NEG37): eval_binding inferred_from_log must be exactly False "
            "(the exit code is literal, from wait())")
    return binding


# ---------------------------------------------------------------------------
# NEG24 / NEG25 guards
# ---------------------------------------------------------------------------
def assert_scaffold_hash_not_global(cert: dict):
    """NEG24: a scaffold certificate must never present its bank hash as the
    GLOBAL_WORLD_SET_HASH (that belongs solely to the seed42 canonical materializer)."""
    label = cert.get("state_bank_hash_label")
    if cert.get("scenario") in (FRONT, BACK):
        require(label in SCAFFOLD_HASH_LABELS,
                "FAIL CLOSED (NEG24): scaffold certificate state_bank_hash_label %r is not a "
                "scaffold bank label" % label)
        require(label != GLOBAL_HASH_LABEL,
                "FAIL CLOSED (NEG24): scaffold certificate claims GLOBAL_WORLD_SET_HASH")
    return True


def assert_scaffold_does_not_claim_full_success(cert: dict):
    """NEG25: a scaffold result must not claim full-task success / breakthrough.

    A scaffold certificate is diagnostic-only; it may report the conditional scaffold
    metric but NEVER DEFEAT_KOBOLD_SR solved, Tier3 broken, SOTA, Persistent>Reset128,
    or a Replay scientific gain.
    """
    claims = set(cert.get("claims", []))
    bad = sorted(claims & FORBIDDEN_OVERCLAIMS)
    require(not bad,
            "FAIL CLOSED (NEG25): scaffold certificate makes forbidden full-task claim(s): %s"
            % bad)
    if cert.get("scenario") in (FRONT, BACK):
        require(cert.get("scaffolded_results_can_replace_full_task") is False,
                "FAIL CLOSED (NEG25): scaffold certificate must declare "
                "scaffolded_results_can_replace_full_task=False")
        # a scaffold cert must not headline the full-task primary metric as achieved
        require(cert.get("headline_metric") != metrics.FULL_PRIMARY_METRIC
                or cert.get("headline_metric_achieved") is not True,
                "FAIL CLOSED (NEG25): scaffold certificate claims full-task %s achieved"
                % metrics.FULL_PRIMARY_METRIC)
    return True


# ---------------------------------------------------------------------------
# Honest status labels (freeze discipline; task §三 student-status split)
# ---------------------------------------------------------------------------
def honest_status_labels(has_real_rollout: bool, student_state: dict, mode: str) -> dict:
    """Produce ONLY labels that the evidence supports. Never over-claim.

    The ambiguous REAL_STUDENT_EVALUATION key is REMOVED (task §三): the split labels
    REAL_STUDENT_INTERFACE_SMOKE / REAL_STUDENT_PERFORMANCE_EVALUATION say exactly
    which Student activity this certificate's run executed, and
    FORMAL_SCIENTIFIC_CLAIM stays NOT_AUTHORIZED_SINGLE_TRAINING_SEED forever this
    round. A real Student rollout is never labelled NOT_RUN again.
    """
    require(mode in CERT_MODES, "FAIL CLOSED: certificate mode %r not in %s"
            % (mode, CERT_MODES))
    labels = {
        "BOUNDARY_SCHEMA": "IMPLEMENTED_STATIC" if True else None,
        "SCAFFOLD_BUILDER": "IMPLEMENTED_STATIC",
        "STATE_BANK_MATERIALIZER": "TESTED_SYNTHETIC",
        "EVALUATOR": "TESTED_SYNTHETIC",
        "NEGATIVE_TESTS": "PASS",
        "REAL_CRAFTAX_SCAFFOLD_TEST": "TESTED_REAL_ENV_RESET" if has_real_rollout else "BLOCKED_ENVIRONMENT",
        "REAL_STUDENT_INTERFACE_SMOKE": "EXECUTED" if mode == "interface_smoke" else "NOT_RUN",
        "REAL_STUDENT_PERFORMANCE_EVALUATION": "EXECUTED" if mode == "performance_evaluation" else "NOT_RUN",
        "FORMAL_SCIENTIFIC_CLAIM": "NOT_AUTHORIZED_SINGLE_TRAINING_SEED",
        "GLOBAL_WORLD_SET_HASH": "BLOCKED_SOURCE_UNVERIFIED",
        "FRONT_SCAFFOLD_STATE_BANK_HASH": "MATERIALIZED" if has_real_rollout else "NOT_MATERIALIZED",
        "BACK_SCAFFOLD_STATE_BANK_HASH": "MATERIALIZED" if has_real_rollout else "NOT_MATERIALIZED",
        "NEW_TRAINING_RUNS": 0,
        "FORMAL_EVALUATION_RUNS": 0,     # single training seed -> never a formal run
        "SCAFFOLDED_RESULTS_CAN_REPLACE_FULL_TASK": False,
    }
    return labels


def _normalize_student_state(student_state) -> dict:
    if student_state is None:
        return {k: False for k in STUDENT_STATE_KEYS}
    ss = dict(student_state)
    missing = [k for k in STUDENT_STATE_KEYS if k not in ss]
    require(not missing, "FAIL CLOSED: student_state missing key(s) %s" % missing)
    for k in STUDENT_STATE_KEYS:
        require(isinstance(ss[k], bool),
                "FAIL CLOSED: student_state.%s %r is not a bool" % (k, ss[k]))
    require(ss["scientific_claim_authorized"] is False,
            "FAIL CLOSED: student_state.scientific_claim_authorized must be False "
            "(single training seed; provisional selection only)")
    # Never label a real rollout NOT_RUN (task §三): if any Student activity executed,
    # the checkpoint must have loaded.
    if ss["student_policy_rollout_executed"] or ss["performance_evaluation_executed"]:
        require(ss["student_checkpoint_loaded"] is True,
                "FAIL CLOSED: student_policy_rollout_executed / "
                "performance_evaluation_executed require student_checkpoint_loaded=True")
    return {k: ss[k] for k in STUDENT_STATE_KEYS}


def build_certificate(evaluation_result: dict, state_bank_hash_label: str = None,
                      claims=None, has_real_rollout: bool = False,
                      student_state: dict = None, mode: str = "synthetic",
                      eval_binding: dict = None, finalized: bool = False) -> dict:
    """Build one scenario's certificate.

    mode: synthetic | interface_smoke | performance_evaluation — drives the split
    Student status labels (task §三).
    eval_binding + finalized: with finalized=False the ENGINE-stage binding is
    validated (no exit provenance allowed); with finalized=True the RUNNER-finalized
    binding (engine fields + literal wait() exit provenance) is validated (task §二).
    """
    scenario = evaluation_result["scenario"]
    primary = evaluation_result["metrics"]["primary"]
    ss = _normalize_student_state(student_state)
    cert = {
        "schema": SCHEMA,
        "cert_version": CERT_VERSION,
        "scenario": scenario,
        "identity_class": evaluation_result.get("contract", {}).get("observation_schema") and {
            FULL: "CANONICAL_S4_EVALUATION",
            FRONT: "TIER3_FRONT_DIAGNOSTIC_SCAFFOLD",
            BACK: "BOSS_COMBAT_SCAFFOLDED",   # 收口: combat only; boss-area search is N/A
        }[scenario],
        "headline_metric": primary["metric"],
        "headline_metric_value": primary["value"],
        "headline_metric_achieved": None,     # never asserted without real Student data
        "valid_starts": primary["valid_starts"],
        "failure_rule_version": evaluation_result.get("failure_rule_version"),
        "terminal_label_counts": evaluation_result.get("terminal_label_counts"),
        "state_bank_hash_label": state_bank_hash_label or (
            "FRONT_SCAFFOLD_STATE_BANK_HASH" if scenario == FRONT
            else "BACK_SCAFFOLD_STATE_BANK_HASH" if scenario == BACK
            else GLOBAL_HASH_LABEL),
        "rollout_status": evaluation_result.get("rollout_status"),
        "claims": list(claims or []),
        "scaffolded_results_can_replace_full_task": False,
        "student_state": ss,
        "status_labels": honest_status_labels(has_real_rollout, ss, mode),
        "source_audit_schema": audit.SCHEMA,
    }
    if eval_binding is not None:
        cert["eval_binding"] = eval_binding
    assert_scaffold_hash_not_global(cert)          # NEG24
    assert_scaffold_does_not_claim_full_success(cert)   # NEG25
    if eval_binding is not None:
        if finalized:
            assert_eval_binding_complete(cert)     # NEG27/NEG29/NEG37 (runner-finalized)
        else:
            assert_engine_binding_complete(cert)   # NEG27/NEG37 (engine stage)
    return cert


# ---------------------------------------------------------------------------
# Self-test (synthetic; runs on this host).
# ---------------------------------------------------------------------------
def _result(scenario, value, n):
    return {
        "schema": "mechanism_UED.tier3_evaluation_result/v1",
        "scenario": scenario,
        "contract": {"observation_schema": "canonical_craftax_symbolic"},
        "metrics": {"primary": {"metric": metrics.PRIMARY_METRIC[scenario],
                                "value": value, "valid_starts": n}},
        "failure_rule_version": taxonomy.FAILURE_RULE_VERSION,
        "terminal_label_counts": {},
        "rollout_status": "BLOCKED_ENVIRONMENT",
    }


def self_test() -> int:
    problems = []

    def check(name, cond):
        if not cond:
            problems.append(name)

    # A clean front certificate builds and carries honest labels.
    c = build_certificate(_result(FRONT, 0.5, 4))
    check("front_cert_label_scaffold",
          c["state_bank_hash_label"] == "FRONT_SCAFFOLD_STATE_BANK_HASH")
    check("honest_not_run",
          c["status_labels"]["REAL_STUDENT_INTERFACE_SMOKE"] == "NOT_RUN"
          and c["status_labels"]["REAL_STUDENT_PERFORMANCE_EVALUATION"] == "NOT_RUN"
          and c["status_labels"]["FORMAL_SCIENTIFIC_CLAIM"]
          == "NOT_AUTHORIZED_SINGLE_TRAINING_SEED"
          and c["status_labels"]["REAL_CRAFTAX_SCAFFOLD_TEST"] == "BLOCKED_ENVIRONMENT"
          and c["status_labels"]["FRONT_SCAFFOLD_STATE_BANK_HASH"] == "NOT_MATERIALIZED"
          and c["status_labels"]["GLOBAL_WORLD_SET_HASH"] == "BLOCKED_SOURCE_UNVERIFIED")
    check("old_ambiguous_label_removed",
          "REAL_STUDENT_EVALUATION" not in c["status_labels"])
    check("student_state_default_quad",
          c["student_state"] == {"student_checkpoint_loaded": False,
                                 "student_policy_rollout_executed": False,
                                 "performance_evaluation_executed": False,
                                 "scientific_claim_authorized": False})
    check("no_breakthrough_claim", c["headline_metric_achieved"] is None)

    # task §三: interface-smoke / performance modes set the split labels correctly.
    smoke = build_certificate(_result(FRONT, 0.5, 4), has_real_rollout=True,
                              student_state={"student_checkpoint_loaded": True,
                                             "student_policy_rollout_executed": True,
                                             "performance_evaluation_executed": False,
                                             "scientific_claim_authorized": False},
                              mode="interface_smoke")
    check("smoke_mode_labels",
          smoke["status_labels"]["REAL_STUDENT_INTERFACE_SMOKE"] == "EXECUTED"
          and smoke["status_labels"]["REAL_STUDENT_PERFORMANCE_EVALUATION"] == "NOT_RUN"
          and smoke["status_labels"]["FORMAL_SCIENTIFIC_CLAIM"]
          == "NOT_AUTHORIZED_SINGLE_TRAINING_SEED")
    perf = build_certificate(_result(FRONT, 0.5, 4), has_real_rollout=True,
                             student_state={"student_checkpoint_loaded": True,
                                            "student_policy_rollout_executed": True,
                                            "performance_evaluation_executed": True,
                                            "scientific_claim_authorized": False},
                             mode="performance_evaluation")
    check("performance_mode_labels",
          perf["status_labels"]["REAL_STUDENT_PERFORMANCE_EVALUATION"] == "EXECUTED"
          and perf["status_labels"]["REAL_STUDENT_INTERFACE_SMOKE"] == "NOT_RUN")
    try:
        build_certificate(_result(FRONT, 0.5, 4),
                          student_state={"student_checkpoint_loaded": False,
                                         "student_policy_rollout_executed": True,
                                         "performance_evaluation_executed": False,
                                         "scientific_claim_authorized": False})
        check("rollout_without_loaded_checkpoint_rejected", False)
    except FailClosed:
        check("rollout_without_loaded_checkpoint_rejected", True)
    try:
        build_certificate(_result(FRONT, 0.5, 4),
                          student_state={"student_checkpoint_loaded": True,
                                         "student_policy_rollout_executed": False,
                                         "performance_evaluation_executed": False,
                                         "scientific_claim_authorized": True})
        check("scientific_claim_authorized_true_rejected", False)
    except FailClosed:
        check("scientific_claim_authorized_true_rejected", True)

    # NEG24: scaffold cert claiming GLOBAL_WORLD_SET_HASH rejected.
    try:
        build_certificate(_result(FRONT, 0.5, 4), state_bank_hash_label=GLOBAL_HASH_LABEL)
        check("NEG24_global_label_rejected", False)
    except FailClosed:
        check("NEG24_global_label_rejected", True)

    # NEG25: scaffold cert claiming breakthrough / full-task success rejected.
    for bad_claim in ["TIER3_FRONT_HALF_BREAKTHROUGH", "DEFEAT_KOBOLD_SOLVED", "SOTA",
                      "PERSISTENT_BEATS_RESET128", "FORMAL_SCIENTIFIC_PASS",
                      "PERSISTENT_PROVEN_BETTER", "RESET128_PROVEN_BETTER",
                      "TIER3_SOLVED"]:
        try:
            build_certificate(_result(FRONT, 0.5, 4), claims=[bad_claim])
            check("NEG25_overclaim_rejected", False)
            break
        except FailClosed:
            check("NEG25_overclaim_rejected", True)

    # NEG25: scaffold cert headlining full-task metric as achieved rejected.
    full_as_scaffold = _result(FRONT, 1.0, 4)
    full_as_scaffold["metrics"]["primary"]["metric"] = metrics.FULL_PRIMARY_METRIC
    try:
        cert = build_certificate(full_as_scaffold)
        cert["headline_metric_achieved"] = True
        assert_scaffold_does_not_claim_full_success(cert)
        check("NEG25_full_metric_achieved_rejected", False)
    except FailClosed:
        check("NEG25_full_metric_achieved_rejected", True)

    # A FULL certificate MAY use the GLOBAL label and full-task metric.
    fc = build_certificate(_result(FULL, 0.25, 8), has_real_rollout=False)
    check("full_cert_global_label_ok", fc["state_bank_hash_label"] == GLOBAL_HASH_LABEL)

    # ---- NEG27: REAL value binding (actual SHAs, never labels) ----
    def _engine_binding(**over):
        b = {
            "state_bank_hash": "2" + "a" * 63,
            "state_payload_hashes": ["b" * 64, "c" * 64],
            "checkpoint_file_sha256": "d" * 64,
            "cc2_params_sha256": "e" * 64,
            "checkpoint_step": 98304,
            "carry_mode": "persistent",
            "run_class": "INTERFACE_SMOKE",
            "episode_records_sha256": "f" * 64,
            "cc2_policy_source_sha256": "0" * 64,
            "evaluator_source_sha256": "1" * 64,
            "predicate_code_sha256": "a4fba86b054d20412fc1df2c79e7000d66b0525decb1801f"
                                     "a474ee7fb0d25b4c",
            "observation_shape": [8335],
            "action_dim": 43,
            "params_unchanged": True,
            "performance_claim_authorized": False,
            "driver_source_sha256": "9" * 64,
            "checkpoint_contract_sha256": "7" * 64,
            "checkpoint_contract_arm": "persistent",
            "action_mode": "greedy_argmax",
            "max_timesteps": 32,
            "evaluation_seed_schedule": {
                FULL: {"kind": "canonical_reset_seeds", "base": 42, "count": 2,
                       "seeds": [42, 43]},
                FRONT: {"kind": "frozen_bank_state", "seed_base": 10000, "stride": 1,
                        "count": 2, "seeds": [10000, 10001]},
                BACK: {"kind": "frozen_bank_state", "seed_base": 10000, "stride": 1,
                       "count": 2, "seeds": [1010000, 1010001]},
            },
            "state_entry_ids": {FULL: ["full-ep0", "full-ep1"],
                                FRONT: ["front_l2-bank0", "front_l2-bank1"],
                                BACK: ["back_l2-bank0", "back_l2-bank1"]},
            "python_version": "3.11.9",
            "jax_version": "0.4.30",
            "jaxlib_version": "0.4.30",
            "numpy_version": "1.26.4",
            "flax_version": "0.8.5",
            "craftax_version": "1.4.5",
            "evaluator_git_commit": "f67675b87ad98b391f82678bc2f937ab30578145",
            "scientific_claim_authorized": False,
            "single_training_seed": True,
            "provisional_selection_only": True,
        }
        b.update(over)
        return b

    def _exit_binding(**over):
        b = {
            "child_process_pid": 12345,
            "child_process_argv": ["python", "-u",
                                   "tools/tier3_scaffolded_evaluation/tier3_evaluator.py",
                                   "--performance-evaluation"],
            "actual_started_at_utc": "2026-07-30T00:00:00+00:00",
            "actual_finished_at_utc": "2026-07-30T01:00:00+00:00",
            "literal_exit_code": 0,
            "exit_source": "wait_pid",
            "inferred_from_log": False,
            "evaluation_runner_source_sha256": "8" * 64,
        }
        b.update(over)
        return b

    cb = build_certificate(_result(FRONT, 0.5, 4), eval_binding=_engine_binding())
    check("NEG27_engine_binding_accepted", cb["eval_binding"]["params_unchanged"] is True)
    # A single-scenario smoke certificate binds ONLY the scenarios it executed —
    # structural validity per present key, strictness reserved for PROVISIONAL.
    single = build_certificate(_result(FRONT, 0.5, 4), eval_binding=_engine_binding(
        evaluation_seed_schedule={FRONT: {"kind": "frozen_bank_state_smoke", "count": 2,
                                          "seeds": [10000, 10001]}},
        state_entry_ids={FRONT: ["front_l2-bank0", "front_l2-bank1"]}))
    check("single_scenario_schedule_accepted",
          single["eval_binding"]["evaluation_seed_schedule"][FRONT]["seeds"]
          == [10000, 10001])
    finalized_b = dict(_engine_binding())
    finalized_b.update(_exit_binding())
    cb2 = build_certificate(_result(FRONT, 0.5, 4), eval_binding=finalized_b,
                            finalized=True)
    check("NEG29_finalized_binding_accepted",
          cb2["eval_binding"]["literal_exit_code"] == 0
          and cb2["eval_binding"]["exit_source"] == "wait_pid")
    for bad_over, tag in (
            ({"state_bank_hash": "FRONT_SCAFFOLD_STATE_BANK_HASH"}, "label_not_sha"),
            ({"checkpoint_file_sha256": None}, "missing_value"),
            ({"state_payload_hashes": []}, "empty_payload_hashes"),
            ({"observation_shape": [67, 7, 7]}, "wrong_obs_shape"),
            ({"action_dim": 42}, "wrong_action_dim"),
            ({"params_unchanged": False}, "params_changed"),
            ({"run_class": "SMOKE_BUT_PERFORMANCE"}, "bad_run_class"),
            ({"performance_claim_authorized": True}, "smoke_claims_performance"),
            ({"carry_mode": "sideways"}, "bad_carry_mode"),
            ({"checkpoint_contract_sha256": "contract-sha-label"}, "contract_sha_label"),
            ({"checkpoint_contract_arm": "sideways"}, "bad_contract_arm"),
            ({"carry_mode": "reset128", "checkpoint_contract_arm": "persistent"},
             "arm_carry_disagree"),
            ({"action_mode": "sampling"}, "wrong_action_mode"),
            ({"max_timesteps": 0}, "bad_max_timesteps"),
            ({"evaluation_seed_schedule": {FULL: {"kind": "x", "seeds": []}}},
             "empty_seeds_schedule"),
            ({"evaluation_seed_schedule": {FULL: {"seeds": [1]}}},
             "missing_kind_schedule"),
            ({"evaluation_seed_schedule": {"sideways": {"kind": "x", "seeds": [1]}}},
             "unknown_scenario_schedule"),
            ({"state_entry_ids": {FULL: ["a"], FRONT: [], BACK: ["b"]}},
             "empty_state_entry_ids"),
            ({"python_version": ""}, "empty_python_version"),
            ({"jax_version": None}, "missing_jax_version"),
            ({"evaluator_git_commit": "not-a-commit"}, "bad_git_commit"),
            ({"scientific_claim_authorized": True}, "scientific_claim_true"),
            ({"single_training_seed": False}, "single_training_seed_false"),
            ({"provisional_selection_only": False}, "provisional_flag_false"),
            ({"driver_source_sha256": "xyz"}, "bad_driver_sha"),
            # ---- NEG37: self-declared / legacy exit provenance never accepted ----
            ({"run_exit_code": 0}, "NEG37_legacy_run_exit_code"),
            ({"process_pid": 123}, "NEG37_legacy_process_pid"),
            ({"run_start_utc": "2026-07-30T00:00:00+00:00"}, "NEG37_legacy_run_start")):
        try:
            build_certificate(_result(FRONT, 0.5, 4), eval_binding=_engine_binding(**bad_over))
            check("NEG27_rejects_%s" % tag, False)
        except FailClosed:
            check("NEG27_rejects_%s" % tag, True)
    # ---- NEG37/NEG29 on the finalized path ----
    for bad_over, tag in (
            ({"literal_exit_code": 137}, "NEG29_nonzero_exit_code"),
            ({"literal_exit_code": None}, "NEG29_missing_exit_code"),
            ({"exit_source": "self_declared"}, "NEG37_self_declared_exit_source"),
            ({"exit_source": "log_inferred"}, "NEG37_log_inferred_exit_source"),
            ({"inferred_from_log": True}, "NEG37_inferred_from_log_true"),
            ({"child_process_pid": 0}, "NEG29_bad_child_pid"),
            ({"child_process_argv": []}, "NEG29_empty_child_argv"),
            ({"actual_started_at_utc": "yesterday"}, "NEG29_bad_started_at"),
            ({"actual_finished_at_utc": ""}, "NEG29_empty_finished_at"),
            ({"evaluation_runner_source_sha256": "nope"}, "NEG29_bad_runner_sha"),
            ({"run_exit_code": 0}, "NEG37_legacy_field_in_finalized")):
        fb = dict(_engine_binding())
        fb.update(_exit_binding())
        fb.update(bad_over)
        try:
            build_certificate(_result(FRONT, 0.5, 4), eval_binding=fb, finalized=True)
            check("NEG29_rejects_%s" % tag, False)
        except FailClosed:
            check("NEG29_rejects_%s" % tag, True)
    # An UNFINALIZED (engine-only) certificate may NOT pass full verification — the
    # runner provenance is missing (NEG37: no self-certified exit code).
    try:
        ctmp = {"eval_binding": _engine_binding(), "scenario": FRONT}
        assert_eval_binding_complete(ctmp)
        check("engine_only_fails_full_verification", False)
    except FailClosed:
        check("engine_only_fails_full_verification", True)
    # ---- PROVISIONAL schedule binding: the frozen held-out schedule is accepted;
    # a drifted schedule is rejected ----
    prov_sched = {
        FULL: {"kind": "canonical_reset_seeds_held_out", "base": 200000, "count": 64,
               "seeds": [200000 + i for i in range(64)]},
        FRONT: {"kind": "frozen_bank_state", "seed_base": 10000, "stride": 1, "count": 8,
                "seeds": [10000 + i for i in range(8)]},
        BACK: {"kind": "frozen_bank_state", "seed_base": 10000, "stride": 1, "count": 8,
               "seeds": [1010000 + i for i in range(8)]},
    }
    prov_ids = {FULL: ["full-seed%d" % (200000 + i) for i in range(64)],
                FRONT: ["front_l2-bank%d" % i for i in range(8)],
                BACK: ["back_l2-bank%d" % i for i in range(8)]}
    prov = _engine_binding(run_class="PROVISIONAL_STRONG_STUDENT_SELECTION",
                           max_timesteps=4096,
                           evaluation_seed_schedule=prov_sched,
                           state_entry_ids=prov_ids)
    cp = build_certificate(_result(FRONT, 0.5, 8), eval_binding=prov)
    check("provisional_frozen_schedule_accepted",
          cp["eval_binding"]["run_class"] == "PROVISIONAL_STRONG_STUDENT_SELECTION")
    drifted = dict(prov_sched)
    drifted[FULL] = dict(prov_sched[FULL])
    drifted[FULL]["seeds"] = [200001 + i for i in range(64)]   # shifted by one
    try:
        build_certificate(_result(FRONT, 0.5, 8),
                          eval_binding=_engine_binding(
                              run_class="PROVISIONAL_STRONG_STUDENT_SELECTION",
                              max_timesteps=4096,
                              evaluation_seed_schedule=drifted,
                              state_entry_ids=prov_ids))
        check("provisional_shifted_schedule_rejected", False)
    except FailClosed:
        check("provisional_shifted_schedule_rejected", True)
    try:
        build_certificate(_result(FRONT, 0.5, 8),
                          eval_binding=_engine_binding(
                              run_class="PROVISIONAL_STRONG_STUDENT_SELECTION",
                              max_timesteps=2048,
                              evaluation_seed_schedule=prov_sched,
                              state_entry_ids=prov_ids))
        check("provisional_wrong_max_timesteps_rejected", False)
    except FailClosed:
        check("provisional_wrong_max_timesteps_rejected", True)
    # A FORMAL_EVALUATION run class MAY (only if separately authorized) carry
    # performance_claim_authorized=True — the completeness gate still passes.
    formal = build_certificate(_result(FRONT, 0.5, 4),
                               eval_binding=_engine_binding(run_class="FORMAL_EVALUATION",
                                                            performance_claim_authorized=True))
    check("NEG27_formal_class_binding_ok",
          formal["eval_binding"]["run_class"] == "FORMAL_EVALUATION")

    if problems:
        print("TIER3_EVALUATION_CERTIFICATE_SELF_TEST_FAIL")
        for p in problems:
            print("  -", p)
        return 1
    print("TIER3_EVALUATION_CERTIFICATE_SELF_TEST_PASS (NEG24/NEG25 guards live; honest labels)")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--self-test" in argv:
        return self_test()
    print("usage: tier3_evaluation_certificate.py --self-test")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
