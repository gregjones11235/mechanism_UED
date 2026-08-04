"""C6 tests: six-role review board (fixed order, all-or-nothing, fail-closed)."""
import json

import pytest

from dicode.teachers.e1_formal import board as B
from dicode.teachers.e1_formal import evidence as E
from dicode.teachers.e1_formal import llm_client as LC
from dicode.teachers.e1_formal import manifest as M
from dicode.teachers.e1_formal.accounting import LLMCallLedger


def _evidence():
    return E.build_evidence_snapshot(
        [
            {
                "source": "training_window.session_metrics",
                "session_idx": 3,
                "provenance": "NORMAL_TRAINING_FEEDBACK",
                "facts": {"success_rate": 0.4, "skill_get_wood": 0.75},
            }
        ],
        "board-test",
    )


def _role_payloads():
    """Valid per-role JSON payloads (all pass guards + contracts)."""
    return {
        "student_modeler": {
            "model_summary": "Student collects wood but fails at bridges",
            "capability_profile": [
                {"skill_id": "get_wood", "success_rate": 0.75},
                {"skill_id": "build_bridge", "success_rate": 0.1},
            ],
        },
        "behavior_auditor": {
            "findings": [
                {
                    "finding_id": "bf1",
                    "description": "repeated failure when crossing gaps",
                }
            ]
        },
        "causal_failure_analyst": {
            "weaknesses": [
                {
                    "weakness_id": "w1",
                    "name": "gap crossing",
                    "evidence_refs": ["bf1"],
                    "priority": 1,
                }
            ],
            "hypotheses": [
                {
                    "hypothesis_id": "h1",
                    "weakness_id": "w1",
                    "statement": "placement skill is undertrained",
                }
            ],
            "reuse_previous_direction": False,
            "overall_confidence": 0.7,
        },
        "intervention_tutor": {
            "families": [
                {
                    "family_id": "fam_a",
                    "description": "reduce enemy pressure near gaps",
                    "target_achievements": ["get_wood"],
                    "axis_changes": [
                        {"axis": "enemy_density", "from_value": "high", "to_value": "low"}
                    ],
                    "constant_axes": ["map_size"],
                    "scaffolding": "narrow the gap",
                    "student_must_do": "collect wood and cross",
                },
                {
                    "family_id": "fam_b",
                    "description": "add a second crossing site",
                    "target_achievements": ["get_wood"],
                    "axis_changes": [
                        {"axis": "crossings", "from_value": "one", "to_value": "two"}
                    ],
                    "constant_axes": ["map_size"],
                    "scaffolding": "mark both sites",
                    "student_must_do": "choose a crossing",
                },
            ],
            "explorations": [],
        },
        "explorer": {
            "exploration_rationale": "probe terrain variety",
            "candidate_axes": ["terrain_roughness"],
        },
        "critic": {"vetoes": [], "notes": "nothing to veto"},
    }


def _build_store(
    evidence,
    overrides=None,
    drop=(),
    window_id="w01",
    session_idx=3,
    trigger_code="FIRST_WINDOW",
):
    """Replay store matching the SEQUENTIAL board (round-3 P0-1).

    Mirrors ``run_review_board`` exactly: the board context is built
    once, and each role's envelope binds every SUCCESSFULLY PARSED
    upstream output. Roles whose payload fails guards/parsing are
    recorded in the store but excluded from the upstream chain of
    later roles — precisely what the board computes at runtime.

    ``window_id`` / ``session_idx`` / ``trigger_code`` must match the
    values the caller hands to ``run_review_board`` (the controller
    derives them from the gate state, hence the parameters).
    """
    payloads = _role_payloads()
    if overrides:
        payloads.update(overrides)
    store = {}
    context = B.make_board_context(
        window_id=window_id,
        session_idx=session_idx,
        trigger_code=trigger_code,
        evidence_hash=evidence.evidence_hash,
    )
    upstream = []
    for role in M.BOARD_ROLE_ORDER:
        if role in drop:
            break  # replay miss => HARD FAIL; the board never goes on
        envelope = B.build_prompt_envelope_hash(
            role, evidence, context=context, upstream=upstream
        )
        key = LC.make_replay_key(
            role=role,
            evidence_hash=evidence.evidence_hash,
            prompt_envelope_hash=envelope,
            prompt_version=M.BOARD_PROMPT_VERSION,
            schema_version=M.ROLE_OUTPUT_SCHEMA_VERSION,
        )
        payload = payloads[role]
        content = payload if isinstance(payload, str) else json.dumps(payload)
        store[key] = content
        # mimic the board's parse outcome for the upstream chain
        try:
            parsed = B._parse_role_content(role, content, f"store-build {role}")
        except Exception:
            continue  # failed role is NOT bound by later roles
        upstream.append(
            B.UpstreamOutput(
                role=role,
                output=parsed,
                output_hash=B.role_output_hash(role, parsed),
            )
        )
    return store


def _run(evidence=None, store=None, window_id="w01"):
    evidence = evidence or _evidence()
    store = store if store is not None else _build_store(
        evidence, window_id=window_id
    )
    llm = LC.ReplayLLMClient(store, "board-test")
    ledger = LLMCallLedger()
    window = B.run_review_board(
        llm,
        window_id=window_id,
        session_idx=3,
        trigger_code="FIRST_WINDOW",
        evidence=evidence,
        ledger=ledger,
    )
    return window, ledger


class TestHappyPath:
    def test_complete_window(self):
        window, ledger = _run()
        assert window.status == B.WINDOW_STATUS_COMPLETE
        assert window.void_code == ""
        assert window.surviving_families == ("fam_a", "fam_b")
        assert window.ignored_vetoes == ()
        assert len(window.window_hash) == 64

    def test_board_runs_six_roles_in_fixed_order(self):
        window, ledger = _run()
        board_records = [
            r for r in ledger.to_records() if r["kind"] == "BOARD"
        ]
        assert [r["role"] for r in board_records] == list(M.BOARD_ROLE_ORDER)
        counts = ledger.reconcile()
        assert counts["G1"] == 1
        assert counts["board_calls"] == 6

    def test_double_run_equality(self):
        evidence = _evidence()
        store = _build_store(evidence)
        w1, _ = _run(evidence=evidence, store=store)
        w2, _ = _run(evidence=evidence, store=store)
        assert w1 == w2
        assert w1.window_hash == w2.window_hash

    def test_role_results_present_for_all_six(self):
        window, _ = _run()
        assert [role for role, _ in window.role_results] == list(M.BOARD_ROLE_ORDER)
        assert all(obj is not None for _, obj in window.role_results)


class TestCriticVerdict:
    def test_valid_veto_removes_family(self):
        evidence = _evidence()
        critic = {"vetoes": [{"family_id": "fam_a", "reason": "GUARD_VETO"}]}
        store = _build_store(evidence, overrides={"critic": critic})
        window, _ = _run(evidence=evidence, store=store)
        assert window.status == B.WINDOW_STATUS_COMPLETE
        assert window.surviving_families == ("fam_b",)

    def test_invented_reason_ignored_and_recorded(self):
        evidence = _evidence()
        critic = {
            "vetoes": [{"family_id": "fam_a", "reason": "BECAUSE_I_SAY_SO"}]
        }
        store = _build_store(evidence, overrides={"critic": critic})
        window, _ = _run(evidence=evidence, store=store)
        # invented reason must NOT remove the family
        assert window.surviving_families == ("fam_a", "fam_b")
        assert window.ignored_vetoes == (
            {
                "family_id": "fam_a",
                "reason": "BECAUSE_I_SAY_SO",
                "why_ignored": "UNKNOWN_VETO_REASON",
            },
        )

    def test_unknown_family_target_ignored(self):
        evidence = _evidence()
        critic = {
            "vetoes": [{"family_id": "fam_z", "reason": "RETENTION_VETO"}]
        }
        store = _build_store(evidence, overrides={"critic": critic})
        window, _ = _run(evidence=evidence, store=store)
        assert window.surviving_families == ("fam_a", "fam_b")
        assert window.ignored_vetoes[0]["why_ignored"] == "UNKNOWN_FAMILY_TARGET"

    def test_all_families_vetoed_voids_window(self):
        evidence = _evidence()
        critic = {
            "vetoes": [
                {"family_id": "fam_a", "reason": "CRITIC_VETO_UNSUPPORTED_CLAIM"},
                {"family_id": "fam_b", "reason": "CRITIC_VETO_EVIDENCE_MISMATCH"},
            ]
        }
        store = _build_store(evidence, overrides={"critic": critic})
        window, _ = _run(evidence=evidence, store=store)
        assert window.status == B.WINDOW_STATUS_VOID
        assert window.void_code == "ALL_FAMILIES_VETOED"
        assert window.surviving_families == ()

    def test_apply_critic_verdict_unit(self):
        critic = {
            "vetoes": [
                {"family_id": "a", "reason": "GUARD_VETO"},
                {"family_id": "b", "reason": "NOT_A_REAL_CODE"},
            ]
        }
        surviving, ignored = B.apply_critic_verdict(["a", "b", "c"], critic, "t")
        assert surviving == ("b", "c")
        assert len(ignored) == 1


class TestVoidWindows:
    def test_bad_role_output_voids_but_all_six_calls_recorded(self):
        evidence = _evidence()
        store = _build_store(evidence, overrides={"behavior_auditor": {"nope": 1}})
        window, ledger = _run(evidence=evidence, store=store)
        assert window.status == B.WINDOW_STATUS_VOID
        assert window.void_code == "INCOMPLETE_REVIEW_WINDOW"
        # all six roles still ran and were accounted
        counts = ledger.reconcile()
        assert counts["board_calls"] == 6
        by_role = dict(window.role_results)
        assert by_role["behavior_auditor"] is None
        assert by_role["student_modeler"] is not None

    def test_guard_tripped_content_voids_window(self):
        evidence = _evidence()
        poisoned = {
            "findings": [
                {"finding_id": "bf1", "description": "add waypoints to the map"}
            ]
        }
        store = _build_store(evidence, overrides={"behavior_auditor": poisoned})
        window, ledger = _run(evidence=evidence, store=store)
        assert window.status == B.WINDOW_STATUS_VOID
        assert window.void_code == "INCOMPLETE_REVIEW_WINDOW"
        assert ledger.counts()["board_calls"] == 6

    def test_invalid_json_voids_window(self):
        evidence = _evidence()
        # the chain builder stores this raw string under the
        # explorer's sequential envelope key (no JSON object inside)
        store = _build_store(
            evidence, overrides={"explorer": "this is not JSON at all"}
        )
        window, _ = _run(evidence=evidence, store=store)
        assert window.status == B.WINDOW_STATUS_VOID
        assert window.void_code == "INCOMPLETE_REVIEW_WINDOW"

    def test_replay_miss_is_hard_fail_and_propagates(self):
        evidence = _evidence()
        store = _build_store(evidence, drop=("critic",))  # critic will miss
        llm = LC.ReplayLLMClient(store, "board-test")
        ledger = LLMCallLedger()
        with pytest.raises(RuntimeError) as excinfo:
            B.run_review_board(
                llm,
                window_id="w01",
                session_idx=3,
                trigger_code="FIRST_WINDOW",
                evidence=evidence,
                ledger=ledger,
            )
        assert str(excinfo.value).startswith(LC.HARD_FAIL_PREFIX)


class TestPersistence:
    def test_round_trip(self):
        window, _ = _run()
        record = B.window_to_record(window)
        assert B.verify_window_record(record, "t") == window

    def test_tampered_record_fails_closed(self):
        window, _ = _run()
        record = B.window_to_record(window)
        record["status"] = "VOID"
        with pytest.raises(B.BoardError) as excinfo:
            B.verify_window_record(record, "t")
        assert excinfo.value.code == "WINDOW_HASH_MISMATCH"

    def test_tampered_hash_fails_closed(self):
        window, _ = _run()
        record = B.window_to_record(window)
        record["window_hash"] = "0" * 64
        with pytest.raises(B.BoardError) as excinfo:
            B.verify_window_record(record, "t")
        assert excinfo.value.code == "WINDOW_HASH_MISMATCH"

    def test_unknown_field_rejected(self):
        window, _ = _run()
        record = B.window_to_record(window)
        record["extra"] = 1
        with pytest.raises(B.BoardError) as excinfo:
            B.verify_window_record(record, "t")
        assert excinfo.value.code == "WINDOW_RECORD_UNKNOWN_FIELD"

    def test_missing_field_rejected(self):
        window, _ = _run()
        record = B.window_to_record(window)
        del record["status"]
        with pytest.raises(B.BoardError) as excinfo:
            B.verify_window_record(record, "t")
        assert excinfo.value.code == "WINDOW_RECORD_MISSING_FIELD"


class TestPrompts:
    def _context(self, evidence):
        return B.make_board_context(
            window_id="w01",
            session_idx=3,
            trigger_code="FIRST_WINDOW",
            evidence_hash=evidence.evidence_hash,
        )

    def test_prompt_is_deterministic(self):
        evidence = _evidence()
        context = self._context(evidence)
        assert B.build_role_prompt(
            "critic", evidence, context=context
        ) == B.build_role_prompt("critic", evidence, context=context)

    def test_unknown_role_rejected(self):
        evidence = _evidence()
        with pytest.raises(B.BoardError):
            B.build_role_prompt(
                "envcoder", evidence, context=self._context(evidence)
            )

    def test_user_prompt_embeds_identity_and_evidence(self):
        evidence = _evidence()
        context = self._context(evidence)
        _, user_prompt = B.build_role_prompt(
            "student_modeler", evidence, context=context
        )
        assert "EVIDENCE_SNAPSHOT hash=" in user_prompt
        assert evidence.evidence_hash in user_prompt
        # round-3 P0-1: window + Student identity are bound into every
        # role prompt
        assert "WINDOW: window_id=w01" in user_prompt
        assert context.student_candidate_id in user_prompt
        assert "UPSTREAM_ROLE_OUTPUTS" in user_prompt

    def test_upstream_output_is_rendered_and_bound(self):
        evidence = _evidence()
        context = self._context(evidence)
        modeler_payload = _role_payloads()["student_modeler"]
        upstream = [
            B.UpstreamOutput(
                role="student_modeler",
                output=modeler_payload,
                output_hash=B.role_output_hash(
                    "student_modeler", modeler_payload
                ),
            )
        ]
        _, user_prompt = B.build_role_prompt(
            "behavior_auditor", evidence, context=context, upstream=upstream
        )
        # the later role literally reads the earlier role's output
        assert modeler_payload["model_summary"] in user_prompt
        assert upstream[0].output_hash in user_prompt
        # and the envelope changes with the chain
        bare = B.build_prompt_envelope_hash(
            "behavior_auditor", evidence, context=context
        )
        chained = B.build_prompt_envelope_hash(
            "behavior_auditor", evidence, context=context, upstream=upstream
        )
        assert bare != chained

    def test_corrupted_upstream_hash_rejected(self):
        evidence = _evidence()
        context = self._context(evidence)
        modeler_payload = _role_payloads()["student_modeler"]
        corrupted = B.UpstreamOutput(
            role="student_modeler",
            output=modeler_payload,
            output_hash="0" * 64,
        )
        with pytest.raises(B.BoardError) as excinfo:
            B.build_role_prompt(
                "behavior_auditor",
                evidence,
                context=context,
                upstream=[corrupted],
            )
        assert excinfo.value.code == "BOARD_CHAIN_HASH_MISMATCH"

    def test_envelope_hash_differs_per_role(self):
        evidence = _evidence()
        context = self._context(evidence)
        hashes = {
            B.build_prompt_envelope_hash(role, evidence, context=context)
            for role in M.BOARD_ROLE_ORDER
        }
        assert len(hashes) == len(M.BOARD_ROLE_ORDER)
