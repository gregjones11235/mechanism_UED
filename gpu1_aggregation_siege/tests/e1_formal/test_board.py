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


def _build_store(evidence, overrides=None, drop=()):
    payloads = _role_payloads()
    if overrides:
        payloads.update(overrides)
    store = {}
    for role, payload in payloads.items():
        if role in drop:
            continue
        envelope = B.build_prompt_envelope_hash(role, evidence)
        key = LC.make_replay_key(
            role=role,
            evidence_hash=evidence.evidence_hash,
            prompt_envelope_hash=envelope,
            prompt_version=M.BOARD_PROMPT_VERSION,
            schema_version=M.ROLE_OUTPUT_SCHEMA_VERSION,
        )
        store[key] = json.dumps(payload)
    return store


def _run(evidence=None, store=None, window_id="w01"):
    evidence = evidence or _evidence()
    store = store if store is not None else _build_store(evidence)
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
        store = _build_store(evidence)
        envelope = B.build_prompt_envelope_hash("explorer", evidence)
        key = LC.make_replay_key(
            role="explorer",
            evidence_hash=evidence.evidence_hash,
            prompt_envelope_hash=envelope,
            prompt_version=M.BOARD_PROMPT_VERSION,
            schema_version=M.ROLE_OUTPUT_SCHEMA_VERSION,
        )
        store[key] = "this is not JSON at all"
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
    def test_prompt_is_deterministic(self):
        evidence = _evidence()
        assert B.build_role_prompt("critic", evidence) == B.build_role_prompt(
            "critic", evidence
        )

    def test_unknown_role_rejected(self):
        with pytest.raises(B.BoardError):
            B.build_role_prompt("envcoder", _evidence())

    def test_user_prompt_embeds_evidence_rendering(self):
        evidence = _evidence()
        _, user_prompt = B.build_role_prompt("student_modeler", evidence)
        assert "EVIDENCE_SNAPSHOT hash=" in user_prompt
        assert evidence.evidence_hash in user_prompt

    def test_envelope_hash_differs_per_role(self):
        evidence = _evidence()
        hashes = {
            B.build_prompt_envelope_hash(role, evidence)
            for role in M.BOARD_ROLE_ORDER
        }
        assert len(hashes) == len(M.BOARD_ROLE_ORDER)
