"""C8 tests: independent EnvCoder (whitelist prompt, single-pass, K1)."""
import inspect
import json

import pytest

from dicode.teachers.e1_formal import envcoder as EC
from dicode.teachers.e1_formal import llm_client as LC
from dicode.teachers.e1_formal import manifest as M
from dicode.teachers.e1_formal.accounting import LLMCallLedger
from dicode.teachers.e1_formal.schemas import E1SchemaError

from test_task_specs import _family, _window_with_families

SEEDS = [
    {"task_id": "task_1", "description": "collect coal near spawn"},
    {"task_id": "task_2", "description": "cross a narrow gap"},
    {"task_id": "task_3", "description": "defeat the orc solider"},
]


def _specs(n_families=1):
    families = [_family(f"fam_{i}") for i in range(n_families)]
    window = _window_with_families(families)
    from dicode.teachers.e1_formal import task_specs as TS

    return TS.compile_task_specs(window).specs


def _store_for(spec, seeds, payload):
    envelope = EC.build_envcoder_envelope_hash(spec, seed_examples=seeds)
    key = LC.make_replay_key(
        role=M.ENVCODER_ROLE,
        evidence_hash=spec.spec_hash,
        prompt_envelope_hash=envelope,
        prompt_version=M.ENVCODER_PROMPT_VERSION,
        schema_version=M.ENVCODER_OUTPUT_SCHEMA_VERSION,
    )
    return {key: json.dumps(payload)}


def _payload(spec, env_code="def make_env():\n    return env"):
    return {"artifact_id": spec.artifact_id, "env_code": env_code}


class TestPromptWhitelist:
    def test_prompt_is_deterministic(self):
        spec = _specs()[0]
        assert EC.build_envcoder_prompt(spec, seed_examples=SEEDS) == \
            EC.build_envcoder_prompt(spec, seed_examples=SEEDS)

    def test_seed_rotation_by_variant(self):
        seeds = EC.rotate_seeds(tuple(SEEDS), 0)
        assert [s["task_id"] for s in seeds] == ["task_1", "task_2", "task_3"]
        seeds = EC.rotate_seeds(tuple(SEEDS), 1)
        assert [s["task_id"] for s in seeds] == ["task_2", "task_3", "task_1"]
        seeds = EC.rotate_seeds(tuple(SEEDS), 2)
        assert [s["task_id"] for s in seeds] == ["task_3", "task_1", "task_2"]
        # rotation is modulo the seed count
        assert EC.rotate_seeds(tuple(SEEDS), 3) == EC.rotate_seeds(tuple(SEEDS), 0)

    def test_prompt_shows_rotated_order(self):
        window = _window_with_families([_family("fam_0")])
        from dicode.teachers.e1_formal import task_specs as TS

        v0, v1 = TS.compile_task_specs(window).specs
        _, p0 = EC.build_envcoder_prompt(v0, seed_examples=SEEDS)
        _, p1 = EC.build_envcoder_prompt(v1, seed_examples=SEEDS)
        assert "SEED[0] task_id=task_1" in p0
        assert "SEED[0] task_id=task_2" in p1

    def test_builder_signatures_accept_no_board_objects(self):
        # whitelist sentinel: the prompt builder cannot even RECEIVE board
        # windows, evidence or role outputs.
        params = set(inspect.signature(EC.build_envcoder_prompt).parameters)
        assert params == {"spec", "seed_examples"}
        params = set(inspect.signature(EC.run_envcoder).parameters)
        assert params == {"llm", "spec", "seed_examples", "ledger", "window_id"}

    def test_board_content_does_not_leak_into_prompt(self):
        # Sentinels planted in diagnosis hypotheses and audit findings
        # must never reach the EnvCoder prompt (only the canonical
        # TaskSpec does).
        from dicode.teachers.e1_formal import board as B
        from test_board import _build_store, _evidence, _role_payloads

        evidence = _evidence()
        payloads = _role_payloads()
        payloads["causal_failure_analyst"] = dict(
            payloads["causal_failure_analyst"],
            hypotheses=[
                {
                    "hypothesis_id": "h1",
                    "weakness_id": "w1",
                    "statement": "BOARD_SENTINEL_DIAGNOSIS_XYZ",
                }
            ],
        )
        payloads["behavior_auditor"] = {
            "findings": [
                {
                    "finding_id": "bf1",
                    "description": "BOARD_SENTINEL_FINDING_XYZ",
                }
            ]
        }
        payloads["intervention_tutor"] = {
            "families": [
                _family("fam_a", desc="clean intervention description")
            ],
            "explorations": [],
        }
        store = _build_store(evidence, overrides=payloads)
        ledger = LLMCallLedger()
        window = B.run_review_board(
            LC.ReplayLLMClient(store, "t"),
            window_id="w01",
            session_idx=3,
            trigger_code="FIRST_WINDOW",
            evidence=evidence,
            ledger=ledger,
        )
        from dicode.teachers.e1_formal import task_specs as TS

        spec = TS.compile_task_specs(window).specs[0]
        _, user_prompt = EC.build_envcoder_prompt(spec, seed_examples=SEEDS)
        assert "BOARD_SENTINEL_DIAGNOSIS_XYZ" not in user_prompt
        assert "BOARD_SENTINEL_FINDING_XYZ" not in user_prompt
        assert "EVIDENCE_SNAPSHOT" not in user_prompt
        # the whitelist content IS present
        assert "ENV CONTRACT" in user_prompt
        assert "TASK_SPEC spec_id=" in user_prompt
        assert spec.artifact_id in user_prompt

    def test_bad_seeds_rejected(self):
        spec = _specs()[0]
        with pytest.raises(EC.EnvCoderError) as excinfo:
            EC.build_envcoder_prompt(
                spec, seed_examples=[{"task_id": "t", "description": "d", "x": 1}]
            )
        assert excinfo.value.code == "ENVCODER_UNKNOWN_FIELD"
        with pytest.raises(EC.EnvCoderError) as excinfo:
            EC.build_envcoder_prompt(spec, seed_examples=[{"task_id": "t"}])
        assert excinfo.value.code == "ENVCODER_MISSING_FIELD"
        with pytest.raises(EC.EnvCoderError) as excinfo:
            EC.build_envcoder_prompt(spec, seed_examples="not-a-list")
        assert excinfo.value.code == "ENVCODER_BAD_SEEDS"


class TestRunEnvCoder:
    def test_happy_path_and_k1_accounting(self):
        spec = _specs()[0]
        store = _store_for(spec, SEEDS, _payload(spec))
        ledger = LLMCallLedger()
        artifact = EC.run_envcoder(
            LC.ReplayLLMClient(store, "t"),
            spec=spec,
            seed_examples=SEEDS,
            ledger=ledger,
            window_id="w01",
        )
        assert artifact.artifact_id == spec.artifact_id
        assert artifact.spec_id == spec.spec_id
        assert artifact.env_code == "def make_env():\n    return env"
        assert len(artifact.prompt_envelope_hash) == 64
        assert ledger.counts()["K1"] == 1
        assert ledger.counts()["F1"] == 0

    def test_double_run_equality(self):
        spec = _specs()[0]
        store = _store_for(spec, SEEDS, _payload(spec))
        a1 = EC.run_envcoder(
            LC.ReplayLLMClient(store, "t"),
            spec=spec, seed_examples=SEEDS, ledger=LLMCallLedger(), window_id="w01",
        )
        a2 = EC.run_envcoder(
            LC.ReplayLLMClient(store, "t"),
            spec=spec, seed_examples=SEEDS, ledger=LLMCallLedger(), window_id="w01",
        )
        assert a1 == a2

    def test_duplicate_artifact_never_double_counts(self):
        spec = _specs()[0]
        store = _store_for(spec, SEEDS, _payload(spec))
        ledger = LLMCallLedger()
        for _ in range(3):
            EC.run_envcoder(
                LC.ReplayLLMClient(store, "t"),
                spec=spec, seed_examples=SEEDS, ledger=ledger, window_id="w01",
            )
        assert ledger.counts()["K1"] == 1

    def test_artifact_id_mismatch_fails_closed(self):
        spec = _specs()[0]
        bad = {"artifact_id": spec.artifact_id + "x", "env_code": "code"}
        store = _store_for(spec, SEEDS, bad)
        with pytest.raises(EC.EnvCoderError) as excinfo:
            EC.run_envcoder(
                LC.ReplayLLMClient(store, "t"),
                spec=spec, seed_examples=SEEDS, ledger=LLMCallLedger(),
                window_id="w01",
            )
        assert excinfo.value.code == "ENVCODER_ARTIFACT_MISMATCH"

    def test_missing_field_fails_closed(self):
        spec = _specs()[0]
        store = _store_for(spec, SEEDS, {"artifact_id": spec.artifact_id})
        with pytest.raises(EC.EnvCoderError) as excinfo:
            EC.run_envcoder(
                LC.ReplayLLMClient(store, "t"),
                spec=spec, seed_examples=SEEDS, ledger=LLMCallLedger(),
                window_id="w01",
            )
        assert excinfo.value.code == "ENVCODER_MISSING_FIELD"

    def test_unknown_field_fails_closed(self):
        spec = _specs()[0]
        payload = dict(_payload(spec), extra="nope")
        store = _store_for(spec, SEEDS, payload)
        with pytest.raises(EC.EnvCoderError) as excinfo:
            EC.run_envcoder(
                LC.ReplayLLMClient(store, "t"),
                spec=spec, seed_examples=SEEDS, ledger=LLMCallLedger(),
                window_id="w01",
            )
        assert excinfo.value.code == "ENVCODER_UNKNOWN_FIELD"

    def test_non_json_output_fails_closed(self):
        spec = _specs()[0]
        store = _store_for(spec, SEEDS, None)
        envelope = EC.build_envcoder_envelope_hash(spec, seed_examples=SEEDS)
        key = LC.make_replay_key(
            role=M.ENVCODER_ROLE,
            evidence_hash=spec.spec_hash,
            prompt_envelope_hash=envelope,
            prompt_version=M.ENVCODER_PROMPT_VERSION,
            schema_version=M.ENVCODER_OUTPUT_SCHEMA_VERSION,
        )
        store[key] = "no JSON here"
        with pytest.raises(E1SchemaError) as excinfo:
            EC.run_envcoder(
                LC.ReplayLLMClient(store, "t"),
                spec=spec, seed_examples=SEEDS, ledger=LLMCallLedger(),
                window_id="w01",
            )
        assert excinfo.value.code == "JSON_NOT_FOUND"

    def test_forbidden_env_code_content_rejected_by_guards(self):
        spec = _specs()[0]
        payload = _payload(spec, env_code="env.waypoint_list = []")
        store = _store_for(spec, SEEDS, payload)
        ledger = LLMCallLedger()
        with pytest.raises(Exception) as excinfo:
            EC.run_envcoder(
                LC.ReplayLLMClient(store, "t"),
                spec=spec, seed_examples=SEEDS, ledger=ledger, window_id="w01",
            )
        assert getattr(excinfo.value, "code", "") != ""
        # single-pass: the call happened (K1) but NO repair call exists
        assert ledger.counts()["K1"] == 1
        assert ledger.counts()["F1"] == 0

    def test_replay_miss_is_hard_fail(self):
        spec = _specs()[0]
        with pytest.raises(RuntimeError) as excinfo:
            EC.run_envcoder(
                LC.ReplayLLMClient({}, "t"),
                spec=spec, seed_examples=SEEDS, ledger=LLMCallLedger(),
                window_id="w01",
            )
        assert str(excinfo.value).startswith(LC.HARD_FAIL_PREFIX)

    def test_variants_get_distinct_cache_keys(self):
        window = _window_with_families([_family("fam_0")])
        from dicode.teachers.e1_formal import task_specs as TS

        v0, v1 = TS.compile_task_specs(window).specs
        h0 = EC.build_envcoder_envelope_hash(v0, seed_examples=SEEDS)
        h1 = EC.build_envcoder_envelope_hash(v1, seed_examples=SEEDS)
        assert h0 != h1  # seed rotation changes the envelope
