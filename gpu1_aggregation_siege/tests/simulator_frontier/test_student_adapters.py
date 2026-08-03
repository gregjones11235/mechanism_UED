"""Stage 2 tests: shared student_adapters contract layer (CC4 sole owner).

Covers: StudentIdentity fail-closed validation, FakeStudentAdapter protocol
round-trip (contract tests only, explicitly not scientific content), registry
fail-closed resolution, runtime override parsing, checkpoint codec (synthetic
CC2 pkl round-trip + tamper negatives), and profile loading (five real
profiles + path-scan negatives).
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from dicode.student_adapters import (
    FakeStudentAdapter,
    StudentAdapterRegistry,
    default_profile_dir,
    fake_params_sha256,
    load_student_profile,
    resolve_runtime_overrides,
)
from dicode.student_adapters.checkpoint_codec import (
    FORMAT_BAKEOFF_PKL,
    FORMAT_CC2_PKL,
    FORMAT_PHASE4A_PKL,
    CheckpointCodecError,
    CheckpointFormatNotImplementedError,
    cc2_params_sha256,
    file_sha256,
    load_cc2_pkl,
    load_checkpoint,
)
from dicode.student_adapters.identity import (
    StudentIdentity,
    StudentIdentityError,
    identity_field_names,
    identity_to_mapping,
    validate_identity,
)
from dicode.student_adapters.protocol import StudentAdapter
from dicode.student_adapters.registry import StudentRegistryError


def _identity(**overrides):
    base = dict(
        candidate_id="TEST_STUDENT",
        architecture_family="RMT16",
        checkpoint_format=FORMAT_CC2_PKL,
        global_step=98304,
        total_env_steps=98304,
        params_sha256="a" * 64,
        source_commit="src-sha256:" + "b" * 64,
        observation_shape=(8335,),
        action_count=43,
        memory_spec_hash="c" * 64,
    )
    base.update(overrides)
    return StudentIdentity(**base)


class TestStudentIdentity:
    def test_valid_identity_passes_and_hash_is_deterministic(self):
        ident = validate_identity(_identity())
        h1, h2 = ident.identity_hash(), ident.identity_hash()
        assert h1 == h2 and len(h1) == 64

    @pytest.mark.parametrize("field,bad", [
        ("candidate_id", ""),
        ("candidate_id", "PENDING_EVIDENCE"),
        ("candidate_id", "UNKNOWN"),
        ("architecture_family", "TODO"),
        ("checkpoint_format", "N/A"),
        ("source_commit", " "),
        ("params_sha256", "zzz"),
        ("params_sha256", "A" * 64),  # must be lowercase hex
    ])
    def test_placeholder_or_malformed_fields_raise(self, field, bad):
        with pytest.raises(StudentIdentityError):
            validate_identity(_identity(**{field: bad}))

    def test_bad_shape_and_counts_raise(self):
        with pytest.raises(StudentIdentityError):
            validate_identity(_identity(observation_shape=()))
        with pytest.raises(StudentIdentityError):
            validate_identity(_identity(observation_shape=(0,)))
        with pytest.raises(StudentIdentityError):
            validate_identity(_identity(action_count=0))
        with pytest.raises(StudentIdentityError):
            validate_identity(_identity(global_step=-1))

    def test_identity_to_mapping_and_field_names(self):
        mapping = identity_to_mapping(_identity())
        assert set(mapping) == {"identity_hash", "params_sha256", "memory_spec_hash", "candidate_id"}
        assert "memory_spec_hash" in identity_field_names()
        assert "extras" in identity_field_names()


class TestFakeAdapterContract:
    def test_fake_adapter_satisfies_protocol(self):
        assert isinstance(FakeStudentAdapter(), StudentAdapter)

    def test_memory_contract_positive_and_negative(self):
        fake = FakeStudentAdapter()
        mem = fake.initial_memory(2)
        assert fake.validate_memory(mem, 2)["ok"] is True
        assert fake.validate_memory(fake.initial_memory(3), 2)["ok"] is False  # batch mismatch
        assert fake.validate_memory({"nope": 1}, 1)["ok"] is False
        bad_dtype = {"h": np.zeros((2, 8), dtype=np.float64)}
        check = fake.validate_memory(bad_dtype, 2)
        assert check["ok"] is False and check["reasons"]

    def test_policy_step_deterministic_reproducible_and_single_obs(self):
        fake = FakeStudentAdapter()
        mem = fake.initial_memory(2)
        obs = np.random.default_rng(1).standard_normal((2, 4)).astype(np.float32)
        out1 = fake.policy_step(fake._params, obs, mem, None, None, None, True)
        out2 = fake.policy_step(fake._params, obs, mem, None, None, None, True)
        assert np.array_equal(out1["action"], out2["action"])
        assert out1["logits"].shape == (2, 3)
        single = fake.policy_step(fake._params, obs[0], fake.initial_memory(1), None, None, None, True)
        assert single["action"].shape == () and single["logits"].shape == (3,)
        rng = np.random.default_rng(0)
        stoch = fake.policy_step(fake._params, obs, mem, None, None, rng, False)
        assert stoch["action"].shape == (2,)
        with pytest.raises(ValueError):
            fake.policy_step(fake._params, obs, {"wrong": 1}, None, None, None, True)

    def test_save_restore_round_trip_and_tamper_rejection(self, tmp_path):
        fake = FakeStudentAdapter()
        ts = {"params": fake._params, "optimizer": {"m": np.zeros(1)},
              "global_step": 7, "rng_state": 123, "memory": fake.initial_memory(1)}
        path = str(tmp_path / "fake_state.pkl")
        fake.save_full_state(path, ts, {"note": "contract"})
        restored = fake.restore_full_state(path)
        assert restored["train_state"]["global_step"] == 7
        assert restored["params_sha256"] == fake_params_sha256(fake._params)
        # tamper: mutate stored params, keep declared hash -> must raise
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        payload["train_state"]["params"]["b"] = np.ones(3, dtype=np.float32)
        tampered = str(tmp_path / "tampered.pkl")
        with open(tampered, "wb") as fh:
            pickle.dump(payload, fh)
        with pytest.raises(ValueError):
            fake.restore_full_state(tampered)

    def test_load_full_state_identity_gate(self, tmp_path):
        fake = FakeStudentAdapter()
        ts = {"params": fake._params, "optimizer": {}, "global_step": 0,
              "rng_state": 0, "memory": fake.initial_memory(1)}
        path = str(tmp_path / "state.pkl")
        fake.save_full_state(path, ts, {})
        loaded = fake.load_full_state(path, fake.identity())
        assert loaded["params_sha256"] == fake.identity().params_sha256
        foreign = _identity(params_sha256="f" * 64)
        with pytest.raises(ValueError):
            fake.load_full_state(path, foreign)


class TestRegistryFailClosed:
    def test_unknown_candidate_raises(self):
        reg = StudentAdapterRegistry()
        profile = load_student_profile(default_profile_dir() / "rmt16_persistent_98304.yaml")
        with pytest.raises(StudentRegistryError):
            reg.resolve(profile)

    def test_registration_violations_raise(self):
        reg = StudentAdapterRegistry()
        with pytest.raises(StudentRegistryError):
            reg.register("   ", FakeStudentAdapter)
        with pytest.raises(StudentRegistryError):
            reg.register("X", "not-callable")
        reg.register("X", FakeStudentAdapter)
        with pytest.raises(StudentRegistryError):
            reg.register("X", FakeStudentAdapter)

    def test_resolve_checks_protocol_conformance(self):
        reg = StudentAdapterRegistry()
        profile = load_student_profile(default_profile_dir() / "rmt16_persistent_98304.yaml")

        class NotAnAdapter:
            pass

        reg.register(profile.candidate_id, lambda p: NotAnAdapter())
        with pytest.raises(StudentRegistryError):
            reg.resolve(profile)

    def test_resolve_returns_registered_adapter(self):
        reg = StudentAdapterRegistry()
        profile = load_student_profile(default_profile_dir() / "rmt16_persistent_98304.yaml")
        reg.register(profile.candidate_id, lambda p: FakeStudentAdapter(candidate_id=p.candidate_id))
        adapter = reg.resolve(profile)
        assert isinstance(adapter, StudentAdapter)
        assert reg.candidates() == (profile.candidate_id,)


class TestRuntimeOverrides:
    def test_parses_known_keys_and_ignores_others(self):
        argv = ["student.profile=rmt16_persistent_98304",
                "student.checkpoint_path=/tmp/ckpt.pkl",
                "hydra.verbose=true", "flag"]
        out = resolve_runtime_overrides(argv)
        assert out["student.profile"] == "rmt16_persistent_98304"
        assert out["student.checkpoint_path"] == "/tmp/ckpt.pkl"
        assert "hydra.verbose" not in out

    def test_unknown_student_key_raises(self):
        with pytest.raises(StudentRegistryError):
            resolve_runtime_overrides(["student.nope=1"])

    def test_duplicate_and_empty_raise(self):
        with pytest.raises(StudentRegistryError):
            resolve_runtime_overrides(["student.profile=a", "student.profile=b"])
        with pytest.raises(StudentRegistryError):
            resolve_runtime_overrides(["student.profile= "])


class TestCheckpointCodec:
    def _synthetic_cc2_pkl(self, tmp_path, *, extra_top_key=None, manifest=None,
                           declared_sha=None):
        params = {"encoder": {"kernel": np.arange(6, dtype=np.float32).reshape(2, 3),
                              "bias": np.zeros(3, np.float32)},
                  "head": np.ones(4, np.float32)}
        sha = cc2_params_sha256(params)
        top = {"params": params,
               "manifest": manifest if manifest is not None else
               {"params_sha256": declared_sha or sha, "step": 100, "arm": "synthetic"}}
        if extra_top_key is not None:
            top[extra_top_key] = 1
        path = str(tmp_path / "synthetic_full_state.pkl")
        with open(path, "wb") as fh:
            pickle.dump(top, fh, protocol=4)
        return path, sha, params

    def test_cc2_round_trip_and_expected_sha_gate(self, tmp_path):
        path, sha, params = self._synthetic_cc2_pkl(tmp_path)
        loaded = load_cc2_pkl(path)
        assert loaded.format == FORMAT_CC2_PKL
        assert loaded.params_sha256 == sha
        assert loaded.global_step == 100
        assert loaded.contains_optimizer is False
        assert loaded.contains_rng is False
        assert loaded.contains_policy_memory is False
        assert loaded.file_sha256 == file_sha256(path)
        again = load_cc2_pkl(path, expected_params_sha256=sha)
        assert again.params_sha256 == sha
        with pytest.raises(CheckpointCodecError):
            load_cc2_pkl(path, expected_params_sha256="0" * 64)

    def test_cc2_hash_is_deterministic_and_tamper_sensitive(self):
        params = {"w": np.arange(4, dtype=np.float32)}
        assert cc2_params_sha256(params) == cc2_params_sha256(dict(params))
        mutated = {"w": np.arange(4, dtype=np.float32) + 1.0}
        assert cc2_params_sha256(mutated) != cc2_params_sha256(params)

    def test_structural_violations_raise(self, tmp_path):
        # extra top-level key -> not the CC2 contract -> raise (never guess)
        path, base_sha, _ = self._synthetic_cc2_pkl(tmp_path, extra_top_key="optimizer")
        with pytest.raises(CheckpointCodecError):
            load_cc2_pkl(path)
        # manifest missing step
        path2, _, _ = self._synthetic_cc2_pkl(tmp_path, manifest={"params_sha256": base_sha})
        with pytest.raises(CheckpointCodecError):
            load_cc2_pkl(path2)
        # declared sha mismatch
        path3, _, _ = self._synthetic_cc2_pkl(tmp_path, declared_sha="9" * 64)
        with pytest.raises(CheckpointCodecError):
            load_cc2_pkl(path3)
        # missing file
        with pytest.raises(CheckpointCodecError):
            load_cc2_pkl(str(tmp_path / "nope.pkl"))

    def test_dispatch_unknown_and_pending_formats(self, tmp_path):
        with pytest.raises(CheckpointCodecError):
            load_checkpoint("whatever", expected_format="CLOSEST_LOOKING")
        for pending in (FORMAT_BAKEOFF_PKL, FORMAT_PHASE4A_PKL):
            with pytest.raises(CheckpointFormatNotImplementedError):
                load_checkpoint("whatever", expected_format=pending)
        path, sha, _ = self._synthetic_cc2_pkl(tmp_path)
        loaded = load_checkpoint(path, expected_format=FORMAT_CC2_PKL,
                                 expected_params_sha256=sha)
        assert loaded.params_sha256 == sha


EXPECTED_CANDIDATES = {
    "rmt16_persistent_98304": "PERSISTENT_RMT16_ORIGINAL_VTRACE_98304",
    "rmt16_reset128_98304": "RESET128_RMT16_ORIGINAL_VTRACE_98304",
    "gtrxl128_control_98304": "CONTROL_CONTINUOUS_98304",
    "baseline_teacher_17500_smoke": "BASELINE_TEACHER_CKPT17500",
    "slowgru_persistent_24576_compat": "SLOWGRU_PERSISTENT_24576",
}


class TestProfileLoading:
    def test_all_five_profiles_load_with_contracted_interface(self):
        profile_dir = default_profile_dir()
        files = sorted(profile_dir.glob("*.yaml"))
        assert {f.stem for f in files} == set(EXPECTED_CANDIDATES)
        for f in files:
            profile = load_student_profile(f)
            assert profile.candidate_id == EXPECTED_CANDIDATES[f.stem]
            assert profile.observation_shape == (8335,)
            assert profile.action_count == 43
            ident = profile.expected_identity()  # must not raise
            assert ident.params_sha256 == profile.params_sha256
            assert profile.memory_spec().spec_hash() == ident.memory_spec_hash

    def test_absolute_path_in_profile_rejected(self, tmp_path):
        text = (default_profile_dir() / "rmt16_persistent_98304.yaml").read_text(encoding="utf-8")
        bad = tmp_path / "bad.yaml"
        bad.write_text(text + "\nnotes_extra_marker: unused\n", encoding="utf-8")
        # unknown key must raise before anything else
        with pytest.raises(StudentRegistryError):
            load_student_profile(bad)
        bad2 = tmp_path / "bad2.yaml"
        bad2.write_text(text.replace("notes:", "server_ref: /home/oseasy/x\nnotes:"),
                        encoding="utf-8")
        with pytest.raises(StudentRegistryError):
            load_student_profile(bad2)

    def test_checkpoint_path_key_and_bad_format_rejected(self, tmp_path):
        text = (default_profile_dir() / "rmt16_persistent_98304.yaml").read_text(encoding="utf-8")
        bad = tmp_path / "withpath.yaml"
        bad.write_text("checkpoint_path: relative/only\n" + text, encoding="utf-8")
        with pytest.raises(StudentRegistryError):
            load_student_profile(bad)
        bad2 = tmp_path / "badformat.yaml"
        bad2.write_text(text.replace("checkpoint_format: CC2_PKL",
                                     "checkpoint_format: PROBABLY_ORBAX"), encoding="utf-8")
        with pytest.raises(StudentRegistryError):
            load_student_profile(bad2)

    def test_missing_required_key_rejected(self, tmp_path):
        text = (default_profile_dir() / "rmt16_persistent_98304.yaml").read_text(encoding="utf-8")
        bad = tmp_path / "missing.yaml"
        bad.write_text(text.replace("params_sha256: ", "params_sha_GONE: "), encoding="utf-8")
        with pytest.raises(StudentRegistryError):
            load_student_profile(bad)
