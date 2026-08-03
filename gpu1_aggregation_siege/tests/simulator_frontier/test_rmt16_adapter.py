"""Stage 3 tests (R4b): vendored RMT16 provenance + read-only adapter gates.

Hermetic contract tests: the checkpoint used here is a SYNTHETIC CC2-format
pkl whose params come from a random ``network.init`` (never a real
checkpoint, never performance data); the driver source is the committed
repo archive copy.  The real-checkpoint mount runs in the mount driver
(scripts/run_student_mount_smoke.py), not in pytest.
"""

from __future__ import annotations

import pickle
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dicode.student_adapters.architectures.rmt16_provenance import (
    ARCHIVE_DRIVER_LF_SHA256,
    ARCHIVE_DRIVER_RAW_SHA256,
    ARCHIVE_ROOT_RELATIVE,
    FROZEN_RMT16_CFG,
    REQUIRED_CFG_FIELDS,
    VENDORED_FILES,
    DriverSourceError,
    load_rmt16_cfg_from_driver_source,
    verify_frozen_cfg,
)
from dicode.student_adapters.checkpoint_codec import (
    CheckpointCodecError,
    cc2_params_sha256,
    file_sha256,
)
from dicode.student_adapters.protocol import MemoryFieldSpec, StudentAdapter
from dicode.student_adapters.rmt16_adapter import (
    MEMORY_FIELD_KEYS,
    RMT16MountError,
    RMT16StudentAdapter,
)
from dicode.student_adapters.registry import (
    StudentAdapterRegistry,
    StudentProfile,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
ARCHIVE_ROOT = REPO_ROOT / ARCHIVE_ROOT_RELATIVE
ARCHIVE_DRIVER_PATH = ARCHIVE_ROOT / "train_rmt16_p2replay.py"


# --- provenance: byte-fidelity of the vendored subset ------------------------

class TestVendoredProvenance:
    def test_archive_sources_match_recorded_shas(self):
        import hashlib

        for vendored, record in VENDORED_FILES.items():
            blob = (ARCHIVE_ROOT / record["source_file"]).read_bytes()
            assert hashlib.sha256(blob).hexdigest() == record["source_raw_sha256"], vendored
            assert hashlib.sha256(blob.replace(b"\r\n", b"\n")).hexdigest() \
                == record["source_lf_sha256"], vendored
        driver = ARCHIVE_DRIVER_PATH.read_bytes()
        assert hashlib.sha256(driver).hexdigest() == ARCHIVE_DRIVER_RAW_SHA256
        assert hashlib.sha256(driver.replace(b"\r\n", b"\n")).hexdigest() \
            == ARCHIVE_DRIVER_LF_SHA256

    def test_vendored_network_and_memory_are_byte_identical(self):
        here = Path(__file__).resolve().parents[2] / "src" / "dicode" / "student_adapters" / "architectures"
        assert (here / "rmt16_network.py").read_bytes() \
            == (ARCHIVE_ROOT / "network_rmt16.py").read_bytes()
        assert (here / "rmt16_memory.py").read_bytes() \
            == (ARCHIVE_ROOT / "rmt16_memory.py").read_bytes()

    def test_vendored_anchor_is_archive_minus_the_recorded_import_line(self):
        import hashlib

        here = Path(__file__).resolve().parents[2] / "src" / "dicode" / "student_adapters" / "architectures"
        archive_text = (ARCHIVE_ROOT / "rmt_memory_anchor.py").read_text(encoding="utf-8")
        lines = archive_text.splitlines(keepends=True)
        dropped = [l for l in lines if l.startswith("import memory_anchor")]
        assert len(dropped) == 1
        assert "MA." not in archive_text.replace(dropped[0], "")
        # apply the two recorded import fixes (and nothing else)
        expected = "".join(l for l in lines if not l.startswith("import memory_anchor"))
        assert expected.count("import rmt16_memory as rmtm") == 1
        expected = expected.replace("import rmt16_memory as rmtm",
                                    "from . import rmt16_memory as rmtm", 1)
        vendored = (here / "rmt16_anchor.py").read_text(encoding="utf-8")
        assert vendored == expected
        assert hashlib.sha256(vendored.encode("utf-8")).hexdigest() \
            == VENDORED_FILES["rmt16_anchor.py"]["vendored_sha256"]


# --- frozen Cfg recovery from the SHA-bound driver source --------------------

class TestCfgRecovery:
    def test_archive_driver_recovers_frozen_cfg(self):
        cfg, sha = load_rmt16_cfg_from_driver_source(
            str(ARCHIVE_DRIVER_PATH), ARCHIVE_DRIVER_LF_SHA256)
        assert sha == ARCHIVE_DRIVER_LF_SHA256
        assert all(f in cfg for f in REQUIRED_CFG_FIELDS)
        verify_frozen_cfg(cfg)  # must not raise
        assert {k: cfg[k] for k in FROZEN_RMT16_CFG} == FROZEN_RMT16_CFG

    def test_wrong_expected_sha_raises(self):
        with pytest.raises(DriverSourceError):
            load_rmt16_cfg_from_driver_source(str(ARCHIVE_DRIVER_PATH), "0" * 64)

    def test_missing_driver_raises(self):
        with pytest.raises(DriverSourceError):
            load_rmt16_cfg_from_driver_source("/nonexistent/driver.py", "0" * 64)

    def test_non_literal_cfg_value_raises(self, tmp_path):
        bad = tmp_path / "driver.py"
        bad.write_text("class Cfg:\n    embed_size = int(256)\n", encoding="utf-8")
        with pytest.raises(DriverSourceError):
            load_rmt16_cfg_from_driver_source(str(bad), _sha_of(bad))

    def test_missing_cfg_class_or_field_raises(self, tmp_path):
        nocls = tmp_path / "nocls.py"
        nocls.write_text("X = 1\n", encoding="utf-8")
        with pytest.raises(DriverSourceError):
            load_rmt16_cfg_from_driver_source(str(nocls), _sha_of(nocls))
        partial = tmp_path / "partial.py"
        fields = "\n".join(f"    {k} = {v!r}" for k, v in FROZEN_RMT16_CFG.items()
                           if k != "window_mem")
        partial.write_text(f"class Cfg:\n{fields}\n", encoding="utf-8")
        with pytest.raises(DriverSourceError):
            load_rmt16_cfg_from_driver_source(str(partial), _sha_of(partial))

    def test_verify_frozen_cfg_clash_raises(self):
        bad = dict(FROZEN_RMT16_CFG)
        bad["embed_size"] = 999
        with pytest.raises(DriverSourceError):
            verify_frozen_cfg(bad)


def _sha_of(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


# --- hermetic synthetic CC2 pipeline fixture ---------------------------------

MEMORY_FIELDS = {
    "memories": MemoryFieldSpec(shape=(None, 128, 2, 256), dtype="float32"),
    "mem_mask": MemoryFieldSpec(shape=(None, 8, 1, 129), dtype="bool"),
    "mem_idx": MemoryFieldSpec(shape=(None,), dtype="int32"),
    "rmt.mem_tokens": MemoryFieldSpec(shape=(None, 16, 256), dtype="float32"),
    "rmt.seg_buf": MemoryFieldSpec(shape=(None, 128, 256), dtype="float32"),
    "rmt.seg_count": MemoryFieldSpec(shape=(None,), dtype="int32"),
}


def _write_cc2_pkl(path: Path, params, manifest: dict) -> str:
    with open(path, "wb") as fh:
        pickle.dump({"params": params, "manifest": manifest}, fh, protocol=4)
    return file_sha256(str(path))


def _manifest_for(params_sha: str, *, step=7, carry_mode="persistent",
                  config=None, segment_len=128) -> dict:
    return {
        "params_sha256": params_sha, "step": step,
        "arm": "SYNTHETIC-CONTRACT-TEST", "carry_mode": carry_mode,
        "replay_mode": "original_vtrace", "seed": 0,
        "config": {} if config is None else config,
        "phase4a_v2": {"run_class": "contract_test", "sequence_length": 129,
                       "segment_len": segment_len, "crosses_boundary": True,
                       "replay_mode": "original_vtrace"},
    }


def _profile_for(params_sha: str, file_sha: str, *, memory_mode="PERSISTENT",
                 carry="persistent") -> StudentProfile:
    return StudentProfile(
        profile_name="synthetic_rmt16_contract",
        candidate_id="SYNTHETIC_RMT16_CONTRACT_TEST",
        architecture_family="RMT16",
        checkpoint_format="CC2_PKL",
        global_step=7,
        total_env_steps=7,
        params_sha256=params_sha,
        source_commit="src-sha256:" + ARCHIVE_DRIVER_LF_SHA256,
        observation_shape=(8335,),
        action_count=43,
        memory_mode=memory_mode,
        memory_fields=dict(MEMORY_FIELDS),
        notes={"carry_mode": carry, "checkpoint_file_sha256": file_sha},
    )


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    """Random-init params via network.init (contract fixture, NOT training)."""
    import jax
    import jax.numpy as jnp
    from dicode.student_adapters.architectures import rmt16_memory, rmt16_network

    cfg, driver_sha = load_rmt16_cfg_from_driver_source(
        str(ARCHIVE_DRIVER_PATH), ARCHIVE_DRIVER_LF_SHA256)
    network = rmt16_network.ActorCriticTransformerRMT16(
        action_dim=43, activation=cfg["activation"], encoder_size=cfg["embed_size"],
        hidden_layers=cfg["hidden_layers"], num_heads=cfg["num_heads"],
        qkv_features=cfg["qkv_features"], num_layers=cfg["num_layers"],
        gating=cfg["gating"], gating_bias=cfg["gating_bias"],
        rmt_num_tokens=cfg["rmt_num_tokens"])
    ref = network.init(
        jax.random.PRNGKey(0),
        jnp.zeros((2, cfg["window_mem"], cfg["num_layers"], cfg["embed_size"])),
        jnp.zeros((2, 8335)),
        jnp.zeros((2, cfg["num_heads"], 1, cfg["window_mem"] + 1), jnp.bool_),
        mem_tokens=jnp.zeros((2, cfg["rmt_num_tokens"], cfg["embed_size"])),
        seg_buf=jnp.zeros((2, cfg["num_steps"], cfg["embed_size"])),
        method=network.init_all)
    from flax.core import unfreeze
    params = jax.tree_util.tree_map(np.asarray, unfreeze(ref["params"]))
    params_sha = cc2_params_sha256(params)
    out = tmp_path_factory.mktemp("rmt16_contract")
    path = out / "synthetic_full_state.pkl"
    file_sha = _write_cc2_pkl(path, params, _manifest_for(params_sha))
    return SimpleNamespace(path=str(path), dir=out, params=params,
                           params_sha=params_sha, file_sha=file_sha, cfg=cfg,
                           driver_sha=driver_sha)


def _adapter(profile) -> RMT16StudentAdapter:
    return RMT16StudentAdapter(
        profile, driver_source_path=str(ARCHIVE_DRIVER_PATH),
        expected_driver_sha256=ARCHIVE_DRIVER_LF_SHA256)


# --- adapter gates on the synthetic pipeline ----------------------------------

class TestLoadGatesHermetic:
    def test_registry_resolves_and_protocol_holds(self, synthetic):
        profile = _profile_for(synthetic.params_sha, synthetic.file_sha)
        reg = StudentAdapterRegistry()
        reg.register(profile.candidate_id,
                     lambda p: _adapter(p))
        adapter = reg.resolve(profile)
        assert isinstance(adapter, StudentAdapter)
        assert adapter.identity().params_sha256 == synthetic.params_sha
        assert adapter.observation_spec().shape == (8335,)
        assert adapter.action_spec().count == 43
        spec = adapter.checkpoint_spec()
        assert spec.contains_optimizer is False and spec.contains_rng is False
        assert spec.contains_memory is False

    def test_load_full_state_passes_all_gates(self, synthetic):
        profile = _profile_for(synthetic.params_sha, synthetic.file_sha)
        loaded = _adapter(profile).load_full_state(synthetic.path,
                                                   profile.expected_identity())
        assert loaded["params_sha256"] == synthetic.params_sha
        assert loaded["file_sha256"] == synthetic.file_sha
        assert loaded["global_step"] == 7
        assert loaded["carry_mode"] == "persistent"
        assert loaded["contains_optimizer"] is False
        assert loaded["contains_rng"] is False
        assert loaded["contains_policy_memory"] is False
        gates = loaded["gates"]
        assert set(gates) == {"G0_identity", "G1_driver_cfg", "G2_file_sha256",
                              "G3_params_sha256", "G4_manifest", "G5_structure",
                              "G6_absent_components"}
        assert gates["G5_structure"]["observation_dim"] == 8335
        assert gates["G5_structure"]["action_count"] == 43
        assert gates["G5_structure"]["craftax_len_action"] == 43
        assert gates["G5_structure"]["encoder_kernel_shape"][0] == 8335
        assert "ABSENT_IN_CHECKPOINT" in loaded["r4c_joint_proof_status"]["optimizer"]

    def test_foreign_identity_raises(self, synthetic):
        profile = _profile_for(synthetic.params_sha, synthetic.file_sha)
        foreign = replace(profile.expected_identity(), params_sha256="f" * 64)
        with pytest.raises(RMT16MountError):
            _adapter(profile).load_full_state(synthetic.path, foreign)

    def test_file_sha_gate_raises_on_tamper(self, synthetic):
        profile = _profile_for(synthetic.params_sha, synthetic.file_sha)
        tampered = Path(synthetic.dir) / "tampered.pkl"
        tampered.write_bytes(Path(synthetic.path).read_bytes() + b"0")
        with pytest.raises(RMT16MountError):
            _adapter(profile).load_full_state(str(tampered),
                                              profile.expected_identity())

    def test_params_sha_gate_raises_on_payload_mutation(self, synthetic):
        profile = _profile_for(synthetic.params_sha, synthetic.file_sha)

        def _mutate(tree):
            return {k: (_mutate(v) if isinstance(v, dict) else np.asarray(v) + 1.0)
                    for k, v in tree.items()}

        mutated = _mutate(synthetic.params)
        path = Path(synthetic.dir) / "mutated.pkl"
        file_sha = _write_cc2_pkl(path, mutated, _manifest_for(synthetic.params_sha))
        bad_profile = _profile_for(synthetic.params_sha, file_sha)
        with pytest.raises(CheckpointCodecError):
            _adapter(bad_profile).load_full_state(str(path),
                                                  bad_profile.expected_identity())

    @pytest.mark.parametrize("breakage,manifest_kwargs", [
        ("carry", dict(carry_mode="reset128")),
        ("step", dict(step=8)),
        ("config_clash", dict(config={"embed_size": 999})),
        ("segment_len", dict(segment_len=64)),
    ])
    def test_manifest_gates_raise(self, synthetic, breakage, manifest_kwargs):
        path = Path(synthetic.dir) / f"manifest_{breakage}.pkl"
        file_sha = _write_cc2_pkl(path, synthetic.params,
                                  _manifest_for(synthetic.params_sha, **manifest_kwargs))
        profile = _profile_for(synthetic.params_sha, file_sha)
        with pytest.raises(RMT16MountError):
            _adapter(profile).load_full_state(str(path), profile.expected_identity())

    @pytest.mark.parametrize("field,bad", [
        ("architecture_family", "GTRXL128"),
        ("checkpoint_format", "ORBAX_FLAT_TRAINSTATE"),
        ("memory_mode", "CONTINUOUS"),
    ])
    def test_constructor_rejects_wrong_family(self, synthetic, field, bad):
        profile = _profile_for(synthetic.params_sha, synthetic.file_sha)
        bad_profile = replace(profile, **{field: bad})
        with pytest.raises(RMT16MountError):
            _adapter(bad_profile)

    def test_constructor_requires_carry_mode_note(self, synthetic):
        profile = _profile_for(synthetic.params_sha, synthetic.file_sha)
        stripped = replace(profile, notes={"checkpoint_file_sha256": synthetic.file_sha})
        with pytest.raises(RMT16MountError):
            _adapter(stripped)

    def test_load_without_driver_source_fails_closed(self, synthetic):
        profile = _profile_for(synthetic.params_sha, synthetic.file_sha)
        adapter = RMT16StudentAdapter(profile)  # driver source never configured
        with pytest.raises(RMT16MountError):
            adapter.load_full_state(synthetic.path, profile.expected_identity())


# --- memory contract -----------------------------------------------------------

class TestMemoryContract:
    def test_initial_memory_shapes_and_validate(self, synthetic):
        profile = _profile_for(synthetic.params_sha, synthetic.file_sha)
        adapter = _adapter(profile)
        adapter.load_full_state(synthetic.path, profile.expected_identity())
        mem = adapter.initial_memory(2)
        assert tuple(sorted(mem)) == tuple(sorted(MEMORY_FIELD_KEYS))
        assert mem["memories"].shape == (2, 128, 2, 256)
        assert mem["mem_mask"].shape == (2, 8, 1, 129) and mem["mem_mask"].dtype == np.bool_
        assert (mem["mem_idx"] == 128).all() and mem["mem_idx"].dtype == np.int32
        assert mem["rmt.mem_tokens"].shape == (2, 16, 256)
        assert mem["rmt.seg_buf"].shape == (2, 128, 256)
        assert mem["rmt.seg_count"].shape == (2,)
        assert adapter.validate_memory(mem, 2)["ok"] is True

    def test_validate_negatives(self, synthetic):
        profile = _profile_for(synthetic.params_sha, synthetic.file_sha)
        adapter = _adapter(profile)
        adapter.load_full_state(synthetic.path, profile.expected_identity())
        mem = adapter.initial_memory(2)
        assert adapter.validate_memory(adapter.initial_memory(3), 2)["ok"] is False
        missing = {k: v for k, v in mem.items() if k != "mem_idx"}
        assert adapter.validate_memory(missing, 2)["ok"] is False
        extra = dict(mem)
        extra["surprise"] = np.zeros(1)
        assert adapter.validate_memory(extra, 2)["ok"] is False
        bad_dtype = dict(mem)
        bad_dtype["memories"] = mem["memories"].astype(np.float64)
        assert adapter.validate_memory(bad_dtype, 2)["ok"] is False
        bad_idx = dict(mem)
        bad_idx["mem_idx"] = np.full((2,), 129, np.int32)
        assert adapter.validate_memory(bad_idx, 2)["ok"] is False
        nan_mem = dict(mem)
        nan_mem["rmt.mem_tokens"] = np.full((2, 16, 256), np.nan, np.float32)
        assert adapter.validate_memory(nan_mem, 2)["ok"] is False

    def test_memory_ops_before_load_fail_closed(self, synthetic):
        profile = _profile_for(synthetic.params_sha, synthetic.file_sha)
        adapter = _adapter(profile)
        with pytest.raises(RMT16MountError):
            adapter.initial_memory(1)
        assert adapter.validate_memory({}, 1)["ok"] is False


# --- read-only forward (zero update) -------------------------------------------

class TestForwardNoUpdate:
    def _loaded(self, synthetic):
        profile = _profile_for(synthetic.params_sha, synthetic.file_sha)
        adapter = _adapter(profile)
        loaded = adapter.load_full_state(synthetic.path, profile.expected_identity())
        return adapter, loaded

    def test_deterministic_reproducible_and_zero_update(self, synthetic):
        adapter, loaded = self._loaded(synthetic)
        params = loaded["params"]
        sha_before = cc2_params_sha256(params)
        rng_obs = np.random.default_rng(20260803)
        for batch in (1, 4):
            obs = rng_obs.normal(size=(batch, 8335)).astype(np.float32)
            mem = adapter.initial_memory(batch)
            out1 = adapter.policy_step(params, obs, mem, None, None, None, True)
            out2 = adapter.policy_step(params, obs, mem, None, None, None, True)
            assert out1["action"].shape == (batch,)
            assert out1["logits"].shape == (batch, 43)
            assert np.isfinite(out1["logits"]).all() and np.isfinite(out1["value"]).all()
            assert ((out1["action"] >= 0) & (out1["action"] < 43)).all()
            assert np.array_equal(out1["action"], out2["action"])
            assert np.array_equal(out1["logits"], out2["logits"])
        assert cc2_params_sha256(params) == sha_before  # ZERO update

    def test_single_obs_shapes(self, synthetic):
        adapter, loaded = self._loaded(synthetic)
        obs = np.random.default_rng(1).normal(size=8335).astype(np.float32)
        out = adapter.policy_step(loaded["params"], obs, adapter.initial_memory(1),
                                  None, None, None, True)
        assert isinstance(out["action"], int) and 0 <= out["action"] < 43
        assert out["logits"].shape == (43,)

    def test_stochastic_seeded_reproducible(self, synthetic):
        adapter, loaded = self._loaded(synthetic)
        obs = np.random.default_rng(2).normal(size=(4, 8335)).astype(np.float32)
        mem = adapter.initial_memory(4)
        a1 = adapter.policy_step(loaded["params"], obs, mem, None, None,
                                 np.random.default_rng(7), False)["action"]
        a2 = adapter.policy_step(loaded["params"], obs, mem, None, None,
                                 np.random.default_rng(7), False)["action"]
        assert np.array_equal(a1, a2)
        assert ((a1 >= 0) & (a1 < 43)).all()
        with pytest.raises(RMT16MountError):
            adapter.policy_step(loaded["params"], obs, mem, None, None, None, False)

    def test_memory_progression_matches_rmt16_contract(self, synthetic):
        adapter, loaded = self._loaded(synthetic)
        params = loaded["params"]
        obs = np.random.default_rng(3).normal(size=(1, 8335)).astype(np.float32)
        mem = adapter.initial_memory(1)
        for step in range(1, 6):
            out = adapter.policy_step(params, obs, mem, None, None, None, True)
            mem = out["memory"]
            assert int(np.asarray(mem["rmt.seg_count"])[0]) == step
            # before the 128-step segment boundary mem_tokens stay untouched
            assert np.all(np.asarray(mem["rmt.mem_tokens"]) == 0.0)
            assert int(np.asarray(mem["mem_idx"])[0]) == max(128 - step, 0)
        # window memory rolled: last slot carries the fresh transformer output
        assert np.any(np.asarray(mem["memories"]) != 0.0)

    def test_policy_step_before_load_raises(self, synthetic):
        profile = _profile_for(synthetic.params_sha, synthetic.file_sha)
        adapter = _adapter(profile)
        with pytest.raises(RMT16MountError):
            adapter.policy_step({}, np.zeros(8335, np.float32), {}, None, None, None, True)

    def test_bad_obs_rejected(self, synthetic):
        adapter, loaded = self._loaded(synthetic)
        with pytest.raises(RMT16MountError):
            adapter.policy_step(loaded["params"], np.zeros(10, np.float32),
                                adapter.initial_memory(1), None, None, None, True)
        with pytest.raises(RMT16MountError):
            adapter.policy_step(loaded["params"],
                                np.full(8335, np.nan, np.float32),
                                adapter.initial_memory(1), None, None, None, True)


# --- training surface is explicitly PENDING ------------------------------------

class TestTrainingSurfacePending:
    def test_save_and_restore_raise_not_implemented(self, synthetic):
        profile = _profile_for(synthetic.params_sha, synthetic.file_sha)
        adapter = _adapter(profile)
        with pytest.raises(NotImplementedError):
            adapter.save_full_state("x.pkl", {}, {})
        with pytest.raises(NotImplementedError):
            adapter.restore_full_state(synthetic.path)
