"""Fail-closed tests for real Craftax EnvState restore and dynamics parity (R4a).

These tests run against the REAL minicraftax/craftax environment (no skips,
no importorskip): if craftax is missing the collection itself fails, which is
the honest fail-closed behaviour for this gate.

Scope honesty: a green run here proves ONLY the R4a env-side restore/parity.
It is not the R4c combined fresh-process proof and not a performance result.
"""

from __future__ import annotations

import dataclasses

import jax
import numpy as np
import pytest

from craftax.craftax.constants import Action

from dicode.simulator_frontier import craftax_checks as cc
from dicode.simulator_frontier.env_restore import (
    build_template,
    encode_env_state,
    flatten_env_state,
    restore_env_state,
    slice_env_state,
    stack_env_states,
)
from dicode.simulator_frontier.errors import SchemaMismatchError
from dicode.simulator_frontier.state_codec import StateCodec


RESET_SEED = 20260803
RUNNER_SEED = 777
ACTION_SEED = 0


@pytest.fixture(scope="module")
def core_setup():
    return cc.build_core_setup(max_timesteps=64, reset_seed=RESET_SEED)


@pytest.fixture(scope="module")
def eager_template(core_setup):
    _, state_ref = core_setup["env"].reset_env(
        jax.random.PRNGKey(RESET_SEED + 1), core_setup["params"])
    return build_template(state_ref)


# ---------------------------------------------------------------------------
# Class 1: normal restore round-trip (G1 + bootstrap + lineage discipline)
# ---------------------------------------------------------------------------

class TestRestoreRoundTrip:
    def test_bootstrap_gate_passes(self):
        boot = cc.bootstrap_environment()
        assert boot["pass"], boot
        assert boot["action_dim"] == len(Action) == 43
        assert not boot["x64_enabled"]
        assert boot["env_params_hashable"]

    def test_roundtrip_leaf_equality(self, core_setup, eager_template):
        encoded, _ = encode_env_state(
            core_setup["state0"], next_step_key=jax.random.PRNGKey(RUNNER_SEED),
            previous_action=0, previous_reward=0.0)
        restored = restore_env_state(encoded, eager_template)
        cmp = cc.compare_env_states(core_setup["state0"], restored.env_state)
        assert cmp["ok"], cmp["mismatched"][:5]
        assert cmp["n_leaves"] > 100  # real EnvState is large; guard against trivial states

    def test_payload_hash_is_deterministic(self, core_setup):
        a, _ = encode_env_state(core_setup["state0"], next_step_key=jax.random.PRNGKey(RUNNER_SEED),
                                previous_action=0, previous_reward=0.0)
        b, _ = encode_env_state(core_setup["state0"], next_step_key=jax.random.PRNGKey(RUNNER_SEED),
                                previous_action=0, previous_reward=0.0)
        assert a.payload_hash == b.payload_hash

    def test_none_leaves_survive_roundtrip(self, core_setup, eager_template):
        encoded, _ = encode_env_state(core_setup["state0"], next_step_key=jax.random.PRNGKey(RUNNER_SEED),
                                      previous_action=0, previous_reward=0.0)
        restored = restore_env_state(encoded, eager_template)
        assert restored.env_state.fractal_noise_angles == (None, None, None, None)
        none_paths = [p for p in eager_template.leaf_paths if p.startswith("fractal_noise_angles")]
        assert len(none_paths) == 4

    def test_template_lineage_is_enforced(self, core_setup, eager_template):
        # A jit-stepped state is a different lineage than the eager-reset template:
        # restore must fail closed instead of silently coercing leaf kinds.
        key = jax.random.PRNGKey(RUNNER_SEED)
        key, step_key = jax.random.split(key)
        _, stepped, _, _, _ = core_setup["step_fn"](
            step_key, core_setup["state0"], 0, core_setup["params"])
        encoded, _ = encode_env_state(stepped, next_step_key=key,
                                      previous_action=0, previous_reward=0.0)
        with pytest.raises(SchemaMismatchError):
            restore_env_state(encoded, eager_template)


# ---------------------------------------------------------------------------
# Class 2: dynamics parity (G2) and terminal restore (G3)
# ---------------------------------------------------------------------------

class TestDynamicsParity:
    def test_restore_then_rollout_is_bit_identical(self, core_setup):
        result = cc.check_dynamics_parity(core_setup, k_steps=40, restore_at=13, seed=RUNNER_SEED)
        assert result["capture_restore_ok"]
        assert result["first_divergence"] is None, result["first_divergence"]
        assert len(result["steps"]) == 40
        assert result["pass"]

    def test_terminal_state_restore_and_post_terminal_step(self):
        result = cc.check_terminal_restore(seed=RESET_SEED, max_timesteps=12)
        assert result["terminal_done"]
        assert result["terminal_restore_ok"]
        assert result["post_terminal_parity"]
        assert result["post_terminal_done_equal"]
        assert result["pass"]


# ---------------------------------------------------------------------------
# Class 3: corrupted payload and version drift are fail-closed (G5 + G6)
# ---------------------------------------------------------------------------

class TestCorruptedPayloadFailClosed:
    def test_every_corruption_case_is_rejected(self, core_setup):
        result = cc.check_corrupted_payload(core_setup)
        cases = {c["case"]: c for c in result["cases"]}
        expected = [
            "c1_hash_tamper",
            "c2_bitflip_no_rehash",
            "c3_bitflip_with_rehash_detected_by_parity",
            "c4_truncated_payload_with_rehash",
            "c5a_drop_leaf",
            "c5b_add_leaf",
            "c5c_wrong_shape",
            "c5d_wrong_dtype",
            "c5e_none_replaced_by_array",
        ]
        assert sorted(cases) == sorted(expected), sorted(cases)
        for name in expected:
            assert cases[name]["ok"], cases[name]

    def test_direct_hash_tamper_raises_schema_mismatch(self, core_setup, eager_template):
        encoded, _ = encode_env_state(core_setup["state0"], next_step_key=jax.random.PRNGKey(RUNNER_SEED),
                                      previous_action=0, previous_reward=0.0)
        tampered = dataclasses.replace(encoded, payload_hash="0" * 64)
        with pytest.raises(SchemaMismatchError):
            StateCodec().decode(tampered)

    def test_every_version_drift_case_is_rejected(self, core_setup):
        result = cc.check_version_mismatch(core_setup)
        cases = {c["case"]: c for c in result["cases"]}
        expected = ["v1_schema_version_swap", "v2_field_set_tamper",
                    "v3_treedef_fingerprint_drift", "v4_env_state_type_drift"]
        assert sorted(cases) == sorted(expected), sorted(cases)
        for name in expected:
            assert cases[name]["ok"], cases[name]
            assert cases[name]["raised"] == "SchemaMismatchError"


# ---------------------------------------------------------------------------
# Class 4: auto-reset terminal evidence chain (G4)
# ---------------------------------------------------------------------------

class TestAutoResetTerminalEvidence:
    def test_wrapper_replay_and_adapter_chain(self):
        result = cc.check_autoreset_evidence(seed=RESET_SEED, max_timesteps=8)
        assert result["done_detected"]
        assert result["replay_faithful"]
        assert result["returned_differs_from_terminal"]
        assert result["adapter_negative_raised"]
        assert result["goal_state_is_terminal"]
        assert result["terminal_restorable"]
        assert result["pass"]


# ---------------------------------------------------------------------------
# Batch slice/restore/stack parity (G7) and multitask secondary
# ---------------------------------------------------------------------------

class TestBatchAndMultitask:
    def test_batch_slice_restore_stack_parity(self, core_setup):
        result = cc.check_batch_parity(core_setup, batch=2, slice_index=1, steps=12, seed=RUNNER_SEED)
        assert result["capture_restore_ok"]
        assert result["first_divergence"] is None, result["first_divergence"]
        assert len(result["steps_detail"]) == 12
        assert result["pass"]

    def test_stack_then_slice_is_identity_for_leaf_values(self, core_setup):
        state0 = core_setup["state0"]
        batched = stack_env_states([state0, state0])
        sliced = slice_env_state(batched, 0)
        # stack turns python-scalar leaves into arrays: that is the documented
        # lineage change of batched states.  Values must still be bit-equal
        # after normalisation to arrays.
        flat_a = flatten_env_state(state0)
        flat_b = flatten_env_state(sliced)
        assert flat_a["leaf_paths"] == flat_b["leaf_paths"]
        for path in flat_a["leaf_paths"]:
            va, vb = flat_a["leaves"][path], flat_b["leaves"][path]
            if va is None:
                assert vb is None
            elif isinstance(va, float):
                # stack promotes python floats to jnp float32: exact round-trip
                # through float32 is the documented behaviour, not bit-equality
                # in float64.
                assert np.array_equal(np.asarray(vb), np.asarray(va, dtype="float32")), path
            else:
                assert np.array_equal(np.asarray(va), np.asarray(vb)), path

    def test_multitask_task0_roundtrip(self):
        result = cc.check_multitask_secondary(seed=RESET_SEED, steps=8, max_timesteps=64)
        assert result["restore_ok"]
        assert result["pass"]
