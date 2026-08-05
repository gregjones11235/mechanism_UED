# -*- coding: utf-8 -*-
"""TEST_ONLY / SYNTHETIC fixtures.  NOT_REAL_EXECUTION.

CC4 follow-up (P0-11): the compiled distributions' seed / stochasticity /
taskparams fields EXECUTE.  ``resolve_distribution_binding`` consumes them
into an immutable per-episode binding; empty, unknown, malformed or
out-of-range content is refused — a distribution whose fields cannot execute
is never silently rolled out.
"""

import dataclasses

import pytest

from dicode.simulator_frontier.distribution_runtime import (
    DISTRIBUTION_RUNTIME_VERSION,
    DistributionRuntimeBinding,
    resolve_distribution_binding,
    verify_distribution_binding,
)
from dicode.simulator_frontier.errors import InvalidEvidenceError
from dicode.simulator_frontier.frontier_distributions import FrontierDistribution

SYNTHETIC_FIXTURE_NOT_SCIENTIFIC_CONTENT = True


def _distribution(*, seed=None, stochasticity=None, taskparams=None,
                  distribution_id="plan::D00") -> FrontierDistribution:
    return FrontierDistribution(
        distribution_id=distribution_id,
        bucket=(1, "mid", "low", "mid", "stone", False),
        eligible_states=("s1", "s2"),
        start_state_weights={"s1": 0.6, "s2": 0.4},
        taskparam_ranges={"max_timesteps": 128} if taskparams is None else taskparams,
        seed_distribution={"kind": "canonical", "base": 7} if seed is None else seed,
        stochasticity_range={"epsilon": 0.1} if stochasticity is None else stochasticity,
        memory_mode="SAVED_POLICY_MEMORY",
        goal_family="FRONTIER:LEARNABLE_FRONTIER",
        evidence_hash="a" * 64,
        retention_constraint="anchor_ratio>=0.250000",
    )


class TestResolveExecutesTheFields:
    def test_seed_distribution_executes_into_episode_seed(self):
        binding = resolve_distribution_binding(
            _distribution(), episode_index=3, seed_base=42)
        verify_distribution_binding(binding)
        assert binding.runtime_version == DISTRIBUTION_RUNTIME_VERSION
        assert 0 <= binding.episode_seed < 2 ** 31
        # The seed is a canonical function of the seed distribution + identity.
        other = resolve_distribution_binding(
            _distribution(), episode_index=4, seed_base=42)
        assert other.episode_seed != binding.episode_seed

    def test_stochasticity_executes_into_epsilon_temperature(self):
        binding = resolve_distribution_binding(
            _distribution(stochasticity={"epsilon": [0.0, 0.25],
                                         "temperature": [0.5, 2.0]}),
            episode_index=0, seed_base=1)
        assert binding.epsilon == 0.0 and binding.temperature == 0.5
        # Defaults when a key is absent: epsilon=0.0, temperature=1.0.
        temperature_only = resolve_distribution_binding(
            _distribution(stochasticity={"temperature": 1.5}),
            episode_index=0, seed_base=1)
        assert temperature_only.epsilon == 0.0 and temperature_only.temperature == 1.5

    def test_taskparams_execute_as_pass_through(self):
        binding = resolve_distribution_binding(
            _distribution(taskparams={"max_timesteps": 64, "level": 2}),
            episode_index=0, seed_base=1)
        assert dict(binding.taskparams) == {"max_timesteps": 64, "level": 2}

    def test_deterministic_remint(self):
        a = resolve_distribution_binding(_distribution(), episode_index=0, seed_base=7)
        b = resolve_distribution_binding(_distribution(), episode_index=0, seed_base=7)
        assert a.binding_hash == b.binding_hash

    def test_mint_only_binding_hash(self):
        with pytest.raises(TypeError):
            DistributionRuntimeBinding(
                distribution_id="x", episode_index=0, episode_seed=1,
                epsilon=0.0, temperature=1.0, taskparams={"a": 1},
                binding_hash="f" * 64)


class TestFieldsCannotBeSilentlySkipped:
    def test_empty_seed_distribution_refused(self):
        with pytest.raises(InvalidEvidenceError):
            resolve_distribution_binding(
                _distribution(seed={}), episode_index=0, seed_base=1)

    def test_empty_stochasticity_refused(self):
        with pytest.raises(InvalidEvidenceError):
            resolve_distribution_binding(
                _distribution(stochasticity={}), episode_index=0, seed_base=1)

    def test_unknown_stochasticity_key_refused(self):
        with pytest.raises(InvalidEvidenceError):
            resolve_distribution_binding(
                _distribution(stochasticity={"noise_scale": 0.5}),
                episode_index=0, seed_base=1)

    def test_empty_taskparams_refused(self):
        with pytest.raises(InvalidEvidenceError):
            resolve_distribution_binding(
                _distribution(taskparams={}), episode_index=0, seed_base=1)

    def test_out_of_range_stochasticity_refused(self):
        with pytest.raises(InvalidEvidenceError):
            resolve_distribution_binding(
                _distribution(stochasticity={"epsilon": 1.5}),
                episode_index=0, seed_base=1)
        with pytest.raises(InvalidEvidenceError):
            resolve_distribution_binding(
                _distribution(stochasticity={"temperature": 0.0}),
                episode_index=0, seed_base=1)
        with pytest.raises(InvalidEvidenceError):
            resolve_distribution_binding(
                _distribution(stochasticity={"epsilon": [0.5, 0.1]}),
                episode_index=0, seed_base=1)

    def test_negative_episode_index_and_bool_seed_refused(self):
        with pytest.raises(InvalidEvidenceError):
            resolve_distribution_binding(
                _distribution(), episode_index=-1, seed_base=1)
        with pytest.raises(InvalidEvidenceError):
            resolve_distribution_binding(
                _distribution(), episode_index=0, seed_base=True)


class TestFakeMappingRefused:
    def test_mapping_distribution_refused(self):
        with pytest.raises(InvalidEvidenceError):
            resolve_distribution_binding(
                {"distribution_id": "x"}, episode_index=0, seed_base=1)

    def test_foreign_distribution_refused(self):
        with pytest.raises(InvalidEvidenceError):
            resolve_distribution_binding("distribution", episode_index=0, seed_base=1)

    def test_mapping_binding_refused_by_verify(self):
        with pytest.raises(InvalidEvidenceError):
            verify_distribution_binding({"binding_hash": "f" * 64})

    def test_foreign_binding_refused_by_verify(self):
        with pytest.raises(InvalidEvidenceError):
            verify_distribution_binding("binding")

    def test_tampered_binding_refused_by_verify(self):
        binding = resolve_distribution_binding(
            _distribution(), episode_index=0, seed_base=1)
        verify_distribution_binding(binding)
        tampered = dataclasses.replace(binding)
        object.__setattr__(tampered, "epsilon", 0.99)
        with pytest.raises(InvalidEvidenceError):
            verify_distribution_binding(tampered)
