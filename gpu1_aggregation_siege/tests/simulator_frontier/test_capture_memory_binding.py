# -*- coding: utf-8 -*-
"""TEST_ONLY / SYNTHETIC fixtures.  NOT_REAL_EXECUTION.

CC4 follow-up (P0-3): a frontier capture is never detached from its memory.
``encode_env_state`` MUST forward the live rollout policy memory
(SAVED_POLICY_MEMORY) or the recorded rollout history reference
(HISTORY_BURN_IN) into the encoded bundle; a capture that omits them leaves
the bundle fields empty and the production archive guard chain refuses the
write.  These tests pin the forwarding contract of the reachable surface.
"""

import pytest

from dicode.simulator_frontier.env_restore import (
    encode_env_state,
    make_state_bundle,
)
from dicode.simulator_frontier.state_codec import StateCodec

SYNTHETIC_FIXTURE_NOT_SCIENTIFIC_CONTENT = True

_MEMORY = {"h": [[0.25, 0.5]]}
_HISTORY = {"mode": "HISTORY_BURN_IN", "history_length": 3, "steps": [1, 2, 3]}


def _state():
    import jax.numpy as jnp
    return {"pos": jnp.zeros((2,), dtype=jnp.float32), "t": 0}


class TestCaptureMemoryBinding:
    def test_policy_memory_forwarded_into_bundle(self):
        import jax
        encoded, bundle = encode_env_state(
            _state(), next_step_key=jax.random.PRNGKey(0),
            previous_action=0, previous_reward=0.0,
            policy_memory=_MEMORY)
        assert bundle.policy_memory == _MEMORY
        decoded = StateCodec().decode(encoded)
        assert decoded.policy_memory == _MEMORY

    def test_history_reference_forwarded_into_bundle(self):
        import jax
        encoded, bundle = encode_env_state(
            _state(), next_step_key=jax.random.PRNGKey(0),
            previous_action=1, previous_reward=0.5,
            history_reference=_HISTORY)
        assert bundle.history_reference == _HISTORY
        decoded = StateCodec().decode(encoded)
        assert decoded.history_reference == _HISTORY

    def test_omitted_memory_leaves_bundle_fields_empty(self):
        import jax
        _encoded, bundle = encode_env_state(
            _state(), next_step_key=jax.random.PRNGKey(0),
            previous_action=0, previous_reward=0.0)
        # A capture that omits the mode-conditional memory surface must leave
        # BOTH fields empty — the archive guard chain then refuses the write
        # (never a silent default, never a shared object).
        assert bundle.policy_memory is None
        assert bundle.history_reference is None

    def test_make_state_bundle_does_not_share_memory_across_calls(self):
        import jax
        b1 = make_state_bundle(_state(), next_step_key=jax.random.PRNGKey(0),
                               previous_action=0, previous_reward=0.0,
                               policy_memory=dict(_MEMORY))
        b2 = make_state_bundle(_state(), next_step_key=jax.random.PRNGKey(0),
                               previous_action=0, previous_reward=0.0,
                               policy_memory=dict(_MEMORY))
        # Distinct bundle objects and distinct memory dictionaries — no shared
        # mutable memory object crosses bundles.
        assert b1 is not b2
        assert b1.policy_memory is not b2.policy_memory

    def test_fake_mapping_codec_rejected(self):
        import jax
        with pytest.raises((AttributeError, TypeError)):
            encode_env_state(_state(), next_step_key=jax.random.PRNGKey(0),
                             previous_action=0, previous_reward=0.0,
                             codec={"encode": lambda x: x})
