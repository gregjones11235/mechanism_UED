"""Integration tests for the dspro substitution baseline production path.

Verifies that call_role_judge(use_dspro_substitution=True) routes ALL three
roles (tutor, critic, explorer) through deepseek-v4-pro, preserves logical
role labels, and never uses legacy aliases.

Uses a fake transport (mocked call_llm_api) to avoid real API calls.
No API key required. No network access.
"""

import json
import os
import sys
from unittest.mock import MagicMock, patch

# Ensure source is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ==============================================================================
# Fake API response — simulates a successful deepseek-v4-pro completion
# ==============================================================================

FAKE_TASK = {
    "task_id": "task_test_001",
    "description": "Collect 5 wood and craft a pickaxe",
    "source": "generated",
    "skills": "collecting, crafting",
    "recent_success": 0.75,
    "best_success": 0.82,
}


def fake_api_success(provider_name, messages, model=None, max_tokens=256, temperature=0.0, timeout=60, _override_max_tokens=False):
    """Fake call_llm_api that returns a valid JSON judgment."""
    # Extract role from the user prompt for verification.
    # In dspro mode, messages[0] may be a system message; look for the
    # user message that contains the role-specific prompt.
    role = "unknown"
    for msg in messages:
        content = msg.get("content", "")
        for r in ("tutor", "critic", "explorer"):
            if f'"role":"{r}"' in content:
                role = r
                break
        if role != "unknown":
            break

    fake_judgment = {
        "task_id": "task_test_001",
        "role": role,
        "provider": provider_name,
        "model": model or "unknown",
        "scores": {"progression_score": 0.7, "learnability_score": 0.6} if role == "tutor"
        else {"critic_penalty": 0.1} if role == "critic"
        else {"novelty_score": 0.8, "diversity_score": 0.5},
        "flags": {},
        "skill_tag": "test_skill",
        "decision": "accept",
        "short_reason": "Test judgment",
    }
    return {
        "success": True,
        "content": json.dumps(fake_judgment),
        "provider": provider_name,
        "model": model or "deepseek-v4-pro",
        "returned_model": model or "deepseek-v4-pro",
        "finish_reason": "stop",
        "input_tokens_est": 50,
        "output_tokens_est": 100,
        "estimated_cost": 0.000035,
    }


def fake_api_empty():
    """Fake call_llm_api that returns empty content."""
    return {
        "success": True,
        "content": "",
        "provider": "deepseek",
        "model": "deepseek-v4-pro",
        "returned_model": "deepseek-v4-pro",
        "finish_reason": "stop",
        "input_tokens_est": 10,
        "output_tokens_est": 0,
        "estimated_cost": 0.0,
    }


# ==============================================================================
# Tests
# ==============================================================================

class TestDsproProductionPath:
    """Verify call_role_judge with use_dspro_substitution=True."""

    @patch("dicode.mechanisms.llm_roles.call_llm_api", side_effect=fake_api_success)
    def test_tutor_routes_to_deepseek_v4_pro(self, mock_api):
        """Tutor role must route to deepseek, deepseek-v4-pro."""
        from dicode.mechanisms.llm_roles import call_role_judge

        result = call_role_judge(
            role="tutor",
            task_summary=FAKE_TASK,
            use_dspro_substitution=True,
        )

        assert result["success"], f"Tutor call failed: {result.get('error')}"
        assert result["provider"] == "deepseek", \
            f"Expected provider 'deepseek', got '{result['provider']}'"
        assert result["judgment"]["role"] == "tutor", \
            f"Expected role 'tutor', got '{result['judgment']['role']}'"

        # Verify the API was called with correct provider and model
        call_args = mock_api.call_args
        assert call_args is not None, "call_llm_api was not called"
        assert call_args[1]["provider_name"] == "deepseek", \
            f"API called with wrong provider: {call_args[1]['provider_name']}"
        assert call_args[1]["model"] == "deepseek-v4-pro", \
            f"API called with wrong model: {call_args[1]['model']}"

    @patch("dicode.mechanisms.llm_roles.call_llm_api", side_effect=fake_api_success)
    def test_critic_routes_to_deepseek_v4_pro(self, mock_api):
        """Critic role must route to deepseek, deepseek-v4-pro."""
        from dicode.mechanisms.llm_roles import call_role_judge

        result = call_role_judge(
            role="critic",
            task_summary=FAKE_TASK,
            use_dspro_substitution=True,
        )

        assert result["success"], f"Critic call failed: {result.get('error')}"
        assert result["provider"] == "deepseek"
        assert result["judgment"]["role"] == "critic"

        call_args = mock_api.call_args
        assert call_args[1]["provider_name"] == "deepseek"
        assert call_args[1]["model"] == "deepseek-v4-pro"

    @patch("dicode.mechanisms.llm_roles.call_llm_api", side_effect=fake_api_success)
    def test_explorer_routes_to_deepseek_v4_pro(self, mock_api):
        """Explorer role must route to deepseek, deepseek-v4-pro."""
        from dicode.mechanisms.llm_roles import call_role_judge

        result = call_role_judge(
            role="explorer",
            task_summary=FAKE_TASK,
            use_dspro_substitution=True,
        )

        assert result["success"], f"Explorer call failed: {result.get('error')}"
        assert result["provider"] == "deepseek"
        assert result["judgment"]["role"] == "explorer"

        call_args = mock_api.call_args
        assert call_args[1]["provider_name"] == "deepseek"
        assert call_args[1]["model"] == "deepseek-v4-pro"


class TestDsproNoLegacyAliases:
    """Verify no role uses legacy aliases."""

    @patch("dicode.mechanisms.llm_roles.call_llm_api", side_effect=fake_api_success)
    def test_no_legacy_alias_in_tutor(self, mock_api):
        """Tutor must NOT use deepseek-chat or deepseek-reasoner."""
        from dicode.mechanisms.llm_roles import call_role_judge

        call_role_judge(role="tutor", task_summary=FAKE_TASK, use_dspro_substitution=True)
        call_args = mock_api.call_args
        model = call_args[1]["model"]
        assert model not in ("deepseek-chat", "deepseek-reasoner"), \
            f"Legacy alias '{model}' used for tutor!"

    @patch("dicode.mechanisms.llm_roles.call_llm_api", side_effect=fake_api_success)
    def test_no_legacy_alias_in_critic(self, mock_api):
        """Critic must NOT use deepseek-chat or deepseek-reasoner."""
        from dicode.mechanisms.llm_roles import call_role_judge

        call_role_judge(role="critic", task_summary=FAKE_TASK, use_dspro_substitution=True)
        call_args = mock_api.call_args
        model = call_args[1]["model"]
        assert model not in ("deepseek-chat", "deepseek-reasoner"), \
            f"Legacy alias '{model}' used for critic!"

    @patch("dicode.mechanisms.llm_roles.call_llm_api", side_effect=fake_api_success)
    def test_no_legacy_alias_in_explorer(self, mock_api):
        """Explorer must NOT use deepseek-chat or deepseek-reasoner."""
        from dicode.mechanisms.llm_roles import call_role_judge

        call_role_judge(role="explorer", task_summary=FAKE_TASK, use_dspro_substitution=True)
        call_args = mock_api.call_args
        model = call_args[1]["model"]
        assert model not in ("deepseek-chat", "deepseek-reasoner"), \
            f"Legacy alias '{model}' used for explorer!"


class TestDsproRoleLabelPreservation:
    """Verify logical role labels preserved even with single underlying model."""

    @patch("dicode.mechanisms.llm_roles.call_llm_api", side_effect=fake_api_success)
    def test_all_three_roles_preserved_in_response(self, mock_api):
        """All three roles must preserve their logical labels."""
        from dicode.mechanisms.llm_roles import call_role_judge

        roles_tested = []
        for role in ("tutor", "critic", "explorer"):
            result = call_role_judge(
                role=role,
                task_summary=FAKE_TASK,
                use_dspro_substitution=True,
            )
            assert result["success"], f"{role} failed: {result.get('error')}"
            assert result["judgment"]["role"] == role, \
                f"Role label mismatch: expected '{role}', got '{result['judgment']['role']}'"
            assert result["provider"] == "deepseek", \
                f"Provider mismatch for {role}: {result['provider']}"
            roles_tested.append(role)

        assert roles_tested == ["tutor", "critic", "explorer"], \
            f"Not all roles tested: {roles_tested}"

        # Verify all three mock calls used deepseek-v4-pro
        for call_args in mock_api.call_args_list:
            assert call_args[1]["provider_name"] == "deepseek"
            assert call_args[1]["model"] == "deepseek-v4-pro"


class TestDsproBackwardCompatibility:
    """Verify original behavior unchanged when use_dspro_substitution=False."""

    @patch("dicode.mechanisms.llm_roles.call_llm_api", side_effect=fake_api_success)
    def test_original_tutor_still_uses_qwen(self, mock_api):
        """Without dspro flag, tutor should still use original qwen provider."""
        from dicode.mechanisms.llm_roles import call_role_judge

        result = call_role_judge(
            role="tutor",
            task_summary=FAKE_TASK,
            use_dspro_substitution=False,
        )
        assert result["success"]
        call_args = mock_api.call_args
        assert call_args[1]["provider_name"] == "qwen", \
            f"Original tutor should use qwen, got: {call_args[1]['provider_name']}"

    @patch("dicode.mechanisms.llm_roles.call_llm_api", side_effect=fake_api_success)
    def test_original_explorer_still_uses_glm(self, mock_api):
        """Without dspro flag, explorer should still use original glm provider."""
        from dicode.mechanisms.llm_roles import call_role_judge

        result = call_role_judge(
            role="explorer",
            task_summary=FAKE_TASK,
            use_dspro_substitution=False,
        )
        assert result["success"]
        call_args = mock_api.call_args
        assert call_args[1]["provider_name"] == "glm", \
            f"Original explorer should use glm, got: {call_args[1]['provider_name']}"


class TestDsproApiKeyResolution:
    """Verify EXP_DEEPSEEK_API_KEY is in the key resolution chain."""

    def test_exp_deepseek_key_in_alt_keys(self):
        """EXP_DEEPSEEK_API_KEY must be in deepseek's alternate key list."""
        from dicode.mechanisms.llm_providers import get_api_key

        # Check that the alt keys include EXP_DEEPSEEK_API_KEY
        # We verify by reading the source of get_api_key
        import inspect
        source = inspect.getsource(get_api_key)
        assert "EXP_DEEPSEEK_API_KEY" in source, \
            "EXP_DEEPSEEK_API_KEY not in deepseek key resolution chain"

    def test_get_api_key_deepseek_checks_exp_var(self):
        """get_api_key for deepseek should find EXP_DEEPSEEK_API_KEY."""
        from dicode.mechanisms.llm_providers import get_api_key

        # With a dummy key set, verify resolution works
        os.environ["EXP_DEEPSEEK_API_KEY"] = "sk-test-dspro-integration"
        try:
            key = get_api_key("deepseek")
            assert key == "sk-test-dspro-integration", \
                f"get_api_key didn't resolve EXP_DEEPSEEK_API_KEY: got {key}"
        finally:
            del os.environ["EXP_DEEPSEEK_API_KEY"]


class TestDsproConfigIntegration:
    """Verify DsproConfig integrates with the production path."""

    @patch("dicode.mechanisms.llm_roles.call_llm_api", side_effect=fake_api_success)
    def test_dspro_config_model_matches_production(self, mock_api):
        """DsproConfig.model_id must match what call_role_judge sends."""
        from dicode.dspro.config import DSPRO_CONFIG
        from dicode.mechanisms.llm_roles import call_role_judge

        call_role_judge(role="tutor", task_summary=FAKE_TASK, use_dspro_substitution=True)
        call_args = mock_api.call_args
        assert call_args[1]["model"] == DSPRO_CONFIG.model_id, \
            f"Config model '{DSPRO_CONFIG.model_id}' != production model '{call_args[1]['model']}'"

    def test_legacy_aliases_in_config(self):
        """DsproConfig must list deepseek-chat as a legacy alias."""
        from dicode.dspro.config import DSPRO_CONFIG
        assert "deepseek-chat" in DSPRO_CONFIG.legacy_aliases
        assert "deepseek-reasoner" in DSPRO_CONFIG.legacy_aliases

    def test_dspro_pinned_model_in_providers(self):
        """DSPRO_PINNED_MODEL must match DsproConfig.model_id."""
        from dicode.dspro.config import DSPRO_CONFIG
        from dicode.mechanisms.llm_providers import DSPRO_PINNED_MODEL
        assert DSPRO_PINNED_MODEL == DSPRO_CONFIG.model_id, \
            f"DSPRO_PINNED_MODEL={DSPRO_PINNED_MODEL} != config={DSPRO_CONFIG.model_id}"
        assert DSPRO_PINNED_MODEL == "deepseek-v4-pro"


# ==============================================================================
# Standalone runner
# ==============================================================================

if __name__ == "__main__":
    passed = 0
    failed = 0
    errors = []

    tests = [
        # Production path tests
        ("tutor→deepseek-v4-pro", TestDsproProductionPath().test_tutor_routes_to_deepseek_v4_pro),
        ("critic→deepseek-v4-pro", TestDsproProductionPath().test_critic_routes_to_deepseek_v4_pro),
        ("explorer→deepseek-v4-pro", TestDsproProductionPath().test_explorer_routes_to_deepseek_v4_pro),
        # No legacy aliases
        ("no legacy alias tutor", TestDsproNoLegacyAliases().test_no_legacy_alias_in_tutor),
        ("no legacy alias critic", TestDsproNoLegacyAliases().test_no_legacy_alias_in_critic),
        ("no legacy alias explorer", TestDsproNoLegacyAliases().test_no_legacy_alias_in_explorer),
        # Role label preservation
        ("all 3 roles preserved", TestDsproRoleLabelPreservation().test_all_three_roles_preserved_in_response),
        # Backward compatibility
        ("original tutor→qwen", TestDsproBackwardCompatibility().test_original_tutor_still_uses_qwen),
        ("original explorer→glm", TestDsproBackwardCompatibility().test_original_explorer_still_uses_glm),
        # API key resolution
        ("exp key in alt keys", TestDsproApiKeyResolution().test_exp_deepseek_key_in_alt_keys),
        ("get_api_key deepseek", TestDsproApiKeyResolution().test_get_api_key_deepseek_checks_exp_var),
        # Config integration
        ("config model matches prod", TestDsproConfigIntegration().test_dspro_config_model_matches_production),
        ("legacy aliases in config", TestDsproConfigIntegration().test_legacy_aliases_in_config),
        ("DSPRO_PINNED_MODEL consistency", TestDsproConfigIntegration().test_dspro_pinned_model_in_providers),
    ]

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
            print(f"  PASS: {name}")
        except AssertionError as e:
            failed += 1
            msg = f"  FAIL: {name} — {e}"
            print(msg)
            errors.append(msg)
        except Exception as e:
            failed += 1
            msg = f"  ERROR: {name} — {e}"
            print(msg)
            errors.append(msg)
            import traceback
            traceback.print_exc()

    print(f"\nIntegration tests: {passed} passed, {failed} failed")
    if failed:
        print("FAILURES:")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
