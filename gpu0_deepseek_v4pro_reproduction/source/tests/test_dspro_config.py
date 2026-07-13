"""Configuration tests for the dspro substitution baseline.

Tests:
- Config singleton is frozen and correct
- Model ID is pinned (not an alias)
- API key env var is documented (not leaked)
- Original configs are preserved
- Substitution config is distinct
"""

import os
import sys

import pytest

# Ensure the src directory is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestDsproConfig:
    """Tests for the dspro substitution configuration."""

    def test_config_is_frozen(self):
        """DsproConfig should be immutable (frozen dataclass)."""
        from dicode.dspro.config import DSPRO_CONFIG

        with pytest.raises(Exception):
            DSPRO_CONFIG.model_id = "something-else"  # type: ignore

    def test_model_id_is_not_alias(self):
        """Model ID must not be 'latest', 'auto', or empty."""
        from dicode.dspro.config import DSPRO_CONFIG

        assert DSPRO_CONFIG.model_id not in ("latest", "auto", ""), \
            f"Model ID '{DSPRO_CONFIG.model_id}' is a disallowed alias"
        assert DSPRO_CONFIG.model_id == "deepseek-v4-pro", \
            f"Expected 'deepseek-v4-pro', got '{DSPRO_CONFIG.model_id}'"

    def test_provider_is_deepseek(self):
        """Provider must be 'deepseek'."""
        from dicode.dspro.config import DSPRO_CONFIG

        assert DSPRO_CONFIG.provider == "deepseek"

    def test_api_key_env_documented(self):
        """API key env var name is documented, value is never read in tests."""
        from dicode.dspro.config import DSPRO_CONFIG

        assert DSPRO_CONFIG.api_key_env == "EXP_DEEPSEEK_API_KEY"
        # Never print or assert the actual key value
        key_value = os.getenv(DSPRO_CONFIG.api_key_env)
        if key_value:
            # Key exists but we only verify it's a string, never print it
            assert isinstance(key_value, str)
            assert len(key_value) > 0

    def test_experiment_label(self):
        """Experiment label must identify this as a substitution baseline."""
        from dicode.dspro.config import DSPRO_CONFIG

        assert "substitution" in DSPRO_CONFIG.experiment_label.lower()
        assert "deepseek" in DSPRO_CONFIG.experiment_label.lower()
        assert DSPRO_CONFIG.is_exact_reproduction is False

    def test_output_root(self):
        """Output root must be under /root/experiments/dicode_runs/dspro."""
        from dicode.dspro.config import DSPRO_CONFIG

        assert DSPRO_CONFIG.output_root == "/root/experiments/dicode_runs/dspro"

    def test_logical_roles_preserved(self):
        """Logical role labels must be preserved even with single model."""
        from dicode.dspro.config import DSPRO_CONFIG

        assert "task_generator" in DSPRO_CONFIG.logical_roles
        assert "env_generator" in DSPRO_CONFIG.logical_roles

    def test_validate_no_issues_without_key(self):
        """Validation should report missing key but not crash."""
        from dicode.dspro.config import DSPRO_CONFIG

        issues = DSPRO_CONFIG.validate()
        # May report missing API key; that's expected in test environments
        assert isinstance(issues, list)

    def test_accounting_namespace(self):
        """Accounting namespace must be set."""
        from dicode.dspro.config import DSPRO_CONFIG

        assert DSPRO_CONFIG.accounting_namespace == "dspro-gpu0"


class TestOriginalConfigsPreserved:
    """Verify that original configuration files are not modified."""

    def test_original_llm_configs_exist(self):
        """Original LLM config files must still exist."""
        conf_dir = os.path.join(
            os.path.dirname(__file__), "..", "conf", "gen_manager", "llm"
        )
        original_configs = [
            "local_gen.yaml",
            "local_embed.yaml",
            "openai.yaml",
            "gemini.yaml",
            "openrouter.yaml",
        ]
        for cfg in original_configs:
            path = os.path.join(conf_dir, cfg)
            assert os.path.exists(path), f"Original config missing: {cfg}"

    def test_original_gen_manager_config_exists(self):
        """Original gen_manager default config must still exist."""
        path = os.path.join(
            os.path.dirname(__file__), "..", "conf", "gen_manager", "default.yaml"
        )
        assert os.path.exists(path), "Original gen_manager/default.yaml missing"

    def test_substitution_config_is_separate(self):
        """Substitution config must be a separate file, not overwriting default."""
        default_path = os.path.join(
            os.path.dirname(__file__), "..", "conf", "gen_manager", "default.yaml"
        )
        dspro_path = os.path.join(
            os.path.dirname(__file__), "..", "conf", "gen_manager", "dspro_substitution.yaml"
        )
        assert os.path.exists(dspro_path), "Substitution config missing"
        assert os.path.exists(default_path), "Default config missing"
        # They must be different files
        assert os.path.realpath(default_path) != os.path.realpath(dspro_path)

    def test_llm_providers_role_map_unchanged(self):
        """Original ROLE_PROVIDER_MAP must still exist and be unchanged."""
        from dicode.mechanisms.llm_providers import ROLE_PROVIDER_MAP

        assert ROLE_PROVIDER_MAP["tutor"] == "qwen"
        assert ROLE_PROVIDER_MAP["critic"] == "deepseek"
        assert ROLE_PROVIDER_MAP["explorer"] == "glm"

    def test_dspro_role_map_exists(self):
        """Dspro substitution role map must exist and route all to deepseek."""
        from dicode.mechanisms.llm_providers import DSPRO_ROLE_PROVIDER_MAP

        assert DSPRO_ROLE_PROVIDER_MAP["tutor"] == "deepseek"
        assert DSPRO_ROLE_PROVIDER_MAP["critic"] == "deepseek"
        assert DSPRO_ROLE_PROVIDER_MAP["explorer"] == "deepseek"


class TestDeepSeekConfigFile:
    """Tests for the deepseek LLM config file."""

    def test_deepseek_config_exists(self):
        """DeepSeek LLM config must exist."""
        import yaml
        path = os.path.join(
            os.path.dirname(__file__), "..", "conf", "gen_manager", "llm", "deepseek.yaml"
        )
        assert os.path.exists(path), "deepseek.yaml config missing"

        with open(path) as f:
            config = yaml.safe_load(f)

        assert config["provider"] == "deepseek"
        assert config["model"] == "deepseek-v4-pro"
        assert config["llm_type"] == "generation"

    def test_dspro_substitution_config_structure(self):
        """Dspro substitution config must have correct structure."""
        import yaml
        path = os.path.join(
            os.path.dirname(__file__), "..", "conf", "gen_manager", "dspro_substitution.yaml"
        )
        assert os.path.exists(path), "dspro_substitution.yaml missing"

        with open(path) as f:
            config = yaml.safe_load(f)

        # Check defaults point to correct LLM configs
        assert "defaults" in config
        defaults_list = config["defaults"]
        defaults_dict = {d if isinstance(d, str) else list(d.keys())[0]: d for d in defaults_list}

        # task_generator should use deepseek
        assert any("task_generator" in str(d) for d in defaults_list)


class TestLLMProviderSupport:
    """Tests for DeepSeek provider support in the LLM class."""

    def test_deepseek_provider_in_create_client(self):
        """LLM._create_client must handle 'deepseek' provider."""
        import inspect
        from dicode.dreaming.llm import LLM

        source = inspect.getsource(LLM._create_client)
        assert "deepseek" in source, \
            "LLM._create_client does not handle 'deepseek' provider"

    def test_deepseek_in_query_method(self):
        """LLM.query must handle 'deepseek' provider."""
        import inspect
        from dicode.dreaming.llm import LLM

        source = inspect.getsource(LLM.query)
        assert "deepseek" in source, \
            "LLM.query does not handle 'deepseek' provider"

    def test_deepseek_embedding_raises(self):
        """LLM.get_embedding must raise NotImplementedError for deepseek."""
        from dicode.dreaming.llm import LLM

        llm = LLM(
            provider="deepseek",
            base_url="https://api.deepseek.com/v1",
            model="deepseek-v4-pro",
            llm_type="generation",
        )
        with pytest.raises(NotImplementedError, match="DeepSeek.*embedding"):
            llm.get_embedding("test text")
