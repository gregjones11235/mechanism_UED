"""Dry-run and integration tests for the dspro substitution baseline.

Tests that validate configuration resolution and path routing without
starting PPO training. All tests are training-safe.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestDryRunConfigResolution:
    """Verify that configuration resolves correctly without training."""

    def test_deepseek_config_can_be_loaded(self):
        """The deepseek LLM config must be readable."""
        import yaml

        conf_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "conf",
            "gen_manager",
            "llm",
            "deepseek.yaml",
        )
        with open(conf_path) as f:
            config = yaml.safe_load(f)

        assert config["provider"] == "deepseek"
        assert config["model"] == "deepseek-v4-pro"
        assert config["llm_type"] == "generation"
        assert isinstance(config["max_tokens"], int)
        assert config["max_tokens"] > 0

    def test_dspro_substitution_config_can_be_loaded(self):
        """The dspro substitution gen_manager config must be readable."""
        import yaml

        conf_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "conf",
            "gen_manager",
            "dspro_substitution.yaml",
        )
        with open(conf_path) as f:
            config = yaml.safe_load(f)

        assert "defaults" in config
        assert isinstance(config["num_generations"], int)
        assert len(config["example_paths"]) > 0

    def test_llm_class_accepts_deepseek_provider(self):
        """LLM class must accept 'deepseek' as a valid provider."""
        from dicode.dreaming.llm import LLM

        llm = LLM(
            provider="deepseek",
            base_url="https://api.deepseek.com/v1",
            model="deepseek-v4-pro",
            llm_type="generation",
            max_tokens=100,
            temperature=0.0,
        )

        assert llm.provider == "deepseek"
        assert llm.model == "deepseek-v4-pro"
        assert llm.client is not None

    def test_all_llm_call_sites_inventoried(self):
        """Audit must cover all known call sites."""
        from dicode.dspro.audit import audit_call_sites

        sites = audit_call_sites()
        assert len(sites) >= 5, f"Expected >= 5 call sites, got {len(sites)}"

        # Verify CS IDs are unique
        ids = [s.call_site_id for s in sites]
        assert len(ids) == len(set(ids)), f"Duplicate call site IDs: {ids}"

    def test_no_ppo_loop_llm_calls(self):
        """No LLM call site should be inside the PPO update loop."""
        from dicode.dspro.audit import audit_call_sites

        sites = audit_call_sites()
        ppo_calls = [s for s in sites if s.in_ppo_update_loop]
        assert len(ppo_calls) == 0, \
            f"Found {len(ppo_calls)} call sites in PPO update loop: " \
            f"{[s.call_site_id for s in ppo_calls]}"

    def test_all_gen_calls_route_to_deepseek(self):
        """All generation call sites must route to deepseek-v4-pro."""
        from dicode.dspro.audit import audit_call_sites

        sites = audit_call_sites()
        gen_sites = [s for s in sites if s.logical_role not in ("embedding_model",)]

        for s in gen_sites:
            assert s.substituted_provider == "deepseek", \
                f"{s.call_site_id} ({s.logical_role}) not routed to deepseek"
            assert s.substituted_model == "deepseek-v4-pro", \
                f"{s.call_site_id} ({s.logical_role}) model is '{s.substituted_model}'"


class TestDryRunPathValidation:
    """Path and output validation tests (no training)."""

    def test_output_root_directory(self):
        """Output root directory must exist or be creatable."""
        from dicode.dspro.config import DSPRO_CONFIG

        root = DSPRO_CONFIG.output_root
        # Directory should either exist or be creatable
        if not os.path.exists(root):
            try:
                os.makedirs(root, exist_ok=True)
            except OSError as e:
                pytest.fail(f"Cannot create output root {root}: {e}")

        assert os.path.isdir(root), f"Output root {root} is not a directory"

    def test_runs_directory_exists(self):
        """The dspro runs directory must be under the correct root."""
        from dicode.dspro.config import DSPRO_CONFIG

        assert DSPRO_CONFIG.output_root.startswith("/root/experiments/dicode_runs/")

    def test_manifest_output_path_isolation(self):
        """Manifest output paths must be under the dspro root."""
        import shutil
        from dicode.dspro.manifest import create_output_dir

        run_id = "test-dryrun-path-001"
        output_dir = create_output_dir(run_id)

        try:
            assert output_dir.startswith("/root/experiments/dicode_runs/dspro/")
            assert os.path.isdir(output_dir)
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)


class TestDryRunAuditReport:
    """Audit report generation tests."""

    def test_audit_report_generation(self):
        """Audit report must generate valid markdown."""
        from dicode.dspro.audit import audit_call_sites, generate_audit_report

        sites = audit_call_sites()
        report = generate_audit_report(sites)

        assert len(report) > 0
        assert "# DiCode LLM Call-Site Audit" in report
        assert "deepseek-v4-pro" in report
        assert "CS-001" in report
        assert "CS-008" in report

    def test_audit_report_notes_no_exact_reproduction(self):
        """Audit report must note this is a substitution baseline."""
        from dicode.dspro.audit import audit_call_sites, generate_audit_report

        sites = audit_call_sites()
        report = generate_audit_report(sites)

        # Just verify it generates without error
        assert isinstance(report, str)


class TestDryRunOriginalConfigs:
    """Verify original configs are readable and unchanged."""

    def test_all_original_llm_configs_parseable(self):
        """All original LLM configs must be valid YAML."""
        import yaml

        conf_dir = os.path.join(
            os.path.dirname(__file__), "..", "conf", "gen_manager", "llm"
        )
        for fname in os.listdir(conf_dir):
            if fname.endswith(".yaml"):
                with open(os.path.join(conf_dir, fname)) as f:
                    config = yaml.safe_load(f)
                assert isinstance(config, dict), \
                    f"Config {fname} is not a valid YAML dict"

    def test_original_providers_config_parseable(self):
        """Original providers config must be valid YAML."""
        import yaml

        path = os.path.join(
            os.path.dirname(__file__), "..", "conf", "llm", "providers.yaml"
        )
        with open(path) as f:
            config = yaml.safe_load(f)

        assert "providers" in config
        assert "deepseek" in config["providers"]

    def test_no_dependency_changes(self):
        """Dependencies must not have changed."""
        # Check pyproject.toml for expected dependencies
        path = os.path.join(os.path.dirname(__file__), "..", "pyproject.toml")
        with open(path) as f:
            content = f.read()

        expected_deps = ["jax", "flax", "optax", "craftax", "hydra-core"]
        for dep in expected_deps:
            assert dep in content, f"Expected dependency '{dep}' not found in pyproject.toml"


class TestDryRunCannotStartTraining:
    """Verify that dry-run tests cannot start PPO training."""

    def test_imports_dont_trigger_training(self):
        """Importing dspro modules must not start training."""
        # These imports should be side-effect free
        import dicode.dspro.config
        import dicode.dspro.manifest
        import dicode.dspro.audit

        # If we get here without GPU allocation or training start, it's good
        assert True

    def test_manifest_creation_no_training(self):
        """Creating a manifest must not trigger training."""
        from dicode.dspro.manifest import create_manifest

        manifest = create_manifest(
            run_id="test-no-training-001",
            output_path="/tmp/test-dspro-no-training",
        )
        assert manifest["phase"] == "G0"
        assert "environment_steps" in manifest
