"""Manifest system tests for the dspro substitution baseline.

Tests:
- Manifest creation and validation
- Output directory collision detection
- Manifest JSON serialization
- Schema version validation
"""

import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestManifestCreation:
    """Tests for manifest creation and basic validation."""

    def test_create_manifest_minimal(self):
        """Create a manifest with default parameters."""
        from dicode.dspro.manifest import create_manifest, validate_manifest
        from dicode.dspro.config import DSPRO_CONFIG

        manifest = create_manifest(
            run_id="test-run-001",
            seed=42,
            phase="G0",
            output_path="/tmp/test-dspro-output",
        )

        assert manifest["run_id"] == "test-run-001"
        assert manifest["seed"] == 42
        assert manifest["phase"] == "G0"
        assert manifest["provider"] == DSPRO_CONFIG.provider
        assert manifest["model_id"] == DSPRO_CONFIG.model_id
        assert manifest["manifest_schema_version"] == "1.0.0"

        errors = validate_manifest(manifest)
        assert len(errors) == 0, f"Manifest validation errors: {errors}"

    def test_create_manifest_with_extra_fields(self):
        """Create a manifest with extra custom fields."""
        from dicode.dspro.manifest import create_manifest

        manifest = create_manifest(
            run_id="test-run-002",
            seed=0,
            output_path="/tmp/test-dspro-output-2",
            custom_field="custom_value",
            another_field=123,
        )

        assert manifest["custom_field"] == "custom_value"
        assert manifest["another_field"] == 123

    def test_generate_run_id(self):
        """Run ID generation must produce unique IDs."""
        from dicode.dspro.manifest import generate_run_id

        id1 = generate_run_id()
        id2 = generate_run_id()

        assert id1 != id2
        assert id1.startswith("dspro-gpu0-")
        assert len(id1) > 20

    def test_manifest_required_fields(self):
        """Manifest must contain all required fields."""
        from dicode.dspro.manifest import (
            REQUIRED_MANIFEST_FIELDS,
            create_manifest,
            validate_manifest,
        )

        manifest = create_manifest(
            run_id="test-run-003",
            output_path="/tmp/test-dspro-output-3",
        )

        for field in REQUIRED_MANIFEST_FIELDS:
            assert field in manifest, f"Missing required field: {field}"

        errors = validate_manifest(manifest)
        assert len(errors) == 0, f"Validation errors: {errors}"


class TestManifestValidation:
    """Tests for manifest validation edge cases."""

    def test_missing_required_field(self):
        """Missing required field must be caught."""
        from dicode.dspro.manifest import validate_manifest

        manifest = {"run_id": "test"}  # Missing most required fields
        errors = validate_manifest(manifest)
        assert len(errors) > 0
        assert any("run_id" not in e for e in errors) or any("Missing" in e for e in errors)

    def test_disallowed_model_alias(self):
        """Manifest with 'latest' or 'auto' model must fail validation."""
        from dicode.dspro.manifest import create_manifest, validate_manifest

        manifest = create_manifest(
            run_id="test-run-004",
            output_path="/tmp/test-dspro-output-4",
        )
        manifest["model_id"] = "latest"
        errors = validate_manifest(manifest)
        assert len(errors) > 0
        assert any("alias" in e.lower() for e in errors)

    def test_wrong_provider(self):
        """Manifest with wrong provider must fail validation."""
        from dicode.dspro.manifest import create_manifest, validate_manifest

        manifest = create_manifest(
            run_id="test-run-005",
            output_path="/tmp/test-dspro-output-5",
        )
        manifest["provider"] = "openai"
        errors = validate_manifest(manifest)
        assert len(errors) > 0

    def test_wrong_model_id(self):
        """Manifest with wrong model_id must fail validation."""
        from dicode.dspro.manifest import create_manifest, validate_manifest

        manifest = create_manifest(
            run_id="test-run-006",
            output_path="/tmp/test-dspro-output-6",
        )
        manifest["model_id"] = "gpt-4"
        errors = validate_manifest(manifest)
        assert len(errors) > 0

    def test_wrong_schema_version(self):
        """Manifest with wrong schema version must fail."""
        from dicode.dspro.manifest import create_manifest, validate_manifest

        manifest = create_manifest(
            run_id="test-run-007",
            output_path="/tmp/test-dspro-output-7",
        )
        manifest["manifest_schema_version"] = "0.0.1"
        errors = validate_manifest(manifest)
        assert len(errors) > 0


class TestOutputSafety:
    """Tests for output directory collision detection."""

    def test_resolve_output_dir_no_collision(self):
        """Resolving a new output dir should succeed."""
        from dicode.dspro.manifest import resolve_output_dir

        # Use a temp subdirectory
        output_dir = resolve_output_dir(
            "test-no-collision-001",
        )
        # Should not exist yet
        assert not os.path.exists(output_dir)

    def test_create_output_dir(self):
        """Creating a new output dir should succeed."""
        import shutil
        from dicode.dspro.manifest import create_output_dir

        run_id = "test-create-dir-001"
        output_dir = create_output_dir(run_id)

        try:
            assert os.path.isdir(output_dir)
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_output_dir_collision_detected(self):
        """Re-creating an existing output dir must raise FileExistsError."""
        import shutil
        from dicode.dspro.manifest import create_output_dir

        run_id = "test-collision-001"
        output_dir = create_output_dir(run_id)

        try:
            with pytest.raises(FileExistsError):
                create_output_dir(run_id)
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_output_root_isolation(self):
        """Output must be under the dspro root."""
        from dicode.dspro.config import DSPRO_CONFIG
        from dicode.dspro.manifest import resolve_output_dir

        output_dir = resolve_output_dir("test-isolation-001")

        assert output_dir.startswith(DSPRO_CONFIG.output_root), \
            f"Output {output_dir} not under {DSPRO_CONFIG.output_root}"


class TestManifestSerialization:
    """Tests for manifest JSON write/read."""

    def test_write_and_read_manifest(self):
        """Write manifest to temp dir and read it back."""
        import shutil
        from dicode.dspro.manifest import (
            create_manifest,
            read_manifest,
            write_manifest,
        )

        tmpdir = tempfile.mkdtemp()
        try:
            manifest = create_manifest(
                run_id="test-serialize-001",
                output_path=tmpdir,
            )
            manifest_path = write_manifest(manifest, tmpdir)
            assert os.path.exists(manifest_path)

            # Read back
            loaded = read_manifest(manifest_path)
            assert loaded["run_id"] == manifest["run_id"]
            assert loaded["model_id"] == manifest["model_id"]
            assert loaded["provider"] == manifest["provider"]
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_manifest_json_is_valid(self):
        """Manifest must be valid JSON."""
        import shutil
        from dicode.dspro.manifest import create_manifest, write_manifest

        tmpdir = tempfile.mkdtemp()
        try:
            manifest = create_manifest(
                run_id="test-json-001",
                output_path=tmpdir,
            )
            manifest_path = write_manifest(manifest, tmpdir)

            with open(manifest_path) as f:
                data = json.load(f)

            assert isinstance(data, dict)
            assert "run_id" in data
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_compute_artifact_hash(self):
        """Artifact hash computation must be deterministic."""
        from dicode.dspro.manifest import compute_artifact_hash

        tmpdir = tempfile.mkdtemp()
        try:
            path = os.path.join(tmpdir, "test.txt")
            with open(path, "w") as f:
                f.write("test content")

            hash1 = compute_artifact_hash(path)
            hash2 = compute_artifact_hash(path)
            assert hash1 == hash2
            assert len(hash1) == 64  # SHA-256
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)
