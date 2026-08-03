"""G1 tests: ReferenceIdentityContract — configurable, fail-closed, no guessing.

Every identity field is required with NO default; placeholder/wildcard
values are rejected as GUESSED; an unfrozen block blocks the whole
evaluation seam. Offline only; the module performs no file I/O.
"""
import pytest

from dicode.teachers.e1_formal import reference_contract as R

_SHA_FILE = "a" * 64
_SHA_PARAMS = "b" * 64
_SHA_ARCH = "c" * 64
_SHA_MEM = "d" * 64
_SHA_PROTO = "e" * 64
_SHA_MANIFEST = "f" * 64


def _block(**overrides):
    obj = {
        "frozen": True,
        "candidate_id": "REFERENCE_CANDIDATE_FROZEN_BY_SUPERVISOR",
        "checkpoint_ref": "cc4-registry://reference/frozen-v1",
        "file_sha256": _SHA_FILE,
        "params_sha256": _SHA_PARAMS,
        "architecture_family": "rmt16",
        "architecture_version": "reference_v1",
        "architecture_config_hash": _SHA_ARCH,
        "memory_semantics": "original craftax reset; no memory replay",
        "memory_semantics_hash": _SHA_MEM,
        "global_step": 131072,
        "source_commit": "9eca2de914068a33e500e2ad90d50f48e6e4e632",
        "seed": 42,
        "episode_reset_protocol_id": "standard_reset_v1",
        "episode_reset_protocol_hash": _SHA_PROTO,
        "frozen_manifest_hash": _SHA_MANIFEST,
    }
    obj.update(overrides)
    return obj


class TestFrozenGate:
    def test_unfrozen_block_rejected(self):
        with pytest.raises(R.ReferenceContractError) as excinfo:
            R.consume_reference_identity_contract(
                _block(frozen=False), "unit-test"
            )
        assert excinfo.value.code == "REFERENCE_CONTRACT_UNFROZEN"

    def test_missing_frozen_key_rejected(self):
        block = _block()
        del block["frozen"]
        with pytest.raises(R.ReferenceContractError) as excinfo:
            R.consume_reference_identity_contract(block, "unit-test")
        assert excinfo.value.code == "REFERENCE_CONTRACT_UNFROZEN"

    @pytest.mark.parametrize("bad", ["true", 1, None, "yes"])
    def test_frozen_must_be_literal_bool_true(self, bad):
        with pytest.raises(R.ReferenceContractError) as excinfo:
            R.consume_reference_identity_contract(_block(frozen=bad), "t")
        assert excinfo.value.code == "REFERENCE_CONTRACT_UNFROZEN"

    def test_non_mapping_rejected(self):
        with pytest.raises(R.ReferenceContractError) as excinfo:
            R.consume_reference_identity_contract(None, "t")
        assert excinfo.value.code == "REFERENCE_CONTRACT_BAD_TYPE"


class TestIdentityFields:
    def test_valid_contract_parses(self):
        c = R.consume_reference_identity_contract(_block(), "unit-test")
        assert c.candidate_id == "REFERENCE_CANDIDATE_FROZEN_BY_SUPERVISOR"
        assert c.global_step == 131072
        assert c.seed == 42
        assert c.schema_version == R.REFERENCE_CONTRACT_SCHEMA_VERSION

    def test_parse_is_deterministic_double_run(self):
        block = _block()
        assert R.consume_reference_identity_contract(
            block, "t"
        ) == R.consume_reference_identity_contract(block, "t")

    @pytest.mark.parametrize("field", sorted(R._REQUIRED_FIELDS))
    def test_missing_field_rejected(self, field):
        block = _block()
        del block[field]
        with pytest.raises(R.ReferenceContractError) as excinfo:
            R.consume_reference_identity_contract(block, "t")
        assert excinfo.value.code == "REFERENCE_CONTRACT_MISSING_FIELD"

    def test_unknown_field_rejected(self):
        with pytest.raises(R.ReferenceContractError) as excinfo:
            R.consume_reference_identity_contract(
                _block(extra_key="x"), "t"
            )
        assert excinfo.value.code == "REFERENCE_CONTRACT_UNKNOWN_FIELD"

    @pytest.mark.parametrize("field", sorted(R._STRING_FIELDS))
    def test_empty_string_rejected(self, field):
        with pytest.raises(R.ReferenceContractError) as excinfo:
            R.consume_reference_identity_contract(_block(**{field: "  "}), "t")
        assert excinfo.value.code == "REFERENCE_CONTRACT_EMPTY_FIELD"

    @pytest.mark.parametrize("field", sorted(R._STRING_FIELDS))
    @pytest.mark.parametrize(
        "placeholder",
        sorted(["TODO", "latest", "auto", "TBD", "unknown", "<fill-me>", "${REF_ID}"]),
    )
    def test_placeholder_values_rejected_as_guessed(self, field, placeholder):
        with pytest.raises(R.ReferenceContractError) as excinfo:
            R.consume_reference_identity_contract(
                _block(**{field: placeholder}), "t"
            )
        assert excinfo.value.code == "REFERENCE_CONTRACT_GUESSED_FORBIDDEN"

    @pytest.mark.parametrize("field", sorted(R._HASH_FIELDS))
    @pytest.mark.parametrize("bad", ["A" * 64, "z" * 64, "a" * 63])
    def test_bad_hash_rejected(self, field, bad):
        with pytest.raises(R.ReferenceContractError) as excinfo:
            R.consume_reference_identity_contract(_block(**{field: bad}), "t")
        assert excinfo.value.code == "REFERENCE_CONTRACT_BAD_HASH"

    @pytest.mark.parametrize("field", ["global_step", "seed"])
    def test_bool_counter_rejected_no_coercion(self, field):
        with pytest.raises(R.ReferenceContractError) as excinfo:
            R.consume_reference_identity_contract(_block(**{field: True}), "t")
        assert excinfo.value.code == "REFERENCE_CONTRACT_BAD_TYPE"

    @pytest.mark.parametrize("field", ["global_step", "seed"])
    def test_negative_counter_rejected(self, field):
        with pytest.raises(R.ReferenceContractError) as excinfo:
            R.consume_reference_identity_contract(_block(**{field: -3}), "t")
        assert excinfo.value.code == "REFERENCE_CONTRACT_BAD_STEP"

    def test_optional_total_env_steps_validated(self):
        c = R.consume_reference_identity_contract(
            _block(total_env_steps=1000), "t"
        )
        assert c.total_env_steps == 1000
        with pytest.raises(R.ReferenceContractError) as excinfo:
            R.consume_reference_identity_contract(
                _block(total_env_steps=-1), "t"
            )
        assert excinfo.value.code == "REFERENCE_CONTRACT_BAD_STEP"

    def test_formal_provenance_rejected(self):
        from dicode.teachers.static_llm.schemas import SchemaError

        with pytest.raises(SchemaError) as excinfo:
            R.consume_reference_identity_contract(
                _block(provenance="FORMAL_FULL"), "t"
            )
        assert excinfo.value.code == SchemaError.FORMAL_PROVENANCE_REJECTED


class TestManifestBinding:
    def test_manifest_bytes_matching_hash_passes(self):
        contract = R.consume_reference_identity_contract(
            _block(frozen_manifest_hash=R.sha256_hex(b"FROZEN MANIFEST v1")),
            "t",
        )
        R.verify_reference_manifest_bytes(contract, b"FROZEN MANIFEST v1", "t")

    def test_manifest_hash_mismatch_fails_closed(self):
        contract = R.consume_reference_identity_contract(_block(), "t")
        with pytest.raises(R.ReferenceContractError) as excinfo:
            R.verify_reference_manifest_bytes(contract, b"tampered", "t")
        assert excinfo.value.code == "REFERENCE_CONTRACT_MANIFEST_HASH_MISMATCH"


class TestNoIOSurface:
    def test_module_does_not_import_io_modules(self):
        import ast
        import os

        path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "src",
            "dicode",
            "teachers",
            "e1_formal",
            "reference_contract.py",
        )
        with open(path, "r", encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), filename=path)
        forbidden = {"os", "pathlib", "io", "jax", "torch", "pickle", "subprocess"}
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert imported & forbidden == set(), imported & forbidden
