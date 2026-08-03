"""C2 tests: StudentInitContract identity-only thin consumer.

Asserts the fail-closed consumer contract AND the structural property
that checkpoint loading is inexpressible: no path field exists and the
module imports no jax/flax/optax/orbax/pickle/torch. Offline only.
"""
import ast
import os

import pytest

from dicode.teachers.static_llm import student_init_contract as C
from dicode.teachers.static_llm import schemas as S

_SHA_A = "a" * 64
_SHA_B = "0123456789abcdef" * 4


def _contract(**overrides):
    obj = {
        "candidate_id": C.PINNED_STUDENT_CANDIDATE_ID,
        "architecture_family": "rmt16",
        "architecture_version": "original_vtrace",
        "checkpoint_format": "orbax_zip_v1",
        "checkpoint_global_step": 98304,
        "total_env_steps": 12_582_912,
        "source_commit": "9eca2de914068a33e500e2ad90d50f48e6e4e632",
        "parameter_tree_hash": _SHA_A,
        "optimizer_tree_hash": _SHA_B,
        "adapter_id": "cc4_student_adapter",
        "adapter_version": "v1",
    }
    obj.update(overrides)
    return obj


class TestConsumeContract:
    def test_valid_contract_parses(self):
        c = C.consume_student_init_contract(_contract(), "unit-test")
        assert c.candidate_id == C.PINNED_STUDENT_CANDIDATE_ID
        assert c.checkpoint_global_step == 98304
        assert c.parameter_tree_hash == _SHA_A
        assert c.provenance is None
        assert c.schema_version == C.CONTRACT_SCHEMA_VERSION

    def test_parse_is_deterministic_double_run(self):
        payload = _contract(provenance="TRAINING")
        assert C.consume_student_init_contract(
            payload, "t"
        ) == C.consume_student_init_contract(payload, "t")

    @pytest.mark.parametrize("field", sorted(C._REQUIRED_FIELDS))
    def test_missing_field_rejected(self, field):
        payload = _contract()
        del payload[field]
        with pytest.raises(C.StudentContractError) as excinfo:
            C.consume_student_init_contract(payload, "unit-test")
        assert excinfo.value.code == "STUDENT_CONTRACT_MISSING_FIELD"

    @pytest.mark.parametrize("field", sorted(C._STRING_FIELDS))
    def test_empty_string_field_rejected(self, field):
        with pytest.raises(C.StudentContractError) as excinfo:
            C.consume_student_init_contract(_contract(**{field: "   "}), "t")
        assert excinfo.value.code == "STUDENT_CONTRACT_EMPTY_FIELD"

    def test_non_mapping_rejected(self):
        with pytest.raises(C.StudentContractError) as excinfo:
            C.consume_student_init_contract(["not", "mapping"], "t")
        assert excinfo.value.code == "STUDENT_CONTRACT_BAD_TYPE"

    @pytest.mark.parametrize("field", sorted(C._STEP_FIELDS))
    def test_bool_step_rejected_no_coercion(self, field):
        with pytest.raises(C.StudentContractError) as excinfo:
            C.consume_student_init_contract(_contract(**{field: True}), "t")
        assert excinfo.value.code == "STUDENT_CONTRACT_BAD_TYPE"

    @pytest.mark.parametrize("field", sorted(C._STEP_FIELDS))
    def test_negative_step_rejected(self, field):
        with pytest.raises(C.StudentContractError) as excinfo:
            C.consume_student_init_contract(_contract(**{field: -1}), "t")
        assert excinfo.value.code == "STUDENT_CONTRACT_BAD_STEP"

    @pytest.mark.parametrize("field", sorted(C._HASH_FIELDS))
    @pytest.mark.parametrize("bad", ["A" * 64, "g" * 64, "a" * 63, 12])
    def test_bad_hash_rejected(self, field, bad):
        with pytest.raises(C.StudentContractError) as excinfo:
            C.consume_student_init_contract(_contract(**{field: bad}), "t")
        assert excinfo.value.code == "STUDENT_CONTRACT_BAD_HASH"

    def test_unknown_field_rejected_fail_closed(self):
        with pytest.raises(C.StudentContractError) as excinfo:
            C.consume_student_init_contract(
                _contract(checkpoint_path="/tmp/ckpt"), "t"
            )
        assert excinfo.value.code == "STUDENT_CONTRACT_UNKNOWN_FIELD"

    def test_formal_provenance_rejected(self):
        with pytest.raises(S.SchemaError) as excinfo:
            C.consume_student_init_contract(
                _contract(provenance="FORMAL_FULL"), "t"
            )
        assert excinfo.value.code == S.SchemaError.FORMAL_PROVENANCE_REJECTED

    @pytest.mark.parametrize("label", sorted(["TRAINING", "NORMAL_TRAINING_FEEDBACK"]))
    def test_admissible_provenance_accepted(self, label):
        c = C.consume_student_init_contract(_contract(provenance=label), "t")
        assert c.provenance == label


class TestPinnedCandidate:
    def test_pinned_candidate_passes(self):
        c = C.consume_student_init_contract(_contract(), "t")
        C.assert_pinned_candidate(c)  # no raise

    def test_other_candidate_rejected(self):
        c = C.consume_student_init_contract(
            _contract(candidate_id="SOME_OTHER_STUDENT"), "t"
        )
        with pytest.raises(C.StudentContractError) as excinfo:
            C.assert_pinned_candidate(c)
        assert excinfo.value.code == "STUDENT_ID_MISMATCH"


class TestNoLoaderByConstruction:
    def test_no_path_like_field_exists(self):
        names = set(C.contract_field_names())
        path_like = {n for n in names if "path" in n or "dir" in n or "file" in n}
        assert path_like == set(), path_like

    def test_module_has_no_heavy_imports(self):
        path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "src",
            "dicode",
            "teachers",
            "static_llm",
            "student_init_contract.py",
        )
        with open(path, "r", encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), filename=path)
        forbidden = {"jax", "flax", "optax", "orbax", "pickle", "torch", "numpy"}
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert imported & forbidden == set(), imported & forbidden

    def test_pinned_id_is_the_directive_candidate(self):
        assert C.PINNED_STUDENT_CANDIDATE_ID == (
            "PERSISTENT_RMT16_ORIGINAL_VTRACE_98304"
        )
