"""C3 tests: canonical encoding, provenance tiers, flags, json parse.

Pure standard library, offline, deterministic.
"""
import pytest

from dicode.teachers.e1_formal import canonical as C
from dicode.teachers.e1_formal import flags as F
from dicode.teachers.e1_formal import json_parse as J
from dicode.teachers.e1_formal import schemas as S


# ---------------------------------------------------------------------------
# canonical encoding
# ---------------------------------------------------------------------------
class TestCanonical:
    def test_key_order_irrelevant(self):
        assert C.canonical_json({"a": 1, "b": 2}) == C.canonical_json(
            {"b": 2, "a": 1}
        )

    def test_sha256_stable_vector(self):
        import hashlib

        # fixed vector: pins the exact canonical encoding format
        assert C.canonical_json({"a": 1}) == '{"a":1}'
        assert C.canonical_sha256({"a": 1}) == hashlib.sha256(
            b'{"a":1}'
        ).hexdigest()

    def test_sha256_deterministic_double_run(self):
        payload = {"z": [1, 2, {"k": "v"}], "a": None, "f": 0.5, "t": True}
        assert C.canonical_sha256(payload) == C.canonical_sha256(payload)

    def test_tuple_encodes_as_list(self):
        assert C.canonical_json((1, 2)) == C.canonical_json([1, 2])

    def test_non_finite_float_rejected(self):
        with pytest.raises(S.E1SchemaError) as excinfo:
            C.canonical_json({"x": float("nan")})
        assert excinfo.value.code == S.E1Code.CANONICAL_UNSUPPORTED_TYPE

    def test_unsupported_type_rejected(self):
        with pytest.raises(S.E1SchemaError) as excinfo:
            C.canonical_json({"x": object()})
        assert excinfo.value.code == S.E1Code.CANONICAL_UNSUPPORTED_TYPE

    def test_non_string_dict_key_rejected(self):
        with pytest.raises(S.E1SchemaError) as excinfo:
            C.canonical_json({1: "x"})
        assert excinfo.value.code == S.E1Code.CANONICAL_UNSUPPORTED_TYPE


# ---------------------------------------------------------------------------
# provenance tiering (section seven / G gates)
# ---------------------------------------------------------------------------
class TestProvenanceTiers:
    @pytest.mark.parametrize("label", sorted(["TRAINING", "NORMAL_TRAINING_FEEDBACK"]))
    def test_llm_role_admissible(self, label):
        assert S.assert_llm_role_admissible(label, "t") == label
        assert S.assert_selector_admissible(label, "t") == label

    def test_candidate_evaluation_selector_only(self):
        # admissible on the selection side...
        assert (
            S.assert_selector_admissible("CANDIDATE_EVALUATION", "t")
            == "CANDIDATE_EVALUATION"
        )
        # ...but NEVER in an LLM prompt.
        with pytest.raises(S.E1SchemaError) as excinfo:
            S.assert_llm_role_admissible("CANDIDATE_EVALUATION", "t")
        assert excinfo.value.code == S.E1Code.LLM_PROVENANCE_VIOLATION

    @pytest.mark.parametrize("label", sorted(["FORMAL_FRONT", "FORMAL_BACK", "FORMAL_FULL"]))
    def test_formal_rejected_at_both_tiers(self, label):
        for fn in (S.assert_llm_role_admissible, S.assert_selector_admissible):
            with pytest.raises(S.E1SchemaError) as excinfo:
                fn(label, "t")
            assert excinfo.value.code == "FORMAL_PROVENANCE_REJECTED"

    def test_missing_provenance_fails_closed(self):
        with pytest.raises(S.E1SchemaError) as excinfo:
            S.assert_llm_role_admissible(None, "t")
        assert excinfo.value.code == "PROVENANCE_MISSING"

    def test_unknown_provenance_fails_closed(self):
        with pytest.raises(S.E1SchemaError) as excinfo:
            S.assert_selector_admissible("PROBE_FEEDBACK", "t")
        assert excinfo.value.code == "UNKNOWN_PROVENANCE"


# ---------------------------------------------------------------------------
# flags (D7)
# ---------------------------------------------------------------------------
class TestFlags:
    def _all_false(self):
        return {
            "real_envcoder_used": False,
            "real_student_reference_eval": False,
            "real_training_update_executed": False,
        }

    def test_defaults_are_false(self):
        flags = F.E1Flags()
        assert flags == F.E1Flags(False, False, False)

    def test_parse_all_false(self):
        flags = F.parse_flags(self._all_false(), "t")
        assert flags.real_envcoder_used is False
        assert flags.real_student_reference_eval is False
        assert flags.real_training_update_executed is False

    def test_missing_flag_rejected(self):
        payload = self._all_false()
        del payload["real_envcoder_used"]
        with pytest.raises(S.E1SchemaError) as excinfo:
            F.parse_flags(payload, "t")
        assert excinfo.value.code == S.E1Code.FLAGS_MISSING_FIELD

    def test_unknown_flag_rejected(self):
        payload = self._all_false()
        payload["real_something_else"] = False
        with pytest.raises(S.E1SchemaError) as excinfo:
            F.parse_flags(payload, "t")
        assert excinfo.value.code == S.E1Code.FLAGS_UNKNOWN_FIELD

    @pytest.mark.parametrize("bad", [0, 1, "false", "no", None])
    def test_non_bool_flag_rejected_no_coercion(self, bad):
        payload = self._all_false()
        payload["real_envcoder_used"] = bad
        with pytest.raises(S.E1SchemaError) as excinfo:
            F.parse_flags(payload, "t")
        assert excinfo.value.code == S.E1Code.FLAGS_BAD_TYPE

    def test_manifest_match_passes(self):
        flags = F.parse_flags(self._all_false(), "t")
        manifest = {"flags": self._all_false()}
        F.assert_flags_match_manifest(flags, manifest, "t")  # no raise

    def test_manifest_mismatch_fails_closed(self):
        flags = F.parse_flags(self._all_false(), "t")
        manifest = {"flags": self._all_false()}
        manifest["flags"]["real_envcoder_used"] = True
        with pytest.raises(S.E1SchemaError) as excinfo:
            F.assert_flags_match_manifest(flags, manifest, "t")
        assert excinfo.value.code == S.E1Code.FLAG_MANIFEST_MISMATCH

    def test_manifest_missing_flags_block_fails_closed(self):
        flags = F.parse_flags(self._all_false(), "t")
        with pytest.raises(S.E1SchemaError) as excinfo:
            F.assert_flags_match_manifest(flags, {}, "t")
        assert excinfo.value.code == S.E1Code.FLAG_MANIFEST_MISMATCH


# ---------------------------------------------------------------------------
# json extraction
# ---------------------------------------------------------------------------
class TestJsonParse:
    def test_extracts_object_from_surrounding_text(self):
        text = 'Here is the plan:\n```json\n{"a": 1, "b": [2, 3]}\n```'
        assert J.extract_json_block(text) == {"a": 1, "b": [2, 3]}

    def test_extracts_array(self):
        assert J.extract_json_block("noise [1, 2] tail") == [1, 2]

    def test_string_braces_do_not_confuse_balance(self):
        assert J.extract_json_block('{"s": "}{"}') == {"s": "}{"}

    def test_no_json_fails_closed(self):
        with pytest.raises(S.E1SchemaError) as excinfo:
            J.extract_json_block("no json here")
        assert excinfo.value.code == S.E1Code.JSON_NOT_FOUND

    def test_unbalanced_fails_closed(self):
        with pytest.raises(S.E1SchemaError) as excinfo:
            J.extract_json_block('{"a": 1')
        assert excinfo.value.code == S.E1Code.JSON_PARSE_FAILED

    def test_invalid_json_fails_closed(self):
        with pytest.raises(S.E1SchemaError) as excinfo:
            J.extract_json_block("{a: 1}")
        assert excinfo.value.code == S.E1Code.JSON_PARSE_FAILED

    def test_non_string_fails_closed(self):
        with pytest.raises(S.E1SchemaError) as excinfo:
            J.extract_json_block(None)
        assert excinfo.value.code == S.E1Code.JSON_NOT_FOUND

    def test_deterministic_double_run(self):
        text = 'preface {"x": {"y": [1]}} epilogue'
        assert J.extract_json_block(text) == J.extract_json_block(text)
