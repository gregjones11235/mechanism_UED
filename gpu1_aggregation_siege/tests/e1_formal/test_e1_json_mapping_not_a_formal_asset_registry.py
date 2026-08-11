"""CC2-Repair-2: JSON mapping rejected."""
"""A JSON Mapping is never a FormalAssetRegistry."""
from dicode.teachers.e1_formal import runtime_object_resolution as ROR


class TestJsonMappingRejected:
    def test_plain_mapping_rejected(self):
        try:
            ROR.require_real_registry({"student_identity": "x"}, "t")
            raise AssertionError("expected MAPPING_IMPERSONATION")
        except ROR.RuntimeObjectResolutionError as e:
            assert e.code == ROR.OBJ_RESOLUTION_MAPPING_IMPERSONATION

    def test_non_protocol_object_rejected(self):
        try:
            ROR.require_real_registry(object(), "t")
            raise AssertionError("expected REGISTRY_NOT_PROTOCOL")
        except ROR.RuntimeObjectResolutionError as e:
            assert e.code == ROR.OBJ_RESOLUTION_REGISTRY_NOT_PROTOCOL
