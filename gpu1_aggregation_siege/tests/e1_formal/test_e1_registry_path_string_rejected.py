"""CC2-Repair-2: registry path string."""
"""A registry path string is never a FormalAssetRegistry."""
from dicode.teachers.e1_formal import runtime_object_resolution as ROR


class TestRegistryPathRejected:
    def test_string_path_rejected(self):
        try:
            ROR.require_real_registry("formal_asset_registry.json", "t")
            raise AssertionError("expected STRING_PLACEHOLDER")
        except ROR.RuntimeObjectResolutionError as e:
            assert e.code == ROR.OBJ_RESOLUTION_STRING_PLACEHOLDER

    def test_none_is_unbound(self):
        try:
            ROR.require_real_registry(None, "t")
            raise AssertionError("expected FORMAL_ASSET_REGISTRY_UNBOUND")
        except ROR.RuntimeObjectResolutionError as e:
            assert e.code == ROR.OBJ_RESOLUTION_REGISTRY_UNBOUND
