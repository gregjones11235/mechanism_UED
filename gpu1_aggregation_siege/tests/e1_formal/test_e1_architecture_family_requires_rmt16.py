"""CC2-Repair-2: architecture RMT16."""
"""architecture_family must be RMT16."""
from types import SimpleNamespace
import pytest
from dicode.teachers.e1_formal import runtime_bundle as RB


class TestArchitectureRmt16:
    def test_non_rmt16_rejected(self):
        b = RB.build_test_only_runtime_bundle(
            source_commit="TEST_ONLY_SYNTHETIC_SOURCE_COMMIT",
            capabilities={c: SimpleNamespace(kind=c, identity_id=f"t-{c}")
                          for c in RB.RUNTIME_CAPABILITY_CONTRACTS})
        block = dict(b.student_selection_mapping)
        block["architecture_family"] = "GPT"
        with pytest.raises(RB.RuntimeBundleError) as e:
            RB.parse_student_selection(block, "t")
        assert e.value.code == RB.RUNTIME_BUNDLE_STUDENT_SELECTION_BAD
