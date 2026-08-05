"""CC2-Repair-2: lowercase hex."""
"""Hash fields require strict lowercase hex (not just length 64)."""
from dicode.teachers.e1_formal import runtime_bundle as RB


class TestLowercaseHex:
    def test_uppercase_hex_rejected(self):
        try:
            RB._require_sha64("A" * 64, "x", "t")
            raise AssertionError("expected lowercase-hex rejection")
        except RB.RuntimeBundleError as e:
            assert e.code == RB.RUNTIME_BUNDLE_STUDENT_SELECTION_BAD

    def test_valid_lowercase_hex_accepted(self):
        assert RB._require_sha64("ab" * 32, "x", "t") == "ab" * 32
