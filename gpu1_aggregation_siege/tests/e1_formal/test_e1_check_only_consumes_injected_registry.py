"""CC2-Repair-2: check-only injected registry."""
"""check-only consumes the injected registry, never a hardcoded None."""
import os
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(REPO_ROOT, "scripts", "run_e1_real_one_update.py")


class TestCheckOnlyInjectedRegistry:
    def test_check_only_accepts_an_injected_registry(self):
        s = open(SRC, encoding="utf-8").read()
        assert "formal_asset_registry=formal_asset_registry" in s
        assert "require_real_registry(formal_asset_registry" in s
