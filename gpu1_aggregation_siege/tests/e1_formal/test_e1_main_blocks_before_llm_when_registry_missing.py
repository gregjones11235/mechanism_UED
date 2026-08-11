"""CC2-Repair-2: blocks before llm."""
"""Without the registry, main blocks before any LLM."""
import os
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(REPO_ROOT, "scripts", "run_e1_real_one_update.py")


class TestMainBlocksWithoutRegistry:
    def test_registry_unbound_blocks(self):
        s = open(SRC, encoding="utf-8").read()
        assert "FORMAL_ASSET_REGISTRY_UNBOUND" in s
        idx = s.index("require_real_registry(formal_asset_registry")
        assert "run_director_one_window_pipeline(" not in s[:idx]
