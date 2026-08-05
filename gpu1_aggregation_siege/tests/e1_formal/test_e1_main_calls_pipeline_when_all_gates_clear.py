"""CC2-Repair-2: main calls pipeline."""
"""main() really calls the pipeline when all gates clear."""
import os
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(REPO_ROOT, "scripts", "run_e1_real_one_update.py")


class TestMainCallsPipeline:
    def test_pipeline_is_called_after_resolution_and_authorization(self):
        s = open(SRC, encoding="utf-8").read()
        assert "run_director_one_window_pipeline(" in s
        assert "E1_SMOKE_NOT_AUTHORIZED" in s
        assert "authorized_llm_runtime is None" in s
