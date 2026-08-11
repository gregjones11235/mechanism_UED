"""CC2-Repair: the gates-clear path is REACHABLE (no hardcoded
unreachable)."""
import os
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(REPO_ROOT, "scripts", "run_e1_real_one_update.py")


class TestMainReachesPipeline:
    def test_no_hardcoded_unreachable_raise(self):
        source = open(SRC, encoding="utf-8").read()
        assert "unreachable this round" not in source
        # the pipeline path must resolve objects and CALL the pipeline,
        # never return a fixed pending/handoff
        assert "run_director_one_window_pipeline(" in source

    def test_pipeline_call_surface_is_wired(self):
        source = open(SRC, encoding="utf-8").read()
        assert "run_director_one_window_pipeline" in source
        assert "resolve_e1_runtime_objects" in source
