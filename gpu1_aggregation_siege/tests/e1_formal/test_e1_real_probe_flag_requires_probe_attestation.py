"""CC2-Repair-2: real probe flag."""
"""REAL_CANDIDATE_PROBE_EXECUTED derives from probe coverage proof."""
import inspect
import run_e1_real_one_update as ENT


class TestRealProbeFlag:
    def test_probe_flag_uses_coverage_proof(self):
        src = inspect.getsource(ENT.run_director_one_window_pipeline)
        assert "probe_coverage_proof is not None" in src
