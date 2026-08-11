"""CC2-Repair: REAL_* flags derive from signed evidence, never from
local booleans."""
import inspect
import run_e1_real_one_update as ENT


class TestRealFlagsDerivedFromAttestations:
    def test_test_only_path_keeps_all_real_flags_false(self):
        src = inspect.getsource(ENT.run_director_one_window_pipeline)
        assert "allow_test_only" in src
        assert "not allow_test_only" in src

    def test_production_derivation_uses_whitelists(self):
        src = inspect.getsource(ENT.run_director_one_window_pipeline)
        assert "AUTHORIZED_TRAINING_RUNTIMES" in src
        assert "AUTHORIZED_ROUNDTRIP_SIGNERS" in src
        assert "AUTHORIZED_SMOKE_SIGNERS" in src
