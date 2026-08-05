"""CC2-Repair-2: REAL_LLM_EXECUTED derives from an LLM attestation, not
a whitelist."""
import inspect

import run_e1_real_one_update as ENT


class TestRealLLMFlag:
    def test_derivation_uses_llm_attestation(self):
        src = inspect.getsource(ENT.run_director_one_window_pipeline)
        assert "llm_call_attestation is not None" in src

    def test_not_derived_from_training_whitelist(self):
        src = inspect.getsource(ENT.run_director_one_window_pipeline)
        # REAL_LLM_EXECUTED must NOT be derived from the training
        # runtime whitelist (which is about updates, not LLM calls)
        assert "AUTHORIZED_TRAINING_RUNTIMES" not in src.split(
            "REAL_LLM_EXECUTED"
        )[1].split("REAL_ENVCODER_EXECUTED")[0]
