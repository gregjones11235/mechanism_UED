"""CC2-Repair: missing real objects => BLOCKED before any LLM."""
import inspect


class TestMainBlocksBeforeLLM:
    def test_entry_blocks_before_llm_without_objects(self):
        import run_e1_real_one_update as ENT
        src = inspect.getsource(ENT.main)
        assert "E1_PIPELINE_OBJECTS_NOT_INJECTED" in src
        assert "if not resolution[" in src

    def test_no_llm_call_without_resolution(self):
        import run_e1_real_one_update as ENT
        src = inspect.getsource(ENT.main)
        # the pipeline runs only AFTER resolution succeeds
        idx = src.index("if not resolution[")
        assert "run_director_one_window_pipeline" not in src[:idx]
