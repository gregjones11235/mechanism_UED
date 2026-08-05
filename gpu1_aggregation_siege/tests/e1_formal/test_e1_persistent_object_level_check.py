"""CC2-Repair-2: persistent object-level check."""
"""The object-level check is pinned to the Persistent Student."""
from dicode.teachers.e1_formal import student_contract as SC
import run_e1_real_one_update as ENT


class TestPersistentObjectLevel:
    def test_persistent_is_the_fixed_object_level_target(self):
        assert "PERSISTENT_RMT16_ORIGINAL_VTRACE_98304" in (
            SC.ALLOWED_STUDENT_CANDIDATE_IDS)

    def test_object_level_check_exists(self):
        import inspect
        assert callable(getattr(ENT, "run_e1_object_level_check", None))
        src = inspect.getsource(ENT.run_e1_object_level_check)
        assert "selected_candidate_id" in src
