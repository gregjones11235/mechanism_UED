"""CC2-Repair-2: contract before mount order."""
"""Production resolves the real contract BEFORE mounting the Student."""
import os
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(REPO_ROOT, "scripts", "run_e1_real_one_update.py")


class TestResolveContractBeforeMount:
    def test_main_order_has_registry_resolution_before_mount(self):
        s = open(SRC, encoding="utf-8").read()
        i_registry = s.index("require_real_registry(formal_asset_registry")
        i_mount = s.index("mount_student_from_director_bundle")
        # the real-contract resolution happens before the Student mount
        assert i_registry < i_mount
