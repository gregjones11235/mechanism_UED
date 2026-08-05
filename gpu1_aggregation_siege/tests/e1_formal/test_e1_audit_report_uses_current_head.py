"""CC2-Repair-2: the audit report derives head_after from the real git
HEAD (not a stale hand-written SHA)."""
import json
import os
import subprocess

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REPORT = os.path.join(
    REPO_ROOT,
    "reports",
    "e1_formal_ued",
    "e1_persistent_runtime_object_audit.json",
)


def _git_head():
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


class TestAuditCurrentHead:
    def test_report_head_after_equals_the_real_git_head(self):
        with open(REPORT, encoding="utf-8") as fh:
            report = json.load(fh)
        head = _git_head()
        assert head  # git must resolve
        assert report["head_after"] == head
        assert report["head_after"] != (
            "d4e9173948ce427b097a8a53ee2304d508c00d9b"
        )
