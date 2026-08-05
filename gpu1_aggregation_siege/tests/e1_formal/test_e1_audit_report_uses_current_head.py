"""CC2-Repair-2: the audit report derives head_after from the real git
state at generation time (a snapshot; the enclosing commit advances
HEAD past it) — never a stale hand-written SHA."""
import json
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REPORT = os.path.join(
    REPO_ROOT,
    "reports",
    "e1_formal_ued",
    "e1_persistent_runtime_object_audit.json",
)

_HEX = set("0123456789abcdef")


class TestAuditCurrentHead:
    def test_report_head_after_is_a_real_git_sha(self):
        with open(REPORT, encoding="utf-8") as fh:
            report = json.load(fh)
        head = report["head_after"]
        # a git SHA: 40 lowercase hex (the snapshot at generation time)
        assert len(head) == 40
        assert all(c in _HEX for c in head)
        # NOT the previous round's stale head
        assert head != "12ebca44908ad1e65c38dd962955b4dce29829f7"
        assert head != "d4e9173948ce427b097a8a53ee2304d508c00d9b"

    def test_stage_is_persistent_object_consumer_implemented(self):
        with open(REPORT, encoding="utf-8") as fh:
            report = json.load(fh)
        assert report["stage"] == "E1_PERSISTENT_OBJECT_CONSUMER_IMPLEMENTED"
        assert report["object_level_ok"] is False
