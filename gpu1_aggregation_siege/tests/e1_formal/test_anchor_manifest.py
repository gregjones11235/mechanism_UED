"""C9 tests: shared anchor manifest gate for retention (G3 gate)."""
import ast
import os
import re

import pytest

from dicode.teachers.e1_formal import anchor_manifest as AM
from dicode.teachers.e1_formal.canonical import canonical_sha256

ANCHOR_IDS = ("anchor_alpha", "anchor_beta", "anchor_gamma", "anchor_delta")


def _anchor_mapping(aid, frozen_by="", frozen_at=""):
    return {
        "anchor_id": aid,
        "source_task_id": f"source_of_{aid}",
        "task_params_hash": "a" * 64,
        "seed_protocol": "fixed-seed-protocol-v1",
        "code_hash": "b" * 64,
        "reset_protocol": "standard-reset-v1",
        "frozen_by": frozen_by,
        "frozen_at": frozen_at,
    }


def _manifest_mapping(status=AM.STATUS_DRAFT_UNFROZEN, anchors=None):
    if anchors is None:
        signed = status == AM.STATUS_FROZEN
        anchors = [
            _anchor_mapping(
                aid,
                frozen_by="supervisor" if signed else "",
                frozen_at="2026-08-03T00:00:00Z" if signed else "",
            )
            for aid in ANCHOR_IDS
        ]
    payload = {"status": status, "anchors": anchors}
    return {
        "status": status,
        "anchors": anchors,
        "manifest_sha256": canonical_sha256(payload),
    }


def _draft():
    return AM.consume_anchor_manifest(_manifest_mapping(), "test")


def _frozen():
    return AM.consume_anchor_manifest(
        _manifest_mapping(status=AM.STATUS_FROZEN), "test"
    )


class TestConsume:
    def test_draft_manifest_is_structurally_valid_but_unfrozen(self):
        manifest = _draft()
        assert manifest.status == AM.STATUS_DRAFT_UNFROZEN
        assert manifest.is_frozen is False
        assert manifest.anchor_ids == ANCHOR_IDS
        assert len(manifest.manifest_sha256) == 64

    def test_frozen_manifest_consumes(self):
        manifest = _frozen()
        assert manifest.is_frozen is True

    def test_hash_is_recomputed_and_verified(self):
        manifest = _draft()
        assert manifest.manifest_sha256 == AM.compute_manifest_sha256(
            manifest.status, manifest.anchors
        )

    def test_tampered_content_fails_closed_with_hash_mismatch(self):
        mapping = _manifest_mapping()
        mapping["anchors"][0]["task_params_hash"] = "c" * 64  # tamper
        with pytest.raises(AM.AnchorManifestError) as excinfo:
            AM.consume_anchor_manifest(mapping, "test")
        assert excinfo.value.code == AM.ANCHOR_MANIFEST_HASH_MISMATCH

    def test_tampered_status_fails_closed(self):
        mapping = _manifest_mapping()
        mapping["status"] = AM.STATUS_FROZEN  # escalate without re-signing
        with pytest.raises(AM.AnchorManifestError) as excinfo:
            AM.consume_anchor_manifest(mapping, "test")
        assert excinfo.value.code in (
            AM.ANCHOR_MANIFEST_HASH_MISMATCH,
            AM.ANCHOR_MANIFEST_EMPTY_FIELD,
        )

    def test_wrong_anchor_count_fails_closed(self):
        mapping = _manifest_mapping()
        mapping["anchors"] = mapping["anchors"][:3]
        payload = {"status": mapping["status"], "anchors": mapping["anchors"]}
        mapping["manifest_sha256"] = canonical_sha256(payload)
        with pytest.raises(AM.AnchorManifestError) as excinfo:
            AM.consume_anchor_manifest(mapping, "test")
        assert excinfo.value.code == AM.ANCHOR_COUNT_MISMATCH

    def test_duplicate_anchor_id_fails_closed(self):
        anchors = [
            _anchor_mapping(aid)
            for aid in (ANCHOR_IDS[0], ANCHOR_IDS[0], ANCHOR_IDS[2], ANCHOR_IDS[3])
        ]
        mapping = _manifest_mapping(anchors=anchors)
        with pytest.raises(AM.AnchorManifestError) as excinfo:
            AM.consume_anchor_manifest(mapping, "test")
        assert excinfo.value.code == AM.DUPLICATE_ANCHOR_ID

    def test_unknown_manifest_field_fails_closed(self):
        mapping = dict(_manifest_mapping(), extra=1)
        with pytest.raises(AM.AnchorManifestError) as excinfo:
            AM.consume_anchor_manifest(mapping, "test")
        assert excinfo.value.code == AM.ANCHOR_MANIFEST_UNKNOWN_FIELD

    def test_unknown_anchor_field_fails_closed(self):
        mapping = _manifest_mapping()
        mapping["anchors"][0]["bonus"] = 1
        with pytest.raises(AM.AnchorManifestError) as excinfo:
            AM.consume_anchor_manifest(mapping, "test")
        assert excinfo.value.code == AM.ANCHOR_MANIFEST_UNKNOWN_FIELD

    def test_draft_anchor_with_signature_fails_closed(self):
        mapping = _manifest_mapping()
        mapping["anchors"][0]["frozen_by"] = "someone"
        with pytest.raises(AM.AnchorManifestError):
            AM.consume_anchor_manifest(mapping, "test")

    def test_frozen_anchor_requires_signature(self):
        anchors = [_anchor_mapping(aid) for aid in ANCHOR_IDS]  # unsigned
        mapping = _manifest_mapping(status=AM.STATUS_FROZEN, anchors=anchors)
        with pytest.raises(AM.AnchorManifestError) as excinfo:
            AM.consume_anchor_manifest(mapping, "test")
        assert excinfo.value.code == AM.ANCHOR_MANIFEST_EMPTY_FIELD

    def test_bad_status_fails_closed(self):
        mapping = _manifest_mapping()
        mapping["status"] = "PROVISIONAL"
        with pytest.raises(AM.AnchorManifestError) as excinfo:
            AM.consume_anchor_manifest(mapping, "test")
        assert excinfo.value.code == AM.ANCHOR_MANIFEST_BAD_TYPE


class TestRetentionGate:
    PRE = {aid: 0.5 for aid in ANCHOR_IDS}
    POST = {aid: 0.75 for aid in ANCHOR_IDS}

    def test_draft_manifest_blocks_retention_with_gate_code(self):
        manifest = _draft()
        with pytest.raises(AM.AnchorManifestError) as excinfo:
            AM.evaluate_retention(manifest, self.PRE, self.POST)
        assert excinfo.value.code == AM.BLOCKED_SHARED_ANCHOR_MANIFEST

    def test_assert_manifest_frozen_uses_not_frozen_code(self):
        manifest = _draft()
        with pytest.raises(AM.AnchorManifestError) as excinfo:
            AM.assert_manifest_frozen(manifest, "test")
        assert excinfo.value.code == AM.ANCHOR_MANIFEST_NOT_FROZEN

    def test_frozen_manifest_evaluates_real_pre_post_scores(self):
        manifest = _frozen()
        report = AM.evaluate_retention(manifest, self.PRE, self.POST)
        assert len(report.entries) == AM.NUM_SHARED_ANCHORS
        assert [e.anchor_id for e in report.entries] == list(ANCHOR_IDS)
        for entry in report.entries:
            assert entry.delta == pytest.approx(0.25)
        assert report.mean_delta == pytest.approx(0.25)
        assert report.manifest_sha256 == manifest.manifest_sha256

    def test_scores_must_cover_exactly_the_four_anchors(self):
        manifest = _frozen()
        missing = {aid: 0.5 for aid in ANCHOR_IDS[:3]}
        with pytest.raises(AM.AnchorManifestError) as excinfo:
            AM.evaluate_retention(manifest, missing, self.POST)
        assert excinfo.value.code == AM.RETENTION_MISSING_ANCHOR
        extra = dict(self.PRE, anchor_ghost=0.5)
        with pytest.raises(AM.AnchorManifestError) as excinfo:
            AM.evaluate_retention(manifest, extra, self.POST)
        assert excinfo.value.code == AM.RETENTION_UNKNOWN_ANCHOR

    @pytest.mark.parametrize("bad", ["high", True, None, 1.5, -0.1])
    def test_bad_scores_fail_closed(self, bad):
        manifest = _frozen()
        post = dict(self.POST)
        post[ANCHOR_IDS[0]] = bad
        with pytest.raises(AM.AnchorManifestError):
            AM.evaluate_retention(manifest, self.PRE, post)

    def test_negative_delta_is_reported_honestly(self):
        manifest = _frozen()
        post = {aid: 0.25 for aid in ANCHOR_IDS}
        report = AM.evaluate_retention(manifest, self.PRE, post)
        assert report.mean_delta == pytest.approx(-0.25)


class TestDraftBuilder:
    def test_draft_builder_hashes_honestly(self):
        anchors = tuple(
            AM.consume_anchor_identity(
                _anchor_mapping(aid), "test", require_signing=False
            )
            for aid in ANCHOR_IDS
        )
        manifest = AM.draft_anchor_manifest(anchors)
        assert manifest.status == AM.STATUS_DRAFT_UNFROZEN
        assert manifest.manifest_sha256 == AM.compute_manifest_sha256(
            AM.STATUS_DRAFT_UNFROZEN, anchors
        )

    def test_draft_builder_rejects_signed_anchors(self):
        anchors = [
            AM.consume_anchor_identity(
                _anchor_mapping(aid), "test", require_signing=False
            )
            for aid in ANCHOR_IDS
        ]
        anchors[0] = AM.AnchorIdentity(
            **{**anchors[0].as_canonical_dict(), "frozen_by": "someone"}
        )
        with pytest.raises(AM.AnchorManifestError):
            AM.draft_anchor_manifest(tuple(anchors))

    def test_draft_builder_requires_four_anchors(self):
        anchors = tuple(
            AM.consume_anchor_identity(
                _anchor_mapping(aid), "test", require_signing=False
            )
            for aid in ANCHOR_IDS[:3]
        )
        with pytest.raises(AM.AnchorManifestError) as excinfo:
            AM.draft_anchor_manifest(anchors)
        assert excinfo.value.code == AM.ANCHOR_COUNT_MISMATCH


class TestNoAchievementCountSubstitute:
    """G3: retention must never be approximated by counting achieved/
    mastered achievements. Audit the E1 runtime modules' CODE (imports
    and identifiers) — docstrings may mention the prohibition."""

    MODULES = ("metrics.py", "selector.py", "anchor_manifest.py")

    def _identifiers(self, relpath):
        path = os.path.join(
            os.path.dirname(__file__), "..", "..", "src", "dicode",
            "teachers", "e1_formal", relpath,
        )
        with open(path, "r", encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.add(node.name)
            elif isinstance(node, ast.ImportFrom):
                names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.Import):
                names.update(alias.name for alias in node.names)
        return names

    def test_no_achievement_identifier_in_retention_path(self):
        for relpath in self.MODULES:
            names = self._identifiers(relpath)
            offenders = [
                name for name in names
                if re.search(r"achievement", name, re.IGNORECASE)
            ]
            assert offenders == [], f"{relpath}: {offenders}"

    def test_no_d052_import_in_runtime_modules(self):
        # the ONLY sanctioned d052 runtime import is REGISTRY in
        # task_specs.py; retention/selection/metrics import nothing.
        for relpath in self.MODULES:
            names = self._identifiers(relpath)
            assert "d052" not in names, relpath
            assert "REGISTRY" not in names, relpath
