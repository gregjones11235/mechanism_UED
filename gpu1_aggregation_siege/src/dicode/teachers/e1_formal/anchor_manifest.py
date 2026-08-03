"""Stage 7 (part): shared anchor manifest gate for retention (G3).

Retention may ONLY be measured as the SAME Student's real pre/post
update evaluation on the four cross-direction shared standard-reset
anchors, bound to a supervisor-frozen manifest. Until that manifest is
frozen, every retention query fails closed with
``BLOCKED_SHARED_ANCHOR_MANIFEST``.

There is NO substitute retention metric. In particular, counting
already-mastered achievements is NOT retention and no such path exists
anywhere in the E1 teacher (grep-audited by tests). The 12+4 batch
structure is preserved (anchors enter the batch as registered; the
teacher never modifies them) — only the retention EVALUATION is gated.

Pure standard library; fail-closed with greppable codes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from .canonical import canonical_sha256
from .schemas import E1SchemaError

#: manifest lifecycle states
STATUS_DRAFT_UNFROZEN = "DRAFT_UNFROZEN"
STATUS_FROZEN = "FROZEN"
_VALID_STATUSES = frozenset({STATUS_DRAFT_UNFROZEN, STATUS_FROZEN})

#: the E1 batch carries exactly four shared standard-reset anchors
NUM_SHARED_ANCHORS = 4

# fail-closed codes
BLOCKED_SHARED_ANCHOR_MANIFEST = "BLOCKED_SHARED_ANCHOR_MANIFEST"
ANCHOR_MANIFEST_NOT_FROZEN = "ANCHOR_MANIFEST_NOT_FROZEN"
ANCHOR_MANIFEST_HASH_MISMATCH = "ANCHOR_MANIFEST_HASH_MISMATCH"
ANCHOR_MANIFEST_BAD_TYPE = "ANCHOR_MANIFEST_BAD_TYPE"
ANCHOR_MANIFEST_MISSING_FIELD = "ANCHOR_MANIFEST_MISSING_FIELD"
ANCHOR_MANIFEST_UNKNOWN_FIELD = "ANCHOR_MANIFEST_UNKNOWN_FIELD"
ANCHOR_MANIFEST_EMPTY_FIELD = "ANCHOR_MANIFEST_EMPTY_FIELD"
ANCHOR_COUNT_MISMATCH = "ANCHOR_COUNT_MISMATCH"
DUPLICATE_ANCHOR_ID = "DUPLICATE_ANCHOR_ID"
RETENTION_MISSING_ANCHOR = "RETENTION_MISSING_ANCHOR"
RETENTION_UNKNOWN_ANCHOR = "RETENTION_UNKNOWN_ANCHOR"
RETENTION_BAD_TYPE = "RETENTION_BAD_TYPE"
RETENTION_OUT_OF_RANGE = "RETENTION_OUT_OF_RANGE"

#: identity fields of one shared anchor (all strings)
_ANCHOR_IDENTITY_FIELDS = (
    "anchor_id",
    "source_task_id",
    "task_params_hash",
    "seed_protocol",
    "code_hash",
    "reset_protocol",
)
#: signing fields: non-empty iff FROZEN, exactly "" while DRAFT_UNFROZEN
_ANCHOR_SIGNING_FIELDS = ("frozen_by", "frozen_at")
_ANCHOR_FIELDS = _ANCHOR_IDENTITY_FIELDS + _ANCHOR_SIGNING_FIELDS

#: top-level manifest mapping fields
_MANIFEST_FIELDS = ("status", "anchors", "manifest_sha256")


class AnchorManifestError(E1SchemaError):
    """Fail-closed anchor-manifest violation; ``code`` is greppable."""


@dataclass(frozen=True)
class AnchorIdentity:
    """Frozen identity of ONE cross-direction shared anchor."""

    anchor_id: str
    source_task_id: str
    task_params_hash: str
    seed_protocol: str
    code_hash: str
    reset_protocol: str
    frozen_by: str  # empty string while DRAFT_UNFROZEN
    frozen_at: str  # empty string while DRAFT_UNFROZEN

    def as_canonical_dict(self) -> Dict[str, str]:
        return {name: getattr(self, name) for name in _ANCHOR_FIELDS}


@dataclass(frozen=True)
class SharedAnchorManifest:
    """The four shared anchors plus lifecycle status and content hash."""

    status: str
    anchors: Tuple[AnchorIdentity, ...]
    manifest_sha256: str

    @property
    def anchor_ids(self) -> Tuple[str, ...]:
        return tuple(a.anchor_id for a in self.anchors)

    @property
    def is_frozen(self) -> bool:
        return self.status == STATUS_FROZEN


def _require_nonempty_str(mapping: Mapping, name: str, ctx: str) -> str:
    if name not in mapping:
        raise AnchorManifestError(
            ANCHOR_MANIFEST_MISSING_FIELD, f"{ctx}: missing {name!r}"
        )
    value = mapping[name]
    if not isinstance(value, str):
        raise AnchorManifestError(
            ANCHOR_MANIFEST_BAD_TYPE,
            f"{ctx}: {name!r} must be str, got {type(value).__name__}",
        )
    if not value.strip():
        raise AnchorManifestError(
            ANCHOR_MANIFEST_EMPTY_FIELD, f"{ctx}: {name!r} is empty"
        )
    return value.strip()


def consume_anchor_identity(
    mapping: Any, ctx: str, *, require_signing: bool
) -> AnchorIdentity:
    """Parse one anchor identity fail-closed (all fields required).

    ``require_signing=True`` (FROZEN manifest): frozen_by/frozen_at must
    be non-empty. ``require_signing=False`` (DRAFT): they must be
    EXACTLY empty strings — an unfrozen anchor is never half-signed.
    """
    if not isinstance(mapping, Mapping):
        raise AnchorManifestError(
            ANCHOR_MANIFEST_BAD_TYPE,
            f"{ctx}: anchor must be a mapping, got {type(mapping).__name__}",
        )
    unknown = sorted(k for k in mapping if k not in _ANCHOR_FIELDS)
    if unknown:
        raise AnchorManifestError(
            ANCHOR_MANIFEST_UNKNOWN_FIELD,
            f"{ctx}: unknown anchor field(s) {unknown}",
        )
    values = {
        name: _require_nonempty_str(mapping, name, ctx)
        for name in _ANCHOR_IDENTITY_FIELDS
    }
    for name in _ANCHOR_SIGNING_FIELDS:
        if name not in mapping:
            raise AnchorManifestError(
                ANCHOR_MANIFEST_MISSING_FIELD, f"{ctx}: missing {name!r}"
            )
        value = mapping[name]
        if not isinstance(value, str):
            raise AnchorManifestError(
                ANCHOR_MANIFEST_BAD_TYPE,
                f"{ctx}: {name!r} must be str, got {type(value).__name__}",
            )
        if require_signing:
            if not value.strip():
                raise AnchorManifestError(
                    ANCHOR_MANIFEST_EMPTY_FIELD,
                    f"{ctx}: frozen anchor requires non-empty {name!r}",
                )
            values[name] = value.strip()
        else:
            if value != "":
                raise AnchorManifestError(
                    ANCHOR_MANIFEST_BAD_TYPE,
                    f"{ctx}: DRAFT anchor {name!r} must be exactly '' "
                    "(unfrozen), got a non-empty value",
                )
            values[name] = ""
    return AnchorIdentity(**values)


def manifest_hash_payload(status: str, anchors: Tuple[AnchorIdentity, ...]) -> Dict[str, Any]:
    """Canonical payload whose sha256 is the manifest identity."""
    return {
        "status": status,
        "anchors": [anchor.as_canonical_dict() for anchor in anchors],
    }


def compute_manifest_sha256(
    status: str, anchors: Tuple[AnchorIdentity, ...]
) -> str:
    return canonical_sha256(manifest_hash_payload(status, anchors))


def consume_anchor_manifest(mapping: Any, ctx: str) -> SharedAnchorManifest:
    """Parse + verify a shared anchor manifest fail-closed.

    Structural verification only: a manifest whose hash does not match
    its own content fails closed (``ANCHOR_MANIFEST_HASH_MISMATCH``).
    FREEZING is a supervisor act: a DRAFT manifest is structurally
    valid but blocks every retention query.
    """
    if not isinstance(mapping, Mapping):
        raise AnchorManifestError(
            ANCHOR_MANIFEST_BAD_TYPE,
            f"{ctx}: manifest must be a mapping, got "
            f"{type(mapping).__name__}",
        )
    unknown = sorted(k for k in mapping if k not in _MANIFEST_FIELDS)
    if unknown:
        raise AnchorManifestError(
            ANCHOR_MANIFEST_UNKNOWN_FIELD,
            f"{ctx}: unknown manifest field(s) {unknown}",
        )
    for name in _MANIFEST_FIELDS:
        if name not in mapping:
            raise AnchorManifestError(
                ANCHOR_MANIFEST_MISSING_FIELD, f"{ctx}: missing {name!r}"
            )
    status = mapping["status"]
    if not isinstance(status, str) or status not in _VALID_STATUSES:
        raise AnchorManifestError(
            ANCHOR_MANIFEST_BAD_TYPE,
            f"{ctx}: status {status!r} not in {sorted(_VALID_STATUSES)}",
        )
    raw_anchors = mapping["anchors"]
    if not isinstance(raw_anchors, (list, tuple)):
        raise AnchorManifestError(
            ANCHOR_MANIFEST_BAD_TYPE, f"{ctx}: anchors must be a sequence"
        )
    if len(raw_anchors) != NUM_SHARED_ANCHORS:
        raise AnchorManifestError(
            ANCHOR_COUNT_MISMATCH,
            f"{ctx}: manifest must carry exactly {NUM_SHARED_ANCHORS} "
            f"shared anchors, got {len(raw_anchors)}",
        )
    require_signing = status == STATUS_FROZEN
    anchors = tuple(
        consume_anchor_identity(
            raw, f"{ctx}.anchors[{i}]", require_signing=require_signing
        )
        for i, raw in enumerate(raw_anchors)
    )
    ids = [anchor.anchor_id for anchor in anchors]
    if len(set(ids)) != len(ids):
        raise AnchorManifestError(
            DUPLICATE_ANCHOR_ID, f"{ctx}: duplicate anchor_id in {ids}"
        )
    manifest_sha256 = mapping["manifest_sha256"]
    if not isinstance(manifest_sha256, str) or not manifest_sha256.strip():
        raise AnchorManifestError(
            ANCHOR_MANIFEST_MISSING_FIELD,
            f"{ctx}: manifest_sha256 must be a non-empty str",
        )
    expected = compute_manifest_sha256(status, anchors)
    if manifest_sha256.strip() != expected:
        raise AnchorManifestError(
            ANCHOR_MANIFEST_HASH_MISMATCH,
            f"{ctx}: manifest_sha256 {manifest_sha256!r} != recomputed "
            f"{expected!r}; refusing a tampered or stale manifest",
        )
    return SharedAnchorManifest(
        status=status, anchors=anchors, manifest_sha256=expected
    )


def draft_anchor_manifest(anchors: Tuple[AnchorIdentity, ...]) -> SharedAnchorManifest:
    """Build a structurally valid DRAFT manifest (hash computed honestly)."""
    if len(anchors) != NUM_SHARED_ANCHORS:
        raise AnchorManifestError(
            ANCHOR_COUNT_MISMATCH,
            f"draft requires exactly {NUM_SHARED_ANCHORS} anchors, got "
            f"{len(anchors)}",
        )
    for anchor in anchors:
        if not isinstance(anchor, AnchorIdentity):
            raise AnchorManifestError(
                ANCHOR_MANIFEST_BAD_TYPE,
                "draft anchors must be AnchorIdentity instances",
            )
        if anchor.frozen_by != "" or anchor.frozen_at != "":
            raise AnchorManifestError(
                ANCHOR_MANIFEST_BAD_TYPE,
                f"draft anchor {anchor.anchor_id!r} must be unsigned "
                "(frozen_by/frozen_at exactly '')",
            )
    return SharedAnchorManifest(
        status=STATUS_DRAFT_UNFROZEN,
        anchors=anchors,
        manifest_sha256=compute_manifest_sha256(STATUS_DRAFT_UNFROZEN, anchors),
    )


def assert_manifest_frozen(manifest: SharedAnchorManifest, ctx: str) -> None:
    """Internal frozen-assertion (raises ANCHOR_MANIFEST_NOT_FROZEN)."""
    if not isinstance(manifest, SharedAnchorManifest):
        raise AnchorManifestError(
            ANCHOR_MANIFEST_BAD_TYPE,
            f"{ctx}: manifest must be SharedAnchorManifest",
        )
    if not manifest.is_frozen:
        raise AnchorManifestError(
            ANCHOR_MANIFEST_NOT_FROZEN,
            f"{ctx}: anchor manifest status is {manifest.status!r}; a "
            "frozen supervisor-signed manifest is required",
        )


@dataclass(frozen=True)
class AnchorRetentionEntry:
    """Pre/post real-evaluation retention delta for ONE anchor."""

    anchor_id: str
    pre_score: float
    post_score: float
    delta: float  # post - pre


@dataclass(frozen=True)
class RetentionReport:
    """Same-Student pre/post retention over the four frozen anchors."""

    entries: Tuple[AnchorRetentionEntry, ...]
    mean_delta: float
    manifest_sha256: str


def _require_anchor_scores(
    scores: Any, anchor_ids: Tuple[str, ...], phase: str, ctx: str
) -> Dict[str, float]:
    if not isinstance(scores, Mapping):
        raise AnchorManifestError(
            RETENTION_BAD_TYPE,
            f"{ctx}: {phase} scores must be a mapping anchor_id -> float",
        )
    unknown = sorted(k for k in scores if k not in anchor_ids)
    if unknown:
        raise AnchorManifestError(
            RETENTION_UNKNOWN_ANCHOR,
            f"{ctx}: {phase} scores for unknown anchor(s) {unknown}",
        )
    missing = sorted(set(anchor_ids) - set(scores))
    if missing:
        raise AnchorManifestError(
            RETENTION_MISSING_ANCHOR,
            f"{ctx}: {phase} scores missing anchor(s) {missing}",
        )
    cleaned = {}
    for anchor_id in anchor_ids:
        value = scores[anchor_id]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise AnchorManifestError(
                RETENTION_BAD_TYPE,
                f"{ctx}: {phase} score for {anchor_id!r} must be a number",
            )
        value = float(value)
        if value < 0.0 or value > 1.0:
            raise AnchorManifestError(
                RETENTION_OUT_OF_RANGE,
                f"{ctx}: {phase} score for {anchor_id!r} outside [0, 1]: "
                f"{value}",
            )
        cleaned[anchor_id] = value
    return cleaned


def evaluate_retention(
    manifest: SharedAnchorManifest,
    pre_scores: Any,
    post_scores: Any,
    ctx: str = "retention",
) -> RetentionReport:
    """Same-Student pre/post retention on the four shared anchors (G3).

    BLOCKED (``BLOCKED_SHARED_ANCHOR_MANIFEST``) until the manifest is
    supervisor-frozen. Inputs are real evaluation scores ONLY (one
    float in [0, 1] per anchor, per phase); no achievement counting or
    any other substitute is admissible here.
    """
    if not isinstance(manifest, SharedAnchorManifest):
        raise AnchorManifestError(
            ANCHOR_MANIFEST_BAD_TYPE,
            f"{ctx}: manifest must be SharedAnchorManifest",
        )
    if not manifest.is_frozen:
        raise AnchorManifestError(
            BLOCKED_SHARED_ANCHOR_MANIFEST,
            f"{ctx}: retention evaluation is BLOCKED — the shared anchor "
            f"manifest status is {manifest.status!r}; the supervisor has "
            "not frozen the cross-direction anchor identities/TaskParams/"
            "seeds/hashes. No substitute retention metric exists.",
        )
    anchor_ids = manifest.anchor_ids
    pre = _require_anchor_scores(pre_scores, anchor_ids, "pre", ctx)
    post = _require_anchor_scores(post_scores, anchor_ids, "post", ctx)
    entries = tuple(
        AnchorRetentionEntry(
            anchor_id=anchor_id,
            pre_score=pre[anchor_id],
            post_score=post[anchor_id],
            delta=post[anchor_id] - pre[anchor_id],
        )
        for anchor_id in anchor_ids
    )
    mean_delta = sum(entry.delta for entry in entries) / len(entries)
    return RetentionReport(
        entries=entries,
        mean_delta=mean_delta,
        manifest_sha256=manifest.manifest_sha256,
    )
