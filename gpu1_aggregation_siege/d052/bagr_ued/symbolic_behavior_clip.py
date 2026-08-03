"""Bounded, de-identified SYMBOLIC behavior clips for the Review Board
(CC3 fix2, task §9-§12).

fix1 handed the six review roles only anomaly dumps + clip metadata (clip id /
episode / span / anomaly provenance). fix2 adds the missing evidence layer:
per-step SYMBOLIC behavior evidence with provenance — so the board reasons
about WHAT the Student actually did (action semantic classes, threat distance
bands, safety status, health/resource/progress delta bands, event semantics,
terminal category) instead of only the anomaly label.

Hard data boundary (task §10):

  ALLOWED  — action semantic CLASSES, threat distance band, safety summary,
             health/resource/progress delta BANDS, event semantics, terminal
             category, a limited step window;
  FORBIDDEN — raw action integers, raw observation vectors, raw state leaf
             keys, full coordinate trajectories, full maps, exact ladder
             positions, formal FRONT/BACK/FULL state, Reference action
             sequences, expert actions, correct-action labels, policy logits,
             hidden state, reward-shaping targets, next-step advice.

Every payload runs through FormalEvaluationLeakageGuard +
TrajectorySupervisionGuard + source admissibility + payload hash validation +
step-count / serialized-size limits BEFORE the board may consume it. Over-limit
payloads are explicitly truncated (truncation_applied=true) or fail closed —
a full 4096-step trajectory NEVER reaches an LLM role.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from pydantic import Field, model_validator

from d052.bagr_ued import constants as C
from d052.bagr_ued.formal_evaluation_leakage_guard import (
    FormalEvaluationLeakageGuard)
from d052.bagr_ued.hashing import canonical_sha256
from d052.bagr_ued.trajectory_supervision_guard import TrajectorySupervisionGuard
from d052.schemas.common import CanonicalModel, is_sha256_hex

CLIP_SCHEMA_VERSION = "bagr_ued.symbolic_clip.v1"

#: deterministic MOCK provenance (declared mock — no real checkpoint / runner
#: identity exists this round; the fields are required so the schema +
#: tamper-detection contract is exercised end-to-end).
_MOCK_PROVENANCE_LABELS = (
    "student_checkpoint_sha256",
    "environment_descriptor_hash",
    "taskparams_hash",
    "generator_provenance_hash",
    "rollout_runner_sha256",
    "environment_lock_sha256",
)

#: the canonical provenance field set — every value MUST be a LOWER-CASE
#: full-64 sha256 hex digest (CC3 fix3 §8; enforced at the schema layer AND
#: re-checked over serialized dicts).
PROVENANCE_FIELDS = tuple(_MOCK_PROVENANCE_LABELS)

#: the ONLY legal safety_status vocabulary (CC3 fix3 §6)
SAFETY_STATUS_VOCABULARY = frozenset({"safe", "unsafe", "unknown"})


def mock_clip_provenance() -> Dict[str, str]:
    """Deterministic mock provenance hashes (labeled mock; not real identity)."""
    import hashlib
    return {label: hashlib.sha256(
        f"bagr_ued.mock_provenance.{label}".encode("utf-8")).hexdigest()
        for label in _MOCK_PROVENANCE_LABELS}


class SymbolicBehaviorClipError(Exception):
    CLIP_SOURCE_NOT_ADMISSIBLE = "CLIP_SOURCE_NOT_ADMISSIBLE"
    RAW_ACTION_INTEGER_EXPOSED = "RAW_ACTION_INTEGER_EXPOSED"
    RAW_STATE_EXPOSED = "RAW_STATE_EXPOSED"
    CLIP_STEP_LIMIT_EXCEEDED = "CLIP_STEP_LIMIT_EXCEEDED"
    CLIP_EVENT_LIMIT_EXCEEDED = "CLIP_EVENT_LIMIT_EXCEEDED"
    CLIP_RESOURCE_FIELD_LIMIT_EXCEEDED = "CLIP_RESOURCE_FIELD_LIMIT_EXCEEDED"
    CLIP_PAYLOAD_TOO_LARGE = "CLIP_PAYLOAD_TOO_LARGE"
    CLIP_PAYLOAD_HASH_MISMATCH = "CLIP_PAYLOAD_HASH_MISMATCH"
    CLIP_PROVENANCE_MISMATCH = "CLIP_PROVENANCE_MISMATCH"
    CLIP_GUARD_VIOLATION = "CLIP_GUARD_VIOLATION"
    #: CC3 fix3 codes
    CLIP_PAYLOAD_HASH_MISSING = "CLIP_PAYLOAD_HASH_MISSING"
    CLIP_PROVENANCE_FORMAT_INVALID = "CLIP_PROVENANCE_FORMAT_INVALID"
    CLIP_SAFETY_STATUS_INVALID = "CLIP_SAFETY_STATUS_INVALID"
    CLIP_TERMINAL_FABRICATED = "CLIP_TERMINAL_FABRICATED"
    CLIP_PER_EPISODE_LIMIT_EXCEEDED = "CLIP_PER_EPISODE_LIMIT_EXCEEDED"
    CLIP_PER_WINDOW_LIMIT_EXCEEDED = "CLIP_PER_WINDOW_LIMIT_EXCEEDED"
    CLIP_EXPECTED_PROVENANCE_NOT_BOUND = "CLIP_EXPECTED_PROVENANCE_NOT_BOUND"

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"[{code}] {message}")


@dataclass(frozen=True)
class ClipProvenanceIdentity:
    """IDENTITY_BOUND (CC3 fix3 §9): the ONLY admissible carrier of EXPECTED
    clip provenance.

    A plain dict of expected values is refused by validate_symbolic_clip_payload
    — expected provenance must be constructed through this bound type, which
    format-verifies EVERY value as a lower-case full-64 sha256 hex digest at
    construction. This closes the hole where a caller could "expect" a
    malformed / truncated / upper-case identity and silently compare strings.
    """

    student_checkpoint_sha256: str
    environment_descriptor_hash: str
    taskparams_hash: str
    generator_provenance_hash: str
    rollout_runner_sha256: str
    environment_lock_sha256: str

    def __post_init__(self) -> None:
        for label in PROVENANCE_FIELDS:
            value = getattr(self, label)
            if not is_sha256_hex(value):
                raise ValueError(
                    f"CLIP_IDENTITY_NOT_BOUND: {label} must be a lower-case "
                    f"full-64 sha256 hex digest, got {value!r}")

    def as_dict(self) -> Dict[str, str]:
        return {label: getattr(self, label) for label in PROVENANCE_FIELDS}


def identity_bound(provenance: Dict[str, str]) -> ClipProvenanceIdentity:
    """Construct the IDENTITY_BOUND expected-provenance carrier from a dict."""
    return ClipProvenanceIdentity(**{label: provenance[label]
                                     for label in PROVENANCE_FIELDS})


# raw-exposure detectors run over the SERIALIZED payload (task §10/§14):
# a raw action integer value under an action-ish key, or a raw state leaf /
# observation / inventory key anywhere, fails closed.
_ACTION_INT_KEY = re.compile(
    r"^(action|raw_action|action_int|action_integer|action_id)$", re.I)
_RAW_STATE_KEY = re.compile(
    r"(observation|obs_vector|state_vector|state_leaf|^leaf_|^idx_|"
    r"^index_|^inventory_\d+$|^i\d+$|^\d+$|^state\[|logits|hidden_state|"
    r"coordinate|trajectory_points|map_grid|ladder_position)", re.I)


class SymbolicBehaviorStep(CanonicalModel):
    """One bounded, de-identified step of Student behavior evidence."""

    step_offset: int = Field(ge=0)
    #: semantic CLASSES only — never a raw action integer
    action_semantic_classes: List[str] = Field(default_factory=list)
    hostile_distance_band: str = "unknown"
    #: safe / unsafe / unknown — the ONLY legal vocabulary (CC3 fix3 §6);
    #: a missing safety signal is "unknown", NEVER a defaulted "unsafe"
    safety_status: str = "unknown"
    health_delta_band: str = "unknown"
    resource_delta_bands: Dict[str, str] = Field(default_factory=dict)
    progress_delta_band: str = "unknown"
    event_semantics: List[str] = Field(default_factory=list)
    #: none / death / timeout / success / abandoned — set ONLY on the true
    #: terminal step of the episode; a truncated clip never carries it
    terminal_category: str = "none"

    @model_validator(mode="after")
    def _safety_vocabulary(self) -> "SymbolicBehaviorStep":
        if self.safety_status not in SAFETY_STATUS_VOCABULARY:
            raise SymbolicBehaviorClipError(
                SymbolicBehaviorClipError.CLIP_SAFETY_STATUS_INVALID,
                f"step {self.step_offset}: safety_status {self.safety_status!r}"
                f" not in {sorted(SAFETY_STATUS_VOCABULARY)}")
        return self


class SymbolicBehaviorClipPayload(CanonicalModel):
    """The full bounded symbolic clip the board receives (with provenance)."""

    clip_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    #: must be an ALLOWED training source (generative training env / synthetic
    #: test trace); formal FRONT/BACK/FULL sources fail closed
    source: str = Field(min_length=1)
    start_step: int = Field(ge=0)
    end_step: int = Field(ge=0)
    steps: List[SymbolicBehaviorStep] = Field(default_factory=list)
    student_checkpoint_sha256: str = Field(min_length=1)
    environment_descriptor_hash: str = Field(min_length=1)
    taskparams_hash: str = Field(min_length=1)
    generator_provenance_hash: str = Field(min_length=1)
    rollout_runner_sha256: str = Field(min_length=1)
    environment_lock_sha256: str = Field(min_length=1)
    #: content hash over the payload minus itself; tampering breaks it
    clip_payload_sha256: str = ""
    truncation_applied: bool = False
    schema_version: str = CLIP_SCHEMA_VERSION

    @model_validator(mode="after")
    def _limits_and_hash(self) -> "SymbolicBehaviorClipPayload":
        # CC3 fix3 §8: every provenance value is a LOWER-CASE full-64 sha256
        # hex digest — upper-case / truncated / non-hex identities are refused
        # at the schema layer (defense in depth: the validator below re-checks
        # serialized dicts that bypass model construction).
        for label in PROVENANCE_FIELDS:
            value = getattr(self, label)
            if not is_sha256_hex(value):
                raise SymbolicBehaviorClipError(
                    SymbolicBehaviorClipError.CLIP_PROVENANCE_FORMAT_INVALID,
                    f"clip {self.clip_id}: provenance {label} must be a "
                    f"lower-case full-64 sha256 hex digest, got {value!r}")
        if self.source not in C.ALLOWED_EVIDENCE_SOURCES:
            raise SymbolicBehaviorClipError(
                SymbolicBehaviorClipError.CLIP_SOURCE_NOT_ADMISSIBLE,
                f"clip {self.clip_id}: source {self.source!r} is not an "
                f"allowed training source "
                f"({sorted(C.ALLOWED_EVIDENCE_SOURCES)})")
        if self.end_step < self.start_step:
            raise ValueError(
                f"CLIP_SPAN_INVERTED: end_step={self.end_step} < "
                f"start_step={self.start_step}")
        if len(self.steps) > C.MAX_CLIP_STEPS:
            raise SymbolicBehaviorClipError(
                SymbolicBehaviorClipError.CLIP_STEP_LIMIT_EXCEEDED,
                f"clip {self.clip_id}: {len(self.steps)} steps > "
                f"MAX_CLIP_STEPS={C.MAX_CLIP_STEPS}")
        for s in self.steps:
            if len(s.event_semantics) > C.MAX_EVENT_SEMANTICS_PER_STEP:
                raise SymbolicBehaviorClipError(
                    SymbolicBehaviorClipError.CLIP_EVENT_LIMIT_EXCEEDED,
                    f"clip {self.clip_id} step {s.step_offset}: "
                    f"{len(s.event_semantics)} events > "
                    f"MAX_EVENT_SEMANTICS_PER_STEP="
                    f"{C.MAX_EVENT_SEMANTICS_PER_STEP}")
            if len(s.resource_delta_bands) > C.MAX_RESOURCE_FIELDS:
                raise SymbolicBehaviorClipError(
                    SymbolicBehaviorClipError.CLIP_RESOURCE_FIELD_LIMIT_EXCEEDED,
                    f"clip {self.clip_id} step {s.step_offset}: "
                    f"{len(s.resource_delta_bands)} resource fields > "
                    f"MAX_RESOURCE_FIELDS={C.MAX_RESOURCE_FIELDS}")
        if not self.clip_payload_sha256:
            object.__setattr__(self, "clip_payload_sha256",
                               clip_payload_hash(self))
        return self


def clip_payload_hash(payload: SymbolicBehaviorClipPayload) -> str:
    """Content hash over the payload minus the hash field itself."""
    dump = payload.model_dump()
    dump.pop("clip_payload_sha256", None)
    return canonical_sha256(dump)


# ---------------------------------------------------------------------------
# builder: BehaviorClip + evidence bundle -> bounded symbolic payload
# ---------------------------------------------------------------------------

_HEALTH_ORDER = {"critical": 0, "low": 1, "mid": 2, "high": 3}
_OUTCOME_TO_TERMINAL = {"death": "death", "timeout": "timeout",
                        "success": "success", "abandoned": "abandoned"}


def _delta_band(cur: str, prev: Optional[str], order: dict) -> str:
    if prev is None or cur not in order or prev not in order:
        return "unknown"
    d = order[cur] - order[prev]
    return "improved" if d > 0 else "worsened" if d < 0 else "unchanged"


def _ordinal_delta(cur, prev) -> str:
    if prev is None:
        return "unknown"
    try:
        d = float(cur) - float(prev)
    except (TypeError, ValueError):
        return "unknown"
    return "increased" if d > 0 else "decreased" if d < 0 else "unchanged"


def build_symbolic_clip_payload(bundle, clip, *,
                                provenance: Dict[str, str] | None = None
                                ) -> SymbolicBehaviorClipPayload:
    """Build a bounded, de-identified symbolic payload for one BehaviorClip.

    Reads ONLY the symbolic StepRecord contract (semantic classes +
    state_summary bands + env events) — never raw ints, never leaf indices
    (StepRecord already refuses those at intake). Truncates to MAX_CLIP_STEPS
    with truncation_applied=true.
    """
    prov = provenance if provenance is not None else mock_clip_provenance()
    episode = bundle.episode(clip.episode_id)
    span_steps = sorted(
        (s for s in episode.steps
         if clip.span.start_step <= s.step_index <= clip.span.end_step),
        key=lambda s: s.step_index)

    truncated = len(span_steps) > C.MAX_CLIP_STEPS
    span_steps = span_steps[:C.MAX_CLIP_STEPS]

    last_episode_step = max((s.step_index for s in episode.steps), default=0)
    terminal = _OUTCOME_TO_TERMINAL.get(episode.outcome or "", "none") \
        if episode.outcome else "none"

    steps: List[SymbolicBehaviorStep] = []
    prev_health, prev_resource, prev_progress = None, None, None
    for s in span_steps:
        summary = s.state_summary
        health = str(summary.get("health_band", "unknown"))
        resource = str(summary.get("resource_band", "unknown"))
        progress = summary.get("progress_ordinal")

        events = list(s.env_events)[:C.MAX_EVENT_SEMANTICS_PER_STEP]
        # CC3 fix3 §5: the terminal category rides ONLY on the episode's TRUE
        # last step. A truncated clip window that happens to end mid-episode
        # must NOT fake a terminal (no fabricated death / timeout / success
        # timing) — truncation is recorded via truncation_applied, never via
        # a synthesized terminal step.
        is_terminal_step = (s.step_index == last_episode_step)
        if truncated and is_terminal_step:
            # structurally unreachable (truncation implies the true last step
            # is outside the window) — keep the invariant explicit
            raise SymbolicBehaviorClipError(
                SymbolicBehaviorClipError.CLIP_TERMINAL_FABRICATED,
                f"clip {clip.clip_id}: truncated clip may not reach the true "
                f"terminal step {last_episode_step}")
        if summary.get("env_confirmed_safe") is True:
            safety = "safe"
        elif summary.get("env_confirmed_safe") is False or \
                summary.get("env_confirmed_unsafe") is True:
            safety = "unsafe"
        else:
            safety = "unknown"      # CC3 fix3 §6: no evidence -> unknown
        steps.append(SymbolicBehaviorStep(
            step_offset=s.step_index - clip.span.start_step,
            action_semantic_classes=list(s.action_semantic_classes),
            hostile_distance_band=str(
                summary.get("hostile_distance_band", "unknown")),
            safety_status=safety,
            health_delta_band=_delta_band(health, prev_health, _HEALTH_ORDER),
            resource_delta_bands={"resource": _delta_band(
                resource, prev_resource, _HEALTH_ORDER)},
            progress_delta_band=_ordinal_delta(progress, prev_progress),
            event_semantics=events,
            terminal_category=terminal if is_terminal_step else "none"))
        prev_health, prev_resource, prev_progress = health, resource, progress

    return SymbolicBehaviorClipPayload(
        clip_id=clip.clip_id,
        episode_id=clip.episode_id,
        source=bundle.source.value,
        start_step=clip.span.start_step,
        end_step=clip.span.end_step,
        steps=steps,
        truncation_applied=truncated,
        **prov)


# ---------------------------------------------------------------------------
# validation: guards + raw-exposure scan + provenance + hash + size (task §10)
# ---------------------------------------------------------------------------

def _raw_exposure_findings(dump: dict) -> List[dict]:
    """Scan the serialized payload for raw action integers / raw state keys."""
    findings: List[dict] = []

    def walk(node, path):
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(k, str):
                    if _ACTION_INT_KEY.match(k) and isinstance(v, int) \
                            and not isinstance(v, bool):
                        findings.append(dict(
                            code=SymbolicBehaviorClipError.
                                RAW_ACTION_INTEGER_EXPOSED,
                            path=f"{path}.{k}",
                            detail=f"raw action integer {v!r} under key "
                                   f"{k!r} — only semantic CLASSES allowed"))
                    if _RAW_STATE_KEY.search(k):
                        findings.append(dict(
                            code=SymbolicBehaviorClipError.RAW_STATE_EXPOSED,
                            path=f"{path}.{k}",
                            detail=f"raw state / observation key {k!r} — "
                                   f"only symbolic summary fields allowed"))
                walk(v, f"{path}.{k}")
        elif isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    walk(dump, "$")
    return findings


def dict_clip_payload_hash(dump: dict) -> str:
    """CC3 fix3 §4: content hash recomputed DIRECTLY over a serialized dict
    payload (minus the hash field). Dicts bypass model construction — the
    validator MUST recompute their SHA instead of trusting a carried value."""
    body = {k: v for k, v in dump.items() if k != "clip_payload_sha256"}
    return canonical_sha256(body)


def validate_symbolic_clip_payload(payload, *,
                                   expected_provenance:
                                       ClipProvenanceIdentity | None = None,
                                   leakage_guard:
                                       FormalEvaluationLeakageGuard | None =
                                       None,
                                   supervision_guard:
                                       TrajectorySupervisionGuard | None =
                                       None) -> dict:
    """Fail-closed validation of a symbolic clip payload (builder output or
    tampered dump). Returns a report dict {passed, findings} — raises
    SymbolicBehaviorClipError via ``assert_valid``.

    CC3 fix3 contracts:
      * §4 — a DICT payload has its payload SHA RECOMPUTED from the dict
        content (a missing/empty recorded hash fails closed);
      * §8 — every provenance field is re-verified as a lower-case full-64
        sha256 hex digest over the SERIALIZED payload;
      * §9 — expected_provenance must be an IDENTITY_BOUND
        ClipProvenanceIdentity; a plain dict fails closed.
    """
    findings: List[dict] = []
    dump = payload.model_dump() if hasattr(payload, "model_dump") else payload

    # 0. provenance FORMAT re-check over the serialized payload (catches
    #    dicts that never passed the model validators)
    for label in PROVENANCE_FIELDS:
        value = dump.get(label)
        if not is_sha256_hex(value):
            findings.append(dict(
                code=SymbolicBehaviorClipError.CLIP_PROVENANCE_FORMAT_INVALID,
                path=f"$.{label}",
                detail=f"provenance {label} must be a lower-case full-64 "
                       f"sha256 hex digest, got {value!r}"))

    # 1. raw-exposure scan over the serialized payload
    findings.extend(_raw_exposure_findings(dump))

    # 2. both guards over the payload
    leak = (leakage_guard or FormalEvaluationLeakageGuard()).scan(
        dump, label="symbolic_clip_payload")
    if not leak["passed"]:
        findings.extend(dict(code=SymbolicBehaviorClipError.
                             CLIP_GUARD_VIOLATION,
                             path=f.get("path", "$"),
                             detail=f"FormalEvaluationLeakageGuard: "
                                    f"{f.get('code')} {f.get('detail', '')}")
                        for f in leak["findings"])
    sup = (supervision_guard or TrajectorySupervisionGuard()).scan(
        dump, label="symbolic_clip_payload")
    if not sup["passed"]:
        findings.extend(dict(code=SymbolicBehaviorClipError.
                             CLIP_GUARD_VIOLATION,
                             path=f.get("path", "$"),
                             detail=f"TrajectorySupervisionGuard: "
                                    f"{f.get('code')} {f.get('detail', '')}")
                        for f in sup["findings"])

    # 3. limits (step window + serialized byte cap — never a full trajectory)
    steps = dump.get("steps", [])
    if len(steps) > C.MAX_CLIP_STEPS:
        findings.append(dict(
            code=SymbolicBehaviorClipError.CLIP_STEP_LIMIT_EXCEEDED,
            path="$.steps",
            detail=f"{len(steps)} steps > MAX_CLIP_STEPS={C.MAX_CLIP_STEPS}"))
    serialized = json.dumps(dump, sort_keys=True, separators=(",", ":"),
                            ensure_ascii=False, default=str)
    if len(serialized.encode("utf-8")) > C.MAX_SERIALIZED_PAYLOAD_BYTES:
        findings.append(dict(
            code=SymbolicBehaviorClipError.CLIP_PAYLOAD_TOO_LARGE,
            path="$",
            detail=f"serialized payload bytes exceed "
                   f"MAX_SERIALIZED_PAYLOAD_BYTES="
                   f"{C.MAX_SERIALIZED_PAYLOAD_BYTES}"))

    # 4. payload hash validation (tamper detection) — model and dict paths
    #    BOTH get a recomputed SHA (CC3 fix3 §4: a dict never gets a free
    #    pass on its carried hash; a missing/empty hash fails closed)
    if hasattr(payload, "clip_payload_sha256") and \
            not isinstance(payload, dict):
        if payload.clip_payload_sha256:
            recomputed = clip_payload_hash(payload)
            if recomputed != payload.clip_payload_sha256:
                findings.append(dict(
                    code=SymbolicBehaviorClipError.CLIP_PAYLOAD_HASH_MISMATCH,
                    path="$.clip_payload_sha256",
                    detail="recorded payload hash does not match payload "
                           "content (tampering or stale hash)"))
        else:
            findings.append(dict(
                code=SymbolicBehaviorClipError.CLIP_PAYLOAD_HASH_MISSING,
                path="$.clip_payload_sha256",
                detail="model payload carries no payload hash"))
    else:
        recorded = dump.get("clip_payload_sha256")
        if not (isinstance(recorded, str) and len(recorded) == 64):
            findings.append(dict(
                code=SymbolicBehaviorClipError.CLIP_PAYLOAD_HASH_MISSING,
                path="$.clip_payload_sha256",
                detail="dict symbolic clip carries no recorded payload hash — "
                       "the SHA must be recomputed and bound, never omitted"))
        else:
            recomputed = dict_clip_payload_hash(dump)
            if recomputed != recorded:
                findings.append(dict(
                    code=SymbolicBehaviorClipError.CLIP_PAYLOAD_HASH_MISMATCH,
                    path="$.clip_payload_sha256",
                    detail="dict clip payload hash recomputed from content "
                           "does not match the recorded hash (tampering or "
                           "stale hash)"))

    # 5. provenance validation (expected values, when supplied) — CC3 fix3
    #    §9: expected provenance must be IDENTITY_BOUND; a plain dict is
    #    refused fail-closed (never silently compared as strings)
    if expected_provenance is not None:
        if not isinstance(expected_provenance, ClipProvenanceIdentity):
            findings.append(dict(
                code=SymbolicBehaviorClipError.
                    CLIP_EXPECTED_PROVENANCE_NOT_BOUND,
                path="$",
                detail="expected_provenance must be an IDENTITY_BOUND "
                       "ClipProvenanceIdentity (construct via "
                       "identity_bound(...)); a plain dict is refused"))
        else:
            for key, expected in expected_provenance.as_dict().items():
                actual = dump.get(key)
                if actual != expected:
                    findings.append(dict(
                        code=SymbolicBehaviorClipError.CLIP_PROVENANCE_MISMATCH,
                        path=f"$.{key}",
                        detail=f"provenance {key} does not match the expected"
                               f" IDENTITY_BOUND value"))
    return dict(guard="SymbolicBehaviorClipValidation",
                passed=not findings,
                findings=findings)


def assert_valid_symbolic_clip_payload(payload, **kwargs) -> dict:
    report = validate_symbolic_clip_payload(payload, **kwargs)
    if not report["passed"]:
        first = report["findings"][0]
        raise SymbolicBehaviorClipError(
            first["code"],
            f"{first['code']} at {first['path']} ({first['detail']}); "
            f"total_findings={len(report['findings'])}")
    return report
