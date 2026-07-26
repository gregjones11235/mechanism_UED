"""Read-only adapter: REAL bundle judgments -> canonical RoleJudgment + audit envelope.

Spec section 6: preserve raw records untouched; record glm role-label normalization
explicitly (raw_role_label / canonical_role_label / normalization_reason /
normalization_log_hash). Nothing is silently coerced:

  * parse_status != "ok"              -> AdapterError (fail closed)
  * critic_reject has NO legacy bit   -> DERIVED ONLY under a caller-supplied,
                                         explicitly named rule (decision_reject |
                                         flags_too_hard); flagged derived=True.
                                         There is NO implicit default:
                                         CRITIC_REJECT_POLICY=UNDECIDED, so a
                                         missing policy for critic records ->
                                         AdapterError CRITIC_POLICY_REQUIRED
                                         (fail closed; spec D052_PREMERGE_CORRECTION_V2)
  * unknown rule string               -> AdapterError UNKNOWN_RULE
  * fields with no canonical home     -> audit envelope (never dropped)

The historical legacy replay (replay.py) NEVER uses this adapter: it consumes the
raw critic_penalty exactly as the legacy selector did, so no derivation rule can
ever alter a historical replay anchor.

The adapter NEVER reads or mutates bundle files except to parse them; the original
record is embedded verbatim in the envelope for audit.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from d052.reconciliation.real_bundle import sha256_hex
from d052.schemas.roles import RoleJudgment, ScoringRole


class AdapterError(Exception):
    PARSE_NOT_OK = "PARSE_NOT_OK"
    UNKNOWN_RULE = "UNKNOWN_RULE"
    COVERAGE_GAP = "COVERAGE_GAP"
    CRITIC_POLICY_REQUIRED = "CRITIC_POLICY_REQUIRED"

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"[{code}] {message}")


#: critic_reject CANDIDATE derivation rules. The legacy schema has no hard-veto
#: bit, so any canonical critic_reject is a DERIVATION, never a raw field. Both
#: rules are CANDIDATES only: CRITIC_REJECT_POLICY=UNDECIDED — the future
#: canonical science protocol must freeze ONE of them explicitly before any real
#: canonical judgment conversion is authorized. Neither rule is used by the
#: historical legacy replay.
CRITIC_REJECT_RULES: Dict[str, Callable[[dict], bool]] = {
    "decision_reject": lambda r: r.get("decision") == "reject",
    "flags_too_hard": lambda r: bool((r.get("flags") or {}).get("too_hard")),
}

#: FAIL CLOSED: there is NO implicit default critic_reject policy. Callers must
#: name a rule explicitly; critic records without one raise CRITIC_POLICY_REQUIRED.
DEFAULT_CRITIC_REJECT_RULE: Optional[str] = None

#: fields that have no canonical RoleJudgment home; kept verbatim for audit
ENVELOPE_FIELDS = ("anon_id", "arm", "attempts", "decision", "flags",
                   "judgment_hash_sha256", "model_requested", "parse_status",
                   "role_label_in_raw", "role_label_normalized_to",
                   "source_file", "task_id")


class RoleNormalizationRecord(BaseModel):
    """Explicit glm role-echo normalization record (spec section 6)."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    arm: str
    raw_role_label: str
    canonical_role_label: str
    normalized: bool
    normalization_reason: str
    #: stable hash over this record's content (minus this field)
    normalization_log_hash: str = ""

    def compute_log_hash(self) -> str:
        payload = {
            "task_id": self.task_id, "arm": self.arm,
            "raw_role_label": self.raw_role_label,
            "canonical_role_label": self.canonical_role_label,
            "normalized": self.normalized,
            "normalization_reason": self.normalization_reason,
        }
        return sha256_hex(payload)


class AdaptedJudgment(BaseModel):
    """One bundle record adapted to canonical form, with full audit trail."""

    model_config = ConfigDict(extra="forbid")

    role_judgment: RoleJudgment
    #: the ORIGINAL bundle record, embedded verbatim (read-only preservation)
    original_record: dict
    #: fields with no canonical home, preserved verbatim
    audit_envelope: dict
    normalization: RoleNormalizationRecord
    #: derivation provenance (empty unless something was derived)
    derived: dict = Field(default_factory=dict)


def _normalization(record: dict) -> RoleNormalizationRecord:
    raw = str(record.get("role_label_in_raw", ""))
    canon = str(record.get("role_label_normalized_to", record.get("role", "")))
    normalized = raw != canon
    if normalized:
        reason = (f"glm role-echo quirk: raw output labeled role {raw!r}; request "
                  f"schema/role pins require {canon!r}; selector consumes ONLY the "
                  f"pinned role, the echo is decorative")
    else:
        reason = "raw role label already canonical; no normalization applied"
    rec = RoleNormalizationRecord(
        task_id=record["task_id"], arm=record["arm"],
        raw_role_label=raw, canonical_role_label=canon,
        normalized=normalized, normalization_reason=reason)
    rec.normalization_log_hash = rec.compute_log_hash()
    return rec


def adapt_judgment(record: dict, *,
                   critic_reject_rule: Optional[str] = DEFAULT_CRITIC_REJECT_RULE,
                   prompt_version: Optional[str] = None) -> AdaptedJudgment:
    """Adapt ONE flattened bundle judgment record (read-only; fail-closed).

    critic_reject_rule MUST be named explicitly whenever a critic record is
    adapted: None (the only default) raises CRITIC_POLICY_REQUIRED; any string
    outside CRITIC_REJECT_RULES raises UNKNOWN_RULE. Non-critic records never
    carry a critic_reject derivation, regardless of the rule argument.
    """
    if record.get("parse_status") != "ok":
        raise AdapterError(AdapterError.PARSE_NOT_OK,
                           f"{record.get('arm')}/{record.get('task_id')}/"
                           f"{record.get('role')}: parse_status="
                           f"{record.get('parse_status')!r} != 'ok'")
    if critic_reject_rule is not None and critic_reject_rule not in CRITIC_REJECT_RULES:
        raise AdapterError(AdapterError.UNKNOWN_RULE,
                           f"unknown critic_reject_rule {critic_reject_rule!r}; "
                           f"legal: {sorted(CRITIC_REJECT_RULES)}")

    role = record["role"]
    kw = dict(role=ScoringRole(role), candidate_id=record["task_id"],
              scores={k: float(v) for k, v in record["raw_scores"].items()},
              rationale=record.get("short_reason", ""),
              provider=record.get("provider"),
              exact_model_id=record.get("model_returned"),
              prompt_version=prompt_version)
    derived: dict = {}
    if role == "critic":
        if critic_reject_rule is None:
            raise AdapterError(
                AdapterError.CRITIC_POLICY_REQUIRED,
                f"{record.get('arm')}/{record.get('task_id')}/critic: "
                f"critic_reject_rule must be specified explicitly; "
                f"CRITIC_REJECT_POLICY=UNDECIDED, there is no implicit default "
                f"(candidate rules: {sorted(CRITIC_REJECT_RULES)})")
        rule = CRITIC_REJECT_RULES[critic_reject_rule]
        kw["critic_reject"] = bool(rule(record))
        derived = {"critic_reject_rule": critic_reject_rule,
                   "critic_reject_value": kw["critic_reject"],
                   "derived": True,
                   "note": "legacy schema has no raw critic_reject bit; "
                           "DERIVED under the explicitly named rule, not raw"}

    rj = RoleJudgment.model_validate(kw)
    envelope = {k: record.get(k) for k in ENVELOPE_FIELDS}
    envelope["model_returned"] = record.get("model_returned")
    return AdaptedJudgment(role_judgment=rj, original_record=record,
                           audit_envelope=envelope,
                           normalization=_normalization(record), derived=derived)


def adapt_arm(records: List[dict], *,
              critic_reject_rule: Optional[str] = DEFAULT_CRITIC_REJECT_RULE,
              prompt_version: Optional[str] = None) -> List[AdaptedJudgment]:
    """Adapt one arm (96 records expected; 32 candidates x 3 roles).

    Fail closed for the WHOLE arm: if any record is a critic judgment and no
    explicit critic_reject_rule was given, nothing is adapted.
    """
    if critic_reject_rule is None and any(r.get("role") == "critic" for r in records):
        raise AdapterError(
            AdapterError.CRITIC_POLICY_REQUIRED,
            f"arm contains critic judgments but critic_reject_rule was not "
            f"specified; CRITIC_REJECT_POLICY=UNDECIDED -> fail closed "
            f"(candidate rules: {sorted(CRITIC_REJECT_RULES)})")
    out = [adapt_judgment(r, critic_reject_rule=critic_reject_rule,
                          prompt_version=prompt_version) for r in records]
    by_role: Dict[str, int] = {}
    for a in out:
        by_role[a.role_judgment.role.value] = by_role.get(a.role_judgment.role.value, 0) + 1
    if len(out) != 96 or any(v != 32 for v in by_role.values()) or len(by_role) != 3:
        raise AdapterError(AdapterError.COVERAGE_GAP,
                           f"expected 96 records (32x3 roles); got {len(out)}, {by_role}")
    return out


def normalization_log(adapted: List[AdaptedJudgment]) -> dict:
    """The deterministic normalization log + its stable hash (spec section 6)."""
    entries = [a.normalization.model_dump() for a in adapted]
    normalized = [e for e in entries if e["normalized"]]
    return {"records": entries,
            "n_normalized": len(normalized),
            "normalization_log_hash": sha256_hex(entries)}
