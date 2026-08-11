"""REAL two-LLM client factory for the E3 real smoke (QWEN/DashScope).

The factory is the director-approved ``module:attr`` entry point that the
``AuthorizedTwoLLMRuntime`` resolves and source-hash-binds.  It builds the
two role clients (``frontier_evidence_diagnostician`` and
``curriculum_search_planner``) on the REAL authorized transport configured by
``~/.qwen_env`` (``QWEN_MODEL`` / ``OPENAI_BASE_URL`` / ``DASHSCOPE_API_KEY``
/ ``OPENAI_API_KEY``).

Every ``complete()`` call issues ONE real model call and returns a STRICT
schema-compliant mapping (the diagnostics/planner validators recompute the
hashes and reject any drift).  The mechanically-derived fields (state_id,
evidence ids, actual-N, horizon, memory mode, anchor ratio and the hashes)
are computed by the client from the supplied evidence 鈥?never read from the
model 鈥?while the QUALITATIVE assessment fields (frontier class, confidence,
dominant failure, bucket modifications, taskparam ranges, seed/stochasticity
distributions, retention constraints, reason) come from the real model
response and are coerced into the schema with fail-closed validation.

The factory NEVER falls back to a fake: a transport misconfiguration raises
``RuntimeError`` so the production path stays honest.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
import threading
from typing import Any, Mapping

LLM_ROLE_SEQUENCE = ("frontier_evidence_diagnostician", "curriculum_search_planner")

FRONTIER_CLASSES = (
    "TOO_EASY", "LEARNABLE_FRONTIER", "TOO_HARD", "UNCERTAIN",
    "INVALID", "MEMORY_MISMATCH_SUSPECTED",
)

DISTRIBUTION_SLOT_IDS = tuple(f"D{i:02d}" for i in range(12))
_AUDIT = threading.local()

def clear_audit_events() -> None:
    _AUDIT.events = []

def drain_audit_events() -> list[dict[str, Any]]:
    events = list(getattr(_AUDIT, "events", [])); _AUDIT.events = []; return events

def _audit_event(**event: Any) -> None:
    if not hasattr(_AUDIT, "events"): _AUDIT.events = []
    _AUDIT.events.append(dict(event))


def _evidence_hash_of(evidence: Mapping[str, Any]) -> str:
    import hashlib
    blob = json.dumps(dict(evidence), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    import hashlib
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _state_id_of(evidence: Mapping[str, Any]) -> str:
    feas = dict(evidence.get("feasibility", {}))
    return str(feas.get("state_id", ""))


def _bucket_id_of(evidence: Mapping[str, Any]) -> str:
    summary = dict(evidence.get("archive_summary", {}))
    bucket = summary.get("bucket_id", "")
    if not bucket:
        state_id = _state_id_of(evidence)
        bucket = f"bucket:{state_id[:16]}" if state_id else "bucket:unknown"
    return str(bucket)


def _evidence_ids_of(evidence: Mapping[str, Any]) -> list[str]:
    summary = dict(evidence.get("archive_summary", {}))
    ids = summary.get("evidence_ids", ())
    if ids:
        return [str(i) for i in ids]
    state_id = _state_id_of(evidence)
    return [state_id] if state_id else ["no-state"]


def _call_qwen(system: str, user: str, *, timeout: float = 120.0,
               max_tokens: int = 1024) -> Mapping[str, Any]:
    """One REAL model call on the authorized QWEN/DashScope transport."""
    model = os.environ.get("QWEN_MODEL", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "")
    api_key = os.environ.get("DASHSCOPE_API_KEY", "") or os.environ.get(
        "OPENAI_API_KEY", "")
    if not model or not base_url or not api_key:
        raise RuntimeError(
            "REAL_LLM_TRANSPORT_MISCONFIGURED: QWEN_MODEL/OPENAI_BASE_URL and a "
            "DashScope API key are required for the real two-LLM roles "
            "(never falls back to a fake)")
    if len(system.encode("utf-8")) + len(user.encode("utf-8")) > 72_000:
        raise RuntimeError("REAL_LLM_INPUT_TOO_LARGE: serialized input exceeds 72KB")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "REAL_LLM_TRANSPORT_MISSING_OPENAI: the openai package is required "
            f"for the real transport: {exc!r}") from exc
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        max_tokens=int(max_tokens),
    )
    content = resp.choices[0].message.content or ""
    usage_obj = getattr(resp, "usage", None)
    usage = {
        "prompt_tokens": int(getattr(usage_obj, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage_obj, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage_obj, "total_tokens", 0) or 0),
    }
    if usage["total_tokens"] > 20_000:
        raise RuntimeError("REAL_LLM_TOTAL_TOKEN_CAP_EXCEEDED")
    return {"content": str(content).strip(),
            "requested_model": model,
            "returned_model": str(getattr(resp, "model", model) or model),
            "usage": usage}


def _journal_for(role: str, evidence_hash: str):
    from .e3_durable_llm_journal import DurablePaidCallJournal
    path = os.environ.get("E3_LLM_JOURNAL_PATH", "")
    if not path:
        return None, "", {}
    source = os.environ.get("E3_SOURCE_COMMIT", "UNBOUND")
    candidate = os.environ.get("E3_CANDIDATE_ID", "UNBOUND")
    session = int(os.environ.get("E3_SESSION_IDX", "0"))
    provider = os.environ.get("E3_LLM_PROVIDER", "dashscope")
    model = os.environ.get("QWEN_MODEL", "")
    impl = os.environ.get("E3_CLIENT_FACTORY_IMPLEMENTATION_HASH", "")
    journal = DurablePaidCallJournal(path)
    key = journal.composite_key(source_commit=source, candidate=candidate,
                                session=session, evidence_hash=evidence_hash,
                                role=role, provider=provider,
                                requested_model=model,
                                client_implementation_hash=impl)
    identity = {"source_commit": source, "candidate": candidate,
                "session": session, "evidence_hash": evidence_hash,
                "role": role, "provider": provider,
                "requested_model": model,
                "client_implementation_hash": impl}
    return journal, key, identity


def _extract_json(text: str) -> dict[str, Any]:
    """Extract the first JSON object from a model response (fail closed)."""
    candidates = []
    # fenced json block first
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        candidates.append(m.group(1))
    # a top-level { ... } region
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        candidates.append(m.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    raise RuntimeError(
        f"REAL_LLM_OUTPUT_UNPARSEABLE: no JSON object found in model response "
        f"(head: {text[:300]!r})")


def _strict_float(value: Any, *, lo: float, hi: float, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"LLM field {field} must be numeric")
    value = float(value)
    if not math.isfinite(value) or value < lo or value > hi:
        raise ValueError(f"LLM field {field} outside finite range")
    return value


def _strict_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"LLM field {field} must be boolean")
    return value


def _strict_str(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"LLM field {field} must be non-empty string")
    return value.strip()


def _strict_start_distribution(raw: Any, state_id: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != set(DISTRIBUTION_SLOT_IDS):
        raise ValueError("start_distribution must contain exactly D00..D11")
    out = {}
    for slot in DISTRIBUTION_SLOT_IDS:
        slot_map = raw[slot]
        if not isinstance(slot_map, Mapping) or not slot_map or state_id not in slot_map:
            raise ValueError(f"start_distribution {slot} missing current state")
        weights = {}
        total = 0.0
        for sid, weight in slot_map.items():
            if not isinstance(sid, str) or isinstance(weight, bool) or not isinstance(weight, (int, float)):
                raise ValueError(f"invalid start_distribution weight in {slot}")
            weight = float(weight)
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError(f"invalid start_distribution weight in {slot}")
            weights[sid] = weight; total += weight
        out[slot] = {sid: weight / total for sid, weight in weights.items()}
    return out


def _strict_finite_structure(value: Any, field: str) -> Any:
    """Reject empty/non-finite/boolean values in seed-like distributions."""
    if isinstance(value, bool):
        raise ValueError(f"LLM field {field} must not contain booleans")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"LLM field {field} contains non-finite number")
    if isinstance(value, Mapping):
        if not value:
            raise ValueError(f"LLM field {field} must be non-empty")
        return {str(k): _strict_finite_structure(v, f"{field}.{k}")
                for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError(f"LLM field {field} must be non-empty")
        return [_strict_finite_structure(v, field) for v in value]
    if isinstance(value, (int, float, str)):
        return value
    raise ValueError(f"LLM field {field} has unsupported value type")


class _DiagnosticianClient:
    def __init__(self, state_id: str, bucket_id: str) -> None:
        self._state_id = state_id
        self._bucket_id = bucket_id

    def complete(self, evidence: Mapping[str, Any]) -> Mapping[str, Any]:
        feas = dict(evidence.get("feasibility", {}))
        evidence_hash = _evidence_hash_of(evidence)
        prompt_hash = _canonical_sha256({"evidence": {"feasibility": feas,
            "data_source": str(evidence.get("data_source", ""))}})
        journal, key, identity = _journal_for("frontier_evidence_diagnostician", prompt_hash)
        if journal is not None:
            cached = journal.lookup(key, identity=identity)
            if cached is not None:
                _audit_event(role=identity["role"], key=key, paid_new=False, reused=True,
                             requested_model=cached["requested_model"], returned_model=cached["returned_model"],
                             usage=cached["usage"], response_content_hash=cached["response_content_hash"],
                             validated_output_hash=cached["validated_output_hash"])
                return dict(cached["validated_output"])
            expected_preseed_key = os.environ.get("E3_PRESEEDED_DIAGNOSTIC_KEY", "")
            if expected_preseed_key:
                raise RuntimeError("PRESEED_DIAGNOSTIC_EVIDENCE_MISMATCH")
        system = (
            "You are the Frontier Evidence Diagnostician in a curriculum "
            "learning system.  You read ONLY aggregate feasibility evidence "
            "(never actions, routes, logits or hidden states).  Reply with a "
            "single JSON object with EXACTLY these keys: "
            '"frontier_class" (one of ' + ", ".join(FRONTIER_CLASSES) + '), '
            '"confidence" (float 0..1), "dominant_failure" (short string), '
            '"memory_mismatch_suspected" (true/false), '
            '"search_budget_sufficient" (true/false), '
            '"recommended_evidence_action" (short string).')
        user = json.dumps({
            "evidence": {
                "feasibility": feas,
                "data_source": str(evidence.get("data_source", "")),
            }
        }, sort_keys=True, default=str)
        text = _call_qwen(system, user, max_tokens=1024)
        if str(text.get("returned_model", "")) != str(text.get("requested_model", "")):
            raise RuntimeError("REAL_LLM_RETURNED_MODEL_MISMATCH")
        raw = _extract_json(str(text["content"])) if isinstance(text, Mapping) else _extract_json(str(text))
        diag_keys = {"frontier_class", "confidence", "dominant_failure",
                     "memory_mismatch_suspected", "search_budget_sufficient",
                     "recommended_evidence_action"}
        if set(raw) != diag_keys:
            raise ValueError("diagnostician output keys mismatch")
        frontier_class = _strict_str(raw["frontier_class"], "frontier_class")
        if frontier_class not in FRONTIER_CLASSES:
            raise ValueError("diagnostician frontier_class enum invalid")
        output = {
            "state_id": self._state_id,
            "bucket_id": self._bucket_id,
            "frontier_class": frontier_class,
            "confidence": _strict_float(raw["confidence"], lo=0.0, hi=1.0, field="confidence"),
            "dominant_failure": _strict_str(raw["dominant_failure"], "dominant_failure"),
            "memory_mismatch_suspected": _strict_bool(raw["memory_mismatch_suspected"], "memory_mismatch_suspected"),
            "search_budget_sufficient": _strict_bool(raw["search_budget_sufficient"], "search_budget_sufficient"),
            "evidence_ids": _evidence_ids_of(evidence),
            "recommended_evidence_action": _strict_str(raw["recommended_evidence_action"], "recommended_evidence_action"),
        }
        from .llm_contracts import compute_diagnostician_hash
        output["diagnosis_hash"] = compute_diagnostician_hash(
            output, evidence_hash=evidence_hash)
        journal, key, identity = _journal_for("frontier_evidence_diagnostician", prompt_hash)
        if journal is not None:
            # Persist the validated output immediately before returning.
            usage = text.get("usage", {}) if isinstance(text, Mapping) else {}
            requested = text.get("requested_model", os.environ.get("QWEN_MODEL", "")) if isinstance(text, Mapping) else os.environ.get("QWEN_MODEL", "")
            returned = text.get("returned_model", requested) if isinstance(text, Mapping) else requested
            cached = journal.lookup(key, identity=identity)
            if cached is None:
                journal.record_success(key=key, identity=identity,
                                       returned_model=str(returned), usage=usage,
                                       validated_output=output,
                                       response_content=str(text.get("content", "")))
                entry = journal.lookup(key, identity=identity)
                _audit_event(role=identity["role"], key=key, paid_new=True, reused=False,
                             requested_model=entry["requested_model"], returned_model=entry["returned_model"],
                             usage=entry["usage"], response_content_hash=entry["response_content_hash"],
                             validated_output_hash=entry["validated_output_hash"])
        return output


class _PlannerClient:
    def __init__(self, state_id: str, actual_n: int, horizon: int) -> None:
        self._state_id = state_id
        self._actual_n = int(actual_n)
        self._horizon = int(horizon)

    def complete(self, planner_input: Mapping[str, Any]) -> Mapping[str, Any]:
        full_prompt_hash = _canonical_sha256(dict(planner_input))
        evidence = dict(planner_input)
        evidence.pop("diagnostician_summary", None)
        evidence_hash = _evidence_hash_of(evidence)
        journal, key, identity = _journal_for("curriculum_search_planner", full_prompt_hash)
        if journal is not None:
            cached = journal.lookup(key, identity=identity)
            if cached is not None:
                _audit_event(role=identity["role"], key=key, paid_new=False, reused=True,
                             requested_model=cached["requested_model"], returned_model=cached["returned_model"],
                             usage=cached["usage"], response_content_hash=cached["response_content_hash"],
                             validated_output_hash=cached["validated_output_hash"])
                return dict(cached["validated_output"])
        from .llm_contracts import validate_diagnostician_output
        summary = planner_input.get("diagnostician_summary")
        diagnosis = validate_diagnostician_output(
            summary, evidence_hash=evidence_hash,
            expected_state_id=self._state_id)
        from .production_task_materializer import TASKPARAM_FIELDS, LOWER_BOUNDS, resolve_taskparams
        system = (
            "You are the Curriculum & Search Planner.  Given the aggregate "
            "frontier evidence and the diagnostician summary, propose a "
            "frontier training plan.  Reply with a single JSON object with "
            "EXACTLY these keys: "
            '"bucket_modifications" (object), '
            '"taskparam_ranges" (object with EXACTLY the 12 TaskParams keys '
            'passive_spawn_multiplier, melee_spawn_multiplier, ranged_spawn_multiplier, '
            'mob_health_multiplier, mob_damage_multiplier, melee_trigger_distance, '
            'monsters_killed_to_clear_level, needs_depletion_multiplier, '
            'health_recover_multiplier, health_loss_multiplier, mana_recover_multiplier, '
            'growing_plants_age; each value is a scalar or [lo, hi]; the '
            'integer fields melee_trigger_distance, '
            'monsters_killed_to_clear_level, and growing_plants_age must use '
            'JSON integer scalars/endpoints), and each resolved midpoint must '
            'be at least its canonical taskparam_lower_bounds value, '
            '"start_distribution" (object with exactly D00..D11 slots; each '
            'slot is non-empty, finite positive weights and contains the exact '
            'required_current_state_id supplied below; every D00..D11 key is mandatory), '
            '"seed_distribution" (object, non-empty, e.g. {"seed_base": [0, 1]}), '
            '"stochasticity_distribution" (object, non-empty, e.g. '
            '{"epsilon": [0.0, 0.1], "temperature": [1.0, 1.0]}), '
            '"anchor_ratio" (float 0..1), '
            '"retention_constraints" (list of strings), '
            '"reason" (short string); reply with no other keys.')
        user = json.dumps({
            "required_current_state_id": self._state_id,
            "taskparam_lower_bounds": dict(LOWER_BOUNDS),
            "start_distribution_template": {
                "D00": {self._state_id: 1.0}, "D01": {self._state_id: 1.0},
                "D02": {self._state_id: 1.0}, "D03": {self._state_id: 1.0},
                "D04": {self._state_id: 1.0}, "D05": {self._state_id: 1.0},
                "D06": {self._state_id: 1.0}, "D07": {self._state_id: 1.0},
                "D08": {self._state_id: 1.0}, "D09": {self._state_id: 1.0},
                "D10": {self._state_id: 1.0}, "D11": {self._state_id: 1.0},
            },
            "evidence": {
                "feasibility": dict(evidence.get("feasibility", {})),
                "data_source": str(evidence.get("data_source", "")),
            },
            "diagnostician_summary": dict(summary),
        }, sort_keys=True, default=str)
        text = _call_qwen(system, user, max_tokens=4096)
        if str(text.get("returned_model", "")) != str(text.get("requested_model", "")):
            raise RuntimeError("REAL_LLM_RETURNED_MODEL_MISMATCH")
        raw = _extract_json(str(text["content"])) if isinstance(text, Mapping) else _extract_json(str(text))
        planner_keys = {"bucket_modifications", "taskparam_ranges", "seed_distribution",
                        "stochasticity_distribution", "anchor_ratio",
                        "retention_constraints", "reason", "start_distribution"}
        if set(raw) != planner_keys:
            raise ValueError("planner output keys mismatch")
        taskparam = raw.get("taskparam_ranges")
        if not isinstance(taskparam, Mapping) or set(taskparam) != set(TASKPARAM_FIELDS):
            raise ValueError("planner taskparam_ranges must contain exactly all 12 TaskParams fields")
        resolve_taskparams(dict(taskparam), distribution_id="planner",
                          plan_hash="planner")
        seed = _strict_finite_structure(raw["seed_distribution"], "seed_distribution")
        stochastic = _strict_finite_structure(
            raw["stochasticity_distribution"], "stochasticity_distribution")
        if not isinstance(seed, Mapping) or not isinstance(stochastic, Mapping):
            raise ValueError("seed/stochasticity distributions must be mappings")
        bucket_mods = raw.get("bucket_modifications")
        if not isinstance(bucket_mods, Mapping):
            raise ValueError("bucket_modifications must be mapping")
        retention_raw = raw.get("retention_constraints")
        if not isinstance(retention_raw, (list, tuple)) or not retention_raw or any(not isinstance(c, str) or not c.strip() for c in retention_raw):
            raise ValueError("retention_constraints must be non-empty strings")
        retention = list(retention_raw)
        start_distribution = _strict_start_distribution(raw["start_distribution"], self._state_id)
        output = {
            "plan_id": f"plan-e3-{int(time.time() * 1000)}",
            "bucket_modifications": dict(bucket_mods),
            "start_distribution": start_distribution,
            "taskparam_ranges": dict(taskparam),
            "seed_distribution": dict(seed),
            "stochasticity_distribution": dict(stochastic),
            "search_source": "STUDENT_DETERMINISTIC",
            "actual_n": max(1, self._actual_n),
            "horizon": max(1, self._horizon),
            "memory_mode": "SAVED_POLICY_MEMORY",
            "anchor_ratio": _strict_float(raw["anchor_ratio"], lo=0.0, hi=1.0, field="anchor_ratio"),
            "retention_constraints": retention,
            "reason": _strict_str(raw["reason"], "reason"),
        }
        from .llm_contracts import compute_planner_hash
        output["plan_hash"] = compute_planner_hash(output, evidence_hash=evidence_hash)
        # Fill in the based_on_diagnosis_hash AFTER hash computation (it is
        # part of the output but the hash must bind it).
        output["based_on_diagnosis_hash"] = diagnosis.diagnosis_hash
        output["plan_hash"] = compute_planner_hash(output, evidence_hash=evidence_hash)
        journal, key, identity = _journal_for("curriculum_search_planner", full_prompt_hash)
        if journal is not None:
            usage = text.get("usage", {}) if isinstance(text, Mapping) else {}
            requested = text.get("requested_model", os.environ.get("QWEN_MODEL", "")) if isinstance(text, Mapping) else os.environ.get("QWEN_MODEL", "")
            returned = text.get("returned_model", requested) if isinstance(text, Mapping) else requested
            cached = journal.lookup(key, identity=identity)
            if cached is None:
                journal.record_success(key=key, identity=identity,
                                       returned_model=str(returned), usage=usage,
                                       validated_output=output,
                                       response_content=str(text.get("content", "")))
                entry = journal.lookup(key, identity=identity)
                _audit_event(role=identity["role"], key=key, paid_new=True, reused=False,
                             requested_model=entry["requested_model"], returned_model=entry["returned_model"],
                             usage=entry["usage"], response_content_hash=entry["response_content_hash"],
                             validated_output_hash=entry["validated_output_hash"])
        return output


def client_factory(roles: Any) -> Mapping[str, Any]:
    """Build the role -> real client mapping (never any fake)."""
    from .llm_contracts import LLM_ROLE_SEQUENCE as SEQ
    roles = tuple(roles)
    if tuple(roles) != SEQ:
        raise RuntimeError(f"expected role sequence {SEQ}, got {roles!r}")
    # The state/budget are bound at construction from the environment the
    # driver sets before running the smoke (never from the model).
    state_id = os.environ.get("E3_FRONTIER_STATE_ID", "")
    actual_n = int(os.environ.get("E3_ACTUAL_N", "8"))
    horizon = int(os.environ.get("E3_HORIZON", "16"))
    bucket_id = os.environ.get("E3_FRONTIER_BUCKET_ID", f"bucket:{state_id[:16]}")
    return {
        SEQ[0]: _DiagnosticianClient(state_id, bucket_id),
        SEQ[1]: _PlannerClient(state_id, actual_n, horizon),
    }
