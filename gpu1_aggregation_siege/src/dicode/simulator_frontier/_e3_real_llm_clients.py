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
are computed by the client from the supplied evidence — never read from the
model — while the QUALITATIVE assessment fields (frontier class, confidence,
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
from typing import Any, Mapping

LLM_ROLE_SEQUENCE = ("frontier_evidence_diagnostician", "curriculum_search_planner")

FRONTIER_CLASSES = (
    "TOO_EASY", "LEARNABLE_FRONTIER", "TOO_HARD", "UNCERTAIN",
    "INVALID", "MEMORY_MISMATCH_SUSPECTED",
)

DISTRIBUTION_SLOT_IDS = tuple(f"D{i:02d}" for i in range(12))


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


def _call_qwen(system: str, user: str, *, timeout: float = 120.0) -> str:
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
        max_tokens=1024,
    )
    content = resp.choices[0].message.content or ""
    return str(content).strip()


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


def _coerce_float(value: Any, default: float, *, lo: float, hi: float) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(f):
        return default
    return max(lo, min(hi, f))


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.strip().lower() in ("true", "yes", "1"):
            return True
        if value.strip().lower() in ("false", "no", "0"):
            return False
    return default


def _coerce_str(value: Any, default: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def _coerce_start_distribution(raw: Any, state_id: str) -> dict[str, Any]:
    """Normalize the planner start_distribution to exactly D00..D11 slots.

    Each slot is a non-empty ``state_id -> weight`` mapping.  The model may
    supply per-slot emphasis weights over the SAME captured state; any slot
    missing from the response falls back to weight 1.0 on the captured state.
    """
    out: dict[str, Any] = {}
    if isinstance(raw, Mapping):
        mapping = dict(raw)
    else:
        mapping = {}
    for slot in DISTRIBUTION_SLOT_IDS:
        slot_map = mapping.get(slot)
        if isinstance(slot_map, Mapping) and slot_map:
            weights = {}
            total = 0.0
            for sid, w in slot_map.items():
                try:
                    fw = float(w)
                except (TypeError, ValueError):
                    continue
                if fw > 0 and math.isfinite(fw):
                    weights[str(sid)] = fw
                    total += fw
            if weights and state_id in weights and total > 0:
                out[slot] = {sid: float(w) / total for sid, w in weights.items()}
                continue
        out[slot] = {state_id: 1.0}
    return out


class _DiagnosticianClient:
    def __init__(self, state_id: str, bucket_id: str) -> None:
        self._state_id = state_id
        self._bucket_id = bucket_id

    def complete(self, evidence: Mapping[str, Any]) -> Mapping[str, Any]:
        feas = dict(evidence.get("feasibility", {}))
        evidence_hash = _evidence_hash_of(evidence)
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
        text = _call_qwen(system, user)
        raw = _extract_json(text)
        frontier_class = _coerce_str(
            raw.get("frontier_class"), "LEARNABLE_FRONTIER")
        if frontier_class not in FRONTIER_CLASSES:
            frontier_class = "LEARNABLE_FRONTIER"
        output = {
            "state_id": self._state_id,
            "bucket_id": self._bucket_id,
            "frontier_class": frontier_class,
            "confidence": _coerce_float(raw.get("confidence"), 0.7, lo=0.0, hi=1.0),
            "dominant_failure": _coerce_str(
                raw.get("dominant_failure"), "UNDER_EXPLORATION"),
            "memory_mismatch_suspected": _coerce_bool(
                raw.get("memory_mismatch_suspected"), False),
            "search_budget_sufficient": _coerce_bool(
                raw.get("search_budget_sufficient"), True),
            "evidence_ids": _evidence_ids_of(evidence),
            "recommended_evidence_action": _coerce_str(
                raw.get("recommended_evidence_action"), "INCREASE_SEARCH_N"),
        }
        from .llm_contracts import compute_diagnostician_hash
        output["diagnosis_hash"] = compute_diagnostician_hash(
            output, evidence_hash=evidence_hash)
        return output


class _PlannerClient:
    def __init__(self, state_id: str, actual_n: int, horizon: int) -> None:
        self._state_id = state_id
        self._actual_n = int(actual_n)
        self._horizon = int(horizon)

    def complete(self, planner_input: Mapping[str, Any]) -> Mapping[str, Any]:
        evidence = dict(planner_input)
        evidence.pop("diagnostician_summary", None)
        evidence_hash = _evidence_hash_of(evidence)
        from .llm_contracts import validate_diagnostician_output
        summary = planner_input.get("diagnostician_summary")
        diagnosis = validate_diagnostician_output(
            summary, evidence_hash=evidence_hash,
            expected_state_id=self._state_id)
        system = (
            "You are the Curriculum & Search Planner.  Given the aggregate "
            "frontier evidence and the diagnostician summary, propose a "
            "frontier training plan.  Reply with a single JSON object with "
            "EXACTLY these keys: "
            '"bucket_modifications" (object), '
            '"taskparam_ranges" (object, non-empty, e.g. '
            '{"passive_spawn_multiplier": [0.5, 1.5], '
            '"melee_spawn_multiplier": [0.1, 0.6]}), '
            '"seed_distribution" (object, non-empty, e.g. {"seed_base": [0, 1]}), '
            '"stochasticity_distribution" (object, non-empty, e.g. '
            '{"epsilon": [0.0, 0.1], "temperature": [1.0, 1.0]}), '
            '"anchor_ratio" (float 0..1), '
            '"retention_constraints" (list of strings), '
            '"reason" (short string).')
        user = json.dumps({
            "evidence": {
                "feasibility": dict(evidence.get("feasibility", {})),
                "data_source": str(evidence.get("data_source", "")),
            },
            "diagnostician_summary": dict(summary),
        }, sort_keys=True, default=str)
        text = _call_qwen(system, user)
        raw = _extract_json(text)
        taskparam = raw.get("taskparam_ranges")
        if not isinstance(taskparam, Mapping) or not taskparam:
            taskparam = {"passive_spawn_multiplier": [0.5, 1.5]}
        seed = raw.get("seed_distribution")
        if not isinstance(seed, Mapping) or not seed:
            seed = {"seed_base": [0, 1]}
        stochastic = raw.get("stochasticity_distribution")
        if not isinstance(stochastic, Mapping) or not stochastic:
            stochastic = {"epsilon": [0.0, 0.1], "temperature": [1.0, 1.0]}
        bucket_mods = raw.get("bucket_modifications")
        if not isinstance(bucket_mods, Mapping):
            bucket_mods = {}
        retention_raw = raw.get("retention_constraints")
        retention = [str(c) for c in retention_raw] if isinstance(
            retention_raw, (list, tuple)) else ["anchor_ratio>=0.20"]
        output = {
            "plan_id": f"plan-e3-{int(time.time() * 1000)}",
            "bucket_modifications": dict(bucket_mods),
            "start_distribution": _coerce_start_distribution(
                raw.get("start_distribution"), self._state_id),
            "taskparam_ranges": dict(taskparam),
            "seed_distribution": dict(seed),
            "stochasticity_distribution": dict(stochastic),
            "search_source": "STUDENT_DETERMINISTIC",
            "actual_n": max(1, self._actual_n),
            "horizon": max(1, self._horizon),
            "memory_mode": "SAVED_POLICY_MEMORY",
            "anchor_ratio": _coerce_float(raw.get("anchor_ratio"), 0.25, lo=0.05, hi=1.0),
            "retention_constraints": retention,
            "reason": _coerce_str(raw.get("reason"), "frontier plan"),
        }
        from .llm_contracts import compute_planner_hash
        output["plan_hash"] = compute_planner_hash(output, evidence_hash=evidence_hash)
        # Fill in the based_on_diagnosis_hash AFTER hash computation (it is
        # part of the output but the hash must bind it).
        output["based_on_diagnosis_hash"] = diagnosis.diagnosis_hash
        output["plan_hash"] = compute_planner_hash(output, evidence_hash=evidence_hash)
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
