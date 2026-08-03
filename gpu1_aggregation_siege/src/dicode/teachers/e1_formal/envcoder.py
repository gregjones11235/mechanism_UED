"""Stage 4: independent EnvCoder prompt builder (OUTSIDE the board).

The EnvCoder is an independent artifact producer, NOT a board member.
Its prompt is assembled from a strict WHITELIST:

1. the fixed environment-authoring contract text;
2. the canonical TaskSpec rendering (identity + goals + axes);
3. seed examples, rotated deterministically by the spec variant.

NO board content (diagnoses, hypotheses, audit findings, critic
verdicts, evidence) can enter the prompt: the builder's signature
accepts no board/window objects at all, and sentinel tests pin this.

Discipline:
* single-pass code generation — a parse/guard failure raises; there
  is NO repair loop this round (F1 == 0 by construction);
* compilation outcomes are never fed back into any LLM;
* the call is accounted per UNIQUE artifact (K1) via the ledger.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from ..static_llm.guards import raise_if_forbidden
from .canonical import canonical_sha256
from .json_parse import extract_json_block
from .llm_client import make_replay_key
from .manifest import (
    ENVCODER_OUTPUT_SCHEMA_VERSION,
    ENVCODER_PROMPT_VERSION,
    ENVCODER_ROLE,
)
from .schemas import E1SchemaError
from .task_specs import TaskSpec

ENV_CONTRACT_TEXT = (
    "ENV CONTRACT: produce a self-contained Craftax task module. The "
    "module must define deterministic reset and step semantics, use only "
    "the sanctioned environment API, terminate within the episode budget, "
    "and never modify Student or Reference state. Output exactly one "
    "JSON object with fields 'artifact_id' and 'env_code'."
)


class EnvCoderError(E1SchemaError):
    """Fail-closed EnvCoder violation; ``code`` is greppable."""


class _ECCode:
    BAD_SEEDS = "ENVCODER_BAD_SEEDS"
    MISSING_FIELD = "ENVCODER_MISSING_FIELD"
    UNKNOWN_FIELD = "ENVCODER_UNKNOWN_FIELD"
    BAD_TYPE = "ENVCODER_BAD_TYPE"
    ARTIFACT_MISMATCH = "ENVCODER_ARTIFACT_MISMATCH"
    REPLY_BAD_SHAPE = "ENVCODER_REPLY_BAD_SHAPE"


@dataclass(frozen=True)
class EnvCoderArtifact:
    """One generated env-code artifact (identity-bound to its spec)."""

    spec_id: str
    artifact_id: str
    env_code: str
    prompt_envelope_hash: str


# ---------------------------------------------------------------------------
# Seed examples (whitelist-validated, variant-rotated)
# ---------------------------------------------------------------------------
def _validate_seed_examples(
    seed_examples: Sequence[Mapping[str, Any]], ctx: str
) -> Tuple[Dict[str, str], ...]:
    if not isinstance(seed_examples, (list, tuple)):
        raise EnvCoderError(
            _ECCode.BAD_SEEDS, f"{ctx}: seed_examples must be a sequence"
        )
    cleaned = []
    for i, example in enumerate(seed_examples):
        if not isinstance(example, Mapping):
            raise EnvCoderError(
                _ECCode.BAD_SEEDS,
                f"{ctx}: seed example [{i}] must be a mapping",
            )
        unknown = sorted(k for k in example if k not in ("task_id", "description"))
        if unknown:
            raise EnvCoderError(
                _ECCode.UNKNOWN_FIELD,
                f"{ctx}: seed example [{i}] unknown field(s) {unknown}",
            )
        task_id = example.get("task_id")
        description = example.get("description")
        if not isinstance(task_id, str) or not task_id.strip():
            raise EnvCoderError(
                _ECCode.MISSING_FIELD,
                f"{ctx}: seed example [{i}] needs non-empty task_id",
            )
        if not isinstance(description, str) or not description.strip():
            raise EnvCoderError(
                _ECCode.MISSING_FIELD,
                f"{ctx}: seed example [{i}] needs non-empty description",
            )
        cleaned.append(
            {"task_id": task_id.strip(), "description": description.strip()}
        )
    return tuple(cleaned)


def rotate_seeds(
    seeds: Sequence[Mapping[str, str]], variant: int
) -> Tuple[Mapping[str, str], ...]:
    """Deterministic variant-based rotation (variant 0 leaves order)."""
    if len(seeds) == 0:
        return ()
    shift = variant % len(seeds)
    return tuple(seeds[shift:]) + tuple(seeds[:shift])


# ---------------------------------------------------------------------------
# Prompt construction (whitelist only)
# ---------------------------------------------------------------------------
def render_spec_for_prompt(spec: TaskSpec) -> str:
    """Deterministic rendering of the canonical TaskSpec."""
    axis_lines = "; ".join(
        f"{c['axis']}: {c['from_value']} -> {c['to_value']}"
        for c in spec.axis_changes
    )
    return (
        f"TASK_SPEC spec_id={spec.spec_id}\n"
        f"window_hash={spec.window_hash}\n"
        f"spec_hash={spec.spec_hash}\n"
        f"artifact_id={spec.artifact_id}\n"
        f"description={spec.description}\n"
        f"target_achievements={','.join(spec.target_achievements)}\n"
        f"axis_changes={axis_lines}\n"
        f"constant_axes={','.join(spec.constant_axes)}\n"
        f"scaffolding={spec.scaffolding}\n"
        f"student_must_do={spec.student_must_do}"
    )


def build_envcoder_prompt(
    spec: TaskSpec, *, seed_examples: Sequence[Mapping[str, Any]]
) -> Tuple[str, str]:
    """Whitelist-only prompt assembly; returns (system_prompt, user_prompt)."""
    if not isinstance(spec, TaskSpec):
        raise EnvCoderError(
            _ECCode.BAD_TYPE,
            f"envcoder prompt requires a TaskSpec, got {type(spec).__name__}",
        )
    seeds = _validate_seed_examples(seed_examples, "envcoder")
    rotated = rotate_seeds(seeds, spec.variant)
    seed_lines = [
        f"SEED[{i}] task_id={s['task_id']} description={s['description']}"
        for i, s in enumerate(rotated)
    ]
    user_prompt = (
        f"{ENV_CONTRACT_TEXT}\n"
        f"{render_spec_for_prompt(spec)}\n"
        f"SEED_EXAMPLES (ordered by variant rotation):\n"
        + ("\n".join(seed_lines) if seed_lines else "(none)")
        + f"\nProduce the env-code artifact {spec.artifact_id}."
    )
    system_prompt = (
        "You are the independent EnvCoder of the E1 teacher. You author "
        "environment task code from the given canonical TaskSpec and seed "
        "examples only. You never see review-board content."
    )
    return system_prompt, user_prompt


def build_envcoder_envelope_hash(
    spec: TaskSpec, *, seed_examples: Sequence[Mapping[str, Any]]
) -> str:
    """Canonical hash of the full EnvCoder prompt envelope."""
    system_prompt, user_prompt = build_envcoder_prompt(
        spec, seed_examples=seed_examples
    )
    return canonical_sha256(
        {
            "role": ENVCODER_ROLE,
            "prompt_version": ENVCODER_PROMPT_VERSION,
            "schema_version": ENVCODER_OUTPUT_SCHEMA_VERSION,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        }
    )


# ---------------------------------------------------------------------------
# Execution (single-pass; accounted per unique artifact)
# ---------------------------------------------------------------------------
def _require_reply_content(reply: Any, ctx: str) -> str:
    if (
        not isinstance(reply, (list, tuple))
        or len(reply) != 1
        or not isinstance(reply[0], Mapping)
        or not isinstance(reply[0].get("content"), str)
    ):
        raise EnvCoderError(
            _ECCode.REPLY_BAD_SHAPE,
            f"{ctx}: replay reply must be [{'content': str}], got {reply!r}",
        )
    return reply[0]["content"]


def parse_envcoder_output(content: str, spec: TaskSpec, ctx: str) -> EnvCoderArtifact:
    """Fail-closed parse; artifact identity must match the spec."""
    raise_if_forbidden(content, ctx)
    obj = extract_json_block(content, ctx)
    raise_if_forbidden(obj, ctx)
    if not isinstance(obj, Mapping):
        raise EnvCoderError(
            _ECCode.BAD_TYPE, f"{ctx}: output must be a JSON object"
        )
    unknown = sorted(k for k in obj if k not in ("artifact_id", "env_code"))
    if unknown:
        raise EnvCoderError(
            _ECCode.UNKNOWN_FIELD, f"{ctx}: unknown output field(s) {unknown}"
        )
    artifact_id = obj.get("artifact_id")
    if not isinstance(artifact_id, str) or not artifact_id.strip():
        raise EnvCoderError(
            _ECCode.MISSING_FIELD, f"{ctx}: missing artifact_id"
        )
    env_code = obj.get("env_code")
    if not isinstance(env_code, str) or not env_code.strip():
        raise EnvCoderError(_ECCode.MISSING_FIELD, f"{ctx}: missing env_code")
    if artifact_id.strip() != spec.artifact_id:
        raise EnvCoderError(
            _ECCode.ARTIFACT_MISMATCH,
            f"{ctx}: output artifact_id {artifact_id!r} != spec artifact "
            f"{spec.artifact_id!r}",
        )
    raise_if_forbidden(env_code, f"{ctx}.env_code")
    return EnvCoderArtifact(
        spec_id=spec.spec_id,
        artifact_id=spec.artifact_id,
        env_code=env_code,
        prompt_envelope_hash="",  # filled by run_envcoder
    )


def run_envcoder(
    llm: Any,
    *,
    spec: TaskSpec,
    seed_examples: Sequence[Mapping[str, Any]],
    ledger: Any,
    window_id: str,
) -> EnvCoderArtifact:
    """Single-pass EnvCoder call, accounted per unique artifact (K1).

    A replay miss is a HARD FAIL and propagates; a parse/guard failure
    raises without any repair attempt (F1 stays 0).
    """
    ledger.record_envcoder_call(window_id, spec.artifact_id)
    system_prompt, user_prompt = build_envcoder_prompt(
        spec, seed_examples=seed_examples
    )
    envelope_hash = build_envcoder_envelope_hash(
        spec, seed_examples=seed_examples
    )
    cache_key = make_replay_key(
        role=ENVCODER_ROLE,
        evidence_hash=spec.spec_hash,
        prompt_envelope_hash=envelope_hash,
        prompt_version=ENVCODER_PROMPT_VERSION,
        schema_version=ENVCODER_OUTPUT_SCHEMA_VERSION,
    )
    reply = llm.query(
        system_prompt, [user_prompt], cache_key=cache_key, role=ENVCODER_ROLE
    )
    content = _require_reply_content(reply, f"envcoder {spec.artifact_id}")
    artifact = parse_envcoder_output(
        content, spec, f"envcoder {spec.artifact_id}"
    )
    return EnvCoderArtifact(
        spec_id=artifact.spec_id,
        artifact_id=artifact.artifact_id,
        env_code=artifact.env_code,
        prompt_envelope_hash=envelope_hash,
    )
