"""The independent EnvCoder — the 7th LLM-family call of every window (C7).

Consumes the board's AxisDirectives (the controlled environment
specifications) and emits candidate environment CODE. Roles are strictly
separated: the six board roles decide WHAT to measure and predict; the
EnvCoder alone decides HOW the environment realizes it. It never sees the
Student, never grades feedback, and its output may only reach the probe
funnel through the compile/reset/step gates (``env_coder_gate``).

This round the coder is a deterministic SYMBOLIC generator
(``SpecEnvCoder`` semantics via the mock rule): every code manifest is a
pure function of the directive batch, so replay and tamper checks hold.
The real seam (``RealEnvCoderSeam``) refuses to exist while
``REAL_LLM_CALLS_AUTHORIZED`` is False — ``REAL_ENVCODER_USED`` stays False.
"""
from __future__ import annotations

import json
from typing import List

from pydantic import Field, model_validator

from d052.bagr_ued.hashing import canonical_sha256
from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.axis_directive import AxisDirective
from d052.feedback_llm_ued.feedback_contracts import (
    CONTEXT_CLOSE,
    CONTEXT_OPEN,
    FeedbackRoleEnvelope,
)
from d052.schemas.common import CanonicalModel

ROLE = C.ROLE_ENV_CODER
PROMPT_VERSION = f"{C.ROLE_PROMPT_VERSION}.{ROLE}"

#: prefix of every symbolic code manifest (the gate requires it)
CODE_SYMBOL_PREFIX = "ENVCODE_SYMBOLIC_V1::"

PROMPT_TEMPLATE = f"""\
You are the independent EnvCoder of the simulator-grounded feedback-adaptive
LLM-UED loop. You receive the review board's AxisDirectives (controlled
environment specifications) and NOTHING else — no Student data, no feedback,
no probe results. For each directive, emit the candidate environment code
manifest that realizes exactly the requested axis setting while holding the
declared axes constant. Do not invent axes; do not touch action, reward, or
policy knobs.
Prompt version: {{prompt_version}}
{CONTEXT_OPEN}
{{context_json}}
{CONTEXT_CLOSE}
Respond with a single JSON object matching the EnvCoderOutput schema.
"""


class CodedDirective(CanonicalModel):
    """One directive's coded environment manifest (audit-grade)."""

    directive_id: str = Field(min_length=1)
    #: content hash of the SOURCE directive — the gate recomputes and
    #: compares, so a substituted directive cannot be coded silently
    directive_hash: str = Field(min_length=1)
    environment_family: str = Field(min_length=1)
    axis: str = Field(min_length=1)
    new_level: str = Field(min_length=1)
    experiment_control_role: str = Field(min_length=1)
    code_symbol: str = Field(min_length=1)
    reset_contract: str = Field(min_length=1)
    step_contract: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate(self) -> "CodedDirective":
        if self.environment_family not in C.ENVIRONMENT_FAMILIES:
            raise ValueError(
                f"UNKNOWN_ENVIRONMENT_FAMILY: {self.environment_family!r}")
        return self


class EnvCoderOutput(CanonicalModel):
    window: int = Field(ge=0)
    coded: List[CodedDirective] = Field(default_factory=list)
    directive_batch_hash: str = ""
    coder_summary: str = ""


def build_prompt(context: dict) -> str:
    return PROMPT_TEMPLATE.format(
        prompt_version=PROMPT_VERSION,
        context_json=json.dumps(context, sort_keys=True, ensure_ascii=False,
                                default=str))


def parse(raw: str) -> EnvCoderOutput:
    return EnvCoderOutput.model_validate_json(raw)


def build_env_coder_context(*, window: int,
                            directives: List[AxisDirective]) -> dict:
    """The EnvCoder's entire world: the directive batch, nothing else."""
    dumps = [d.model_dump() for d in directives]
    return dict(window=window, directives=dumps,
                directive_batch_hash=canonical_sha256(dumps))


def symbolic_code_symbol(directive_dump: dict) -> str:
    """Deterministic symbolic code manifest for one directive.

    Pure function of the directive content (hash field excluded — it is an
    identity stamp, not code). Seeds never enter: reproducibility comes from
    the probe's banked seed schedule, not from the code manifest.
    """
    payload = {k: v for k, v in directive_dump.items()
               if k != "directive_hash"}
    return CODE_SYMBOL_PREFIX + canonical_sha256(payload)


def mock_rule(context: dict) -> dict:
    """Deterministically code every directive (SpecEnvCoder semantics)."""
    window = int(context.get("window", 0))
    coded: List[dict] = []
    for d in sorted(context.get("directives", []),
                    key=lambda x: x.get("directive_id", "")):
        fam = d.get("environment_family", "")
        coded.append(dict(
            directive_id=d.get("directive_id", ""),
            directive_hash=d.get("directive_hash", ""),
            environment_family=fam,
            axis=d.get("axis", ""),
            new_level=d.get("new_level", ""),
            experiment_control_role=d.get("experiment_control_role", ""),
            code_symbol=symbolic_code_symbol(d),
            reset_contract=f"reset(seed)->state::{fam}",
            step_contract=(f"step(action)->(state,reward,terminal,info)"
                           f"::{fam}")))
    summary = (f"window {window}: symbolically coded {len(coded)} "
               f"directive(s); REAL_ENVCODER_USED=False this round")
    return dict(window=window, coded=coded,
                directive_batch_hash=context.get("directive_batch_hash", ""),
                coder_summary=summary)


def run_env_coder(*, window: int, directives: List[AxisDirective], backend,
                  sequence: int):
    """The window's 7th LLM-family call. Returns (output, envelope)."""
    context = build_env_coder_context(window=window, directives=directives)
    prompt = build_prompt(context)
    raw = backend.complete(ROLE, prompt)
    parsed = parse(raw)
    envelope = FeedbackRoleEnvelope.make(
        role=ROLE, prompt_version=PROMPT_VERSION, backend_id=backend.backend_id,
        model_id=backend.model_id, window=window, sequence=sequence,
        prompt=prompt, raw_response=raw, parsed_dump=parsed.model_dump())
    return parsed, envelope


class RealEnvCoderBlocked(RuntimeError):
    """The real-LLM EnvCoder seam refused to exist this round."""


class RealEnvCoderSeam:
    """Seam for a REAL LLM EnvCoder (real backend transport).

    Construction FAILS CLOSED unless the round authorizes real LLM calls;
    this worktree never flips that flag, so ``REAL_ENVCODER_USED`` stays
    False and every window is coded symbolically.
    """

    def __init__(self, *, authorized: bool) -> None:
        if not authorized:
            raise RealEnvCoderBlocked(
                "REAL_ENVCODER_BLOCKED: REAL_LLM_CALLS_AUTHORIZED="
                f"{C.REAL_LLM_CALLS_AUTHORIZED} this round; the EnvCoder "
                "runs symbolically (ENGINEERING_SCAFFOLD)")
