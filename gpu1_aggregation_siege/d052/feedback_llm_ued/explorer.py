"""Board role 5/6: Explorer (six-role Review Board, C6).

Proposes the CONTROLLED environment specifications — AxisDirectives: which
axis moves from which level to which level, what is held constant, and the
predicted signature the probe should observe. One treatment (+ optional
held-control re-measurement) per family with visible feedback, plus bounded
exploration for families with none. The historical ``i % len(axes)`` index
rotation is NOT a source of truth here — every setting derives from visible
evidence.

ENGINEERING_SCAFFOLD: deterministic mock rule; no real LLM call this round.
"""
from __future__ import annotations

import json
from typing import List

from pydantic import Field

from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.axis_directive import (
    DIRECTION_DECREASE,
    DIRECTION_HOLD,
    DIRECTION_INCREASE,
    LEVEL_NONE,
    ROLE_CONTROL,
    ROLE_TREATMENT,
    AxisDirective,
)
from d052.feedback_llm_ued.environment_generator import (
    AXIS_LEVELS,
    FAMILY_AXES,
)
from d052.feedback_llm_ued.feedback_contracts import (
    CONTEXT_CLOSE,
    CONTEXT_OPEN,
    FeedbackRoleEnvelope,
)
from d052.schemas.common import CanonicalModel

ROLE = C.ROLE_EXPLORER
PROMPT_VERSION = f"{C.ROLE_PROMPT_VERSION}.{ROLE}"

_LEVEL_RANK = {level: i for i, level in enumerate(AXIS_LEVELS)}

PROMPT_TEMPLATE = f"""\
You are the Explorer role of the six-role Review Board of the
simulator-grounded feedback-adaptive LLM-UED loop. Propose controlled
environment specifications as AxisDirectives: for each family with visible
probe feedback, move ONE axis (treatment) from its measured level in the
direction the evidence supports, holding the other axes constant, and predict
the probe signature the movement should produce; add a held control
re-measurement where a level already exists; explore families with no visible
feedback at all. Environment-level knobs only.
Prompt version: {{prompt_version}}
{CONTEXT_OPEN}
{{context_json}}
{CONTEXT_CLOSE}
Respond with a single JSON object matching the ExplorerOutput schema.
"""


class ExplorerOutput(CanonicalModel):
    window: int = Field(ge=0)
    directives: List[AxisDirective] = Field(default_factory=list)
    exploration_summary: str = ""


def build_prompt(context: dict) -> str:
    return PROMPT_TEMPLATE.format(
        prompt_version=PROMPT_VERSION,
        context_json=json.dumps(context, sort_keys=True, ensure_ascii=False,
                                default=str))


def parse(raw: str) -> ExplorerOutput:
    return ExplorerOutput.model_validate_json(raw)


#: output class exposed for the board assembler (single source of truth)
OUTPUT_MODEL = ExplorerOutput


def _clamp_sr(value: float) -> float:
    return round(min(0.95, max(0.05, value)), 4)


def _treatment_for(family: str, worst: dict, window: int
                   ) -> List[AxisDirective]:
    """Treatment (+ optional control) directives for one evidenced family."""
    axis = FAMILY_AXES[family][0]
    held = {a: "medium" for a in FAMILY_AXES[family] if a != axis}
    old_level = worst.get("axis_values", {}).get(axis, LEVEL_NONE)
    sr = float(worst.get("student_success_rate", 0.0))
    directives: List[AxisDirective] = []

    if sr < C.PREFLIGHT_LEARNABLE_LOW:
        wanted = DIRECTION_DECREASE            # Student crushed: back off
    elif sr > C.PREFLIGHT_TOO_EASY:
        wanted = DIRECTION_INCREASE            # Student coasting: press
    elif old_level == LEVEL_NONE:
        wanted = DIRECTION_INCREASE
    else:
        # near the frontier: keep moving — up unless already at the top
        wanted = (DIRECTION_INCREASE
                  if _LEVEL_RANK[old_level] < len(AXIS_LEVELS) - 1
                  else DIRECTION_DECREASE)

    treatment = None
    if old_level == LEVEL_NONE:
        treatment = (wanted, AXIS_LEVELS[1])       # first measurement: medium
    else:
        rank = _LEVEL_RANK[old_level]
        if wanted == DIRECTION_INCREASE and rank < len(AXIS_LEVELS) - 1:
            treatment = (wanted, AXIS_LEVELS[rank + 1])
        elif wanted == DIRECTION_DECREASE and rank > 0:
            treatment = (wanted, AXIS_LEVELS[rank - 1])
        # otherwise the axis is at the boundary: no legal movement, control only

    if treatment is not None:
        direction, new_level = treatment
        delta = 0.15 if direction == DIRECTION_INCREASE else -0.15
        directives.append(AxisDirective(
            directive_id=(f"dir-w{window:02d}-{family}-{axis}-treatment"),
            source_window=window,
            environment_family=family,
            axis=axis,
            old_level=old_level,
            new_level=new_level,
            direction=direction,
            experiment_control_role=ROLE_TREATMENT,
            held_constant_axes=held,
            expected_next_signature={
                "student_success_rate": _clamp_sr(sr + delta)},
            rationale=(f"visible feedback {worst.get('feedback_id', '')}: "
                       f"Student success rate {sr:.3f} -> move {axis} "
                       f"{direction}")))

    if old_level in AXIS_LEVELS:
        directives.append(AxisDirective(
            directive_id=f"dir-w{window:02d}-{family}-{axis}-control",
            source_window=window,
            environment_family=family,
            axis=axis,
            old_level=old_level,
            new_level=old_level,
            direction=DIRECTION_HOLD,
            experiment_control_role=ROLE_CONTROL,
            held_constant_axes=held,
            expected_next_signature={
                "student_success_rate": _clamp_sr(sr)},
            rationale=(f"held control re-measurement of {axis}="
                       f"{old_level} for {family}")))
    return directives


def mock_rule(context: dict) -> dict:
    """Deterministically derive controlled directives from visible feedback.

    C10: families listed under ``retired_families`` /
    ``families_in_cooldown`` in the board context emit NO directives —
    they cannot be funded this window (the Reconciler fails closed on any
    proposal targeting them), so specifying axis movements for them would
    only produce dead code. The FUNDED_FAMILY_WITHOUT_DIRECTIVE invariant
    is untouched: funded families are never blocked.
    """
    window = int(context.get("window", 0))
    feedback = context.get("feedback", [])
    board_context = context.get("board_context", {})
    blocked = set(board_context.get("retired_families", [])) | \
        set(board_context.get("families_in_cooldown", []))

    fam_records: dict = {}
    for fb in feedback:
        fam_records.setdefault(fb.get("environment_family", ""),
                               []).append(fb)

    directives: List[dict] = []
    for family in sorted(fam_records):
        if family not in C.ENVIRONMENT_FAMILIES:
            continue
        if family in blocked:
            continue
        worst = sorted(fam_records[family],
                       key=lambda r: (float(r.get("student_success_rate", 0.0))
                                      - float(r.get("reference_success_rate",
                                                    0.0)),
                                      r.get("feedback_id", "")))[0]
        for d in _treatment_for(family, worst, window):
            directives.append(d.model_dump())

    # bounded exploration over families with no visible feedback at all
    explored = 0
    for family in C.ENVIRONMENT_FAMILIES:
        if explored >= C.MAX_EXPLORATION_PROPOSALS:
            break
        if family in fam_records or family in blocked:
            continue
        axis = FAMILY_AXES[family][0]
        held = {a: "medium" for a in FAMILY_AXES[family] if a != axis}
        directives.append(AxisDirective(
            directive_id=f"dir-w{window:02d}-{family}-{axis}-exploration",
            source_window=window,
            environment_family=family,
            axis=axis,
            old_level=LEVEL_NONE,
            new_level=AXIS_LEVELS[1],
            direction=DIRECTION_INCREASE,
            experiment_control_role=ROLE_TREATMENT,
            held_constant_axes=held,
            expected_next_signature={"student_success_rate": 0.5},
            rationale=f"no visible feedback yet for {family}: first "
                      f"controlled measurement").model_dump())
        explored += 1

    summary = (f"window {window}: {len(directives)} directive(s) from "
               f"{len(fam_records)} evidenced family(ies) + "
               f"{explored} exploration; skipped retired/cooldown "
               f"families: {sorted(blocked)}")
    return dict(window=window, directives=directives,
                exploration_summary=summary)


def run(context: dict, backend, window: int, sequence: int
        ) -> FeedbackRoleEnvelope:
    prompt = build_prompt(context)
    raw = backend.complete(ROLE, prompt)
    parsed = parse(raw)
    return FeedbackRoleEnvelope.make(
        role=ROLE, prompt_version=PROMPT_VERSION, backend_id=backend.backend_id,
        model_id=backend.model_id, window=window, sequence=sequence,
        prompt=prompt, raw_response=raw, parsed_dump=parsed.model_dump())
