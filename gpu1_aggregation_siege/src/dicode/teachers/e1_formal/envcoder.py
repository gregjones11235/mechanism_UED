"""Stage 4: independent EnvCoder prompt builder (OUTSIDE the board).

The EnvCoder is an independent artifact producer, NOT a board member.
ROUND-3 P0-2: it is invoked ONCE per UNIQUE task TEMPLATE (family),
not per variant — variants are deterministic derivations of a template
and share one generated env-code artifact. Identities:

* ``template_hash``            — family content identity;
* ``template_artifact_id``     — ``{template_hash}::tpl``; the replay
  key's evidence hash AND the ledger K1 counter are keyed on it;
* per-variant compiled artifacts are derived downstream from the one
  template artifact (see ``gen_manager``).

Its prompt is assembled from a strict WHITELIST:

1. the fixed environment-authoring contract text;
2. the canonical TEMPLATE rendering (identity + goals + axes);
3. seed examples in FIXED order (no variant rotation — the call is
   variant-independent by construction).

NO board content (diagnoses, hypotheses, audit findings, critic
verdicts, evidence) can enter the prompt: the builder's signature
accepts no board/window objects at all, and sentinel tests pin this.

Discipline:
* validation outcomes are never fed back into any LLM EXCEPT through
  the BOUNDED whitelist-safe repair prompt
  (``run_envcoder_with_repair``): contract text + template rendering +
  the reported validation error, nothing else;
* the primary call is accounted per UNIQUE template artifact (K1) via
  the ledger; repair calls are accounted separately (F1) and bounded
  by ``max_repairs`` (never unbounded, never silent acceptance).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ..static_llm.guards import raise_if_forbidden
from .canonical import canonical_sha256
from .json_parse import extract_json_block
from .llm_client import make_replay_key
from .manifest import (
    ENVCODER_OUTPUT_SCHEMA_VERSION,
    ENVCODER_PROMPT_VERSION,
    ENVCODER_ROLE,
)
from .schemas import E1SchemaError, SchemaError
from .task_specs import TaskSpec

ENV_CONTRACT_TEXT = (
    "ENV CONTRACT: produce a self-contained Craftax task module as a "
    "PARAMETRIZED VARIANT of the REFERENCE_BASE_ENV_CODE below. Output "
    "exactly one JSON object with fields 'artifact_id' and 'env_code'.\n"
    "STRUCTURE (validation REJECTS violations): keep the class name "
    "exactly `Env`, keep EVERY module-level import and helper unchanged, "
    "and keep the `make_env()` entry surface present and intact — modify "
    "ONLY task parameters / world generation / mechanics inside the task "
    "methods (get_task_params / generate_world / relevant_achievements / "
    "completed_achievements / label / docstring).\n"
    "API LOCK (validation REJECTS violations): the parametrized variant "
    "may ONLY reference BlockType constants, inventory item keys, mob "
    "names/type_ids, TaskParams fields, world-builder methods and craftax "
    "API names that appear VERBATIM in REFERENCE_BASE_ENV_CODE. You may "
    "change ONLY the NUMERIC values already present (counts like n=, "
    "distances like min_dist/max_dist, spawn multipliers, the achievement "
    "lists, label, description) — NEVER invent or rename any API element, "
    "never add a block type, inventory item, mob, TaskParams field or "
    "import that is absent from the base.\n"
    "STRICTLY FORBIDDEN anywhere in the module (code, comments, "
    "docstrings, strings; validation REJECTS it): any waypoint / route / "
    "coordinate-navigation or step-by-step action sequence, and NEVER "
    "write numeric coordinates in (x, y) form; any reward modification / "
    "reward shaping / reward assignment — never write the word `reward` "
    "next to an assignment or inside a function-definition name; any "
    "access to logits / hidden states / policy weights / network weights "
    "/ gradients; any FORMAL_* evaluation data; any expert trajectory / "
    "demonstration / imitation content; and any reference to Student or "
    "Reference state. The module must only define the task world and its "
    "dynamics via the sanctioned minicraftax API (BaseTask / world "
    "builder / TaskParams)."
)

#: round-3 P0-4: bounded repair protocol (replay-key identity).
#: v2 — the repair prompt now re-supplies the whitelist-validated base
#: seed code (same REFERENCE_BASE_ENV_CODE as the primary prompt), so a
#: repair attempt repairs the KNOWN-GOOD variant instead of
#: reconstructing a whole craftax module from memory (the primary
#: source of structure/import ladder failures).
ENVCODER_REPAIR_PROMPT_VERSION = "e1-envcoder-repair-prompt-v3"

#: hard upper bound for ``max_repairs`` (supervisor-sanctioned range
#: 0..2; the teacher config ``teacher.envcoder.max_repairs`` selects
#: within it, absent => 2, i.e. <= 2 hard validations per template)
MAX_ENVCODER_REPAIRS = 2

# fail-closed repair codes (greppable)
ENVCODER_REPAIR_EXHAUSTED = "ENVCODER_REPAIR_EXHAUSTED"
ENVCODER_REPAIR_BAD_TYPE = "ENVCODER_REPAIR_BAD_TYPE"
ENVCODER_REPAIR_OUT_OF_RANGE = "ENVCODER_REPAIR_OUT_OF_RANGE"


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
    """One generated env-code artifact (identity-bound to its template)."""

    template_hash: str
    artifact_id: str  # == template_artifact_id of the template
    env_code: str
    prompt_envelope_hash: str


@dataclass(frozen=True)
class RepairRecord:
    """Audit record of ONE bounded repair attempt (round-3 P0-4).

    ``repaired_artifact_hash`` is "" when the repair reply itself
    failed to parse (no artifact was produced by that attempt).
    """

    retry_index: int
    previous_artifact_hash: str
    validation_error: str
    reflection_prompt_hash: str
    response_hash: str
    repaired_artifact_hash: str


# ---------------------------------------------------------------------------
# Seed examples (whitelist-validated)
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
        unknown = sorted(
            k for k in example if k not in ("task_id", "description", "code"))
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
        entry = {
            "task_id": task_id.strip(),
            "description": description.strip(),
        }
        code = example.get("code")
        if code is not None:
            if not isinstance(code, str) or not code.strip():
                raise EnvCoderError(
                    _ECCode.BAD_TYPE,
                    f"{ctx}: seed example [{i}] code must be a non-empty "
                    "str when present",
                )
            entry["code"] = code
        cleaned.append(entry)
    return tuple(cleaned)


def _base_env_code_block(seeds: Sequence[Mapping[str, Any]]) -> str:
    """The whitelist-validated REFERENCE_BASE_ENV_CODE prompt block.

    Uses the FIRST seed example that carries ``code`` (the real,
    known-good craftax module); returns "" when no seed carries code.
    Shared by the primary and the repair prompt so a repair attempt
    repairs the SAME known-good base the primary call saw.
    """
    for seed in seeds:
        if seed.get("code"):
            return (
                "REFERENCE_BASE_ENV_CODE (a REAL, known-good Craftax "
                "task module; keep its structure and the sanctioned "
                "minicraftax API intact — parametrize it, never rewrite "
                "it):\n```python\n"
                + seed["code"]
                + "\n```\n"
            )
    return ""


def rotate_seeds(
    seeds: Sequence[Mapping[str, str]], variant: int
) -> Tuple[Mapping[str, str], ...]:
    """Deterministic rotation helper (pure; pinned by tests).

    Kept as a tested pure helper; the template-keyed EnvCoder prompt
    itself uses the FIXED seed order (variant independence).
    """
    if len(seeds) == 0:
        return ()
    shift = variant % len(seeds)
    return tuple(seeds[shift:]) + tuple(seeds[:shift])


# ---------------------------------------------------------------------------
# Prompt construction (whitelist only; template-level rendering)
# ---------------------------------------------------------------------------
def render_template_for_prompt(spec: TaskSpec) -> str:
    """Deterministic rendering of the TEMPLATE behind a TaskSpec.

    Variant fields (variant index, variant_params, spec_hash,
    artifact_id) are deliberately ABSENT: every variant of the same
    template renders identically, so the EnvCoder envelope is
    variant-independent by construction.
    """
    axis_lines = "; ".join(
        f"{c['axis']}: {c['from_value']} -> {c['to_value']}"
        for c in spec.axis_changes
    )
    return (
        f"TASK_TEMPLATE template_hash={spec.template_hash}\n"
        f"window_hash={spec.window_hash}\n"
        f"template_artifact_id={spec.template_artifact_id}\n"
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
    """Whitelist-only prompt assembly; returns (system_prompt, user_prompt).

    Variant-independent: seeds are rendered in FIXED order and only the
    template-level rendering enters the prompt.
    """
    if not isinstance(spec, TaskSpec):
        raise EnvCoderError(
            _ECCode.BAD_TYPE,
            f"envcoder prompt requires a TaskSpec, got {type(spec).__name__}",
        )
    seeds = _validate_seed_examples(seed_examples, "envcoder")
    seed_lines = [
        f"SEED[{i}] task_id={s['task_id']} description={s['description']}"
        for i, s in enumerate(seeds)
    ]
    base_code_block = _base_env_code_block(seeds)
    if base_code_block:
        user_prompt = (
            f"{ENV_CONTRACT_TEXT}\n"
            f"{render_template_for_prompt(spec)}\n"
            f"SEED_EXAMPLES (fixed order):\n"
            + ("\n".join(seed_lines) if seed_lines else "(none)")
            + "\n\n"
            + base_code_block
            + "Produce the env-code artifact "
            + f"{spec.template_artifact_id}. The env_code MUST be a "
            "PARAMETRIZED VARIANT of the REFERENCE_BASE_ENV_CODE module "
            "that implements the TASK_TEMPLATE description, "
            "target_achievements and axis_changes above — modify only "
            "task parameters / world generation / mechanics, NEVER the "
            "class/API structure. Respond with exactly one JSON object: "
            '{"artifact_id": "<id>", "env_code": "<the complete module '
            'source as a single JSON string>"} and nothing else.'
        )
    else:
        user_prompt = (
            f"{ENV_CONTRACT_TEXT}\n"
            f"{render_template_for_prompt(spec)}\n"
            f"SEED_EXAMPLES (fixed order):\n"
            + ("\n".join(seed_lines) if seed_lines else "(none)")
            + f"\nProduce the env-code artifact {spec.template_artifact_id}."
        )
    system_prompt = (
        "You are the independent EnvCoder of the E1 teacher. You author "
        "environment task code from the given canonical task template and "
        "seed examples only. You never see review-board content. When a "
        "REFERENCE_BASE_ENV_CODE is provided you produce a parametrized "
        "variant of it, keeping the sanctioned minicraftax API structure "
        "exactly intact."
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
# Execution (one call per unique template artifact)
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
    """Fail-closed parse; artifact identity must match the TEMPLATE."""
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
    if artifact_id.strip() != spec.template_artifact_id:
        raise EnvCoderError(
            _ECCode.ARTIFACT_MISMATCH,
            f"{ctx}: output artifact_id {artifact_id!r} != template "
            f"artifact {spec.template_artifact_id!r}",
        )
    raise_if_forbidden(env_code, f"{ctx}.env_code")
    return EnvCoderArtifact(
        template_hash=spec.template_hash,
        artifact_id=spec.template_artifact_id,
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
    """EnvCoder call for ONE unique template, accounted per template (K1).

    The replay key's evidence hash is the ``template_hash``, so every
    variant of the same template maps to the SAME call — the caller
    invokes this once per unique template only. A replay miss is a HARD
    FAIL and propagates; a parse/guard failure raises.
    """
    ledger.record_envcoder_call(window_id, spec.template_artifact_id)
    system_prompt, user_prompt = build_envcoder_prompt(
        spec, seed_examples=seed_examples
    )
    envelope_hash = build_envcoder_envelope_hash(
        spec, seed_examples=seed_examples
    )
    cache_key = make_replay_key(
        role=ENVCODER_ROLE,
        evidence_hash=spec.template_hash,
        prompt_envelope_hash=envelope_hash,
        prompt_version=ENVCODER_PROMPT_VERSION,
        schema_version=ENVCODER_OUTPUT_SCHEMA_VERSION,
    )
    reply = llm.query(
        system_prompt, [user_prompt], cache_key=cache_key, role=ENVCODER_ROLE
    )
    content = _require_reply_content(
        reply, f"envcoder {spec.template_artifact_id}"
    )
    artifact = parse_envcoder_output(
        content, spec, f"envcoder {spec.template_artifact_id}"
    )
    return EnvCoderArtifact(
        template_hash=artifact.template_hash,
        artifact_id=artifact.artifact_id,
        env_code=artifact.env_code,
        prompt_envelope_hash=envelope_hash,
    )


# ---------------------------------------------------------------------------
# Bounded repair protocol (round-3 P0-4; F1 gets its runtime caller)
# ---------------------------------------------------------------------------
def previous_artifact_hash_for_code(env_code: str) -> str:
    """Content identity of a PARSED artifact (validation-failure chain)."""
    return canonical_sha256({"env_code": env_code})


def previous_artifact_hash_for_failure(spec: TaskSpec, error: SchemaError) -> str:
    """Deterministic identity of a failed ATTEMPT that produced no
    parseable artifact (parse/guard failure chain)."""
    return canonical_sha256(
        {
            "template_hash": spec.template_hash,
            "envcoder_error_code": getattr(error, "code", ""),
            "envcoder_error": str(error),
        }
    )


def _sanitize_validation_error(text: Any) -> str:
    """Sanitize a reflected validation-error string for the repair prompt.

    The error message quotes the rejected artifact snippet, which would
    otherwise re-trigger the output guard when reflected. Every forbidden
    match (guard code names, scan names and the forbidden content itself)
    is REDACTED so the repair prompt is both safe to scan and useful to
    the LLM (it still sees that forbidden content was the reason)."""
    import re

    cleaned = str(text)
    try:
        from ..static_llm.guards import GuardCode, SCANNERS

        for name in dir(GuardCode):
            if name.isupper():
                code = getattr(GuardCode, name)
                if isinstance(code, str) and code:
                    cleaned = cleaned.replace(code, "[guard-marker]")
        for _scan, _code, patterns in SCANNERS:
            for pattern in patterns:
                cleaned = pattern.sub("[redacted]", cleaned)
    except Exception:
        pass
    # scan-name forms like F1_waypoint / F1_action_sequence
    cleaned = re.sub(r"\bF\d+_[a-z_]+\b", "[scan-name]", cleaned)
    return cleaned


def build_repair_prompt(
    spec: TaskSpec,
    *,
    previous_artifact_hash: str,
    validation_error: str,
    retry_index: int,
    seed_examples: Sequence[Mapping[str, Any]] = (),
) -> Tuple[str, str]:
    """WHITELIST-SAFE repair prompt; returns (system_prompt, user_prompt).

    Admissible inputs are EXACTLY: the fixed environment contract text,
    the canonical template rendering, the previous artifact hash, the
    reported validation error, the retry index and — when the caller
    supplies seed examples — the whitelist-validated base env code
    (seeds are an authorized prompt input; the base lets a repair
    attempt repair the KNOWN-GOOD variant instead of reconstructing a
    whole craftax module from memory). Board content, evidence facts
    and any other role output NEVER enter a repair prompt.
    """
    if not isinstance(spec, TaskSpec):
        raise EnvCoderError(
            ENVCODER_REPAIR_BAD_TYPE,
            f"repair prompt requires a TaskSpec, got {type(spec).__name__}",
        )
    if not isinstance(previous_artifact_hash, str) or not previous_artifact_hash:
        raise EnvCoderError(
            ENVCODER_REPAIR_BAD_TYPE,
            "repair prompt requires a non-empty previous_artifact_hash",
        )
    if not isinstance(validation_error, str) or not validation_error.strip():
        raise EnvCoderError(
            ENVCODER_REPAIR_BAD_TYPE,
            "repair prompt requires a non-empty validation_error",
        )
    if (
        isinstance(retry_index, bool)
        or not isinstance(retry_index, int)
        or retry_index < 1
    ):
        raise EnvCoderError(
            ENVCODER_REPAIR_BAD_TYPE,
            f"repair prompt retry_index must be an int >= 1, got {retry_index!r}",
        )
    ctx = f"envcoder repair {spec.template_artifact_id} retry {retry_index}"
    # guard-scan the reflected error text BEFORE it re-enters any prompt.
    # The reflected message carries OUR OWN guard code names (e.g.
    # WAYPOINT_DETECTED) verbatim, which would self-trigger the scan; the
    # text is sanitized first (our markers are stripped, never the LLM's
    # actual forbidden content).
    validation_error = _sanitize_validation_error(validation_error)
    raise_if_forbidden(validation_error, ctx)
    seeds = _validate_seed_examples(seed_examples, "envcoder.repair")
    base_code_block = _base_env_code_block(seeds)
    user_prompt = (
        f"{ENV_CONTRACT_TEXT}\n"
        f"{render_template_for_prompt(spec)}\n"
        f"REPAIR_ATTEMPT retry_index={retry_index}\n"
        f"previous_artifact_hash={previous_artifact_hash}\n"
        f"VALIDATION_ERROR: {validation_error}\n"
        + (base_code_block if base_code_block else "")
        + f"Produce the env-code artifact {spec.template_artifact_id}."
    )
    system_prompt = (
        "You are the independent EnvCoder of the E1 teacher, repair "
        "pass. You repair the previously produced environment task "
        "code using ONLY the environment contract, the canonical task "
        "template and the reported validation error. You never see "
        "review-board content."
    )
    return system_prompt, user_prompt


def build_repair_envelope_hash(
    spec: TaskSpec,
    *,
    previous_artifact_hash: str,
    validation_error: str,
    retry_index: int,
    seed_examples: Sequence[Mapping[str, Any]] = (),
) -> str:
    """Canonical hash of the full repair prompt envelope (replay identity)."""
    system_prompt, user_prompt = build_repair_prompt(
        spec,
        previous_artifact_hash=previous_artifact_hash,
        validation_error=validation_error,
        retry_index=retry_index,
        seed_examples=seed_examples,
    )
    return canonical_sha256(
        {
            "role": ENVCODER_ROLE,
            "prompt_version": ENVCODER_REPAIR_PROMPT_VERSION,
            "schema_version": ENVCODER_OUTPUT_SCHEMA_VERSION,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "template_hash": spec.template_hash,
            "previous_artifact_hash": previous_artifact_hash,
            "validation_error": validation_error,
            "retry_index": retry_index,
        }
    )


def build_repair_evidence_hash(
    spec: TaskSpec,
    *,
    previous_artifact_hash: str,
    validation_error: str,
    retry_index: int,
) -> str:
    """Replay key evidence hash of a repair call (template-keyed chain)."""
    return canonical_sha256(
        {
            "template_hash": spec.template_hash,
            "previous_artifact_hash": previous_artifact_hash,
            "validation_error": validation_error,
            "retry_index": retry_index,
        }
    )


def run_envcoder_with_repair(
    llm: Any,
    *,
    spec: TaskSpec,
    seed_examples: Sequence[Mapping[str, Any]],
    backend: Any,
    max_repairs: int,
    ledger: Any,
    window_id: str,
) -> Tuple[EnvCoderArtifact, Tuple[RepairRecord, ...]]:
    """EnvCoder call for ONE unique template with BOUNDED repair.

    Flow: one primary call via ``run_envcoder`` (K1) -> backend
    validation. Passed => done. Failed => up to ``max_repairs`` repair
    attempts, each accounted via ``ledger.record_repair_call`` (F1) and
    keyed by the repair envelope (previous artifact hash + validation
    error + retry index). A replay miss on ANY call (primary or
    repair) is a HARD FAIL and propagates. Exhaustion raises
    ``EnvCoderError(ENVCODER_REPAIR_EXHAUSTED)`` with the
    ``RepairRecord`` chain attached as ``records``.

    Returns ``(artifact, repair_records)`` — ``repair_records`` is
    empty when the primary artifact validated on the first pass.
    """
    ctx = f"envcoder {spec.template_artifact_id}"
    if not isinstance(spec, TaskSpec):
        raise EnvCoderError(
            _ECCode.BAD_TYPE,
            f"{ctx}: repair loop requires a TaskSpec, got "
            f"{type(spec).__name__}",
        )
    if isinstance(max_repairs, bool) or not isinstance(max_repairs, int):
        raise EnvCoderError(
            ENVCODER_REPAIR_BAD_TYPE,
            f"{ctx}: max_repairs must be an int, got {max_repairs!r}",
        )
    if max_repairs < 0 or max_repairs > MAX_ENVCODER_REPAIRS:
        raise EnvCoderError(
            ENVCODER_REPAIR_OUT_OF_RANGE,
            f"{ctx}: max_repairs must be within [0, {MAX_ENVCODER_REPAIRS}] "
            f"(<= 2 hard validations per template), got {max_repairs}",
        )

    records: List[RepairRecord] = []
    # ---- primary call (K1) + first validation --------------------------
    artifact: EnvCoderArtifact = None
    try:
        artifact = run_envcoder(
            llm,
            spec=spec,
            seed_examples=seed_examples,
            ledger=ledger,
            window_id=window_id,
        )
    except SchemaError as e:  # parse/guard failures (incl. bare guard errors)
        current_error = f"{getattr(e, 'code', '')}: {e}"
        previous = previous_artifact_hash_for_failure(spec, e)
    else:
        report = backend.validate(artifact.env_code)
        if report.passed:
            return artifact, ()
        current_error = report.error or "validation failed"
        previous = previous_artifact_hash_for_code(artifact.env_code)

    # ---- bounded repair attempts (F1) -----------------------------------
    for retry_index in range(1, max_repairs + 1):
        ledger.record_repair_call(window_id, spec.template_artifact_id)
        system_prompt, user_prompt = build_repair_prompt(
            spec,
            previous_artifact_hash=previous,
            validation_error=current_error,
            retry_index=retry_index,
            seed_examples=seed_examples,
        )
        envelope_hash = build_repair_envelope_hash(
            spec,
            previous_artifact_hash=previous,
            validation_error=current_error,
            retry_index=retry_index,
            seed_examples=seed_examples,
        )
        evidence_hash = build_repair_evidence_hash(
            spec,
            previous_artifact_hash=previous,
            validation_error=current_error,
            retry_index=retry_index,
        )
        cache_key = make_replay_key(
            role=ENVCODER_ROLE,
            evidence_hash=evidence_hash,
            prompt_envelope_hash=envelope_hash,
            prompt_version=ENVCODER_REPAIR_PROMPT_VERSION,
            schema_version=ENVCODER_OUTPUT_SCHEMA_VERSION,
        )
        reply = llm.query(
            system_prompt,
            [user_prompt],
            cache_key=cache_key,
            role=ENVCODER_ROLE,
        )  # replay miss = HARD FAIL, exactly like the primary call
        content = _require_reply_content(
            reply, f"{ctx} repair {retry_index}"
        )
        response_hash = canonical_sha256(content)
        try:
            repaired = parse_envcoder_output(
                content, spec, f"{ctx} repair {retry_index}"
            )
        except SchemaError as e:  # parse/guard failures of the repair reply
            records.append(
                RepairRecord(
                    retry_index=retry_index,
                    previous_artifact_hash=previous,
                    validation_error=current_error,
                    reflection_prompt_hash=envelope_hash,
                    response_hash=response_hash,
                    repaired_artifact_hash="",
                )
            )
            current_error = f"{getattr(e, 'code', '')}: {e}"
            previous = previous_artifact_hash_for_failure(spec, e)
            continue
        repaired_hash = previous_artifact_hash_for_code(repaired.env_code)
        report = backend.validate(repaired.env_code)
        if report.passed:
            records.append(
                RepairRecord(
                    retry_index=retry_index,
                    previous_artifact_hash=previous,
                    validation_error=current_error,
                    reflection_prompt_hash=envelope_hash,
                    response_hash=response_hash,
                    repaired_artifact_hash=repaired_hash,
                )
            )
            final = EnvCoderArtifact(
                template_hash=spec.template_hash,
                artifact_id=spec.template_artifact_id,
                env_code=repaired.env_code,
                prompt_envelope_hash=envelope_hash,
            )
            return final, tuple(records)
        records.append(
            RepairRecord(
                retry_index=retry_index,
                previous_artifact_hash=previous,
                validation_error=current_error,
                reflection_prompt_hash=envelope_hash,
                response_hash=response_hash,
                repaired_artifact_hash=repaired_hash,
            )
        )
        current_error = report.error or "validation failed"
        previous = repaired_hash

    exhausted = EnvCoderError(
        ENVCODER_REPAIR_EXHAUSTED,
        f"{ctx}: env-code still fails validation after {max_repairs} "
        "bounded repair(s); the template is refused (no unbounded "
        "retries, no silent acceptance)",
    )
    exhausted.records = tuple(records)
    raise exhausted
