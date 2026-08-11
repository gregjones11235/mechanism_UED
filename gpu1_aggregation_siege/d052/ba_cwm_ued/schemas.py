"""BA-CWM-UED V1 structured world-state CONTRACT schemas (task §八-§九).

These are pydantic CONTRACT types: validation, serialization, hashing and
provenance ONLY (amendment §4). The model training path consumes fixed-shape
integer-id JAX PyTrees (see ``tensors.py``) and never sees these objects or
Python strings.

Design rules enforced here (fail-closed, greppable codes):

  * semantic actions ONLY — a bare action integer (``action=17``), an
    ``action_id`` / ``raw_action_index`` field, or a numeric action string is
    rejected at the schema boundary (``RAW_ACTION_INTEGER_FORBIDDEN``);
  * missing / None is NEVER silently coerced to a concrete semantic value —
    visibility maps to UNKNOWN_OR_OUT_OF_VIEW, safety to UNKNOWN
    (``task §八``: ``missing/None != NOT_VISIBLE/DEAD``);
  * every vital carries its OWN continuous value AND its OWN band — no vital
    reuses another vital's band threshold;
  * every SHA field is a full 64-char lowercase hex digest (``validate_sha256_hex``);
  * single-axis counterfactuals are validated: >1 changed axis while claiming
    ``single_axis_mutation=true`` is rejected, an unchanged control==intervention
    is rejected, and a malformed base hash is rejected.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from pydantic import Field, field_validator, model_validator

from d052.ba_cwm_ued import constants as C
from d052.ba_cwm_ued.vocabularies import (
    AGGRESSION_VOCABULARY, ENTITY_VOCABULARY, RESOURCE_VOCABULARY,
    SEMANTIC_ACTION_VOCABULARY, TERRAIN_VOCABULARY, VISIBILITY_VOCABULARY)
from d052.schemas.common import (CanonicalModel, validate_finite,
                                 validate_sha256_hex)

WORLD_STATE_SCHEMA_VERSION = "ba_cwm_ued.symbolic_world_state.v1"

# raw action integer / raw state exposure detectors (mirrors symbolic_behavior_clip)
_RAW_ACTION_KEY = re.compile(
    r"^(action|raw_action|action_int|action_integer|action_id|raw_action_index)$",
    re.I)
_NUMERIC_STR = re.compile(r"^-?\d+$")


class SchemaError(Exception):
    RAW_ACTION_INTEGER_FORBIDDEN = "RAW_ACTION_INTEGER_FORBIDDEN"
    UNKNOWN_SEMANTIC_ACTION = "UNKNOWN_SEMANTIC_ACTION"
    UNKNOWN_VOCAB_TOKEN = "UNKNOWN_VOCAB_TOKEN"
    UNKNOWN_VISIBILITY = "UNKNOWN_VISIBILITY"
    COUNTERFACTUAL_AXIS_ILLEGAL = "COUNTERFACTUAL_AXIS_ILLEGAL"
    COUNTERFACTUAL_SINGLE_AXIS_VIOLATION = "COUNTERFACTUAL_SINGLE_AXIS_VIOLATION"
    COUNTERFACTUAL_NO_DIFFERENCE = "COUNTERFACTUAL_NO_DIFFERENCE"
    ENTITY_SLOT_LIMIT_EXCEEDED = "ENTITY_SLOT_LIMIT_EXCEEDED"
    TERRAIN_GRID_LIMIT_EXCEEDED = "TERRAIN_GRID_LIMIT_EXCEEDED"
    DUPLICATE_ENTITY_SLOT = "DUPLICATE_ENTITY_SLOT"

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"[{code}] {message}")


def _assert_vocab(token: str, vocab, field: str) -> str:
    """Validate a token against a frozen vocabulary (fail-closed)."""
    try:
        vocab.id_of(token)
    except Exception as e:  # noqa: BLE001 - re-raised with schema code
        raise SchemaError(
            SchemaError.UNKNOWN_VOCAB_TOKEN,
            f"{field}: {token!r} not in frozen vocabulary {vocab.name} "
            f"version={vocab.version}") from e
    return token


# ---------------------------------------------------------------------------
# Semantic action (semantic classes ONLY — no raw integers)
# ---------------------------------------------------------------------------
class SemanticAction(CanonicalModel):
    """A semantic action. Bare action integers are forbidden."""

    action: str = Field(min_length=1)
    #: NONE / LEFT / RIGHT / UP / DOWN
    direction_class: str = "NONE"
    #: semantic target class (e.g. hostile, resource, structure) or NONE
    target_class: str = "NONE"
    #: semantic item class (craft/use) or NONE
    item_class: str = "NONE"

    @field_validator("action")
    @classmethod
    def _action_is_semantic(cls, v) -> str:
        if isinstance(v, bool) or isinstance(v, int):
            raise SchemaError(
                SchemaError.RAW_ACTION_INTEGER_FORBIDDEN,
                f"action must be a semantic class, got raw integer {v!r}")
        if not isinstance(v, str):
            raise SchemaError(
                SchemaError.RAW_ACTION_INTEGER_FORBIDDEN,
                f"action must be a semantic class string, got {type(v).__name__}")
        if _NUMERIC_STR.match(v.strip()):
            raise SchemaError(
                SchemaError.RAW_ACTION_INTEGER_FORBIDDEN,
                f"action {v!r} is a numeric string; hardcoded Craftax action "
                f"integers are forbidden — use a semantic class")
        if v not in C.SEMANTIC_ACTION_CLASSES:
            raise SchemaError(
                SchemaError.UNKNOWN_SEMANTIC_ACTION,
                f"action {v!r} not in semantic action classes "
                f"{list(C.SEMANTIC_ACTION_CLASSES)}")
        return v


def assert_no_raw_action_int(obj, *, label: str = "input") -> None:
    """Defense-in-depth scan: reject any raw action integer / raw_action_index
    key hiding inside a nested mapping (mirrors symbolic_behavior_clip)."""
    def walk(node, path):
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(k, str) and _RAW_ACTION_KEY.match(k):
                    if isinstance(v, int) and not isinstance(v, bool):
                        raise SchemaError(
                            SchemaError.RAW_ACTION_INTEGER_FORBIDDEN,
                            f"{label}: raw action integer {v!r} under key "
                            f"{path}.{k}")
                    if isinstance(v, str) and _NUMERIC_STR.match(v.strip()):
                        raise SchemaError(
                            SchemaError.RAW_ACTION_INTEGER_FORBIDDEN,
                            f"{label}: numeric action string {v!r} under key "
                            f"{path}.{k}")
                walk(v, f"{path}.{k}")
        elif isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")
    walk(obj, "$")


# ---------------------------------------------------------------------------
# Player / progress / terrain / entity / resource tokens
# ---------------------------------------------------------------------------
class VitalReading(CanonicalModel):
    """One vital with its OWN continuous value and its OWN band.

    Bands are NOT derived from a shared threshold: each vital's band is supplied
    by the (external) symbolic adapter from that vital's own scale.
    """

    value: Optional[float] = None
    band: str = "unknown"

    @field_validator("value")
    @classmethod
    def _finite(cls, v):
        if v is None:
            return None
        return validate_finite(v, "vital.value")


class PlayerStateToken(CanonicalModel):
    health: VitalReading = Field(default_factory=VitalReading)
    food: VitalReading = Field(default_factory=VitalReading)
    drink: VitalReading = Field(default_factory=VitalReading)
    energy: VitalReading = Field(default_factory=VitalReading)
    sleep_pressure: VitalReading = Field(default_factory=VitalReading)
    #: symbolic inventory/equipment present-flags (counts banded, never raw leaf)
    inventory_bands: Dict[str, str] = Field(default_factory=dict)
    equipment_bands: Dict[str, str] = Field(default_factory=dict)
    current_floor: int = 1
    progress_ordinal: int = 0
    alive: bool = True
    step_index: int = Field(default=0, ge=0)


class GlobalProgressToken(CanonicalModel):
    current_floor: int = 1
    progress_ordinal: int = 0
    achievement_count: int = Field(default=0, ge=0)
    front_reached: bool = False
    deepest_floor_seen: int = 1


class LocalTerrainToken(CanonicalModel):
    #: relative coordinates inside the LOCAL_GRID_SIZE x LOCAL_GRID_SIZE grid
    relative_x: int
    relative_y: int
    terrain_type: str = "EMPTY"
    walkable: Optional[bool] = None
    diggable: Optional[bool] = None
    recently_changed: bool = False
    contains_ladder: bool = False
    #: optional resource / entity occupancy (vocab token or None)
    resource: Optional[str] = None
    entity_slot: Optional[int] = None
    visibility_status: str = "UNKNOWN_OR_OUT_OF_VIEW"

    @field_validator("terrain_type")
    @classmethod
    def _terrain(cls, v):
        return _assert_vocab(v, TERRAIN_VOCABULARY, "terrain_type")

    @field_validator("resource")
    @classmethod
    def _resource(cls, v):
        if v is None:
            return None
        return _assert_vocab(v, RESOURCE_VOCABULARY, "terrain.resource")

    @field_validator("visibility_status")
    @classmethod
    def _vis(cls, v):
        return _assert_vocab(v, VISIBILITY_VOCABULARY, "visibility_status")


class EntityToken(CanonicalModel):
    """One bounded entity slot. missing/None -> explicit UNKNOWN, never DEAD."""

    entity_slot_id: int = Field(ge=0)
    entity_type: str
    relative_position_band: str = "unknown"
    distance_band: str = "unknown"
    health_band: str = "unknown"
    visibility_status: str = "UNKNOWN_OR_OUT_OF_VIEW"
    aggression_status: str = "UNKNOWN"
    last_seen_age: Optional[int] = Field(default=None, ge=0)
    #: stable cross-step identity for correspondence matching (None if unknown)
    persistent_entity_id: Optional[str] = None
    is_hostile: Optional[bool] = None
    is_alive: Optional[bool] = None

    @field_validator("entity_type")
    @classmethod
    def _type(cls, v):
        return _assert_vocab(v, ENTITY_VOCABULARY, "entity_type")

    @field_validator("visibility_status")
    @classmethod
    def _vis(cls, v):
        return _assert_vocab(v, VISIBILITY_VOCABULARY, "visibility_status")

    @field_validator("aggression_status")
    @classmethod
    def _agg(cls, v):
        return _assert_vocab(v, AGGRESSION_VOCABULARY, "aggression_status")


class ResourceToken(CanonicalModel):
    resource_type: str
    amount_band: str = "unknown"
    distance_band: str = "unknown"
    visibility_status: str = "UNKNOWN_OR_OUT_OF_VIEW"

    @field_validator("resource_type")
    @classmethod
    def _type(cls, v):
        return _assert_vocab(v, RESOURCE_VOCABULARY, "resource_type")

    @field_validator("visibility_status")
    @classmethod
    def _vis(cls, v):
        return _assert_vocab(v, VISIBILITY_VOCABULARY, "visibility_status")


def resolve_safety_status(env_confirmed_safe: Optional[bool]) -> str:
    """safe True/False/None/missing -> SAFE / UNSAFE / UNKNOWN / UNKNOWN."""
    if env_confirmed_safe is True:
        return "SAFE"
    if env_confirmed_safe is False:
        return "UNSAFE"
    return "UNKNOWN"


class SymbolicWorldState(CanonicalModel):
    """One fully-symbolic world state (player + global + local grid + entities +
    resources). No raw ints, no leaf indices, no coordinates beyond the bounded
    local grid."""

    step_index: int = Field(default=0, ge=0)
    player: PlayerStateToken = Field(default_factory=PlayerStateToken)
    global_progress: GlobalProgressToken = Field(
        default_factory=GlobalProgressToken)
    terrain: List[LocalTerrainToken] = Field(default_factory=list)
    entities: List[EntityToken] = Field(default_factory=list)
    resources: List[ResourceToken] = Field(default_factory=list)
    #: tri-state environmental safety (never silently coerced)
    env_confirmed_safe: Optional[bool] = None
    safety_status: str = "UNKNOWN"
    #: truncation marker: a truncated clip end is NOT a fabricated terminal
    truncation_applied: bool = False
    fabricated_terminal: bool = False
    schema_version: str = WORLD_STATE_SCHEMA_VERSION

    @model_validator(mode="after")
    def _bounds(self) -> "SymbolicWorldState":
        if len(self.terrain) > C.LOCAL_GRID_SIZE * C.LOCAL_GRID_SIZE:
            raise SchemaError(
                SchemaError.TERRAIN_GRID_LIMIT_EXCEEDED,
                f"{len(self.terrain)} terrain cells > "
                f"{C.LOCAL_GRID_SIZE}x{C.LOCAL_GRID_SIZE}")
        if len(self.entities) > C.MAX_ENTITY_SLOTS:
            raise SchemaError(
                SchemaError.ENTITY_SLOT_LIMIT_EXCEEDED,
                f"{len(self.entities)} entities > "
                f"MAX_ENTITY_SLOTS={C.MAX_ENTITY_SLOTS}")
        slots = [e.entity_slot_id for e in self.entities]
        if len(set(slots)) != len(slots):
            raise SchemaError(
                SchemaError.DUPLICATE_ENTITY_SLOT,
                f"duplicate entity_slot_id in {slots}")
        object.__setattr__(self, "safety_status",
                           resolve_safety_status(self.env_confirmed_safe))
        if self.fabricated_terminal:
            raise ValueError(
                "FABRICATED_TERMINAL_FORBIDDEN: a truncated clip end must not "
                "be marked as a real terminal state")
        return self


# ---------------------------------------------------------------------------
# Counterfactual condition (task §九 + amendment §2/§3)
# ---------------------------------------------------------------------------
class CounterfactualCondition(CanonicalModel):
    """A single counterfactual TaskParams intervention condition.

    Carries the environment-condition embedding AND the intervention mask (not
    just the final TaskParams). First phase prefers ``single_axis_mutation``;
    second-order mutations must be explicitly flagged (never the default).
    """

    intervention_id: str = Field(min_length=1)
    control_group_id: str = Field(min_length=1)
    #: axis -> symbolic level (must be legal mutation axes)
    base_axis_values: Dict[str, str] = Field(default_factory=dict)
    counterfactual_axis_values: Dict[str, str] = Field(default_factory=dict)
    changed_axes: List[str] = Field(default_factory=list)
    unchanged_axes: List[str] = Field(default_factory=list)
    single_axis_mutation: bool = True
    second_order_mutation: bool = False
    base_taskparams_sha256: str
    counterfactual_taskparams_sha256: str
    distinguishes_hypothesis_ids: List[str] = Field(default_factory=list)

    @field_validator("base_taskparams_sha256",
                     "counterfactual_taskparams_sha256")
    @classmethod
    def _hashes(cls, v, info):
        return validate_sha256_hex(v, info.field_name)

    @model_validator(mode="after")
    def _legal(self) -> "CounterfactualCondition":
        all_axes = (set(self.base_axis_values) |
                    set(self.counterfactual_axis_values) |
                    set(self.changed_axes) | set(self.unchanged_axes))
        for a in all_axes:
            if a not in C.MUTATION_AXES:
                raise SchemaError(
                    SchemaError.COUNTERFACTUAL_AXIS_ILLEGAL,
                    f"axis {a!r} is not a legal environment-induction mutation "
                    f"axis")
        overlap = set(self.changed_axes) & set(self.unchanged_axes)
        if overlap:
            raise SchemaError(
                SchemaError.COUNTERFACTUAL_AXIS_ILLEGAL,
                f"axis cannot be both changed and unchanged: {sorted(overlap)}")
        if self.single_axis_mutation and len(self.changed_axes) > 1:
            raise SchemaError(
                SchemaError.COUNTERFACTUAL_SINGLE_AXIS_VIOLATION,
                f"single_axis_mutation=true but {len(self.changed_axes)} axes "
                f"changed: {sorted(self.changed_axes)}")
        if self.second_order_mutation and len(self.changed_axes) < 2:
            raise SchemaError(
                SchemaError.COUNTERFACTUAL_SINGLE_AXIS_VIOLATION,
                f"second_order_mutation=true requires >=2 changed axes, got "
                f"{sorted(self.changed_axes)}")
        # control == intervention (no actual difference) is rejected
        if self.base_axis_values == self.counterfactual_axis_values and \
                not self.changed_axes:
            raise SchemaError(
                SchemaError.COUNTERFACTUAL_NO_DIFFERENCE,
                "counterfactual condition identical to control (no changed "
                "axes); nothing to discriminate")
        if self.base_taskparams_sha256 == self.counterfactual_taskparams_sha256 \
                and not self.changed_axes:
            raise SchemaError(
                SchemaError.COUNTERFACTUAL_NO_DIFFERENCE,
                "base and counterfactual TaskParams hashes identical with no "
                "changed axes")
        return self


# ---------------------------------------------------------------------------
# World-model transition / sequence (CONTRACT layer)
# ---------------------------------------------------------------------------
class WorldModelTransition(CanonicalModel):
    """One symbolic transition: state_t + semantic action_t (+ optional next)."""

    state: SymbolicWorldState
    action: SemanticAction
    next_state: Optional[SymbolicWorldState] = None
    #: event labels realized at t+1 (subset of EVENT_VOCABULARY); empty if none
    events: List[str] = Field(default_factory=list)
    #: behavior-outcome labels realized over the window (subset of
    #: BEHAVIOR_VOCABULARY); empty if none
    behaviors: List[str] = Field(default_factory=list)
    valid: bool = True


class WorldModelSequence(CanonicalModel):
    """A bounded sequence of transitions sharing one provenance + condition."""

    sequence_id: str = Field(min_length=1)
    transitions: List[WorldModelTransition] = Field(default_factory=list)
    condition: Optional[CounterfactualCondition] = None
    #: provenance hash binding (see provenance.TrajectoryProvenance.provenance_hash)
    provenance_hash: str = ""
    world_seed: Optional[int] = None
    episode_id: str = ""
    generator_round: int = 0
    taskparams_family: str = ""

    @model_validator(mode="after")
    def _hash(self):
        if self.provenance_hash:
            validate_sha256_hex(self.provenance_hash, "provenance_hash")
        return self
