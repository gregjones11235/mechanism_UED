"""Frozen token vocabularies (amendment §4).

Two-layer design:

  * CONTRACT layer (``schemas.py``) uses pydantic + strings for validation,
    serialization, hashing and provenance;
  * TRAINING layer consumes FIXED-SHAPE integer-id JAX PyTrees and MUST NOT see
    Python strings or pydantic objects.

These vocabularies are the bridge: each maps a closed set of semantic NAMES to
stable integer ids. Reserved ids are identical across every vocabulary:

    PAD_ID     = 0   (padding; never a real token; masked out of attention/loss)
    UNKNOWN_ID = 1   (out-of-vocabulary / unresolved sentinel; NOT a semantic
                      "UNKNOWN_*" class, which are legitimate tokens with id>=2)

Real tokens occupy ids ``2 .. 1 + len(tokens)``; ``vocab_size = 2 + len(tokens)``.

Each vocabulary is content-addressed: ``vocabulary_sha256`` is the canonical-JSON
sha256 over {name, version, tokens, pad_id, unknown_id, source}, so a vocabulary
cannot drift from its recorded identity.

Provenance of the token sets:
  * ``cwm_contract``            — CWM-defined, authoritative for the contract
                                  (semantic action, visibility, aggression,
                                  entity-correspondence, event);
  * ``documented_craftax_fixture`` — extracted from documented Craftax sources
                                  (terrain / entity / resource). These are
                                  DOCUMENTED_COMPATIBILITY_FIXTURE ONLY
                                  (amendment §6): usable for synthetic tests,
                                  NOT for real replay ingestion until the real
                                  Craftax Action/state registry is installed and
                                  its version / names / integer values / registry
                                  SHA256 are verified.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from pydantic import Field, model_validator

from d052.bagr_ued.hashing import canonical_sha256
from d052.ba_cwm_ued import constants as C
from d052.schemas.common import CanonicalModel

PAD_ID = 0
UNKNOWN_ID = 1
_FIRST_REAL_ID = 2


class VocabularyError(Exception):
    UNKNOWN_TOKEN = "UNKNOWN_TOKEN"
    UNKNOWN_TOKEN_ID = "UNKNOWN_TOKEN_ID"
    RESERVED_TOKEN_NAME = "RESERVED_TOKEN_NAME"

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"[{code}] {message}")


class FrozenVocabulary(CanonicalModel):
    """A closed, content-addressed name<->id vocabulary."""

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    #: real semantic tokens, in id order (ids assigned 2..1+len)
    tokens: Tuple[str, ...] = Field(default_factory=tuple)
    pad_id: int = PAD_ID
    unknown_id: int = UNKNOWN_ID
    #: cwm_contract | documented_craftax_fixture
    source: str = Field(pattern=r"^(cwm_contract|documented_craftax_fixture)$")

    @model_validator(mode="after")
    def _well_formed(self) -> "FrozenVocabulary":
        if self.pad_id == self.unknown_id:
            raise ValueError("VOCAB_RESERVED_COLLISION: pad_id == unknown_id")
        seen = set()
        for tok in self.tokens:
            low = tok.lower()
            if low in ("pad", "<pad>", "unknown", "<unk>"):
                raise VocabularyError(
                    VocabularyError.RESERVED_TOKEN_NAME,
                    f"vocabulary {self.name}: token {tok!r} collides with a "
                    f"reserved id name")
            if tok in seen:
                raise ValueError(f"VOCAB_DUPLICATE_TOKEN: {tok!r} in {self.name}")
            seen.add(tok)
        return self

    # -- identity -----------------------------------------------------------
    def identity_payload(self) -> dict:
        return {"name": self.name, "version": self.version,
                "tokens": list(self.tokens), "pad_id": self.pad_id,
                "unknown_id": self.unknown_id, "source": self.source}

    def vocabulary_sha256(self) -> str:
        return canonical_sha256(self.identity_payload())

    # -- sizes / lookup -----------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return _FIRST_REAL_ID + len(self.tokens)

    def id_of(self, token: str, *, allow_unknown: bool = False) -> int:
        """Map a token name to its integer id.

        With ``allow_unknown=False`` an out-of-vocabulary token fails closed
        (``UNKNOWN_TOKEN``); with ``allow_unknown=True`` it maps to UNKNOWN_ID.
        """
        for i, tok in enumerate(self.tokens):
            if tok == token:
                return _FIRST_REAL_ID + i
        if allow_unknown:
            return self.unknown_id
        raise VocabularyError(
            VocabularyError.UNKNOWN_TOKEN,
            f"token {token!r} not in vocabulary {self.name} "
            f"version={self.version}")

    def name_of(self, token_id: int) -> str:
        if token_id == self.pad_id:
            return "<PAD>"
        if token_id == self.unknown_id:
            return "<UNK>"
        idx = token_id - _FIRST_REAL_ID
        if 0 <= idx < len(self.tokens):
            return self.tokens[idx]
        raise VocabularyError(
            VocabularyError.UNKNOWN_TOKEN_ID,
            f"token id {token_id} out of range for vocabulary {self.name} "
            f"(vocab_size={self.vocab_size})")

    def encode(self, tokens: Sequence[str], *, allow_unknown: bool = False
               ) -> List[int]:
        return [self.id_of(t, allow_unknown=allow_unknown) for t in tokens]


# ---------------------------------------------------------------------------
# The frozen vocabories.
# ---------------------------------------------------------------------------

#: Semantic action classes — CWM contract, authoritative.
SEMANTIC_ACTION_VOCABULARY = FrozenVocabulary(
    name="semantic_action", version="cwm.semantic_action.v1",
    tokens=C.SEMANTIC_ACTION_CLASSES, source="cwm_contract")

#: Visibility status — CWM contract, authoritative.
VISIBILITY_VOCABULARY = FrozenVocabulary(
    name="visibility", version="cwm.visibility.v1",
    tokens=C.VISIBILITY_CLASSES, source="cwm_contract")

#: Aggression status — CWM contract, authoritative.
AGGRESSION_VOCABULARY = FrozenVocabulary(
    name="aggression", version="cwm.aggression.v1",
    tokens=C.AGGRESSION_CLASSES, source="cwm_contract")

#: Entity correspondence labels — CWM contract, authoritative.
ENTITY_CORRESPONDENCE_VOCABULARY = FrozenVocabulary(
    name="entity_correspondence", version="cwm.entity_correspondence.v1",
    tokens=C.ENTITY_CORRESPONDENCE_LABELS, source="cwm_contract")

#: Atomic event labels — CWM contract, authoritative. These are the event heads
#: the world model predicts (task §十五). ``achievement_unlocked`` is a single
#: head; per-achievement multi-hot uses the D052 67-dim achievement registry.
EVENT_VOCABULARY = FrozenVocabulary(
    name="event", version="cwm.event.v1",
    tokens=(
        "damage_taken",
        "death",
        "chased",
        "defeat_kobold",
        "floor2_to_floor3_transition",
        "dig_success",
        "resource_collected",
        "achievement_unlocked",
        "episode_done",
    ),
    source="cwm_contract")

#: Behavior-outcome labels the world model predicts as OUTCOME predictions (NOT
#: action supervision) — task §十五 behavior head.
BEHAVIOR_VOCABULARY = FrozenVocabulary(
    name="behavior", version="cwm.behavior.v1",
    tokens=(
        "unsafe_rest_near_hostile",
        "repeated_no_effect_action",
        "oscillation_loop",
        "combat_freeze",
        "resource_neglect",
        "progress_regression",
    ),
    source="cwm_contract")

# --- documented Craftax compatibility FIXTURES (amendment §6) ---------------
# Extracted from documented Craftax sources; NOT verified against an installed
# registry. Usable for synthetic tests only; NOT for real replay ingestion.

#: Documented terrain types (9x9 local grid cell classes).
TERRAIN_VOCABULARY = FrozenVocabulary(
    name="terrain", version="cwm.terrain.documented_fixture.v1",
    tokens=(
        "WATER", "GRASS", "SAND", "PATH", "STONE", "WALL", "TREE",
        "LAVA", "IRON_ORE", "COAL_ORE", "LADDER_DOWN", "LADDER_UP",
        "DOOR", "TORCH", "EMPTY",
    ),
    source="documented_craftax_fixture")

#: Documented (non-player) entity types occupying entity slots.
ENTITY_VOCABULARY = FrozenVocabulary(
    name="entity", version="cwm.entity.documented_fixture.v1",
    tokens=(
        "ZOMBIE", "KOBOLD", "SKELETON", "SPIDER", "MONSTER",
        "BOAR", "CHICKEN", "COW", "ARROW", "FIREBALL", "ICEBALL",
    ),
    source="documented_craftax_fixture")

#: Documented resource types.
RESOURCE_VOCABULARY = FrozenVocabulary(
    name="resource", version="cwm.resource.documented_fixture.v1",
    tokens=(
        "WOOD", "STONE", "IRON", "COAL", "FOOD", "WATER", "TORCH",
        "ARROW", "SWORD", "PICKAXE",
    ),
    source="documented_craftax_fixture")

#: Every vocabulary exported for hashing / reporting.
ALL_VOCABULARIES: Dict[str, FrozenVocabulary] = {
    v.name: v for v in (
        SEMANTIC_ACTION_VOCABULARY, VISIBILITY_VOCABULARY,
        AGGRESSION_VOCABULARY, ENTITY_CORRESPONDENCE_VOCABULARY,
        EVENT_VOCABULARY, BEHAVIOR_VOCABULARY, TERRAIN_VOCABULARY,
        ENTITY_VOCABULARY, RESOURCE_VOCABULARY,
    )
}


def vocabulary_contract_report() -> dict:
    """Serializable report of every vocabulary identity (task §二十七)."""
    return {
        "schema": "ba_cwm_ued.vocabulary_contract.v1",
        "pad_id": PAD_ID,
        "unknown_id": UNKNOWN_ID,
        "real_craftax_action_registry_ready":
            C.REAL_CRAFTAX_ACTION_REGISTRY_READY,
        "vocabularies": {
            name: {"version": v.version, "source": v.source,
                   "vocab_size": v.vocab_size, "tokens": list(v.tokens),
                   "vocabulary_sha256": v.vocabulary_sha256()}
            for name, v in sorted(ALL_VOCABULARIES.items())
        },
    }
