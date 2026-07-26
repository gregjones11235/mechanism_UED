"""Official Craftax-67 achievement registry (canonical_v2 single source).

Loads the committed, source-generated ``official_achievements.json`` and
``explicit_aliases.json`` (sibling data files) and exposes a fail-closed API:

  * canonical names are lowercase; matching is case-sensitive (NO coercion);
  * an unknown target name is an ERROR (unknown_target_policy=error), unless it
    appears in the explicit, audited alias allow-list;
  * an empty goal set is an ERROR (empty_goal_policy=error);
  * ``to_goal_vector`` emits the 67-dim achievement multi-hot whose index j is the
    canonical_id == goal_vector_index == craftax enum value.

Stdlib-only (returns plain ``list[float]`` vectors) so the registry is importable
and deterministic without numpy; numpy conversion is the training adapter's job.

Provenance / drift: GATE 3 (d052/tests/test_achievement_registry.py) re-imports
the canonical source module and asserts the committed JSON matches it exactly, so
any silent drift between source and registry fails CI.
"""
from __future__ import annotations

import json
import os
from typing import Dict, FrozenSet, Iterable, List, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_OFFICIAL_PATH = os.path.join(_HERE, "official_achievements.json")
_ALIASES_PATH = os.path.join(_HERE, "explicit_aliases.json")

ACHIEVEMENT_SCHEMA = "craftax_67_v1"
NUM_ACHIEVEMENTS = 67


class AchievementError(Exception):
    """Fail-closed achievement/target violation. Carries a stable ``code``."""

    UNKNOWN_ACHIEVEMENT = "UNKNOWN_ACHIEVEMENT"
    EMPTY_GOAL_SET = "EMPTY_GOAL_SET"
    REGISTRY_DRIFT = "REGISTRY_DRIFT"
    REGISTRY_CORRUPT = "REGISTRY_CORRUPT"

    def __init__(self, code: str, message: str, *, offending_value=None) -> None:
        self.code = code
        self.offending_value = offending_value
        full = f"[{code}] {message}"
        if offending_value is not None:
            full += f" (offending_value={offending_value!r})"
        super().__init__(full)


class AchievementRegistry:
    """Immutable, fail-closed view of the 67 canonical achievements + aliases."""

    def __init__(self, official_path: str = _OFFICIAL_PATH,
                 aliases_path: str = _ALIASES_PATH) -> None:
        try:
            with open(official_path, encoding="utf-8") as f:
                official = json.load(f)
            with open(aliases_path, encoding="utf-8") as f:
                aliases = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            raise AchievementError(
                AchievementError.REGISTRY_CORRUPT,
                f"cannot load achievement registry: {e}") from e

        if official.get("achievement_schema") != ACHIEVEMENT_SCHEMA:
            raise AchievementError(
                AchievementError.REGISTRY_DRIFT,
                f"achievement_schema != {ACHIEVEMENT_SCHEMA}: "
                f"{official.get('achievement_schema')!r}")

        entries: List[dict] = official["achievements"]
        if len(entries) != NUM_ACHIEVEMENTS:
            raise AchievementError(
                AchievementError.REGISTRY_DRIFT,
                f"registry has {len(entries)} achievements, expected "
                f"{NUM_ACHIEVEMENTS}")

        self._name_to_id: Dict[str, int] = {}
        self._id_to_name: Dict[int, str] = {}
        self._depth: Dict[str, int] = {}
        self._family: Dict[str, str] = {}
        for e in entries:
            cid = e["canonical_id"]
            name = e["name"]
            if e["goal_vector_index"] != cid:
                raise AchievementError(
                    AchievementError.REGISTRY_DRIFT,
                    f"goal_vector_index != canonical_id for {name}")
            if name in self._name_to_id or cid in self._id_to_name:
                raise AchievementError(
                    AchievementError.REGISTRY_DRIFT,
                    f"duplicate canonical entry: {name}/{cid}")
            self._name_to_id[name] = cid
            self._id_to_name[cid] = name
            self._depth[name] = e["depth_tier"]
            self._family[name] = e["family"]

        if sorted(self._id_to_name) != list(range(NUM_ACHIEVEMENTS)):
            raise AchievementError(
                AchievementError.REGISTRY_DRIFT,
                "canonical ids are not exactly 0..66")

        # explicit, audited alias allow-list
        self._alias_to_canonical: Dict[str, str] = {}
        for a in aliases.get("aliases", []):
            alias = a["alias"]
            canon = a["canonical_name"]
            if canon not in self._name_to_id:
                raise AchievementError(
                    AchievementError.REGISTRY_DRIFT,
                    f"alias {alias!r} points to non-canonical name {canon!r}")
            if self._name_to_id[canon] != a["canonical_id"]:
                raise AchievementError(
                    AchievementError.REGISTRY_DRIFT,
                    f"alias {alias!r} canonical_id mismatch for {canon!r}")
            self._alias_to_canonical[alias] = canon

        self.source_provenance: dict = official.get("source", {})

    # --- queries -----------------------------------------------------------
    @property
    def names(self) -> FrozenSet[str]:
        return frozenset(self._name_to_id)

    @property
    def count(self) -> int:
        return len(self._name_to_id)

    def is_canonical(self, name: object) -> bool:
        return isinstance(name, str) and name in self._name_to_id

    def is_alias(self, name: object) -> bool:
        return isinstance(name, str) and name in self._alias_to_canonical

    def canonical_id(self, name: str) -> int:
        if name not in self._name_to_id:
            raise AchievementError(
                AchievementError.UNKNOWN_ACHIEVEMENT,
                f"{name!r} is not a canonical achievement name", offending_value=name)
        return self._name_to_id[name]

    def name_for_id(self, cid: int) -> str:
        if cid not in self._id_to_name:
            raise AchievementError(
                AchievementError.UNKNOWN_ACHIEVEMENT,
                f"canonical_id {cid} out of range 0..66", offending_value=cid)
        return self._id_to_name[cid]

    def depth_tier(self, name: str) -> int:
        return self._depth[self.canonical_id_name(name)]

    def family(self, name: str) -> str:
        return self._family[self.canonical_id_name(name)]

    def canonical_id_name(self, name: str) -> str:
        """Internal: validate name is canonical and return it."""
        if name not in self._name_to_id:
            raise AchievementError(
                AchievementError.UNKNOWN_ACHIEVEMENT,
                f"{name!r} is not canonical", offending_value=name)
        return name

    # --- resolution + goal vector -----------------------------------------
    def resolve(self, name: object) -> str:
        """Resolve a name to its canonical name via the explicit alias allow-list.

        Case-sensitive, exact match only. Unknown -> AchievementError
        (unknown_target_policy=error). Never fuzzy / never silently dropped.
        """
        if not isinstance(name, str):
            raise AchievementError(
                AchievementError.UNKNOWN_ACHIEVEMENT,
                f"achievement name must be str, got {type(name).__name__}",
                offending_value=name)
        if name in self._name_to_id:
            return name
        if name in self._alias_to_canonical:
            return self._alias_to_canonical[name]
        raise AchievementError(
            AchievementError.UNKNOWN_ACHIEVEMENT,
            f"unknown achievement target {name!r}; not canonical and not in the "
            f"explicit alias allow-list (unknown_target_policy=error)",
            offending_value=name)

    def canonicalize_targets(self, targets: Iterable[str]) -> List[str]:
        """Resolve + validate a target set; return sorted canonical names.

        Empty input -> AchievementError (empty_goal_policy=error). Duplicates
        collapse (a multi-hot has one bit per achievement). Order is canonical
        (sorted by canonical_id) for determinism.
        """
        items = list(targets)
        if not items:
            raise AchievementError(
                AchievementError.EMPTY_GOAL_SET,
                "target achievement set is empty (empty_goal_policy=error)")
        ids = sorted({self._name_to_id[self.resolve(t)] for t in items})
        return [self._id_to_name[i] for i in ids]

    def to_goal_vector(self, targets: Iterable[str]) -> List[float]:
        """67-dim achievement multi-hot. Index j == canonical_id == goal_vector_index.

        Deterministic; depends only on the SET of targets. Empty -> error.
        Unknown -> error. Returns plain list[float] (0.0/1.0) of length 67.
        """
        vec = [0.0] * NUM_ACHIEVEMENTS
        for name in self.canonicalize_targets(targets):
            vec[self._name_to_id[name]] = 1.0
        return vec

    def goal_vector_dim(self) -> int:
        return NUM_ACHIEVEMENTS


# Module-level shared instance (importable as `from d052.achievements import REGISTRY`).
REGISTRY = AchievementRegistry()
