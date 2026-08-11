"""Canonical-target firewall for the execution-mapping boundary (Phase 2.5).

Task §禁止迁入 + gate 3: ANY salted-hash / unknown / empty / hash-modulo target that
tries to enter execution mapping MUST fail -- and it must fail with a SPECIFIC,
auditable code, not merely as a generic "unknown" that could one day be aliased
into legitimacy. This module classifies an illegal target into the exact banned
class so the regression tests can assert the precise forbidden path:

  * SALTED_TARGET_FORBIDDEN    -- a target carrying a salt/digest suffix
                                  (``name::salt=..``, ``name#<hex>``, ``sha256:..``).
                                  The old Python-hash-with-salt target scheme is BANNED.
  * HASH_MODULO_TARGET_FORBIDDEN -- a target that is an integer / ``target_<n>`` /
                                  ``id_<n>`` / ``*mod67`` style id, i.e. produced by
                                  ``hash(name) % 67``. hash-modulo mapping is BANNED.
  * EMPTY_TARGET_FORBIDDEN     -- empty goal set (empty_goal_policy=error).
  * UNKNOWN_TARGET_FORBIDDEN   -- a plain string that is neither canonical nor in the
                                  explicit audited alias allow-list.
  * NON_STRING_TARGET_FORBIDDEN -- a non-str target (e.g. a raw int hash value).

The classifier is a strict superset of the achievement registry's own fail-closed
behaviour: the registry already rejects unknown/empty; this layer ADDS the specific
salted / hash-modulo detection so those banned schemes are caught and named even
though they would also happen to be "unknown".

No training, no network, no providers. Pure deterministic classification.
"""
from __future__ import annotations

import re
from typing import Iterable, List, Optional

from d052.achievements import REGISTRY, AchievementError


class TargetFirewallError(Exception):
    """Fail-closed firewall violation with a stable, specific ``code``."""

    SALTED_TARGET_FORBIDDEN = "SALTED_TARGET_FORBIDDEN"
    HASH_MODULO_TARGET_FORBIDDEN = "HASH_MODULO_TARGET_FORBIDDEN"
    EMPTY_TARGET_FORBIDDEN = "EMPTY_TARGET_FORBIDDEN"
    UNKNOWN_TARGET_FORBIDDEN = "UNKNOWN_TARGET_FORBIDDEN"
    NON_STRING_TARGET_FORBIDDEN = "NON_STRING_TARGET_FORBIDDEN"

    def __init__(self, code: str, message: str, *, offending_value=None) -> None:
        self.code = code
        self.offending_value = offending_value
        full = f"[{code}] {message}"
        if offending_value is not None:
            full += f" (offending_value={offending_value!r})"
        super().__init__(full)


# A salt/digest decoration on an otherwise legal-looking name. Matches things like
# "collect_wood::salt=abc", "eat_cow#deadbeef", "sha256:....", "name::0a1b",
# "name 0123456789abcdef0123456789abcdef" (a >=32 hex digest token).
_SALT_PATTERN = re.compile(
    r"(::\s*salt|@salt|salt\s*=|sha256\s*:|sha1\s*:|::\s*[0-9a-f]{4,}|"
    r"#\s*[0-9a-f]{6,}|\b[0-9a-f]{32,}\b|:\s*[0-9a-f]{16,}$)",
    re.IGNORECASE,
)

# A target that is actually an integer / hash-modulo id, not an achievement name.
_HASH_MODULO_PATTERN = re.compile(
    r"(^\s*-?\d+\s*$"          # pure integer string (e.g. "38")
    r"|^target_\d+$"          # target_5
    r"|^id_\d+$"              # id_5
    r"|^ach_\d+$"             # ach_5
    r"|^goal_\d+$"            # goal_5
    r"|^index_\d+$"           # index_5
    r"|mod\s*67"              # mod67 / mod 67
    r"|%\s*67"                # %67
    r"|hash\s*\("            # hash( ...
    r"|hash_mod"             # hash_mod..
    r"|_mod\d+"              # .._mod67
    r"|^0x[0-9a-f]+$)",       # hex literal id
    re.IGNORECASE,
)


def classify_target(name: object) -> Optional[str]:
    """Return the firewall CODE for an illegal target, or None if it is legal.

    Legal == a canonical achievement name OR an entry in the explicit audited
    alias allow-list. Classification priority: non-string > salted > hash-modulo
    > unknown, so a salted hash is named as SALTED (not merely unknown).
    """
    if not isinstance(name, str):
        return TargetFirewallError.NON_STRING_TARGET_FORBIDDEN
    if name.strip() == "":
        return TargetFirewallError.EMPTY_TARGET_FORBIDDEN
    if _SALT_PATTERN.search(name):
        return TargetFirewallError.SALTED_TARGET_FORBIDDEN
    if _HASH_MODULO_PATTERN.search(name):
        return TargetFirewallError.HASH_MODULO_TARGET_FORBIDDEN
    if REGISTRY.is_canonical(name) or REGISTRY.is_alias(name):
        return None
    return TargetFirewallError.UNKNOWN_TARGET_FORBIDDEN


def assert_target_firewall(targets: Iterable[object]) -> List[str]:
    """Assert every target is a legal canonical name/alias; return resolved names.

    Fail-closed with the SPECIFIC banned code:
      * empty goal set            -> EMPTY_TARGET_FORBIDDEN
      * any salted target         -> SALTED_TARGET_FORBIDDEN
      * any hash-modulo target    -> HASH_MODULO_TARGET_FORBIDDEN
      * any non-string target     -> NON_STRING_TARGET_FORBIDDEN
      * any other unknown target  -> UNKNOWN_TARGET_FORBIDDEN
    """
    items = list(targets)
    if not items:
        raise TargetFirewallError(
            TargetFirewallError.EMPTY_TARGET_FORBIDDEN,
            "target achievement set is empty (empty_goal_policy=error)")
    # classify ALL first so the most specific banned class is reported with the
    # offending value, before we fall through to generic unknown.
    for t in items:
        code = classify_target(t)
        if code is not None and code != TargetFirewallError.UNKNOWN_TARGET_FORBIDDEN:
            raise TargetFirewallError(
                code,
                f"banned target scheme detected; canonical_v2 maps targets ONLY by "
                f"official canonical achievement name (canonical_id == goal_vector_"
                f"index); salted-hash / hash-modulo / empty mappings are forbidden",
                offending_value=t)
    for t in items:
        code = classify_target(t)
        if code == TargetFirewallError.UNKNOWN_TARGET_FORBIDDEN:
            raise TargetFirewallError(
                code,
                f"target is neither a canonical achievement name nor in the explicit "
                f"alias allow-list (unknown_target_policy=error)",
                offending_value=t)
    # all legal -> resolve through the registry (alias-aware, deterministic order)
    return REGISTRY.canonicalize_targets(items)  # type: ignore[arg-type]


def assert_execution_mapping_rejects(targets: Iterable[object]) -> str:
    """Prove the banned targets fail at the execution-mapping boundary.

    Attempts the canonical chain (firewall + registry canonicalize + 67-dim
    goal-vector synthesis). MUST raise. Returns the firewall code that caught it
    so a regression test can assert the precise forbidden path. If the chain ever
    succeeds for a banned input, raises AssertionError (the firewall regressed).
    """
    items = list(targets)
    try:
        assert_target_firewall(items)
        # firewall passed -> also drive the registry/vector path to be thorough
        REGISTRY.to_goal_vector(items)  # type: ignore[arg-type]
    except TargetFirewallError as e:
        return e.code
    except AchievementError as e:
        # the registry's own fail-closed code (unknown/empty) is also a valid reject
        return e.code
    raise AssertionError(
        f"FIREWALL_REGRESSION: banned targets {items!r} were NOT rejected by the "
        f"execution-mapping boundary; the canonical target firewall has regressed")
