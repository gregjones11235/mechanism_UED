"""Spawn-kit vocabulary — the single source of truth for legal starting-inventory keys.

v7fix2: the generator LLM writes ``builder.set_player_inventory({...})`` for relay R0 kits, but
nothing ever told it the legal key set, and the old level_meta example even modelled a
non-existent compound name. This module is pure python (no jax / craftax import) so the offline
auction-layer tests can exercise it; the launch sanity block cross-checks ``LEGAL_FIELDS``
against the real ``craftax.craftax.craftax_state.Inventory`` dataclass fields.

Semantics (Craftax full):
  - ``pickaxe`` / ``sword`` hold a TIER int: 0 none, 1 wood, 2 stone, 3 iron, 4 diamond.
  - ``armour`` / ``potions`` are per-slot arrays in the env; a kit gives ONE scalar and the
    builder broadcasts it to every slot (armour: 1 iron, 2 diamond).
  - everything else is a plain item count.
"""

from __future__ import annotations

SCALAR_FIELDS: tuple[str, ...] = (
    "wood", "stone", "coal", "iron", "diamond", "sapling",
    "pickaxe", "sword", "bow", "arrows", "torches", "ruby", "sapphire", "books",
)
ARRAY_FIELDS: tuple[str, ...] = ("armour", "potions")
LEGAL_FIELDS: tuple[str, ...] = SCALAR_FIELDS + ARRAY_FIELDS

_TIER = {"wood": 1, "stone": 2, "iron": 3, "diamond": 4}
_ARMOUR_TIER = {"iron": 1, "diamond": 2}
# singular/plural slips the LLM plausibly makes — mapped, not rejected.
_SYNONYMS = {"torch": "torches", "arrow": "arrows", "potion": "potions", "book": "books"}


def canonicalise_telemetry_label(label: str) -> str | None:
    """Map a fix9-P1 telemetry column label onto its kit field, or None if it is no kit field.

    The telemetry flattens the Inventory struct programmatically: scalar fields keep their name
    (already legal), array fields become ``armour_0..3`` / ``potions_0..N`` — those indexed
    labels are NOT legal ``set_player_inventory`` kwargs and must collapse to their base field
    before they are shown to the LLM as spawn_kit evidence.
    """
    l = str(label).strip().lower()
    if l in LEGAL_FIELDS:
        return l
    base, _, idx = l.rpartition("_")
    if base in ARRAY_FIELDS and idx.isdigit():
        return base
    return None


def normalise_spawn_kit(kit: dict) -> dict[str, int]:
    """Coerce an LLM-written kit dict onto the legal Inventory fields.

    Mapped silently (they carry an unambiguous meaning): material compounds
    (``stone_pickaxe`` -> ``pickaxe: 2``, ``iron_armour`` -> ``armour: 1``; the compound's
    COUNT is ignored, its material IS the tier), telemetry array labels (``armour_2`` ->
    ``armour``), singular/plural synonyms. Collisions max-merge (a kit naming stone AND iron
    pickaxes gets the better tool). Anything else raises ValueError whose message lists the
    legal vocabulary — check_compilation catches it and the reflection loop feeds it back to
    the generator, so the error TEACHES instead of just failing.
    """
    out: dict[str, int] = {}
    unknown: list[str] = []
    for raw_k, raw_v in (kit or {}).items():
        k = str(raw_k).strip().lower().replace(" ", "_")
        k = _SYNONYMS.get(k, k)
        try:
            v = max(0, int(raw_v))
        except (TypeError, ValueError):
            unknown.append(str(raw_k))
            continue
        if k in LEGAL_FIELDS:
            out[k] = max(out.get(k, 0), v)
            continue
        material, _, item = k.rpartition("_")
        if item in ("pickaxe", "sword") and material in _TIER:
            out[item] = max(out.get(item, 0), _TIER[material])
            continue
        if item == "armour" and material in _ARMOUR_TIER:
            out["armour"] = max(out.get("armour", 0), _ARMOUR_TIER[material])
            continue
        mapped = canonicalise_telemetry_label(k)
        if mapped is not None:
            out[mapped] = max(out.get(mapped, 0), v)
            continue
        unknown.append(str(raw_k))
    if unknown:
        raise ValueError(
            f"spawn_kit key(s) {unknown!r} are not inventory fields. Legal keys: "
            f"{', '.join(LEGAL_FIELDS)}. pickaxe/sword take a TIER number "
            f"(1 wood, 2 stone, 3 iron, 4 diamond); armour/potions take one scalar "
            f"applied to every slot."
        )
    return out
