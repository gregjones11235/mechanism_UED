"""v7fix2 — offline tests for the RELAY GENERATION CONTRACT (kit vocabulary + floor bounds).

First-run evidence (job 3791883, killed s6): the generator LLM wrote spawn kits with
non-existent Inventory kwargs (``stone_pickaxe`` 3x, ``wood_pickaxe`` 1x) — taught by the old
level_meta example itself (``{"iron_sword": 1, "torch": 4}``: both keys illegal) while the coder
prompt never listed the legal fields and ``set_player_inventory`` expanded ``**dict`` unchecked.
Every such R0 level died in check_compilation; once a relay opened, its rungs would starve.

The contract now has ONE vocabulary source (minicraftax/spawn_kit.py, pure python):
  P1 level_meta spec shows the legal fields + a legal example (positive vocabulary only);
  P2 set_player_inventory maps compounds/synonyms/telemetry labels and raises a TEACHING
     error (reflection loop) on anything else;
  P3 the WINNER-MEDIAN STOCKPILES hint renders canonical kit field names;
  P4 check_compilation cross-checks declared spawn_floor vs the code's actual reset floor;
  P5 relay_r0_floor bound derives from the sourced MAX_DUNGEON_FLOOR (8), not a literal.

No jax/craftax/LLM needed (spawn_kit.py is loaded by file path, bypassing jax-heavy packages).
"""

import importlib.util
import os

import pytest

from auction.craftax_achievements import MAX_DUNGEON_FLOOR
from auction.level_meta import LEVEL_META_SPEC_TEXT
from auction.modeler import Modeler

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
_SPAWN_KIT = os.path.join(_REPO, "src", "minicraftax", "spawn_kit.py")

_spec = importlib.util.spec_from_file_location("spawn_kit", _SPAWN_KIT)
spawn_kit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(spawn_kit)


# ---- P2: normalise_spawn_kit ----------------------------------------------------------------

def test_material_compounds_map_to_tiers():
    out = spawn_kit.normalise_spawn_kit({"stone_pickaxe": 1, "iron_sword": 1})
    assert out == {"pickaxe": 2, "sword": 3}


def test_compound_count_is_ignored_material_is_tier():
    # {"wood_pickaxe": 5} means five wooden pickaxes to the LLM — the env has ONE tier slot.
    assert spawn_kit.normalise_spawn_kit({"wood_pickaxe": 5}) == {"pickaxe": 1}


def test_collisions_max_merge():
    out = spawn_kit.normalise_spawn_kit({"stone_pickaxe": 1, "diamond_pickaxe": 1, "pickaxe": 1})
    assert out == {"pickaxe": 4}


def test_synonyms_and_passthrough():
    out = spawn_kit.normalise_spawn_kit({"torch": 4, "arrow": 20, "coal": 8, "pickaxe": 3})
    assert out == {"torches": 4, "arrows": 20, "coal": 8, "pickaxe": 3}


def test_armour_compound_and_telemetry_labels():
    out = spawn_kit.normalise_spawn_kit({"iron_armour": 1, "potions_3": 2, "armour_2": 2})
    assert out == {"armour": 2, "potions": 2}


def test_unknown_key_raises_teaching_error():
    with pytest.raises(ValueError) as e:
        spawn_kit.normalise_spawn_kit({"smelted_iron_ingot": 1})
    msg = str(e.value)
    assert "smelted_iron_ingot" in msg
    assert "Legal keys" in msg and "pickaxe" in msg  # the message carries the vocabulary


def test_negative_counts_clamp_to_zero():
    assert spawn_kit.normalise_spawn_kit({"arrows": -5}) == {"arrows": 0}


def test_legal_fields_are_the_16_inventory_fields():
    assert len(spawn_kit.LEGAL_FIELDS) == 16
    assert set(spawn_kit.ARRAY_FIELDS) == {"armour", "potions"}


# ---- P3: telemetry label canonicalisation ----------------------------------------------------

def test_telemetry_labels_canonicalise():
    assert spawn_kit.canonicalise_telemetry_label("armour_2") == "armour"
    assert spawn_kit.canonicalise_telemetry_label("potions_5") == "potions"
    assert spawn_kit.canonicalise_telemetry_label("pickaxe") == "pickaxe"
    assert spawn_kit.canonicalise_telemetry_label("not_a_field") is None


# ---- P1: the spec teaches the positive vocabulary --------------------------------------------

def test_level_meta_spec_shows_legal_example():
    flat = " ".join(LEVEL_META_SPEC_TEXT.split())  # the spec wraps lines mid-sentence
    assert '{"pickaxe": 2' in flat
    assert "1 wood, 2 stone, 3 iron, 4 diamond" in flat
    # the old bad example is gone; the spec never name-drops illegal keys (positive-only).
    assert "iron_sword" not in LEVEL_META_SPEC_TEXT
    assert '"torch"' not in LEVEL_META_SPEC_TEXT


# ---- P5: r0 floor bound derives from the sourced constant ------------------------------------

def _r0(v):
    su = Modeler._validate_siege({"siege_update": {"foci": [
        {"skill": "defeat_kobold", "prereq_tree": [], "relay_r0_floor": v},
    ]}})
    return su["foci"][0]["relay_r0_floor"]


def test_r0_floor_accepts_the_deepest_real_floor():
    assert MAX_DUNGEON_FLOOR == 8
    assert _r0(8) == 8
    assert _r0(3) == 3


def test_r0_floor_rejects_floors_that_do_not_exist():
    assert _r0(9) is None
    assert _r0(12) is None  # the first run's loose literal — never again
    assert _r0(0) is None
