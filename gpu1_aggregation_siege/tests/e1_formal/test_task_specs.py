"""C7 tests: canonical TaskSpec compiler (REGISTRY-bound, dedup, cap)."""
import pytest

from dicode.teachers.e1_formal import task_specs as TS
from dicode.teachers.e1_formal.board import WINDOW_STATUS_VOID

from test_board import _build_store, _evidence, _run


def _family(fid, desc=None, targets=("collect_coal",)):
    return {
        "family_id": fid,
        "description": desc if desc is not None else f"intervention {fid}",
        "target_achievements": list(targets),
        "axis_changes": [
            {"axis": "enemy_density", "from_value": "high", "to_value": "low"}
        ],
        "constant_axes": ["map_size"],
        "scaffolding": "keep the corridor short",
        "student_must_do": "collect coal and return",
    }


def _window_with_families(families, critic=None):
    evidence = _evidence()
    overrides = {
        "intervention_tutor": {"families": families, "explorations": []}
    }
    if critic is not None:
        overrides["critic"] = critic
    store = _build_store(evidence, overrides=overrides)
    window, _ = _run(evidence=evidence, store=store)
    return window


class TestCompile:
    def test_two_families_two_variants_each(self):
        window = _window_with_families([_family("fam_a"), _family("fam_b")])
        result = TS.compile_task_specs(window)
        assert len(result.specs) == 4
        assert [s.spec_id for s in result.specs] == [
            "w01::fam_a::v0",
            "w01::fam_a::v1",
            "w01::fam_b::v0",
            "w01::fam_b::v1",
        ]
        for spec in result.specs:
            assert spec.window_hash == window.window_hash
            assert len(spec.spec_hash) == 64
            assert spec.artifact_id == f"{spec.spec_hash}::v{spec.variant}"
        assert len({s.artifact_id for s in result.specs}) == 4

    def test_compile_is_deterministic(self):
        window = _window_with_families([_family("fam_a")])
        assert TS.compile_task_specs(window) == TS.compile_task_specs(window)

    def test_variants_share_content_hash_but_differ_in_identity(self):
        window = _window_with_families([_family("fam_a")])
        v0, v1 = TS.compile_task_specs(window).specs
        assert v0.spec_hash == v1.spec_hash
        assert v0.artifact_id != v1.artifact_id
        assert v0.spec_id != v1.spec_id

    def test_alias_resolves_to_canonical_registry_name(self):
        # explicit audited alias: defeat_orc_soldier -> defeat_orc_solider
        window = _window_with_families(
            [_family("fam_a", targets=("defeat_orc_soldier",))]
        )
        spec = TS.compile_task_specs(window).specs[0]
        assert spec.target_achievements == ("defeat_orc_solider",)

    def test_targets_sorted_and_deduplicated(self):
        window = _window_with_families(
            [_family("fam_a", targets=("collect_coal", "cast_fireball", "collect_coal"))]
        )
        spec = TS.compile_task_specs(window).specs[0]
        assert spec.target_achievements == ("cast_fireball", "collect_coal")

    def test_vetoed_family_produces_no_specs(self):
        window = _window_with_families(
            [_family("fam_a"), _family("fam_b")],
            critic={"vetoes": [{"family_id": "fam_b", "reason": "GUARD_VETO"}]},
        )
        result = TS.compile_task_specs(window)
        assert {s.family_id for s in result.specs} == {"fam_a"}


class TestFailClosed:
    def test_unknown_achievement_rejected(self):
        window = _window_with_families(
            [_family("fam_a", targets=("not_a_real_achievement_xyz",))]
        )
        with pytest.raises(TS.TaskSpecError) as excinfo:
            TS.compile_task_specs(window)
        assert excinfo.value.code == TS.UNKNOWN_ACHIEVEMENT

    def test_empty_goal_set_rejected(self):
        window = _window_with_families([_family("fam_a", targets=())])
        with pytest.raises(TS.TaskSpecError) as excinfo:
            TS.compile_task_specs(window)
        assert excinfo.value.code == TS.EMPTY_GOAL_SET

    def test_void_window_cannot_compile(self):
        evidence = _evidence()
        store = _build_store(evidence, overrides={"behavior_auditor": {"bad": 1}})
        window, _ = _run(evidence=evidence, store=store)
        assert window.status == WINDOW_STATUS_VOID
        with pytest.raises(TS.TaskSpecError) as excinfo:
            TS.compile_task_specs(window)
        assert excinfo.value.code == TS.TASK_SPEC_VOID_WINDOW

    @pytest.mark.parametrize("bad", [0, -1, True, "2"])
    def test_bad_variants_per_spec_rejected(self, bad):
        window = _window_with_families([_family("fam_a")])
        with pytest.raises(ValueError):
            TS.compile_task_specs(window, variants_per_spec=bad)


class TestDedupAndCap:
    def test_duplicate_content_is_deduped_with_note(self):
        base = _family("fam_a")
        dup = dict(base)
        dup["family_id"] = "fam_dup"  # same CONTENT, different id
        window = _window_with_families([base, dup])
        result = TS.compile_task_specs(window)
        assert {s.family_id for s in result.specs} == {"fam_a"}
        assert len(result.specs) == 2  # only the first family's variants
        assert len(result.notes) == 2
        for note in result.notes:
            assert note["note"] == TS.DEDUPED_SPEC
            assert note["family_id"] == "fam_dup"

    def test_cap_truncates_deterministically_with_notes(self):
        families = [_family(f"fam_{i}") for i in range(6)]  # 12 specs > cap 10
        window = _window_with_families(families)
        result = TS.compile_task_specs(window)
        assert len(result.specs) == TS.MAX_WINDOW_SPECS
        kept = {s.family_id for s in result.specs}
        assert kept == {f"fam_{i}" for i in range(5)}
        truncated = [n for n in result.notes if n["note"] == TS.SPECS_TRUNCATED_TO_CAP]
        assert len(truncated) == 2
        assert {n["family_id"] for n in truncated} == {"fam_5"}
        # deterministic rerun
        assert TS.compile_task_specs(window) == result
