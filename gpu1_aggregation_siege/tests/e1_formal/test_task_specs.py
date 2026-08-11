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

    def test_variants_share_template_but_differ_in_content(self):
        # round-3 P0-2: variants of one family share the TEMPLATE hash
        # (one EnvCoder call) but their spec hashes/artifacts differ —
        # variant params are derived deterministically (no LLM)
        window = _window_with_families([_family("fam_a")])
        result = TS.compile_task_specs(window)
        v0, v1 = result.specs
        assert v0.template_hash == v1.template_hash
        assert v0.template_artifact_id == v1.template_artifact_id
        assert v0.template_artifact_id == f"{v0.template_hash}::tpl"
        assert v0.spec_hash != v1.spec_hash
        assert v0.artifact_id != v1.artifact_id
        assert v0.spec_id != v1.spec_id
        assert v0.variant_params != v1.variant_params
        # exact rational levels between from_value (0) and to_value (1)
        assert dict(v0.variant_params)["enemy_density:level"] == "0"
        assert dict(v1.variant_params)["enemy_density:level"] == "1"
        assert len(result.templates) == 1

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
        # round-3 P0-2: dedup is per TEMPLATE (family content), first
        # family wins; the duplicate yields one note, not per-variant
        base = _family("fam_a")
        dup = dict(base)
        dup["family_id"] = "fam_dup"  # same CONTENT, different id
        window = _window_with_families([base, dup])
        result = TS.compile_task_specs(window)
        assert {s.family_id for s in result.specs} == {"fam_a"}
        assert len(result.specs) == 2  # only the first family's variants
        assert len(result.templates) == 1
        assert len(result.notes) == 1
        assert result.notes[0]["note"] == TS.DEDUPED_TEMPLATE
        assert result.notes[0]["family_id"] == "fam_dup"

    def test_six_families_yield_full_12_pool_without_truncation(self):
        # round-3 P0-2: 6 families x 2 variants = exactly the 12
        # dynamic slots; nothing is truncated and no notes are emitted
        families = [_family(f"fam_{i}") for i in range(6)]
        window = _window_with_families(families)
        result = TS.compile_task_specs(window)
        assert len(result.templates) == 6
        assert len(result.specs) == 12
        assert {s.family_id for s in result.specs} == {
            f"fam_{i}" for i in range(6)
        }
        assert result.notes == ()
        assert TS.MAX_WINDOW_TEMPLATES == 10
        assert TS.MAX_WINDOW_SPEC_POOL == 20

    def test_template_cap_truncates_deterministically_with_notes(self):
        # 11 unique families > MAX_WINDOW_TEMPLATES=10: the 11th is
        # truncated with a recorded note (never silent, never padded).
        # NOTE: the committed board contract caps a real window at
        # MAX_INTERVENTION_FAMILIES=8 families (< the template cap 10),
        # so this truncation is a COMPILER BACKSTOP — exercised here on
        # a window synthesized from a REAL complete board window (the
        # compiler never re-derives families from anywhere else).
        import dataclasses

        real = _window_with_families([_family("fam_0")])
        families = [_family(f"fam_{i}") for i in range(11)]
        window = dataclasses.replace(
            real,
            role_results=tuple(
                ("intervention_tutor", {"families": families, "explorations": []})
                if role == "intervention_tutor"
                else (role, obj)
                for role, obj in real.role_results
            ),
            surviving_families=tuple(f["family_id"] for f in families),
        )
        result = TS.compile_task_specs(window)
        assert len(result.templates) == TS.MAX_WINDOW_TEMPLATES
        kept = {t.family_id for t in result.templates}
        assert kept == {f"fam_{i}" for i in range(10)}
        truncated = [
            n for n in result.notes if n["note"] == TS.TEMPLATES_TRUNCATED_TO_CAP
        ]
        assert len(truncated) == 1
        assert truncated[0]["family_id"] == "fam_10"
        # 10 templates x 2 variants = 20 specs = the pool cap
        assert len(result.specs) == TS.MAX_WINDOW_SPEC_POOL
        # deterministic rerun
        assert TS.compile_task_specs(window) == result
