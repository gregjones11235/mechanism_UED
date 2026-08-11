"""Tests 1-2: static-LLM-UED V1 contract schemas and provenance enum.

Covers design-contract sections 4 (provenance admissibility) and 5 (role
output limits). Everything here is pure standard library — no real external
API calls, no jax, no craftax. Every fail-closed path is asserted through the
greppable ``SchemaError.code``.
"""
import math

import pytest

from dicode.teachers.static_llm import schemas as S


# ---------------------------------------------------------------------------
# Builders for valid minimal payloads
# ---------------------------------------------------------------------------
def _weakness(wid="w1", priority=1):
    return {
        "weakness_id": wid,
        "name": f"weakness {wid}",
        "evidence_refs": [f"session_3/{wid}"],
        "priority": priority,
    }


def _hypothesis(hid="h1", wid="w1"):
    return {
        "hypothesis_id": hid,
        "weakness_id": wid,
        "statement": f"hypothesis statement {hid}",
    }


def _diagnosis(weaknesses=None, hypotheses=None, **overrides):
    obj = {
        "weaknesses": [_weakness()] if weaknesses is None else weaknesses,
        "hypotheses": [_hypothesis()] if hypotheses is None else hypotheses,
        "reuse_previous_direction": False,
        "overall_confidence": 0.5,
    }
    obj.update(overrides)
    return obj


def _axis(axis="mob_count", frm="0", to="1"):
    return {"axis": axis, "from_value": frm, "to_value": to}


def _family(fid="f1", axis_changes=None, **overrides):
    obj = {
        "family_id": fid,
        "description": f"intervention family {fid}",
        "target_achievements": ["make_wooden_pickaxe"],
        "axis_changes": [_axis()] if axis_changes is None else axis_changes,
        "constant_axes": ["world_seed"],
        "scaffolding": "starting inventory contains wood",
        "student_must_do": "craft the target item unaided",
    }
    obj.update(overrides)
    return obj


def _exploration(pid="e1"):
    return {
        "proposal_id": pid,
        "description": f"exploration proposal {pid}",
        "axis_changes": [_axis("weather", "clear", "rain")],
    }


def _plan(families=None, explorations=None, **overrides):
    obj = {
        "families": [_family()] if families is None else families,
        "explorations": [] if explorations is None else explorations,
    }
    obj.update(overrides)
    return obj


def _template(**overrides):
    obj = {
        "template_id": "t1",
        "family_id": "f1",
        "task_description": "collect wood while mobs spawn at night",
        "code_constraints": ["subclass BaseTask"],
        "example_task_ids": ["task_1", "task_2"],
    }
    obj.update(overrides)
    return obj


def _expect_code(excinfo, code):
    assert excinfo.value.code == code


# ---------------------------------------------------------------------------
# Test 1a: Diagnosis schema limits and fail-closed codes
# ---------------------------------------------------------------------------
class TestDiagnosisSchema:
    def test_valid_diagnosis_parses(self):
        d = S.parse_diagnosis(_diagnosis())
        assert len(d.weaknesses) == 1
        assert d.weaknesses[0].weakness_id == "w1"
        assert d.hypotheses[0].weakness_id == "w1"
        assert d.reuse_previous_direction is False
        assert d.overall_confidence == pytest.approx(0.5)
        assert d.schema_version == S.DIAGNOSIS_SCHEMA_VERSION

    def test_parse_is_deterministic_double_run(self):
        payload = _diagnosis()
        assert S.parse_diagnosis(payload) == S.parse_diagnosis(payload)

    def test_limit_constants_match_manifest(self):
        assert S.MAX_WEAKNESSES == 3
        assert S.MAX_HYPOTHESES_PER_WEAKNESS == 3
        assert S.MAX_TOTAL_HYPOTHESES == 6
        assert S.DIAGNOSIS_SCHEMA["schema_version"] == S.DIAGNOSIS_SCHEMA_VERSION
        assert S.DIAGNOSIS_SCHEMA["max_weaknesses"] == S.MAX_WEAKNESSES
        assert (
            S.DIAGNOSIS_SCHEMA["max_hypotheses_per_weakness"]
            == S.MAX_HYPOTHESES_PER_WEAKNESS
        )
        assert S.DIAGNOSIS_SCHEMA["max_total_hypotheses"] == S.MAX_TOTAL_HYPOTHESES

    def test_empty_weaknesses_rejected(self):
        with pytest.raises(S.SchemaError) as excinfo:
            S.parse_diagnosis(_diagnosis(weaknesses=[], hypotheses=[]))
        _expect_code(excinfo, S.SchemaError.EMPTY_FIELD)

    def test_too_many_weaknesses_rejected(self):
        weaknesses = [_weakness(f"w{i}", i) for i in range(1, S.MAX_WEAKNESSES + 2)]
        with pytest.raises(S.SchemaError) as excinfo:
            S.parse_diagnosis(_diagnosis(weaknesses=weaknesses, hypotheses=[]))
        _expect_code(excinfo, S.SchemaError.TOO_MANY_WEAKNESSES)

    def test_too_many_hypotheses_per_weakness_rejected(self):
        hypotheses = [_hypothesis(f"h{i}", "w1") for i in range(1, 5)]  # 4 on w1, total 4 <= 6
        with pytest.raises(S.SchemaError) as excinfo:
            S.parse_diagnosis(_diagnosis(hypotheses=hypotheses))
        _expect_code(excinfo, S.SchemaError.TOO_MANY_HYPOTHESES_PER_WEAKNESS)

    def test_too_many_total_hypotheses_rejected(self):
        # 3 weaknesses (max), 3+3+1 = 7 hypotheses total (> 6, each <= 3).
        weaknesses = [_weakness(f"w{i}", i) for i in range(1, 4)]
        hypotheses = (
            [_hypothesis(f"h{i}", "w1") for i in range(1, 4)]
            + [_hypothesis(f"h{i}", "w2") for i in range(4, 7)]
            + [_hypothesis("h7", "w3")]
        )
        with pytest.raises(S.SchemaError) as excinfo:
            S.parse_diagnosis(_diagnosis(weaknesses=weaknesses, hypotheses=hypotheses))
        _expect_code(excinfo, S.SchemaError.TOO_MANY_TOTAL_HYPOTHESES)

    def test_orphan_hypothesis_rejected(self):
        with pytest.raises(S.SchemaError) as excinfo:
            S.parse_diagnosis(_diagnosis(hypotheses=[_hypothesis("h1", "nope")]))
        _expect_code(excinfo, S.SchemaError.ORPHAN_HYPOTHESIS)

    def test_duplicate_weakness_ids_rejected(self):
        with pytest.raises(S.SchemaError) as excinfo:
            S.parse_diagnosis(_diagnosis(weaknesses=[_weakness(), _weakness()], hypotheses=[]))
        _expect_code(excinfo, S.SchemaError.DUPLICATE_ID)

    def test_duplicate_hypothesis_ids_rejected(self):
        with pytest.raises(S.SchemaError) as excinfo:
            S.parse_diagnosis(_diagnosis(hypotheses=[_hypothesis(), _hypothesis()]))
        _expect_code(excinfo, S.SchemaError.DUPLICATE_ID)

    def test_unknown_field_rejected_fail_closed(self):
        with pytest.raises(S.SchemaError) as excinfo:
            S.parse_diagnosis(_diagnosis(probe_metrics={"success_rate": 0.5}))
        _expect_code(excinfo, S.SchemaError.UNKNOWN_FIELD)

    def test_missing_field_rejected(self):
        payload = _diagnosis()
        del payload["reuse_previous_direction"]
        with pytest.raises(S.SchemaError) as excinfo:
            S.parse_diagnosis(payload)
        _expect_code(excinfo, S.SchemaError.MISSING_FIELD)

    def test_empty_string_field_rejected(self):
        w = _weakness()
        w["name"] = "   "
        with pytest.raises(S.SchemaError) as excinfo:
            S.parse_diagnosis(_diagnosis(weaknesses=[w]))
        _expect_code(excinfo, S.SchemaError.EMPTY_FIELD)

    @pytest.mark.parametrize("bad_confidence", [1.5, -0.1, float("nan")])
    def test_confidence_out_of_range_rejected(self, bad_confidence):
        with pytest.raises(S.SchemaError) as excinfo:
            S.parse_diagnosis(_diagnosis(overall_confidence=bad_confidence))
        _expect_code(excinfo, S.SchemaError.CONFIDENCE_OUT_OF_RANGE)

    def test_confidence_wrong_type_rejected(self):
        with pytest.raises(S.SchemaError) as excinfo:
            S.parse_diagnosis(_diagnosis(overall_confidence="high"))
        _expect_code(excinfo, S.SchemaError.BAD_TYPE)

    def test_priority_non_integer_rejected(self):
        w = _weakness()
        w["priority"] = "high"
        with pytest.raises(S.SchemaError) as excinfo:
            S.parse_diagnosis(_diagnosis(weaknesses=[w]))
        _expect_code(excinfo, S.SchemaError.BAD_TYPE)

    @pytest.mark.parametrize("bad_priority", [0, S.MAX_WEAKNESSES + 1])
    def test_priority_out_of_range_rejected(self, bad_priority):
        w = _weakness()
        w["priority"] = bad_priority
        with pytest.raises(S.SchemaError) as excinfo:
            S.parse_diagnosis(_diagnosis(weaknesses=[w]))
        _expect_code(excinfo, S.SchemaError.BAD_PRIORITY)

    def test_non_mapping_rejected(self):
        with pytest.raises(S.SchemaError) as excinfo:
            S.parse_diagnosis(["not", "a", "mapping"])
        _expect_code(excinfo, S.SchemaError.BAD_TYPE)


# ---------------------------------------------------------------------------
# Test 1b: InterventionPlan schema limits and fail-closed codes
# ---------------------------------------------------------------------------
class TestInterventionPlanSchema:
    def test_valid_plan_parses(self):
        plan = S.parse_intervention_plan(_plan(explorations=[_exploration()]))
        assert len(plan.families) == 1
        assert plan.families[0].family_id == "f1"
        assert plan.families[0].axis_changes[0].axis == "mob_count"
        assert len(plan.explorations) == 1
        assert plan.schema_version == S.INTERVENTION_PLAN_SCHEMA_VERSION

    def test_parse_is_deterministic_double_run(self):
        payload = _plan(explorations=[_exploration()])
        assert S.parse_intervention_plan(payload) == S.parse_intervention_plan(payload)

    def test_limit_constants_match_manifest(self):
        assert S.MAX_INTERVENTION_FAMILIES == 8
        assert S.MAX_AXIS_CHANGES_PER_FAMILY == 3
        assert S.MAX_EXPLORATION_PROPOSALS == 2
        assert (
            S.INTERVENTION_PLAN_SCHEMA["schema_version"]
            == S.INTERVENTION_PLAN_SCHEMA_VERSION
        )
        assert S.INTERVENTION_PLAN_SCHEMA["max_families"] == S.MAX_INTERVENTION_FAMILIES

    def test_empty_families_rejected(self):
        with pytest.raises(S.SchemaError) as excinfo:
            S.parse_intervention_plan(_plan(families=[]))
        _expect_code(excinfo, S.SchemaError.NO_FAMILIES)

    def test_too_many_families_rejected(self):
        families = [_family(f"f{i}") for i in range(1, S.MAX_INTERVENTION_FAMILIES + 2)]
        with pytest.raises(S.SchemaError) as excinfo:
            S.parse_intervention_plan(_plan(families=families))
        _expect_code(excinfo, S.SchemaError.TOO_MANY_FAMILIES)

    def test_too_many_axis_changes_per_family_rejected(self):
        changes = [_axis(f"axis_{i}", "0", "1") for i in range(S.MAX_AXIS_CHANGES_PER_FAMILY + 1)]
        with pytest.raises(S.SchemaError) as excinfo:
            S.parse_intervention_plan(_plan(families=[_family(axis_changes=changes)]))
        _expect_code(excinfo, S.SchemaError.TOO_MANY_AXIS_CHANGES)

    def test_too_many_explorations_rejected(self):
        explorations = [_exploration(f"e{i}") for i in range(1, S.MAX_EXPLORATION_PROPOSALS + 2)]
        with pytest.raises(S.SchemaError) as excinfo:
            S.parse_intervention_plan(_plan(explorations=explorations))
        _expect_code(excinfo, S.SchemaError.TOO_MANY_EXPLORATIONS)

    def test_axis_overlap_rejected(self):
        family = _family(axis_changes=[_axis("mob_count", "0", "1")], constant_axes=["mob_count"])
        with pytest.raises(S.SchemaError) as excinfo:
            S.parse_intervention_plan(_plan(families=[family]))
        _expect_code(excinfo, S.SchemaError.AXIS_OVERLAP)

    def test_duplicate_family_ids_rejected(self):
        with pytest.raises(S.SchemaError) as excinfo:
            S.parse_intervention_plan(_plan(families=[_family(), _family()]))
        _expect_code(excinfo, S.SchemaError.DUPLICATE_ID)

    def test_unknown_field_rejected_fail_closed(self):
        with pytest.raises(S.SchemaError) as excinfo:
            S.parse_intervention_plan(_plan(candidate_probe_scores=[0.1, 0.2]))
        _expect_code(excinfo, S.SchemaError.UNKNOWN_FIELD)

    def test_missing_families_rejected(self):
        payload = _plan()
        del payload["families"]
        with pytest.raises(S.SchemaError) as excinfo:
            S.parse_intervention_plan(payload)
        _expect_code(excinfo, S.SchemaError.MISSING_FIELD)


# ---------------------------------------------------------------------------
# Test 1c: EnvTemplate schema and fail-closed codes
# ---------------------------------------------------------------------------
class TestEnvTemplateSchema:
    def test_valid_template_parses(self):
        t = S.parse_env_template(_template())
        assert t.template_id == "t1"
        assert t.family_id == "f1"
        assert t.example_task_ids == ("task_1", "task_2")
        assert t.schema_version == S.ENV_TEMPLATE_SCHEMA_VERSION

    def test_optional_fields_default_empty(self):
        t = S.parse_env_template(
            {"template_id": "t2", "family_id": "f1", "task_description": "desc"}
        )
        assert t.code_constraints == ()
        assert t.example_task_ids == ()

    def test_missing_required_field_rejected(self):
        payload = _template()
        del payload["task_description"]
        with pytest.raises(S.SchemaError) as excinfo:
            S.parse_env_template(payload)
        _expect_code(excinfo, S.SchemaError.MISSING_FIELD)

    @pytest.mark.parametrize("bad_id", sorted(["task-1", "9task", "task 1", "rm -rf"]))
    def test_bad_example_task_id_rejected(self, bad_id):
        with pytest.raises(S.SchemaError) as excinfo:
            S.parse_env_template(_template(example_task_ids=[bad_id]))
        _expect_code(excinfo, S.SchemaError.BAD_TASK_ID)

    def test_unknown_field_rejected_fail_closed(self):
        with pytest.raises(S.SchemaError) as excinfo:
            S.parse_env_template(_template(success_rate_target=0.7))
        _expect_code(excinfo, S.SchemaError.UNKNOWN_FIELD)


# ---------------------------------------------------------------------------
# Test 2: Provenance enum round-trip and admissibility contract
# ---------------------------------------------------------------------------
class TestProvenance:
    @pytest.mark.parametrize("label", sorted(p.value for p in S.Provenance))
    def test_round_trip_from_string_and_enum(self, label):
        assert S.parse_provenance(label) is S.Provenance(label)
        assert S.parse_provenance(S.Provenance(label)) is S.Provenance(label)

    def test_admissible_and_formal_sets_partition_all_members(self):
        all_members = frozenset(S.Provenance)
        assert S.ADMISSIBLE_TEACHER_PROVENANCES | S.FORMAL_PROVENANCES == all_members
        assert not (S.ADMISSIBLE_TEACHER_PROVENANCES & S.FORMAL_PROVENANCES)
        assert S.ADMISSIBLE_TEACHER_PROVENANCES == frozenset(
            {S.Provenance.TRAINING, S.Provenance.NORMAL_TRAINING_FEEDBACK}
        )

    @pytest.mark.parametrize("label", sorted(["TRAINING", "NORMAL_TRAINING_FEEDBACK"]))
    def test_admissible_provenances_accepted(self, label):
        assert S.assert_admissible_provenance(label, "unit-test") is S.Provenance(label)
        assert S.is_admissible_provenance(label) is True

    @pytest.mark.parametrize("label", sorted(["FORMAL_FRONT", "FORMAL_BACK", "FORMAL_FULL"]))
    def test_formal_provenances_rejected_fail_closed(self, label):
        with pytest.raises(S.SchemaError) as excinfo:
            S.assert_admissible_provenance(label, "resume-path metrics")
        _expect_code(excinfo, S.SchemaError.FORMAL_PROVENANCE_REJECTED)
        assert "formal evaluation" in str(excinfo.value)
        assert S.is_admissible_provenance(label) is False

    def test_missing_provenance_fails_closed(self):
        with pytest.raises(S.SchemaError) as excinfo:
            S.parse_provenance(None)
        _expect_code(excinfo, S.SchemaError.PROVENANCE_MISSING)
        assert S.is_admissible_provenance(None) is False

    @pytest.mark.parametrize("label", sorted(["PROBE_FEEDBACK", "FRONTIER", ""]))
    def test_unknown_provenance_fails_closed(self, label):
        with pytest.raises(S.SchemaError) as excinfo:
            S.parse_provenance(label)
        _expect_code(excinfo, S.SchemaError.UNKNOWN_PROVENANCE)
        assert S.is_admissible_provenance(label) is False
