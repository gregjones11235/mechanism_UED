"""GATE 4 + 7 + 8 (Phase 2.5) — offline matched B/C counterfactual integration.

End-to-end: a NEW legal shared frozen pool; B = S1_THREE_ROLE (modeler OFF),
C = S2_FOUR_ROLE_MODELER (modeler ON); canonical B/C selected-8; every selected
candidate passes the official-67 execution-mapping certificate (gate 4); the run is
fully deterministic (identical manifest/selection hashes across recomputation --
gate 2 at integration level); two DRAFT cells register with 0 intended timesteps;
the modeler context carries NO tier labels; and zero training occurs (gate 8).
"""
import os

import pytest

from d052.counterfactual.pipeline import (
    compute_phase25,
    emit_phase25_artifacts,
    register_phase25_cells,
)
from d052.counterfactual.student_modeler_channel import assert_modeler_firewall


@pytest.fixture(scope="module")
def result():
    return compute_phase25()


# --- gate 1: matched protocol holds end-to-end -----------------------------
def test_matched_protocol_holds(result):
    assert result.verification.passed is True
    assert result.arm_b.modeler_enabled is False
    assert result.arm_c.modeler_enabled is True
    assert result.arm_b.student_profile_hash is None
    assert result.arm_c.student_profile_hash == result.student_profile_hash


# --- gate 4: selected candidates pass the official-67 mapping --------------
def test_selected8_sizes(result):
    assert len(result.b_selected8) == 8
    assert len(result.c_selected8) == 8
    assert len(set(result.b_selected8)) == 8
    assert len(set(result.c_selected8)) == 8


def test_all_certificates_pass_official_67_mapping(result):
    assert result.manifest.all_certificates_executed_as_intended is True
    for cert in result.certificates_b + result.certificates_c:
        assert cert.executed_as_intended is True
        assert cert.goal_vector_dim == 67
        assert cert.student_obs_dim == 8335
        assert cert.conditioning_type == "achievement_multi_hot"
        assert cert.gates["target_is_canonical"] is True
        assert cert.gates["goal_vector_dim_67"] is True
        assert cert.gates["goal_vector_index_aligned"] is True
        assert cert.gates["student_obs_dim_8335"] is True
        assert cert.gates["no_silent_fallback"] is True
        assert cert.gates["task_compiled"] is True


# --- the modeler contrast is live (selection actually changes) -------------
def test_modeler_selection_change_is_live_and_accounted(result):
    assert result.modeler_selection_change >= 1     # the modeler moved a selection
    assert result.manifest.selection_change_over == 8
    assert result.manifest.changed_in            # ids the modeler swapped IN
    assert result.manifest.changed_out           # ids the modeler swapped OUT
    # the swap is exactly the modeler-flagged weakest-skill candidate
    assert result.manifest.changed_in == ["cand_08"]


def test_modeler_bonus_only_on_flagged_candidate(result):
    nz = {k: v for k, v in result.modeler_bonus_by_id.items() if v > 0}
    assert set(nz) == {"cand_08"}                 # only the modeler focus gets a bonus


# --- gate 2 at integration level: fully deterministic recomputation --------
def test_recomputation_is_bit_identical(result):
    again = compute_phase25()
    assert again.manifest.manifest_hash == result.manifest.manifest_hash
    assert again.selection_b.selection_hash == result.selection_b.selection_hash
    assert again.selection_c.selection_hash == result.selection_c.selection_hash
    assert again.b_selected8 == result.b_selected8
    assert again.c_selected8 == result.c_selected8


# --- modeler firewall: no tier labels reach the modeler context ------------
def test_modeler_context_has_no_tier_labels(result):
    assert_modeler_firewall(result.modeler_context)        # raises if any tier leaks
    assert result.modeler_context.firewall_attestation["tier_labels_stripped"] is True
    assert "per_depth_tier_mastery" in \
        result.modeler_context.firewall_attestation["excluded_profile_fields"]


# --- gate 8: zero training --------------------------------------------------
def test_zero_training(result):
    assert result.manifest.training_timesteps == 0
    assert result.manifest.no_training_attestation["timesteps_run"] == 0
    assert result.manifest.no_training_attestation["D052_LONG_TRAINING_RUNS"] == 0


# --- gate 3 attestation recorded on the manifest ---------------------------
def test_firewall_attestation_recorded(result):
    assert result.manifest.canonical_target_firewall == "PASS"


# --- DRAFT cells register legally with 0 timesteps -------------------------
def test_register_two_draft_cells(result, tmp_path):
    cells = register_phase25_cells(result, str(tmp_path / "cells"))
    assert len(cells) == 2
    for c in cells:
        assert c["state"] == "DRAFT"
        assert c["intended_total_timesteps"] == 0
        assert len(c["cell_identity_hash"]) == 64
    # selection hashes on the cells match the live selections
    by_id = {c["cell_id"]: c for c in cells}
    b_cell = next(c for c in cells if c["cell_id"].startswith("phase25_B"))
    c_cell = next(c for c in cells if c["cell_id"].startswith("phase25_C"))
    assert b_cell["selection_hash"] == result.selection_b.selection_hash
    assert c_cell["selection_hash"] == result.selection_c.selection_hash
    assert by_id  # silence unused


def test_cell_registration_is_no_overwrite(result, tmp_path):
    root = str(tmp_path / "cells2")
    register_phase25_cells(result, root)
    with pytest.raises(Exception):          # EXISTS_NO_OVERWRITE on re-register
        register_phase25_cells(result, root)


# --- artifact emission is complete + no-overwrite --------------------------
def test_emit_artifacts_complete_and_no_overwrite(result, tmp_path):
    out = str(tmp_path / "phase25")
    written = emit_phase25_artifacts(result, out)
    names = {os.path.basename(p) for p in written}
    for expected in ("pool.json", "judgment_cache.json", "modeler_context.json",
                     "arm_b.json", "arm_c.json", "selection_b.json",
                     "selection_c.json", "certificates_b.json",
                     "certificates_c.json",
                     "matched_counterfactual_manifest.json", "summary.json"):
        assert expected in names
        assert os.path.exists(os.path.join(out, expected))
    # no-overwrite: a second emit into the same dir must refuse
    with pytest.raises(FileExistsError):
        emit_phase25_artifacts(result, out)
