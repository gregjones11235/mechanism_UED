"""Tests 3-4: deterministic output guards (forbidden content F1-F7).

Test 3 — action suggestion / reward modification detection (+ benign content
passes, recursive scan, determinism, unsupported-type fail-closed).
Test 4 — formal-evaluation provenance rejected fail-closed (including a
simulated resume-path injection). No real external APIs are used.
"""
import pytest

from dicode.teachers.static_llm import guards as G
from dicode.teachers.static_llm import schemas as S


# ---------------------------------------------------------------------------
# Benign content (must PASS the guards)
# ---------------------------------------------------------------------------
BENIGN_DESIGN_TEXT = (
    "Design a curriculum task in which the student must gather wood and craft "
    "a wooden pickaxe. Mob density stays at its default value while the world "
    "seed is randomized. The target achievement chain is get_wood followed by "
    "make_wooden_pickaxe. Scaffolding is limited to a starting inventory that "
    "contains one log."
)

BENIGN_ENV_CODE = '''
class CollectingVariant(BaseTask):
    """
    Objective: Collect wood.
    Description: The student collects wood logs in a forest.
    Relevant Achievements: get_wood
    Completed Achievements: get_wood
    World: default forest
    """

    relevant_achievements = ["get_wood"]
    completed_achievements = ["get_wood"]
    label = "collecting_variant"

    def get_task_params(self):
        return {}

    def generate_world(self, rng, params):
        return default_world(rng)
'''

BENIGN_PLAN = {
    "families": [
        {
            "family_id": "f1",
            "description": BENIGN_DESIGN_TEXT,
            "axis_changes": [{"axis": "mob_count", "from_value": "0", "to_value": "1"}],
            "constant_axes": ["world_seed"],
            "scaffolding": "starting inventory contains one log",
            "student_must_do": "craft the wooden pickaxe unaided",
        }
    ],
    "explorations": [],
}


# ---------------------------------------------------------------------------
# Test 3: forbidden-content detection
# ---------------------------------------------------------------------------
class TestForbiddenContentDetection:
    @pytest.mark.parametrize(
        "text",
        sorted(
            [
                "Step 1: approach the tree. Step 2: press the attack key.",
                "First move north then attack the zombie.",
                "Do the following actions to finish the task quickly.",
                "There is a known action sequence for this task.",
                "Next, press forward and keep walking.",
                "You should press attack as soon as you see the mob.",
            ]
        ),
    )
    def test_action_sequence_instructions_detected(self, text):
        decision = G.scan_text(text)
        assert decision.allowed is False
        assert decision.code == G.GuardCode.ACTION_SEQUENCE_DETECTED

    def test_imperative_chain_detected(self):
        text = "move north, then turn east, then press attack"
        decision = G.scan_text(text)
        assert decision.allowed is False
        assert decision.code == G.GuardCode.ACTION_SEQUENCE_DETECTED

    @pytest.mark.parametrize(
        "text",
        sorted(
            [
                "Set a waypoint near the river crossing.",
                "Go to (5, 3) and wait for nightfall.",
                "The chest spawns at 12, -7 in the valley.",
                "Follow the route from spawn to the cave.",
            ]
        ),
    )
    def test_waypoint_content_detected(self, text):
        decision = G.scan_text(text)
        assert decision.allowed is False
        assert decision.code == G.GuardCode.WAYPOINT_DETECTED

    @pytest.mark.parametrize(
        "text",
        sorted(
            [
                "reward = 1.0 if done else 0.0",
                "total_reward += bonus",
                "def compute_reward(self, state):",
                "we apply reward shaping to accelerate learning",
                "modify the reward so the task is easier",
            ]
        ),
    )
    def test_reward_modification_detected(self, text):
        decision = G.scan_text(text)
        assert decision.allowed is False
        assert decision.code == G.GuardCode.REWARD_MODIFICATION_DETECTED

    @pytest.mark.parametrize(
        "text",
        sorted(
            [
                "Provide an expert trajectory for the crafting task.",
                "an expert demonstration covering the full objective",
                "we seed the buffer via imitation learning",
            ]
        ),
    )
    def test_expert_trajectory_detected(self, text):
        decision = G.scan_text(text)
        assert decision.allowed is False
        assert decision.code == G.GuardCode.EXPERT_TRAJECTORY_DETECTED

    @pytest.mark.parametrize(
        "text",
        sorted(
            [
                "Inspect the logits of the student network.",
                "dump the hidden state of the agent",
                "copy the policy weights into the new agent",
            ]
        ),
    )
    def test_hidden_state_detected(self, text):
        decision = G.scan_text(text)
        assert decision.allowed is False
        assert decision.code == G.GuardCode.HIDDEN_STATE_DETECTED

    @pytest.mark.parametrize(
        "text",
        sorted(
            [
                "Update the student policy after each session.",
                "fine-tune the policy on the new tasks",
                "overwrite the optimizer state to reset progress",
            ]
        ),
    )
    def test_policy_modification_detected(self, text):
        decision = G.scan_text(text)
        assert decision.allowed is False
        assert decision.code == G.GuardCode.POLICY_MODIFICATION_DETECTED

    @pytest.mark.parametrize(
        "text", sorted([BENIGN_DESIGN_TEXT, BENIGN_ENV_CODE, "12 dynamic tasks + 4 anchors"])
    )
    def test_benign_content_passes(self, text):
        decision = G.scan_text(text)
        assert decision.allowed is True
        assert decision.code == G.GuardCode.GUARD_PASS

    def test_benign_plan_structure_passes_recursively(self):
        decision = G.scan_llm_output(BENIGN_PLAN)
        assert decision.allowed is True

    def test_recursive_scan_reports_deep_path(self):
        payload = {
            "families": [
                {
                    "description": "benign family",
                    "scaffolding": ["ok", {"note": "Step 2: press forward"}],
                }
            ]
        }
        decision = G.scan_llm_output(payload)
        assert decision.allowed is False
        assert decision.code == G.GuardCode.ACTION_SEQUENCE_DETECTED
        assert decision.path == "$.families[0].scaffolding[1].note"

    def test_dict_keys_are_scanned_too(self):
        decision = G.scan_llm_output({"waypoint_list": []})
        assert decision.allowed is False
        assert decision.code == G.GuardCode.WAYPOINT_DETECTED

    def test_scan_is_deterministic_double_run(self):
        payload = {"a": [BENIGN_DESIGN_TEXT, {"b": "Step 9: move"}]}
        assert G.scan_llm_output(payload) == G.scan_llm_output(payload)

    def test_scalars_are_ignored(self):
        assert G.scan_llm_output({"n": 12, "rate": 0.5, "ok": True, "none": None}).allowed

    def test_unsupported_type_fails_closed(self):
        decision = G.scan_llm_output(object())
        assert decision.allowed is False
        assert decision.code == G.GuardCode.UNSUPPORTED_TYPE

    def test_non_string_dict_key_fails_closed(self):
        decision = G.scan_llm_output({1: "x"})
        assert decision.allowed is False
        assert decision.code == G.GuardCode.UNSUPPORTED_TYPE

    def test_raise_if_forbidden_raises_with_guard_code(self):
        with pytest.raises(S.SchemaError) as excinfo:
            G.raise_if_forbidden({"note": "Step 3: turn left"}, "role-output")
        assert excinfo.value.code == G.GuardCode.ACTION_SEQUENCE_DETECTED
        # benign input returns a passing decision
        decision = G.raise_if_forbidden(BENIGN_PLAN, "role-output")
        assert decision.allowed is True


# ---------------------------------------------------------------------------
# Test 4: formal-evaluation provenance isolation (fail-closed)
# ---------------------------------------------------------------------------
class TestFormalProvenanceIsolation:
    @pytest.mark.parametrize("label", sorted(["FORMAL_FRONT", "FORMAL_BACK", "FORMAL_FULL"]))
    def test_formal_provenance_rejected_by_guard(self, label):
        decision = G.provenance_guard(label, "resume-path metrics")
        assert decision.allowed is False
        assert decision.code == S.SchemaError.FORMAL_PROVENANCE_REJECTED

    @pytest.mark.parametrize("label", sorted(["TRAINING", "NORMAL_TRAINING_FEEDBACK"]))
    def test_admissible_provenance_passes_guard(self, label):
        decision = G.provenance_guard(label, "training window")
        assert decision.allowed is True
        assert decision.code == G.GuardCode.GUARD_PASS

    def test_missing_provenance_fails_closed(self):
        decision = G.provenance_guard(None, "resume-path metrics")
        assert decision.allowed is False
        assert decision.code == S.SchemaError.PROVENANCE_MISSING

    def test_unknown_provenance_fails_closed(self):
        decision = G.provenance_guard("PROBE_FEEDBACK", "resume-path metrics")
        assert decision.allowed is False
        assert decision.code == S.SchemaError.UNKNOWN_PROVENANCE

    def test_simulated_resume_path_injection_rejected_on_both_layers(self):
        # On the resume path, run_session_evaluation output is shaped exactly
        # like training metrics; the static teacher must reject it twice:
        # provenance layer AND content scanner (FORMAL_* markers).
        formal_metrics = {
            "provenance": "FORMAL_FULL",
            "success_rate": 0.93,
            "achievement_srs": {"get_wood": 1.0},
        }
        provenance_decision = G.provenance_guard(formal_metrics["provenance"], "evolve_tasks")
        assert provenance_decision.allowed is False
        assert provenance_decision.code == S.SchemaError.FORMAL_PROVENANCE_REJECTED

        content_decision = G.scan_llm_output(formal_metrics)
        assert content_decision.allowed is False
        assert content_decision.code == G.GuardCode.FORMAL_DATA_DETECTED

    def test_training_window_metrics_pass_both_layers(self):
        training_metrics = {
            "provenance": "NORMAL_TRAINING_FEEDBACK",
            "success_rate": 0.4,
            "skill_get_wood": 0.75,
        }
        assert G.provenance_guard(training_metrics["provenance"], "observe").allowed
        assert G.scan_llm_output(training_metrics).allowed
