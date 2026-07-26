"""Phase4A-v2 split training counters + precise resolved-step provenance (CC2 directive §二/§三).

Background
----------
The Phase4A probe driver carried a SINGLE overloaded ``update_count`` that was
simultaneously used as:
  * the episode "update index" stamped into collected trajectories / episode records,
  * the pending-episode ``policy_version``,
  * the replay policy-lag reference (``update_count - sample.collected_update_count``),
  * the log "global step" seed.

Once Replay is enabled that single integer is incremented by BOTH the on-policy PPO main
update AND the Replay learner within the same outer iteration, so all four meanings
diverge. This module replaces it with explicitly defined, independently incremented
counters and a precise resolved-environment-step formula.

Pure Python (NO JAX / NO numpy) so it is importable and unit-testable anywhere (GATE 2/3).
"""
from dataclasses import dataclass, asdict


def completion_resolved_env_step(outer_update_index: int, num_envs: int, rollout_steps: int,
                                 rollout_step: int, env_id: int) -> int:
    """PRECISE resolved environment step at which a completion was recorded (directive §二).

    One outer update consumes ``num_envs * rollout_steps`` resolved (parallel) env steps.
    Within an update, rollout_step ``r`` advances all ``num_envs`` envs together, and env
    ``e`` is the ``(r * num_envs + e)``-th resolved step of that update. ``+1`` makes the
    count 1-indexed so the very first resolved step of the run is step 1.

        resolved = outer_update_index * num_envs * rollout_steps
                 + rollout_step * num_envs
                 + env_id
                 + 1

    This SUPERSEDES the deprecated ``completion_global_step`` which was
    ``update_index * (num_envs * rollout_steps) + rollout_step`` — i.e. it dropped the
    ``* num_envs`` on the rollout_step term, the per-env ``env_id`` offset and the ``+1``,
    so it was NOT a precise resolved step (it under-counted and aliased all envs of a
    rollout_step onto one integer).
    """
    return (int(outer_update_index) * int(num_envs) * int(rollout_steps)
            + int(rollout_step) * int(num_envs) + int(env_id) + 1)


def completion_global_step_deprecated(outer_update_index: int, num_envs: int,
                                      rollout_steps: int, rollout_step: int) -> int:
    """DEPRECATED legacy formula (kept ONLY for historical recomparison; NOT a precise
    resolved step). Use :func:`completion_resolved_env_step`."""
    return int(outer_update_index) * (int(num_envs) * int(rollout_steps)) + int(rollout_step)


@dataclass
class Phase4ACounters:
    """Independently incremented training counters (directive §三).

    Each field has exactly ONE meaning; none is reused for another semantic.
    """
    # --- outer loop / environment progress ---
    outer_update_index: int = 0
    # Number of completed OUTER rollout+PPO iterations (the training-loop index). +1 per
    # outer iteration. This is the authoritative episode "update index".
    global_env_steps: int = 0
    # PRECISE resolved environment steps consumed so far. += num_envs*rollout_steps per
    # outer update. The authoritative "global step" (replaces completion_global_step).

    # --- on-policy PPO ---
    online_ppo_update_count: int = 0
    # Number of on-policy PPO MAIN updates executed (one per outer iteration). Independent
    # of Replay. Each PPO main update always commits its policy step.

    # --- Replay ---
    replay_update_count: int = 0
    # Number of Replay GRADIENT updates EXECUTED (an eligible batch was formed and a replay
    # learner step ran), regardless of KL acceptance.
    accepted_replay_policy_update_count: int = 0
    # Number of Replay updates whose policy-affecting parameters were COMMITTED (passed the
    # transactional KL gate). A KL-rolled-back Replay update does NOT increment this.
    replay_attempt_count: int = 0
    # Number of individual Replay sequence SAMPLE attempts (formerly conflated with
    # hindsight_attempts). Independent of relabelability.

    # --- policy version ---
    policy_version: int = 0
    # Increments ONLY after an ACCEPTED update that actually changes the online policy:
    # +1 per PPO main update (always committed) and +1 per Replay update that is
    # policy_committed. A KL-rejected Replay update does NOT advance it. This is the value
    # stamped as the pending-episode policy_version and the Replay policy-lag reference.

    # --- Hindsight / AWR firewall counters (directive §八) ---
    # MUST remain 0 in replay_mode in {off, original_vtrace}. Incremented ONLY by the
    # full_p2_legacy path; the original_vtrace path structurally never touches them.
    hindsight_attempt_count: int = 0
    hindsight_eligible_count: int = 0
    awr_update_count: int = 0
    relabeled_sample_count: int = 0

    # ----- outer loop / PPO -----
    def on_outer_update(self, num_envs: int, rollout_steps: int) -> None:
        self.outer_update_index += 1
        self.global_env_steps += int(num_envs) * int(rollout_steps)

    def on_ppo_accepted(self) -> None:
        """A PPO main update always commits its (policy-affecting) step."""
        self.online_ppo_update_count += 1
        self.policy_version += 1

    # ----- replay -----
    def on_replay_attempt(self, n: int = 1) -> None:
        self.replay_attempt_count += int(n)

    def on_replay_update_executed(self) -> None:
        self.replay_update_count += 1

    def on_replay_policy_committed(self) -> None:
        """Replay update passed the KL gate -> policy changed -> version advances."""
        self.accepted_replay_policy_update_count += 1
        self.policy_version += 1

    def on_replay_kl_rejected(self) -> None:
        """Replay update ran but was KL-rolled-back: policy-affecting side reverted.

        policy_version does NOT advance; accepted count does NOT advance. The executed
        count was already bumped by on_replay_update_executed.
        """
        return None

    # ----- full_p2_legacy-only firewall increments (never called by original_vtrace) -----
    def on_hindsight_attempt(self, n: int = 1) -> None:
        self.hindsight_attempt_count += int(n)

    def on_hindsight_eligible(self, n: int = 1) -> None:
        self.hindsight_eligible_count += int(n)

    def on_relabeled_sample(self, n: int = 1) -> None:
        self.relabeled_sample_count += int(n)

    def on_awr_update(self, n: int = 1) -> None:
        self.awr_update_count += int(n)

    # ----- firewall -----
    def assert_hindsight_awr_disabled(self) -> None:
        """Hard firewall (directive §八): replay_mode in {off, original_vtrace} MUST keep
        all four Hindsight/AWR counters == 0. Raises AssertionError on any breach."""
        assert self.hindsight_attempt_count == 0, \
            f"FIREWALL breach: hindsight_attempt_count={self.hindsight_attempt_count} != 0"
        assert self.hindsight_eligible_count == 0, \
            f"FIREWALL breach: hindsight_eligible_count={self.hindsight_eligible_count} != 0"
        assert self.awr_update_count == 0, \
            f"FIREWALL breach: awr_update_count={self.awr_update_count} != 0"
        assert self.relabeled_sample_count == 0, \
            f"FIREWALL breach: relabeled_sample_count={self.relabeled_sample_count} != 0"

    def snapshot(self) -> dict:
        return asdict(self)
