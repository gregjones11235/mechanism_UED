"""The REAL Canonical DiCode one-update runtime.

Direction code NEVER implements a second PPO/optimizer: the one update
is delegated to ``dicode.training.run_session_training`` (the canonical
chain run_session_training -> run_training_session) and its EIGHT-tuple
return is unpacked + validated fail-closed. Exactly-one-update is
enforced by the session's own update count.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Mapping, Sequence

#: the canonical 8-tuple field order of run_session_training
EIGHT_TUPLE_FIELDS = (
    "rng", "rl_train_state", "global_update_step", "global_env_steps",
    "training_metrics", "num_updates_in_session", "categorized_tasks",
    "evaluation_metrics",
)


class TrainingAssetError(RuntimeError):
    """Fail-closed canonical training violation."""


class CanonicalOneUpdateRuntime:
    """Delegates ONE real update to the canonical DiCode chain."""

    def __init__(self, *, student_adapter: Any,
                 train_state_candidate: str):
        self.student_adapter = student_adapter
        self.selected_candidate_id = train_state_candidate
        self.runtime_id = "canonical_dicode_one_update_runtime.v1"
        self.object_identity_hash = hashlib.sha256(
            b"shared_runtime.canonical_dicode_one_update_runtime.v1"
        ).hexdigest()
        self.registry_identity = self.object_identity_hash

    def execute_one_update(self, *, config: Any, rng: Any,
                           rl_train_state: Any, gen_manager: Any,
                           global_update_step: int, global_env_steps: int,
                           current_session_idx: int,
                           sampled_task_ids: Sequence[str],
                           original_return_prev_session: float = 0.0,
                           require_exactly_one_update: bool = True,
                           ) -> Dict[str, Any]:
        """Run the canonical chain and validate the 8-tuple receipt."""
        from dicode.training import run_session_training

        if "original_craftax" in sampled_task_ids:
            raise TrainingAssetError(
                "ORIGINAL_TASK_IN_SAMPLED_IDS: OriginalTask is appended "
                "by DiCode exactly once and must never enter "
                "sampled_task_ids")
        receipt_tuple = run_session_training(
            config, rng, rl_train_state, gen_manager,
            int(global_update_step), int(global_env_steps),
            int(current_session_idx), list(sampled_task_ids),
            float(original_return_prev_session),
        )
        return self.validate_receipt(
            receipt_tuple,
            require_exactly_one_update=require_exactly_one_update)

    @staticmethod
    def validate_receipt(receipt_tuple: Any, *,
                         require_exactly_one_update: bool = True
                         ) -> Dict[str, Any]:
        if not isinstance(receipt_tuple, tuple) or len(receipt_tuple) != 8:
            raise TrainingAssetError(
                "CANONICAL_RECEIPT_BAD_SHAPE: run_session_training must "
                "return the canonical 8-tuple, got "
                f"{type(receipt_tuple).__name__} of length "
                f"{len(receipt_tuple) if isinstance(receipt_tuple, tuple) else 'n/a'}")
        receipt = dict(zip(EIGHT_TUPLE_FIELDS, receipt_tuple))
        num_updates = receipt["num_updates_in_session"]
        if isinstance(num_updates, bool) or not isinstance(num_updates, int):
            raise TrainingAssetError(
                f"CANONICAL_RECEIPT_BAD_UPDATE_COUNT: {num_updates!r}")
        if require_exactly_one_update and num_updates != 1:
            raise TrainingAssetError(
                "CANONICAL_UPDATE_COUNT_MISMATCH: the smoke requires "
                f"EXACTLY ONE optimizer update, got {num_updates}")
        for counter in ("global_update_step", "global_env_steps"):
            value = receipt[counter]
            if isinstance(value, bool) or not isinstance(value, int) \
                    or value < 0:
                raise TrainingAssetError(
                    f"CANONICAL_RECEIPT_BAD_COUNTER: {counter}={value!r}")
        if not isinstance(receipt["training_metrics"], Mapping):
            raise TrainingAssetError(
                "CANONICAL_RECEIPT_BAD_METRICS: training_metrics must be "
                "a mapping")
        return receipt


def build_full_run_state(*, rl_train_state: Any, rng: Any, env_rng: Any,
                         global_update_step: int, global_env_steps: int,
                         current_session_idx: int,
                         task_archive_identity: str,
                         mechanism_state_identity: str,
                         plan_hash: str, runtime_bundle_hash: str,
                         config_hash: str, source_commit: str,
                         extra: Mapping[str, Any] = ()) -> Dict[str, Any]:
    """Assemble the COMPLETE canonical run state for checkpointing.

    ``rl_train_state`` carries params + opt_state + step; the RNGs,
    counters, task-archive / mechanism identities, plan / runtime-bundle
    / config hashes and the source commit ALL enter the checkpoint.
    """
    params = getattr(rl_train_state, "params", None)
    opt_state = getattr(rl_train_state, "opt_state", None)
    train_step = getattr(rl_train_state, "step", 0)
    if params is None or opt_state is None:
        raise TrainingAssetError(
            "RUNSTATE_SOURCE_INCOMPLETE: rl_train_state must carry "
            "params + opt_state (a params-only source is never a full "
            "run state)")
    run_state: Dict[str, Any] = {
        "params": params,
        "opt_state": opt_state,
        "train_step": int(train_step),
        "training_rng": rng,
        "env_rng": env_rng,
        "global_update_step": int(global_update_step),
        "global_env_steps": int(global_env_steps),
        "current_session_idx": int(current_session_idx),
        "task_archive_identity": task_archive_identity,
        "mechanism_state_identity": mechanism_state_identity,
        "plan_hash": plan_hash,
        "runtime_bundle_hash": runtime_bundle_hash,
        "config_hash": config_hash,
        "source_commit": source_commit,
    }
    run_state.update(dict(extra))
    return run_state
