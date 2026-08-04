"""Full-state checkpoint round-trip evidence + replay equivalence (P0-14).

Before this contract existed, STEP11 verified ONLY the parameter hash after a
save/load round trip and STEP12 ran a single replay step whose action merely
had to be "in range".  A checkpoint could round-trip its params while losing
everything else, and nothing proved the reloaded parameters behaved
identically to the updated ones.

``CheckpointRoundTripEvidence`` closes that gap.  It is MINTED (never
supplied as a mapping, never self-reported) from pipeline-measured facts and
enforces its invariants STRUCTURALLY:

* ``params_sha256_saved == params_sha256_reloaded`` and
  ``global_step_saved == global_step_reloaded`` — a round trip that changed
  anything is never attestable;
* ``replay_action_equal`` / ``replay_logits_equal`` / ``replay_value_equal``
  / ``replay_memory_equal`` all True — one identical deterministic
  next-policy step through the saved and the reloaded parameters must agree
  exactly on action, logits, value and new memory (bit-for-bit where the
  adapter exposes the field);
* ``evidence_hash`` — mint-only (init=False), recomputed from the fields,
  so tampered or self-reported evidence is structurally impossible.

The full-state verifier is never reimplemented here: the reload is performed
by the adapter's own ``restore_full_state`` surface and the evidence records
WHICH driver performed it (``restore_driver``), so a fresh-process driver
can mint the same evidence type without any semantic change.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, fields
from typing import Any, Mapping

from .errors import InvalidEvidenceError, ProductionBlockedError

ROUND_TRIP_EVIDENCE_VERSION = "checkpoint-round-trip-evidence/v1"

# The reload surface used by the in-window round trip: the mounted adapter's
# own restore_full_state, executed in the window process.  A fresh-process
# driver mints the SAME evidence type with its own driver name — the
# semantics never change.
RESTORE_DRIVER_IN_PROCESS_ADAPTER = "IN_PROCESS_ADAPTER_RESTORE"


def _require_sha256(label: str, digest: Any) -> str:
    text = str(digest)
    if len(text) != 64 or any(c not in "0123456789abcdef" for c in text):
        raise ProductionBlockedError(
            f"{label} is not a lowercase sha256 hex digest: {text[:24]!r}… "
            "(fail closed)")
    return text


@dataclass(frozen=True)
class CheckpointRoundTripEvidence:
    """One immutable round-trip evidence record (mint-only, fail closed).

    ``evidence_hash`` is NOT a constructor argument: it is computed in
    ``__post_init__`` from the measured fields only, and every invariant is
    enforced here — evidence for a lossy round trip or a non-equivalent
    replay cannot even be constructed.
    """

    checkpoint_path: str
    restore_driver: str
    params_sha256_saved: str
    params_sha256_reloaded: str
    global_step_saved: int
    global_step_reloaded: int
    replay_action_equal: bool
    replay_logits_equal: bool
    replay_value_equal: bool
    replay_memory_equal: bool
    evidence_hash: str = field(init=False)
    evidence_version: str = ROUND_TRIP_EVIDENCE_VERSION

    def __post_init__(self) -> None:
        if not str(self.checkpoint_path).strip():
            raise InvalidEvidenceError(
                "CheckpointRoundTripEvidence.checkpoint_path is empty")
        if not str(self.restore_driver).strip():
            raise InvalidEvidenceError(
                "CheckpointRoundTripEvidence.restore_driver is empty — the "
                "surface that performed the reload must always be named")
        for label, digest in (("params_sha256_saved", self.params_sha256_saved),
                              ("params_sha256_reloaded",
                               self.params_sha256_reloaded)):
            text = str(digest)
            if len(text) != 64 or any(c not in "0123456789abcdef" for c in text):
                raise InvalidEvidenceError(
                    f"CheckpointRoundTripEvidence.{label} is not a lowercase "
                    f"sha256 hex digest: {text[:24]!r}…")
        if self.params_sha256_saved != self.params_sha256_reloaded:
            raise InvalidEvidenceError(
                "CheckpointRoundTripEvidence invariant violated: the round trip "
                "changed the params hash")
        for label, value in (("global_step_saved", self.global_step_saved),
                             ("global_step_reloaded", self.global_step_reloaded)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise InvalidEvidenceError(
                    f"CheckpointRoundTripEvidence.{label} must be a non-negative "
                    f"int, got {value!r}")
        if self.global_step_saved != self.global_step_reloaded:
            raise InvalidEvidenceError(
                "CheckpointRoundTripEvidence invariant violated: the round trip "
                "changed the global step")
        for label, flag in (("replay_action_equal", self.replay_action_equal),
                            ("replay_logits_equal", self.replay_logits_equal),
                            ("replay_value_equal", self.replay_value_equal),
                            ("replay_memory_equal", self.replay_memory_equal)):
            if not bool(flag):
                raise InvalidEvidenceError(
                    f"CheckpointRoundTripEvidence invariant violated: {label} "
                    "must be True (a non-equivalent replay is never evidence)")
        payload = {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name != "evidence_hash"
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, default=str)
        object.__setattr__(
            self, "evidence_hash",
            hashlib.sha256(blob.encode("utf-8")).hexdigest())


def mint_checkpoint_round_trip_evidence(*, checkpoint_path: Any,
                                        restore_driver: Any,
                                        params_sha256_saved: Any,
                                        params_sha256_reloaded: Any,
                                        global_step_saved: Any,
                                        global_step_reloaded: Any,
                                        replay_action_equal: bool,
                                        replay_logits_equal: bool,
                                        replay_value_equal: bool,
                                        replay_memory_equal: bool
                                        ) -> CheckpointRoundTripEvidence:
    """Mint the evidence from PIPELINE-MEASURED facts only.

    A lossy round trip (params or step changed) or any failed replay
    equivalence raises ``ProductionBlockedError`` — never a forged record.
    """
    path = str(checkpoint_path)
    driver = str(restore_driver)
    if not path.strip():
        raise ProductionBlockedError(
            "round-trip evidence requires a checkpoint path (fail closed)")
    if not driver.strip():
        raise ProductionBlockedError(
            "round-trip evidence requires a named restore driver (the reload "
            "surface is never anonymous; fail closed)")
    saved = _require_sha256("params_sha256_saved", params_sha256_saved)
    reloaded = _require_sha256("params_sha256_reloaded", params_sha256_reloaded)
    if saved != reloaded:
        raise ProductionBlockedError(
            "checkpoint round trip changed params: saved "
            f"{saved[:16]}… != reloaded {reloaded[:16]}… (fail closed)")
    for label, value in (("global_step_saved", global_step_saved),
                         ("global_step_reloaded", global_step_reloaded)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ProductionBlockedError(
                f"{label} must be a non-negative int, got {value!r} (fail closed)")
    if global_step_saved != global_step_reloaded:
        raise ProductionBlockedError(
            f"checkpoint round trip changed the global step: saved "
            f"{global_step_saved} != reloaded {global_step_reloaded} (fail closed)")
    for label, flag in (("replay_action_equal", replay_action_equal),
                        ("replay_logits_equal", replay_logits_equal),
                        ("replay_value_equal", replay_value_equal),
                        ("replay_memory_equal", replay_memory_equal)):
        if not bool(flag):
            raise ProductionBlockedError(
                f"replay equivalence failed: {label} is False — the reloaded "
                "checkpoint does not behave identically to the updated "
                "parameters (fail closed)")
    return CheckpointRoundTripEvidence(
        checkpoint_path=path,
        restore_driver=driver,
        params_sha256_saved=saved,
        params_sha256_reloaded=reloaded,
        global_step_saved=int(global_step_saved),
        global_step_reloaded=int(global_step_reloaded),
        replay_action_equal=True,
        replay_logits_equal=True,
        replay_value_equal=True,
        replay_memory_equal=True,
    )


def verify_checkpoint_round_trip_evidence(evidence: Any) -> None:
    """Recompute the evidence hash + invariants; reject fakes and tamper."""
    if isinstance(evidence, Mapping):
        raise InvalidEvidenceError(
            "verify_checkpoint_round_trip_evidence requires minted "
            "CheckpointRoundTripEvidence, not a mapping")
    if not isinstance(evidence, CheckpointRoundTripEvidence):
        raise InvalidEvidenceError(
            f"verify_checkpoint_round_trip_evidence requires minted "
            f"CheckpointRoundTripEvidence, got {type(evidence).__name__}")
    if evidence.params_sha256_saved != evidence.params_sha256_reloaded:
        raise InvalidEvidenceError(
            "round-trip evidence invariant violated: params hash changed")
    if evidence.global_step_saved != evidence.global_step_reloaded:
        raise InvalidEvidenceError(
            "round-trip evidence invariant violated: global step changed")
    if not (evidence.replay_action_equal and evidence.replay_logits_equal
            and evidence.replay_value_equal and evidence.replay_memory_equal):
        raise InvalidEvidenceError(
            "round-trip evidence invariant violated: replay equivalence flags "
            "must all be True")
    payload = {
        f.name: getattr(evidence, f.name)
        for f in fields(evidence)
        if f.name != "evidence_hash"
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str)
    expected = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    if expected != evidence.evidence_hash:
        raise InvalidEvidenceError(
            "evidence_hash mismatch: the CheckpointRoundTripEvidence was "
            "tampered with or self-reported (fail closed)")


def measure_replay_equivalence(student: Any, *, params_saved: Any,
                               params_reloaded: Any, observation: Any,
                               memory: Any, previous_action: Any = None,
                               previous_reward: Any = None) -> dict[str, bool]:
    """Run ONE identical deterministic next-policy step through both
    parameter sets and measure exact equivalence.

    Deterministic only (``rng=None, deterministic=True``): a stochastic
    replay is uncomparable and never accepted here.  Logits must be exposed
    by the adapter on both sides (equivalence is otherwise unprovable —
    fail closed).  Value is compared when either side exposes it.  New memory
    trees must match structure and every leaf bit-for-bit.
    """
    import jax
    import numpy as np
    out_saved = student.policy_step(params_saved, observation, memory,
                                    previous_action, previous_reward, None, True)
    out_reload = student.policy_step(params_reloaded, observation, memory,
                                     previous_action, previous_reward, None, True)

    action_saved = int(np.asarray(out_saved["action"]).reshape(-1)[0])
    action_reload = int(np.asarray(out_reload["action"]).reshape(-1)[0])
    action_equal = action_saved == action_reload

    if "logits" not in out_saved or "logits" not in out_reload:
        raise ProductionBlockedError(
            "replay equivalence requires logits from the adapter on BOTH "
            "policy steps (equivalence is unprovable without them; fail "
            "closed rather than accept a params-only round trip)")
    logits_equal = bool(np.array_equal(np.asarray(out_saved["logits"]),
                                       np.asarray(out_reload["logits"])))

    value_saved, value_reload = "value" in out_saved, "value" in out_reload
    if value_saved != value_reload:
        value_equal = False
    elif value_saved:
        value_equal = bool(np.array_equal(np.asarray(out_saved["value"]),
                                          np.asarray(out_reload["value"])))
    else:
        value_equal = True

    mem_saved = out_saved.get("new_memory", out_saved.get("memory"))
    mem_reload = out_reload.get("new_memory", out_reload.get("memory"))
    leaves_saved, treedef_saved = jax.tree_util.tree_flatten(mem_saved)
    leaves_reload, treedef_reload = jax.tree_util.tree_flatten(mem_reload)
    memory_equal = (treedef_saved == treedef_reload
                    and len(leaves_saved) == len(leaves_reload)
                    and all(bool(np.array_equal(np.asarray(a), np.asarray(b)))
                            for a, b in zip(leaves_saved, leaves_reload)))

    return {
        "action_equal": bool(action_equal),
        "logits_equal": bool(logits_equal),
        "value_equal": bool(value_equal),
        "memory_equal": bool(memory_equal),
        "action_saved": action_saved,
        "action_reloaded": action_reload,
    }
