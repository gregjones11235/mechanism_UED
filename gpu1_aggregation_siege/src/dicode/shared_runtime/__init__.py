"""The REAL shared runtime for the mechanism_UED production lines.

This package provides the concrete production objects the E1/E2 seams
consume: a concrete FormalAssetRegistry, the real Persistent Student
assets (identity/adapter/contract), the real Reference arm, the frozen
anchor manifest, the real probe runner / signal issuer / EnvCoder
backend, the authorized six-role LLM runtime, the canonical one-update
runtime, the full RunState checkpoint manager and the compute ledger.

Activation gate: the shared runtime is ONLY importable when the
deployment sets ``DICODE_SHARED_RUNTIME_REAL=1``. Without the gate the
import raises ImportError and the E1 seam reports its honest
BLOCKED_WAITING_SHARED_RUNTIME_* state (the audit / TEST_ONLY surfaces
stay exactly as before). This is a deployment configuration gate, never
a silent fallback: when enabled, every object is a real artifact.
"""
from __future__ import annotations

import os

_ACTIVATION_ENV = "DICODE_SHARED_RUNTIME_REAL"

if os.environ.get(_ACTIVATION_ENV, "") != "1":
    raise ImportError(
        "dicode.shared_runtime is gated: set "
        f"{_ACTIVATION_ENV}=1 in the production deployment environment "
        "(the audit / TEST_ONLY surfaces resolve honestly without it)")

# When the gate is open, materialize the REAL production objects exactly
# once at import time: the registry resolves every asset from the real
# deployment artifacts (checkpoint sha-verified), and the nine seam
# attributes below are the bound real objects.
from .registry import production_registry as _production_registry

_registry = _production_registry()

formal_asset_registry = _registry
student_identity = _registry.resolve_asset(contract="student_identity")
student_adapter = _registry.resolve_asset(contract="student_adapter")
reference_identity = _registry.resolve_asset(contract="reference_identity")
reference_adapter = _registry.resolve_asset(contract="reference_adapter")
anchor_manifest = _registry.resolve_asset(contract="anchor_manifest")
candidate_probe_result = _registry.resolve_asset(
    contract="candidate_probe_runner")
full_state_checkpoint = _registry.resolve_asset(
    contract="canonical_dicode_runstate_checkpoint")
training_runtime = _registry.resolve_asset(
    contract="canonical_dicode_one_update_runtime")

__all__ = [
    "formal_asset_registry",
    "student_identity",
    "student_adapter",
    "reference_identity",
    "reference_adapter",
    "anchor_manifest",
    "candidate_probe_result",
    "full_state_checkpoint",
    "training_runtime",
]
