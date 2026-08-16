"""Memory Study: HO reinjection + Floor2->Floor3 probe (tier3-integrated).

Package scope (TASK B1):
  ho_contract      - HOMode / HistoryCapture / IsolationReceipt + hashing
  ho_burnin        - burn-in execution + step_fn adapters (tier3 policy,
                     StudentTrainingBackend.policy_forward_eval)
  ho_capture_bank  - result-blind capture bank generation/persistence/loading
  floor23_probe    - FRONT_L2 probe loop over (state, candidate, ho_mode)

Local runs are SYNTHETIC and JAX-free; REAL runs follow the server RUNBOOK in
docs/memory_study/HO_FLOOR23_DESIGN.md.
"""
from .ho_contract import (
    SCHEMA_ID_CONTRACT,
    FailClosed,
    HOMode,
    HistoryCapture,
    IsolationContext,
    IsolationReceipt,
    canonical_json_bytes,
    canonical_obs_dim,
    hash_pytree,
    sha256_hex,
    structural_form,
)
from .ho_burnin import (
    RNG_STREAM_ID_BURNIN,
    burnin_history,
    wrap_backend_policy_forward_eval,
    wrap_tier3_projection_policy,
)
from .ho_capture_bank import (
    GENERATOR_SYNTHETIC,
    SCHEMA_ID_BANK,
    assign_capture,
    generate_synthetic_capture_bank,
    load_capture_bank,
    write_capture_bank,
)
from .floor23_probe import (
    RESULT_SCHEMA_ID,
    CandidateRuntime,
    load_tier3_library,
    make_synthetic_candidate,
    run_floor23_probe,
    synthetic_states,
)

__all__ = [
    "SCHEMA_ID_CONTRACT", "FailClosed", "HOMode", "HistoryCapture",
    "IsolationContext", "IsolationReceipt", "canonical_json_bytes",
    "canonical_obs_dim", "hash_pytree", "sha256_hex", "structural_form",
    "RNG_STREAM_ID_BURNIN", "burnin_history",
    "wrap_backend_policy_forward_eval", "wrap_tier3_projection_policy",
    "GENERATOR_SYNTHETIC", "SCHEMA_ID_BANK", "assign_capture",
    "generate_synthetic_capture_bank", "load_capture_bank",
    "write_capture_bank",
    "RESULT_SCHEMA_ID", "CandidateRuntime", "load_tier3_library",
    "make_synthetic_candidate", "run_floor23_probe", "synthetic_states",
]