"""D052 canonical mechanism framework (canonical_v2).

A deterministic, auditable curriculum-mechanism framework for Craftax. This
package is the in-place refactor of the legacy "D052" 5x5 experiment grid into
a canonical, reusable, teammate-extensible framework.

Hard invariants enforced across the package (see d052/legacy/protocol_version.py
and the per-module GATE tests):

  * protocol_version is REQUIRED; missing -> hard error (never silently defaulted
    when parsing an existing config).
  * Legal targets are the 67 official Craftax achievements (craftax_67_v1); the
    canonical_id == goal_vector_index == enum value (0..66).
  * Goal conditioning is a 67-dim achievement multi-hot; student obs_dim == 8335.
  * Candidate pools are shared + frozen; unknown/empty/default goals are ERRORS,
    never silently coerced or backfilled.
  * Selectors are deterministic (bit-identical replay) behind a single interface.
  * No training runs without an explicit per-cell authorization. This refactor
    phase performs ZERO long training runs.

Layout (filled in across Commits 1-9):
  d052/legacy/        protocol_version gate + canonical constants + compat layer
  d052/schemas/       pydantic (extra=forbid) data schemas
  d052/achievements/  official 67 registry + explicit aliases
  d052/generation/    shared frozen candidate pool + validator
  d052/profiling/     Student profile (Modeler machine-facts)
  d052/roles/         Tutor/Critic/Explorer/Modeler role protocol
  d052/normalization/ rank_percentile_v1 per-role score normalization
  d052/selectors/     unified selector interface (Soft/Budgeted Copeland, Auction)
  d052/execution/     candidate -> real training-goal mapping certificate
  d052/cells/         cell registry + register/validate/prepare/authorize/launch
  d052/training/      training adapter (authorization-gated; no-op this phase)
  d052/evaluation/    evaluator adapter
  d052/audit/         provenance / manifest / evidence-chain helpers
  d052/cli/           argparse command line (prepare/validate/status NEVER launch)
  d052/tests/         GATE 1..12 test gates
"""

__all__ = ["__protocol_version__", "CANONICAL_PROTOCOL_VERSION"]

from d052.legacy.canonical_constants import CANONICAL_PROTOCOL_VERSION

#: Framework protocol version emitted by this package.
__protocol_version__ = CANONICAL_PROTOCOL_VERSION
