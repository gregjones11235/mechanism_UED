"""Authorization-gated training adapter (canonical_v2).

NO-OP this phase (D052_LONG_TRAINING_RUNS=0). Provides the authorized runner the
cell registry hands to ``launch`` and the contract a future real runner must
satisfy. A training-scope authorization REFUSES (not implemented) rather than
silently running; a no-training authorization can only ever yield 0 timesteps.
"""
from d052.training.adapter import (
    TrainingAdapterError,
    assert_no_training_phase,
    canonical_training_runner,
)

__all__ = [
    "TrainingAdapterError",
    "assert_no_training_phase",
    "canonical_training_runner",
]
