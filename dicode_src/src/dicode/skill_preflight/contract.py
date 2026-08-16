"""Shared preflight-optimization contract exception (R2).

When an enabled preflight optimization (B2 ``preflight_reuse_loaded_tasks`` or
B3 ``compact_preflight_payload``) has a contract violation, the preflight gate
in ``run_dicode.py`` must fail closed instead of degrading to "keep all". This
dedicated exception type lets the outer catch re-raise only the explicit
optimization-contract failures while preserving the historical degradation for
all other (ordinary) preflight errors.
"""


class PreflightOptimizationContractError(RuntimeError):
    """Raised when an enabled preflight optimization contract is violated.

    Propagates through the preflight gate's outer ``except Exception`` (which
    re-raises this type) and terminates the run. Never silently falls back.
    """


def handle_preflight_gate_error(exc):
    """Fail-closed classification for the preflight gate's outer catch (R2).

    ``run_dicode.py``'s preflight gate wraps its body in a broad ``except
    Exception`` that historically degraded any failure to "kept all, gate
    inactive". R2 makes an *enabled* preflight optimization contract violation
    (``PreflightOptimizationContractError``) propagate so the run terminates,
    while ordinary preflight errors keep the historical degradation. This
    function is the single classification point and is unit-testable without
    jax.
    """
    if isinstance(exc, PreflightOptimizationContractError):
        raise exc
    print(f"  [Preflight] ERROR (kept all, gate inactive!): {exc}")
