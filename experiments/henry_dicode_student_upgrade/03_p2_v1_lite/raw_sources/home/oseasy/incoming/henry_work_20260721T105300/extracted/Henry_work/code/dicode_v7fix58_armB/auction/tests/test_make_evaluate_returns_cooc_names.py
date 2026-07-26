"""Regression test for the cooc_names_static NameError (2026-07-05, v6fix2 job 3673162).

Bug: `cooc_names_static` was defined INSIDE the inner evaluate() body of make_evaluate(), but
`make_evaluate` ended with `return evaluate, cooc_names_static` at the OUTER scope. Python therefore
raised `NameError: name 'cooc_names_static' is not defined` the moment craftax_evaluation.main() called
make_evaluate() — which run_dicode.py wraps in a bare `except Exception` labelled "must never break
training", so every session silently printed
    - [cooc/behav] siege collection eval skipped (NameError: name 'cooc_names_static' is not defined).
and the entire (c) co-occurrence + problem-2 behaviour fingerprint collection never ran. The wiring
unit tests all passed because they hand-build a metrics dict instead of ever calling make_evaluate().

This test calls make_evaluate() for real (cheap: eval_shape only, no rollout) and asserts it returns a
non-empty python list of achievement names — i.e. the return-scope binding actually exists.
"""
import jax
import pytest

from omegaconf import OmegaConf


def _min_config():
    # Only the fields make_evaluate / CraftaxAugObsTrain touch. num_envs kept tiny for speed.
    return OmegaConf.create(
        {
            "evaluation": {"num_envs": 2, "num_steps": 4},
            "training": {
                "condition_on_task": False,
                "conditioning_type": "one_hot",
                "activation": "tanh",
                "hidden_layers": 8,
                "embed_size": 16,
                "num_heads": 1,
                "qkv_features": 8,
                "num_layers": 1,
                "gating": False,
                "gating_bias": 0.0,
            },
        }
    )


def test_make_evaluate_returns_nonempty_cooc_names():
    """make_evaluate must return (evaluate_fn, cooc_names, inv_names) without NameError.

    v6fix9 P1 widened the return to a triple: the third element is the STATIC inventory column
    labels for _chain_max_inv, enumerated programmatically from the state's inventory pytree (no
    hand-picked resource list — leak review, user 2026-07-08)."""
    craftax_evaluation = pytest.importorskip("dicode.craftax_evaluation")
    CraftaxAugObsTrain = pytest.importorskip("minicraftax.envs.craftax").CraftaxAugObsTrain
    BatchEnvWrapper = pytest.importorskip("dicode.wrappers").BatchEnvWrapper

    config = _min_config()
    env = CraftaxAugObsTrain()
    env_params = env.default_params.replace(max_timesteps=64)
    env = BatchEnvWrapper(env, num_envs=config.evaluation.num_envs)

    # This is the exact call site from craftax_evaluation.main(). Pre-fix it raised NameError here.
    evaluate_fn, cooc_names, inv_names = craftax_evaluation.make_evaluate(config, env, env_params)

    assert callable(evaluate_fn)
    assert isinstance(cooc_names, list)
    assert len(cooc_names) > 0, "cooc_names_static came back empty — achievement keys not detected"
    # Sanity: these are achievement short-names, so a known early achievement must be present.
    assert "collect_wood" in cooc_names, f"expected collect_wood among names, got {cooc_names[:8]}..."
    # v6fix9 P1: inventory labels — the whole struct, programmatically named; iron must be a column
    # (as a plain field or an expanded array column), and there must be more than a hand-picked few.
    assert isinstance(inv_names, list)
    assert len(inv_names) >= 8, f"inventory enumeration suspiciously small: {inv_names}"
    assert any("iron" in n for n in inv_names), f"no iron column among {inv_names}"
