"""Production integration: wires treatment parameters into real PPO training.

Provides make_train_with_treatments() which:
1. Creates T1/T2/T3 treatment parameters from config flags
2. Calls production make_train() with treatment_params injected
3. Returns augmented train function with treatment losses added to PPO loss

When all flags disabled: numerical identity with unmodified make_train().
"""
import jax
import jax.numpy as jnp
from typing import Dict, Optional


def create_treatment_params(config, obs_feature_dim=None) -> Dict:
    """Create treatment parameter pytrees from config flags.

    Returns {} when all flags are disabled (numerical identity).
    """
    params = {}
    if getattr(config, "enable_lpg_hrl", False):
        from dicode.training.lpg_hrl import LPGHRLWrapper
        wrapper = LPGHRLWrapper(config)
        if wrapper.enabled:
            rng = jax.random.PRNGKey(getattr(config, "treatment_seed", 0))
            params["lpg_hrl"] = wrapper.init_params(rng, obs_feature_dim)
    if getattr(config, "enable_tser", False):
        from dicode.training.tser_ppo import TSERWrapper
        wrapper = TSERWrapper(config)
        if wrapper.enabled:
            rng = jax.random.PRNGKey(getattr(config, "treatment_seed", 1))
            params["tser"] = wrapper.init_params(rng, obs_feature_dim)
    return params


def make_train_with_treatments(
    config, task_classes, num_training_updates,
    task_embeddings=None, task_distribution_proportions=None,
    initial_global_update_step=0,
    _prebuilt_params=None,
):
    """Production entry point with treatment param injection.

    Args:
        _prebuilt_params: if provided, use these instead of auto-creating.
            Enables before/after gradient comparison in tests.

    When no treatment flags are enabled: identical to make_train().
    """
    from dicode.ppo_tr import make_train
    treatment_params = _prebuilt_params if _prebuilt_params is not None else create_treatment_params(config, obs_feature_dim=None)
    return make_train(
        config, task_classes, num_training_updates,
        task_embeddings=task_embeddings,
        task_distribution_proportions=task_distribution_proportions,
        initial_global_update_step=initial_global_update_step,
        treatment_params=treatment_params,
    )


def compute_lpac_controls(config, held_out_progress=0.0, held_out_forgetting=0.0,
                           current_entropy=0.01):
    """Compute LPAC-adapted entropy coef and curriculum temperature."""
    if not getattr(config, "enable_lpac", False):
        return (config.ent_coef, 1.0)
    from dicode.training.lpac import LPACWrapper
    wrapper = LPACWrapper(config)
    if wrapper.enabled:
        return wrapper.update(held_out_progress, held_out_forgetting, current_entropy)
    return (config.ent_coef, 1.0)
