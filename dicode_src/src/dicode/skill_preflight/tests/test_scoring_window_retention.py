from pathlib import Path

import pytest

try:
    import jax
    import jax.numpy as jnp
except ImportError:  # Keep source/config assertions runnable without JAX installed.
    jax = None
    jnp = None


def _runtime():
    if jax is None:
        pytest.skip("JAX is not installed in this environment")
    return jax, jnp, pytest.importorskip("dicode.ppo_tr")


def _random_step(carry, _):
    value, rng = carry
    rng, sample_rng = jax.random.split(rng)
    delta = jax.random.normal(sample_rng, shape=())
    value = value + delta
    return (value, rng), jnp.asarray([value, delta], dtype=jnp.float32)


def _full_scan(init, length):
    return jax.lax.scan(_random_step, init, None, length=length)


def _pytree_step(carry, _):
    value, rng = carry
    rng, sample_rng = jax.random.split(rng)
    delta = jax.random.normal(sample_rng, shape=())
    value = value + delta
    return (value, rng), {
        "scalar": value.astype(jnp.float32),
        "matrix": jnp.stack((value, delta, value - delta)).reshape(3, 1),
        "nested": (
            jnp.asarray([value > 0, delta > 0], dtype=jnp.bool_),
            jnp.asarray([7, 11, 13], dtype=jnp.int32),
        ),
    }


def test_retained_suffix_matches_full_scan_with_rng_step():
    jax, jnp, ppo_tr = _runtime()
    init = (jnp.asarray(0.0, dtype=jnp.float32), jax.random.PRNGKey(7))
    length, keep_last = 9, 4
    full_carry, full_outputs = _full_scan(init, length)
    suffix_carry, suffix_outputs = ppo_tr._scan_with_retained_suffix(
        _random_step, init, length, keep_last
    )

    assert jnp.array_equal(suffix_carry[0], full_carry[0])
    assert jnp.array_equal(suffix_carry[1], full_carry[1])
    assert suffix_outputs.shape == full_outputs[-keep_last:].shape
    assert suffix_outputs.dtype == full_outputs.dtype
    assert jnp.array_equal(suffix_outputs, full_outputs[-keep_last:])


def test_retained_suffix_all_steps_matches_full_scan():
    jax, jnp, ppo_tr = _runtime()
    init = (jnp.asarray(0.0, dtype=jnp.float32), jax.random.PRNGKey(11))
    full_carry, full_outputs = _full_scan(init, 5)
    suffix_carry, suffix_outputs = ppo_tr._scan_with_retained_suffix(
        _random_step, init, 5, 5
    )

    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(jnp.array_equal, suffix_carry, full_carry)
    )
    assert jnp.array_equal(suffix_outputs, full_outputs)


def test_retained_suffix_jit_matches_nested_scoring_pytree():
    jax, jnp, ppo_tr = _runtime()
    init = (jnp.asarray(0.0, dtype=jnp.float32), jax.random.PRNGKey(19))
    length, keep_last = 8, 3

    full_scan = jax.jit(lambda carry: jax.lax.scan(_pytree_step, carry, None, length=length))
    retained_scan = jax.jit(
        lambda carry: ppo_tr._scan_with_retained_suffix(
            _pytree_step, carry, length, keep_last
        )
    )
    full_carry, full_outputs = full_scan(init)
    suffix_carry, suffix_outputs = retained_scan(init)

    for suffix_leaf, full_leaf in zip(
        jax.tree_util.tree_leaves(suffix_carry),
        jax.tree_util.tree_leaves(full_carry),
    ):
        assert suffix_leaf.shape == full_leaf.shape
        assert suffix_leaf.dtype == full_leaf.dtype
        assert jnp.array_equal(suffix_leaf, full_leaf)

    full_tail = jax.tree_util.tree_map(lambda leaf: leaf[-keep_last:], full_outputs)
    for suffix_leaf, full_leaf in zip(
        jax.tree_util.tree_leaves(suffix_outputs),
        jax.tree_util.tree_leaves(full_tail),
    ):
        assert suffix_leaf.shape == full_leaf.shape
        assert suffix_leaf.dtype == full_leaf.dtype
        assert jnp.array_equal(suffix_leaf, full_leaf)


@pytest.mark.parametrize("keep_last", [0, 6])
def test_retained_suffix_rejects_invalid_window(keep_last):
    jax, jnp, ppo_tr = _runtime()
    init = (jnp.asarray(0.0, dtype=jnp.float32), jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match="keep_last"):
        ppo_tr._scan_with_retained_suffix(_random_step, init, 5, keep_last)


def test_default_flag_is_false_and_training_path_is_gated():
    package_root = Path(__file__).parents[4]
    config_text = (package_root / "conf/training/default.yaml").read_text(encoding="utf-8")
    source_text = (package_root / "src/dicode/ppo_tr.py").read_text(encoding="utf-8")

    assert "retain_only_scoring_window: false" in config_text
    assert "_scan_with_retained_suffix" in source_text
    assert 'config.get("retain_only_scoring_window", False)' in source_text
    assert "scoring_window_data = jax.tree.map(lambda x: x[-k:], scan_scoring_data)" in source_text
