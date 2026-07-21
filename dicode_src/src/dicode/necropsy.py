"""[NECROPSY v1] Read-only monster-array-diff instrument for official eval.

Records, per env, FIRST episode only (matching existing forensics semantics):
  kills_melee/kills_ranged  (F,8): mask&health 1->0 transitions by floor x type_id
  dmg_taken_floor (F,): player hp loss attributed to current floor
  dmg_dealt_floor (F,): mob hp loss (melee+ranged) per floor  -> engagement measure
  steps_on_floor  (F,): clock spent per floor
  min_melee/ranged_dist_death: Chebyshev distance to nearest live enemy at death
  food/drink_at_death: starvation-vs-combat discriminator
Pure read-only; no effect on actions/rewards. Output gated by detail flag.
"""
import jax
import jax.numpy as jnp


def necro_init(num_envs, core, detail):
    if not detail:
        return jnp.zeros(())
    F = core.melee_mobs.mask.shape[1]
    assert core.melee_mobs.mask.ndim == 3, f"unexpected mobs layout {core.melee_mobs.mask.shape}"
    z = lambda *s: jnp.zeros((num_envs,) + s, dtype=jnp.float32)
    return dict(
        kills_melee=z(F, 8), kills_ranged=z(F, 8),
        dmg_taken_floor=z(F), dmg_dealt_floor=z(F), steps_on_floor=z(F),
        min_melee_dist_death=jnp.full((num_envs,), 99.0),
        min_ranged_dist_death=jnp.full((num_envs,), 99.0),
        food_at_death=z(), drink_at_death=z(),
    )


def _kills(prev, nxt, active):
    ev = ((prev.mask > 0) & (prev.health > 0.0) & (nxt.health <= 0.0)).astype(jnp.float32)
    t8 = jax.nn.one_hot(jnp.clip(prev.type_id, 0, 7), 8)
    return (ev[..., None] * t8).sum(2) * active[:, None, None]


def _dealt(prev, nxt):
    return (jnp.clip(prev.health - nxt.health, 0.0, None) * (prev.mask > 0)).sum(-1)


def _mindist(mobs, lvl, ppos):
    idx = jnp.arange(ppos.shape[0])
    pos = mobs.position[idx, lvl]
    live = (mobs.mask[idx, lvl] > 0) & (mobs.health[idx, lvl] > 0.0)
    d = jnp.abs(pos - ppos[:, None, :]).max(-1).astype(jnp.float32)
    return jnp.where(live, d, 99.0).min(-1)


def necro_step(necro, prev, nxt, active, first_done, detail):
    if not detail:
        return necro
    lvl = nxt.player_level.astype(jnp.int32)
    F = prev.melee_mobs.mask.shape[1]
    fl = jax.nn.one_hot(lvl, F)
    hp_loss = jnp.clip(prev.player_health - nxt.player_health, 0.0, None)
    dealt = _dealt(prev.melee_mobs, nxt.melee_mobs) + _dealt(prev.ranged_mobs, nxt.ranged_mobs)
    out = dict(necro)
    out["kills_melee"] = necro["kills_melee"] + _kills(prev.melee_mobs, nxt.melee_mobs, active)
    out["kills_ranged"] = necro["kills_ranged"] + _kills(prev.ranged_mobs, nxt.ranged_mobs, active)
    out["dmg_taken_floor"] = necro["dmg_taken_floor"] + fl * hp_loss[:, None] * active[:, None]
    out["dmg_dealt_floor"] = necro["dmg_dealt_floor"] + dealt * active[:, None]
    out["steps_on_floor"] = necro["steps_on_floor"] + fl * active[:, None]
    out["min_melee_dist_death"] = jnp.where(first_done, _mindist(nxt.melee_mobs, lvl, nxt.player_position), necro["min_melee_dist_death"])
    out["min_ranged_dist_death"] = jnp.where(first_done, _mindist(nxt.ranged_mobs, lvl, nxt.player_position), necro["min_ranged_dist_death"])
    out["food_at_death"] = jnp.where(first_done, nxt.player_food.astype(jnp.float32), necro["food_at_death"])
    out["drink_at_death"] = jnp.where(first_done, nxt.player_drink.astype(jnp.float32), necro["drink_at_death"])
    return out
