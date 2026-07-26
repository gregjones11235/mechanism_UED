"""CPU core tests for P7 egomap.py (gates G2/G3 partial + odometry + feature-off).
Run: JAX_PLATFORM_NAME=cpu python test_egomap_core.py
"""
import os
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
import sys
import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import egomap as E


def make_obs(terrain, item=None, light=None, mobs=None, player_level=0):
    """terrain/item/light/mobs: (9,11) int arrays (terrain=BlockType id,
    item=ItemType id 0-4, light=0/1, mobs=0/1). Returns (8335,) float vector."""
    H, W = 9, 11
    space = np.zeros((H, W, 83), dtype=np.float32)
    if light is None:
        light = np.ones((H, W), dtype=np.float32)
    if item is None:
        item = np.zeros((H, W), dtype=np.int32)
    if mobs is None:
        mobs = np.zeros((H, W), dtype=np.float32)
    for i in range(H):
        for j in range(W):
            if light[i, j] > 0.5:
                space[i, j, int(terrain[i, j])] = 1.0       # BlockType one-hot
                space[i, j, 37 + int(item[i, j])] = 1.0      # ItemType one-hot
                if mobs[i, j] > 0.5:
                    space[i, j, 42] = 1.0                    # a mob channel
                space[i, j, 82] = 1.0                        # light
    flat = space.reshape(-1).copy()
    full = np.zeros(8335, dtype=np.float32)
    full[:8217] = flat
    full[8256] = player_level / 10.0                          # player_level
    return full


PASS = []
def check(name, cond):
    PASS.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name)


cfg = E.EgoMapConfig(map_size=16, num_floors=9, enabled=True)

# ---- Test 1: observed cells accumulate, unobserved stay zero (no leak) ---- #
terrain = np.full((9, 11), 2, dtype=np.int32)   # all GRASS (passable)
terrain[4, 5] = 17                               # WALL at center (obstacle)
light = np.zeros((9, 11), dtype=np.float32)
light[3:6, 4:8] = 1.0                            # only a window observed
obs = make_obs(terrain, light=light)
state = E.egomap_init_state(1, cfg)
state = E.egomap_update(state, jnp.asarray(obs)[None], jnp.zeros(1, jnp.int32),
                        jnp.zeros(1, jnp.bool_), cfg)
mb = np.asarray(state["map_bank"][0, 0])         # floor0, (16,16,9)
ever = mb[..., E.CH_EVER]
passable = mb[..., E.CH_PASSABLE]
obstacle = mb[..., E.CH_OBSTACLE]
# observed window mapped around center (8,8); center cell (8,8) is WALL -> obstacle
check("T1 ever_observed nonzero in mapped window", ever.sum() > 0)
check("T1 obstacle set at player-center wall", obstacle[8, 8] == 1.0)
check("T1 center wall NOT passable", passable[8, 8] == 0.0)
# a far corner never observed -> all channels zero
check("T1 unobserved corner all-zero (no leak)",
      np.all(mb[0, 0, :] == 0.0) and ever[0, 0] == 0.0 and passable[0, 0] == 0.0
      and obstacle[0, 0] == 0.0)

# ---- Test 2: torch + stairs recorded only when observed ---- #
item = np.zeros((9, 11), dtype=np.int32)
item[4, 6] = E.ITEM_TORCH            # torch right of center
item[3, 5] = E.ITEM_LADDER_DOWN      # stair up from center
obs2 = make_obs(terrain, item=item, light=light)
s2 = E.egomap_init_state(1, cfg)
s2 = E.egomap_update(s2, jnp.asarray(obs2)[None], jnp.zeros(1, jnp.int32),
                     jnp.zeros(1, jnp.bool_), cfg)
mb2 = np.asarray(s2["map_bank"][0, 0])
# torch at patch (4,6) -> map (8 + (4-4), 8 + (6-5)) = (8,9)
check("T2 torch recorded at correct cell", mb2[8, 9, E.CH_TORCH] == 1.0)
# stair_down at patch (3,5) -> map (8+(3-4), 8+(5-5)) = (7,8)
check("T2 stair_down recorded", mb2[7, 8, E.CH_STAIR_DOWN] == 1.0)
check("T2 no spurious torch elsewhere", mb2[..., E.CH_TORCH].sum() == 1.0)

# ---- Test 3: true_done resets to a FRESH-episode state (auto-reset semantics) ----
# Invariant: update(ANY_state, obs, true_done=True) == update(FRESH, obs, False).
# (old episode cleared, terminal/reset obs stamped as the new episode's first.)
plain = make_obs(np.full((9, 11), 2, dtype=np.int32))   # plain grass, fully lit
s3 = E.egomap_update(s2, jnp.asarray(plain)[None], jnp.zeros(1, jnp.int32),
                     jnp.ones(1, jnp.bool_), cfg)        # true_done=True from dirty s2
fresh = E.egomap_init_state(1, cfg)
fresh = E.egomap_update(fresh, jnp.asarray(plain)[None], jnp.zeros(1, jnp.int32),
                        jnp.zeros(1, jnp.bool_), cfg)
check("T3 true_done resets map to fresh-episode state (== fresh update)",
      np.allclose(np.asarray(s3["map_bank"]), np.asarray(fresh["map_bank"]))
      and np.allclose(np.asarray(s3["ego_pos"]), np.asarray(fresh["ego_pos"]))
      and np.allclose(np.asarray(s3["step"]), np.asarray(fresh["step"])))
# the old episode's torch/stair markers (from s2/obs2) must be GONE after reset
check("T3 old-episode torch cleared by true_done",
      np.asarray(s3["map_bank"][0, 0, ..., E.CH_TORCH]).sum() == 0.0)
check("T3 true_done resets ego_pos to center",
      np.all(np.asarray(s3["ego_pos"][0]) == np.array([cfg.center, cfg.center])))

# ---- Test 4: vector-env isolation ---- #
obsA = make_obs(terrain, light=light)
obsB = make_obs(np.full((9, 11), 13, dtype=np.int32),   # all SAND
                light=np.zeros((9, 11), dtype=np.float32))  # B fully dark
st = E.egomap_init_state(2, cfg)
st = E.egomap_update(st, jnp.stack([jnp.asarray(obsA), jnp.asarray(obsB)]),
                     jnp.zeros(2, jnp.int32), jnp.zeros(2, jnp.bool_), cfg)
envA = np.asarray(st["map_bank"][0]); envB = np.asarray(st["map_bank"][1])
check("T4 env A populated", envA[..., E.CH_EVER].sum() > 0)
# env B is fully dark -> NO observation channel may be set (no leak). visit_count
# at the player's own cell is allowed self-location knowledge (odometry), so we
# check the observation channels specifically, not the whole tensor.
obs_ch = [E.CH_EVER, E.CH_PASSABLE, E.CH_OBSTACLE, E.CH_TORCH,
          E.CH_STAIR_DOWN, E.CH_STAIR_UP, E.CH_FRONTIER]
envB_obs_sum = sum(float(envB[..., c].sum()) for c in obs_ch)
check("T4 env B (dark) observation channels all zero (no leak)", envB_obs_sum == 0.0)
check("T4 env B isolated from env A (no torch/stair bled across envs)",
      float(envB[..., E.CH_TORCH].sum()) == 0.0
      and float(envB[..., E.CH_OBSTACLE].sum()) == 0.0)

# ---- Test 5: odometry — move into open cell advances; into wall blocked ---- #
# open terrain all grass, fully lit
open_obs = make_obs(np.full((9, 11), 2, dtype=np.int32))   # action RIGHT=2 -> (0,1)
s5 = E.egomap_init_state(1, cfg)
pos0 = np.asarray(s5["ego_pos"][0]).copy()
s5 = E.egomap_update(s5, jnp.asarray(open_obs)[None], jnp.array([2], jnp.int32),
                     jnp.zeros(1, jnp.bool_), cfg)
pos1 = np.asarray(s5["ego_pos"][0])
check("T5 move RIGHT advances ego_pos by (0,1)",
      np.all(pos1 - pos0 == np.array([0, 1])))
# wall directly to the right of center -> blocked
wall_obs = make_obs(np.full((9, 11), 2, dtype=np.int32))
# put wall at patch cell right of center (4,6)
terr_w = np.full((9, 11), 2, dtype=np.int32); terr_w[4, 6] = 17
wall_obs = make_obs(terr_w)
s5b = E.egomap_init_state(1, cfg)
p0 = np.asarray(s5b["ego_pos"][0]).copy()
s5b = E.egomap_update(s5b, jnp.asarray(wall_obs)[None], jnp.array([2], jnp.int32),
                      jnp.zeros(1, jnp.bool_), cfg)
p1 = np.asarray(s5b["ego_pos"][0])
check("T5 move RIGHT into wall blocked (no advance)", np.all(p1 == p0))
# non-move action does not move
s5c = E.egomap_init_state(1, cfg)
q0 = np.asarray(s5c["ego_pos"][0]).copy()
s5c = E.egomap_update(s5c, jnp.asarray(open_obs)[None], jnp.array([5], jnp.int32),  # DO
                      jnp.zeros(1, jnp.bool_), cfg)
check("T5 non-move action (DO) no advance", np.all(np.asarray(s5c["ego_pos"][0]) == q0))

# ---- Test 6: feature-off returns zeros and leaves state unchanged ---- #
cfg_off = E.EgoMapConfig(map_size=16, num_floors=9, enabled=False)
so = E.egomap_init_state(1, cfg_off)
feat = E.egomap_read(so, jnp.asarray(open_obs)[None], cfg_off)
so2 = E.egomap_update(so, jnp.asarray(open_obs)[None], jnp.array([2], jnp.int32),
                      jnp.zeros(1, jnp.bool_), cfg_off)
check("T6 feature-off read returns zeros", float(np.asarray(feat).sum()) == 0.0)
check("T6 feature-off update leaves map_bank unchanged",
      float(np.asarray(so2["map_bank"]).sum()) == 0.0)

# ---- Test 7: read normalizes & reflects current floor ---- #
sr = E.egomap_init_state(1, cfg)
sr = E.egomap_update(sr, jnp.asarray(obsA)[None], jnp.zeros(1, jnp.int32),
                     jnp.zeros(1, jnp.bool_), cfg)
feat = np.asarray(E.egomap_read(sr, jnp.asarray(obsA)[None], cfg)[0])
check("T7 read returns (H,W,C) features", feat.shape == (16, 16, E.N_MAP_CH))
check("T7 read visit normalized in [0,1]", feat[..., E.CH_VISIT].max() <= 1.0)
check("T7 read recency in [0,1]", (feat[..., E.CH_LASTVISIT].min() >= 0.0)
      and (feat[..., E.CH_LASTVISIT].max() <= 1.0))

print("\n==== SUMMARY ====")
ok = sum(1 for _, c in PASS if c)
print(f"{ok}/{len(PASS)} passed")
if ok != len(PASS):
    print("FAILED:", [n for n, c in PASS if not c])
    sys.exit(1)
print("ALL_EGOMAP_CORE_TESTS_PASS")
