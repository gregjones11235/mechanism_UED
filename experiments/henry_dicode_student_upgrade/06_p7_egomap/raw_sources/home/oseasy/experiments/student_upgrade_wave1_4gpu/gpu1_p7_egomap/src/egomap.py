"""P7-EGOMAP: egocentric explicit spatial memory built ONLY from the Student's
own observations and actions. No privileged information ever enters the map.

Memory semantics (matches the base GTrXL memory carry):
  carry_t  = map state after incorporating obs_{0..t-1}; ego_pos = pos_t.
  At step t:
      features_t = egomap_read(carry_t, obs_t)        # map up to t-1, around pos_t
      action_t   = policy(obs_t, memory_{t-1}, features_t)
      carry_{t+1}= egomap_update(carry_t, obs_t, action_t, true_done_t)
  Rollout collection, the learner model_forward and the evaluator MUST use this
  identical ordering (enforced in ppo_tr_egomap.py / eval_p7.py). This keeps the
  map a pure function of the agent's own past observations and actions => no
  privileged leakage, no train/eval mismatch.

Obs layout (8335-dim flat vector), verified against craftax renderer.py:
  [0:8217]    spatial local view, reshape (9, 11, 83), world-axis-aligned,
              centered on player, ALL spatial channels masked by light (only
              observed/lit cells are nonzero).
                channels [0:37]  = BlockType one-hot (terrain)
                channels [37:42] = ItemType one-hot: NONE/TORCH(38)/
                                   LADDER_DOWN(39)/LADDER_UP(40)/
                                   LADDER_DOWN_BLOCKED(41)
                channels [42:82] = mob map (5 classes x 8 types)
                channel  [82]    = light / visibility mask (1 = observed)
  [8217:8259] 42 renderer scalars; direction one-hot at [8243:8247];
              special_values[5]=player_level/10.0 at [8256]
  [8259:8335] 76 task embedding (NOT spatial; never used as map source)

Movement (game_logic.move_player): actions LEFT=1/RIGHT=2/UP=3/DOWN=4 move one
cell in absolute world direction DIRECTIONS[action] unless the target cell is
blocked (solid/water/lava/out-of-bounds/mob). Other actions => (0,0).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

# --------------------------------------------------------------------------- #
# Constants (all derived from craftax, no privileged runtime access)
# --------------------------------------------------------------------------- #
OBS_FLAT_DIM = 8335
OBS_H, OBS_W = 9, 11
OBS_CY, OBS_CX = OBS_H // 2, OBS_W // 2  # (4, 5) center of the local patch
N_SPATIAL = OBS_H * OBS_W * 83  # 8217
N_BLOCK = 37
N_ITEM = 5
CH_LIGHT = 82
ITEM_TORCH = 1                 # item one-hot idx -> spatial channel 38
ITEM_LADDER_DOWN = 2           # -> 39
ITEM_LADDER_UP = 3             # -> 40
ITEM_LADDER_DOWN_BLOCKED = 4   # -> 41
PLAYER_LEVEL_IDX = 8217 + 16 + 1 + 9 + 4 + 1 + 3 + 5  # 8256

N_ACTIONS = 43
DIRECTIONS = np.zeros((N_ACTIONS, 2), dtype=np.int32)
DIRECTIONS[1] = [0, -1]   # LEFT  -> -y
DIRECTIONS[2] = [0, 1]    # RIGHT -> +y
DIRECTIONS[3] = [-1, 0]   # UP    -> -x
DIRECTIONS[4] = [1, 0]    # DOWN  -> +x
DIRECTIONS_JNP = jnp.asarray(DIRECTIONS)

SOLID_BLOCKS = [4, 5, 8, 9, 10, 11, 12, 15, 16, 17, 19, 20, 21, 22, 23, 24,
                28, 30, 31, 32, 33, 34, 35]
BLOCKED_VALUES = sorted(set(SOLID_BLOCKS) | {1, 3, 14})  # +OOB(1),WATER(3),LAVA(14)
BLOCKED_LUT = np.zeros(N_BLOCK, dtype=np.float32)
BLOCKED_LUT[BLOCKED_VALUES] = 1.0
BLOCKED_LUT_JNP = jnp.asarray(BLOCKED_LUT)  # (37,)

CH_EVER = 0
CH_VISIT = 1
CH_LASTVISIT = 2
CH_PASSABLE = 3
CH_OBSTACLE = 4
CH_TORCH = 5
CH_STAIR_DOWN = 6
CH_STAIR_UP = 7
CH_FRONTIER = 8
N_MAP_CH = 9
VISIT_TAU = 8.0
RECENCY_CAP = 256.0


class EgoMapConfig:
    def __init__(self, map_size: int = 32, num_floors: int = 9,
                 enabled: bool = True):
        self.map_size = int(map_size)
        self.num_floors = int(num_floors)
        self.enabled = bool(enabled)

    @property
    def center(self) -> int:
        return self.map_size // 2


# --------------------------------------------------------------------------- #
# Obs parsing (pure, leak-free)
# --------------------------------------------------------------------------- #
def parse_obs(obs_flat):
    """obs_flat: (..., 8335) -> (space (...,9,11,83), player_level int (...))."""
    space = obs_flat[..., :N_SPATIAL].reshape(*obs_flat.shape[:-1], OBS_H, OBS_W, 83)
    lvl = jnp.round(obs_flat[..., PLAYER_LEVEL_IDX] * 10.0).astype(jnp.int32)
    return space, lvl


def _decode_patch(space):
    """space: (9,11,83) single env -> per-cell decoded arrays (each (9,11)),
    all gated by the visibility mask (only observed cells)."""
    observed = space[..., CH_LIGHT] > 0.5
    block_id = jnp.argmax(space[..., :N_BLOCK], axis=-1)
    blocked = BLOCKED_LUT_JNP[block_id] > 0.5
    passable = observed & (~blocked)
    obstacle = observed & blocked
    items = space[..., N_BLOCK:N_BLOCK + N_ITEM]
    torch = observed & (items[..., ITEM_TORCH] > 0.5)
    stair_down = observed & ((items[..., ITEM_LADDER_DOWN] > 0.5) |
                             (items[..., ITEM_LADDER_DOWN_BLOCKED] > 0.5))
    stair_up = observed & (items[..., ITEM_LADDER_UP] > 0.5)
    mob = space[..., N_BLOCK + N_ITEM:CH_LIGHT].sum(-1) > 0.5
    return observed, passable, obstacle, torch, stair_down, stair_up, mob, blocked


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
def egomap_init_state(num_envs, cfg: EgoMapConfig):
    c = cfg.center
    return {
        "map_bank": jnp.zeros((num_envs, cfg.num_floors, cfg.map_size,
                               cfg.map_size, N_MAP_CH), dtype=jnp.float32),
        "ego_pos": jnp.tile(jnp.array([c, c], dtype=jnp.int32), (num_envs, 1)),
        "step": jnp.zeros((num_envs,), dtype=jnp.int32),
    }


def _clip_start(ego_pos, cfg: EgoMapConfig):
    start = ego_pos - jnp.array([OBS_CY, OBS_CX], dtype=jnp.int32)
    return jnp.clip(start, 0,
                    jnp.array([cfg.map_size - OBS_H, cfg.map_size - OBS_W]))


def _stamp(floor_ch, ego_pos, patch_ch, cfg):
    """OR-accumulate a (9,11) patch channel into floor_ch (H,W) at ego_pos."""
    start = _clip_start(ego_pos, cfg)
    cur = jax.lax.dynamic_slice(floor_ch, (start[0], start[1]), (OBS_H, OBS_W))
    new = jnp.maximum(cur, patch_ch)
    return jax.lax.dynamic_update_slice(floor_ch, new, (start[0], start[1]))


def _normalize_floor(floor_map, step):
    """(H,W,C) raw floor -> (H,W,C) encoder features (visit/recency normalized)."""
    visit_norm = 1.0 - jnp.exp(-floor_map[..., CH_VISIT:CH_VISIT + 1] / VISIT_TAU)
    never = floor_map[..., CH_EVER:CH_EVER + 1] < 0.5
    recency = jnp.clip((step.astype(jnp.float32) - floor_map[..., CH_LASTVISIT]) /
                       RECENCY_CAP, 0.0, 1.0)
    recency = jnp.where(never[..., 0], 1.0, recency)[..., None]
    feat = floor_map.at[..., CH_VISIT:CH_VISIT + 1].set(visit_norm)
    feat = feat.at[..., CH_LASTVISIT:CH_LASTVISIT + 1].set(recency)
    return feat


# --------------------------------------------------------------------------- #
# Single-env read / update (vmapped over envs)
# --------------------------------------------------------------------------- #
def _read_one(carry, obs_flat, cfg):
    lvl = parse_obs(obs_flat)[1]
    lvl = jnp.clip(lvl, 0, cfg.num_floors - 1)
    floor_map = carry["map_bank"][lvl]
    return _normalize_floor(floor_map, carry["step"])


def _update_one(carry, obs_flat, action, true_done, cfg):
    map_bank = carry["map_bank"]
    ego_pos = carry["ego_pos"]
    step = carry["step"]
    c = cfg.center

    space, lvl = parse_obs(obs_flat)
    (observed, passable, obstacle, torch, stair_down, stair_up,
     mob, blocked) = _decode_patch(space)

    # true done => clear this env's map bank + reset ego_pos/step
    map_bank = jnp.where(true_done, jnp.zeros_like(map_bank), map_bank)
    ego_pos = jnp.where(true_done, jnp.array([c, c], dtype=jnp.int32), ego_pos)
    step = jnp.where(true_done, 0, step)

    lvl = jnp.clip(lvl, 0, cfg.num_floors - 1)
    fm = map_bank[lvl]
    fm = fm.at[..., CH_EVER].set(_stamp(fm[..., CH_EVER], ego_pos, observed.astype(jnp.float32), cfg))
    fm = fm.at[..., CH_PASSABLE].set(_stamp(fm[..., CH_PASSABLE], ego_pos, passable.astype(jnp.float32), cfg))
    fm = fm.at[..., CH_OBSTACLE].set(_stamp(fm[..., CH_OBSTACLE], ego_pos, obstacle.astype(jnp.float32), cfg))
    fm = fm.at[..., CH_TORCH].set(_stamp(fm[..., CH_TORCH], ego_pos, torch.astype(jnp.float32), cfg))
    fm = fm.at[..., CH_STAIR_DOWN].set(_stamp(fm[..., CH_STAIR_DOWN], ego_pos, stair_down.astype(jnp.float32), cfg))
    fm = fm.at[..., CH_STAIR_UP].set(_stamp(fm[..., CH_STAIR_UP], ego_pos, stair_up.astype(jnp.float32), cfg))

    pc = jnp.clip(ego_pos, 0, cfg.map_size - 1)
    fm = fm.at[pc[0], pc[1], CH_VISIT].set(fm[pc[0], pc[1], CH_VISIT] + 1.0)
    fm = fm.at[pc[0], pc[1], CH_LASTVISIT].set(step.astype(jnp.float32))

    ever = fm[..., CH_EVER] > 0.5
    pass_obs = ever & (fm[..., CH_PASSABLE] > 0.5)
    unobs = (~ever).astype(jnp.float32)
    nb = (jnp.roll(unobs, 1, 0) + jnp.roll(unobs, -1, 0) +
          jnp.roll(unobs, 1, 1) + jnp.roll(unobs, -1, 1))
    fm = fm.at[..., CH_FRONTIER].set((pass_obs & (nb > 0.5)).astype(jnp.float32))

    map_bank = map_bank.at[lvl].set(fm)

    # odometry: action + observation-based block correction (no privileged info)
    move_dir = DIRECTIONS_JNP[action]
    is_move = jnp.abs(move_dir).sum() > 0
    tgt = jnp.clip(jnp.array([OBS_CY, OBS_CX]) + move_dir, 0,
                   jnp.array([OBS_H - 1, OBS_W - 1]))
    tgt_blocked = blocked[tgt[0], tgt[1]] | mob[tgt[0], tgt[1]] | (~observed[tgt[0], tgt[1]])
    valid = is_move & (~tgt_blocked)
    ego_pos_new = ego_pos + move_dir * valid.astype(jnp.int32)

    return {"map_bank": map_bank, "ego_pos": ego_pos_new, "step": step + 1}


def egomap_read(state, obs_flat, cfg):
    """(num_envs,8335) -> ego_features (num_envs,H,W,C). Map up to t-1."""
    if not cfg.enabled:
        n = obs_flat.shape[0]
        return jnp.zeros((n, cfg.map_size, cfg.map_size, N_MAP_CH), jnp.float32)
    axes = {"map_bank": 0, "ego_pos": 0, "step": 0}
    return jax.vmap(_read_one, in_axes=(axes, 0, None))(state, obs_flat, cfg)


def egomap_update(state, obs_flat, action, true_done, cfg):
    """Incorporate obs_t/action_t -> carry_{t+1}."""
    if not cfg.enabled:
        return state
    axes = {"map_bank": 0, "ego_pos": 0, "step": 0}
    return jax.vmap(_update_one, in_axes=(axes, 0, 0, 0, None))(
        state, obs_flat, action, true_done, cfg)
