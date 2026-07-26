"""G1 feature-off bit-exact test (+ G6 zero-init-gate initial equivalence).
Run with PYTHONPATH including dicode src and this dir:
  JAX_PLATFORMS=cpu PYTHONPATH=<dicode_src>:<thisdir> python test_network_g1.py
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import sys
import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dicode.network import ActorCriticTransformer
import network_egomap as NE

PASS = []
def check(name, cond):
    PASS.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name)

# small but valid config (bit-exactness is size-independent)
cfg = dict(action_dim=43, activation="relu", hidden_layers=32, encoder_size=32,
           num_heads=2, qkv_features=32, num_layers=2, gating=False, gating_bias=0.0)
B, WM, NL, ES, NH = 2, 8, 2, 32, 2
H = W = 16
key = jax.random.PRNGKey(0)

base = ActorCriticTransformer(**cfg)
new = NE.ActorCriticTransformerEgoMap(**cfg, egomap_channels=9,
                                      egomap_cnn_features=(8, 16))

memories = jnp.zeros((B, WM, NL, ES))
obs = jax.random.normal(key, (B, 8335))
mask = jnp.zeros((B, NH, 1, WM + 1), dtype=jnp.bool_)
ego = jax.random.normal(jax.random.PRNGKey(1), (B, H, W, 9))

# init the NEW (full) module to obtain base+ego params
vars_new = new.init(key, memories, obs, mask, ego_features=ego, egomap_enabled=True,
                    method=new.model_forward_eval)
params_new = vars_new["params"]
base_subset = {k: v for k, v in params_new.items()
               if k not in ("ego_encoder", "ego_gate")}
check("G0 base-subset keys == {transformer,actor_*,critic_*}",
      set(base_subset.keys()) == {"transformer", "actor_ln1", "actor_ln2",
                                  "actor_out", "critic_ln1", "critic_ln2",
                                  "critic_out"})
check("G0 ego_gate initialized to zero",
      float(np.abs(np.asarray(params_new["ego_gate"])).max()) == 0.0)

def run(module, params, ego_features, enabled, method):
    return module.apply({"params": params}, memories, obs, mask,
                        ego_features=ego_features, egomap_enabled=enabled,
                        method=method)

# ---- G1: feature-off bit-exact (eval forward) ---- #
pi_new_off, v_new_off, mem_new_off = run(new, params_new, ego, False,
                                         new.model_forward_eval)
pi_base, v_base, mem_base = base.apply({"params": base_subset}, memories, obs,
                                       mask, method=base.model_forward_eval)
d_logits = float(np.abs(np.asarray(pi_new_off.logits) - np.asarray(pi_base.logits)).max())
d_value = float(np.abs(np.asarray(v_new_off) - np.asarray(v_base)).max())
d_mem = float(np.abs(np.asarray(mem_new_off) - np.asarray(mem_base)).max())
print(f"  feature-off max|dlogits|={d_logits} max|dvalue|={d_value} max|dmem|={d_mem}")
check("G1 feature-off Actor logits bit-exact (diff==0)", d_logits == 0.0)
check("G1 feature-off Value bit-exact (diff==0)", d_value == 0.0)
check("G1 feature-off memory_out bit-exact (diff==0)", d_mem == 0.0)

# ---- G6-init: enabled + zero gate => also bit-exact at init ---- #
pi_new_on, v_new_on, _ = run(new, params_new, ego, True, new.model_forward_eval)
d_on_logits = float(np.abs(np.asarray(pi_new_on.logits) - np.asarray(pi_base.logits)).max())
d_on_value = float(np.abs(np.asarray(v_new_on) - np.asarray(v_base)).max())
print(f"  enabled(zero-gate) max|dlogits|={d_on_logits} max|dvalue|={d_on_value}")
check("G6-init enabled zero-gate Actor logits == base (diff==0)", d_on_logits == 0.0)
check("G6-init enabled zero-gate Value == base (diff==0)", d_on_value == 0.0)

# ---- nonzero gate must CHANGE output (sanity: fusion is actually wired) ---- #
import copy
params_ng = jax.tree_util.tree_map(lambda x: x, params_new)
params_ng["ego_gate"] = jnp.ones((1,))
pi_ng, v_ng, _ = run(new, params_ng, ego, True, new.model_forward_eval)
d_ng = float(np.abs(np.asarray(pi_ng.logits) - np.asarray(pi_base.logits)).max())
check("G-sanity nonzero gate changes logits (fusion wired)", d_ng > 0.0)

# ---- train forward (windowed) feature-off bit-exact ---- #
# forward_train: queries = window (WM), keys = concat(memory 2*WM, window WM) = 3*WM
mem_train = jnp.zeros((B, 2 * WM, NL, ES))
obs_train = jax.random.normal(jax.random.PRNGKey(2), (B, WM, 8335))
mask_train = jnp.zeros((B, NH, WM, 3 * WM), dtype=jnp.bool_)
ego_train = jax.random.normal(jax.random.PRNGKey(3), (B, WM, H, W, 9))
pi_t_new, v_t_new = new.apply({"params": params_new}, mem_train, obs_train,
                              mask_train, ego_features=ego_train,
                              egomap_enabled=False, method=new.model_forward_train)
pi_t_base, v_t_base = base.apply({"params": base_subset}, mem_train, obs_train,
                                 mask_train, method=base.model_forward_train)
dt_logits = float(np.abs(np.asarray(pi_t_new.logits) - np.asarray(pi_t_base.logits)).max())
dt_value = float(np.abs(np.asarray(v_t_new) - np.asarray(v_t_base)).max())
print(f"  train feature-off max|dlogits|={dt_logits} max|dvalue|={dt_value}")
check("G1 train-forward feature-off Actor logits bit-exact", dt_logits == 0.0)
check("G1 train-forward feature-off Value bit-exact", dt_value == 0.0)

print("\n==== SUMMARY ====")
ok = sum(1 for _, c in PASS if c)
print(f"{ok}/{len(PASS)} passed")
if ok != len(PASS):
    print("FAILED:", [n for n, c in PASS if not c])
    sys.exit(1)
print("ALL_NETWORK_G1_TESTS_PASS")
