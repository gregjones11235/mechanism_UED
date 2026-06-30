"""最小复现：诊断 get_learnability_set 加 auction(GAE reverse-scan + hstates emit) 是否炸编译。

模拟 get_learnability_set 的双层 scan 结构（scan over NUM_BATCHES，内层 scan over ROLLOUT_STEPS），
对比 ①原始(只算 success-like 标量) ②加 GAE reverse-scan ③加 hstates emit ④加全部。
测各自的编译时间（首次 .block_until_ready）。若 ②/③ 编译时间暴涨 → 坐实根因。

跑法（minimax env, CPU 足够看相对编译时间）： python _diag_compile.py
缩小规模(BATCH=64, ROLLOUT=100, NB=3)保持结构、压缩到 CPU 几秒可见相对差异。
"""
import time
import jax
import jax.numpy as jnp
from functools import partial

# 缩小但保持结构的维度（CPU 上看相对编译时间）
NB, ROLLOUT, BATCH, H = 3, 100, 64, 64


def make_rollout(rng):
    """模拟一批 rollout 的 (done, value, reward, hidden)。"""
    k1, k2, k3, k4 = jax.random.split(rng, 4)
    done = (jax.random.uniform(k1, (ROLLOUT, BATCH)) < 0.1).astype(jnp.float32)
    value = jax.random.normal(k2, (ROLLOUT, BATCH))
    reward = jax.random.normal(k3, (ROLLOUT, BATCH))
    hidden = jax.random.normal(k4, (ROLLOUT, BATCH, H))
    return done, value, reward, hidden


def variant_base(rng):
    """① 原始：内层 scan 出 rollout，只算 success-like 标量(p(1-p))。"""
    def _batch(unused, r):
        done, value, reward, hidden = make_rollout(r)
        # 模拟 success_rate → learnability
        p = jax.nn.sigmoid(value.mean(0))
        learn = (p * (1 - p))
        return None, learn
    rngs = jax.random.split(rng, NB)
    _, out = jax.lax.scan(_batch, None, rngs, NB)
    return out.flatten()


def variant_gae(rng):
    """② 加 GAE reverse-scan(模拟 PVL 的 _get_adv)。"""
    def _batch(unused, r):
        done, value, reward, hidden = make_rollout(r)
        p = jax.nn.sigmoid(value.mean(0))
        learn = (p * (1 - p))
        # GAE reverse scan over ROLLOUT
        last_val = value[-1]
        def _adv(carry, x):
            gae, nv = carry
            d, v, rw = x
            delta = rw + 0.99 * nv * (1 - d) - v
            gae = delta + 0.99 * 0.95 * (1 - d) * gae
            return (gae, v), gae
        _, adv = jax.lax.scan(_adv, (jnp.zeros_like(last_val), last_val),
                              (done, value, reward), reverse=True, unroll=16)
        pvl = jnp.maximum(adv, 0).mean(0)
        return None, learn + pvl  # 合一个标量返回
    rngs = jax.random.split(rng, NB)
    _, out = jax.lax.scan(_batch, None, rngs, NB)
    return out.flatten()


def variant_hidden_emit(rng):
    """③ 内层 scan emit 全量 hidden(模拟 hstates emit) + 末尾 reshape 用。"""
    def _batch(unused, r):
        # 用真 scan emit hidden(逐步)，模拟 _env_step 返回 (transition, hstate)
        def _step(carry, x):
            h = jax.random.normal(x, (BATCH, H))
            return carry, h
        step_rngs = jax.random.split(r, ROLLOUT)
        _, hstates = jax.lax.scan(_step, None, step_rngs, ROLLOUT)  # (ROLLOUT, BATCH, H)
        # 抽 K 步算个标量(模拟 CENIE 求值)
        K = 16
        t_idx = jnp.linspace(0, ROLLOUT - 1, K).astype(jnp.int32)
        hs = hstates[t_idx]
        score = (hs ** 2).sum(-1).mean(0)  # (BATCH,)
        return None, score
    rngs = jax.random.split(rng, NB)
    _, out = jax.lax.scan(_batch, None, rngs, NB)
    return out.flatten()


def time_compile(name, fn, rng):
    t0 = time.time()
    out = jax.jit(fn)(rng)
    out.block_until_ready()
    t1 = time.time()
    print(f"[{name}] 编译+首run: {t1-t0:.2f}s  out.shape={out.shape}")
    # 二次调用(已编译)看纯运行时间
    t2 = time.time()
    fn_j = jax.jit(fn)
    fn_j(rng).block_until_ready()  # 触发可能的二次编译
    t3 = time.time()
    return t1 - t0


rng = jax.random.PRNGKey(0)
print(f"规模: NB={NB} ROLLOUT={ROLLOUT} BATCH={BATCH} H={H}\n")
tb = time_compile("① base(p(1-p))", variant_base, rng)
tg = time_compile("② +GAE reverse-scan", variant_gae, rng)
th = time_compile("③ +hidden emit(scan over ROLLOUT)", variant_hidden_emit, rng)

print(f"\n=== 相对编译时间 ===")
print(f"② GAE / ① base = {tg/tb:.1f}×")
print(f"③ hidden-emit / ① base = {th/tb:.1f}×")
print("\n判读: 若某变体 >> base，该结构是 get_learnability_set 编译爆炸的元凶。")
