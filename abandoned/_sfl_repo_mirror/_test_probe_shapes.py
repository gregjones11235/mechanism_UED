"""验证 get_probe_signals 的 subsample + per_level_index + 内存预算（CPU，纯形状/索引逻辑）。

不跑 jaxnav，只把 jit 内/外的张量重排逻辑抽出来，用小规模 config 验证：
  ① hidden subsample 后量级（防 OOM 复发）；
  ② per_level_index 正确把每个 hidden 行映射回它的 env 列；
  ③ dones/advantages 的 (T, M) 重排正确。
对照真实 config（NB=5, ROLLOUT=1000, BATCH=1000, H=512, K=16）算 host 内存。
"""
import numpy as np
import jax.numpy as jnp

PASS, FAIL = "PASS", "FAIL"
results = []
def check(name, cond):
    results.append((name, bool(cond))); print(f"[{PASS if cond else FAIL}] {name}")


# ── 小规模 config 验证索引逻辑（NB=2, ROLLOUT=8, BATCH=3, H=4, K=4, num_agents=1）──
NB, ROLLOUT, BATCH, H, K = 2, 8, 3, 4, 4
A = BATCH  # num_agents=1 → BATCH_ACTORS = BATCH

# 模拟 jit 内：hstates (ROLLOUT, A, H) → 抽 K 步
def subsample_hidden(hstates, rollout_steps, K):
    t_idx = jnp.linspace(0, rollout_steps - 1, K).astype(jnp.int32)
    return hstates[t_idx]  # (K, A, H)

# 每个 batch 的 hidden 标记成可识别值：hid[k,a] = a（env 列），便于验 per_level_index
per_batch_hidden = []
for b in range(NB):
    hs = np.zeros((ROLLOUT, A, H), dtype=np.float32)
    for a in range(A):
        hs[:, a, :] = a  # 该 actor 所有步、所有维都 = a
    per_batch_hidden.append(np.asarray(subsample_hidden(jnp.asarray(hs), ROLLOUT, K)))
batched_hidden = np.stack(per_batch_hidden, axis=0)  # (NB, K, A, H)
check("subsample: hidden 形状 (NB,K,A,H)", batched_hidden.shape == (NB, K, A, H))

# 外层：reshape + per_level_index（复刻 jaxnav_sfl 的 broadcasting 构造）
nb, Kd, Ad = batched_hidden.shape[:3]
hid = batched_hidden.reshape(-1, H)  # (NB*K*A, H)
nb_g = jnp.arange(nb)[:, None, None]
a_g = jnp.arange(Ad)[None, None, :]
pli_grid = (nb_g * BATCH + (a_g % BATCH))
pli_grid = jnp.broadcast_to(pli_grid, (nb, Kd, Ad))
per_level_index = np.asarray(pli_grid.reshape(-1).astype(jnp.int32))

check("per_level_index 长度 = NB*K*A", per_level_index.shape[0] == NB * K * A)
# 关键正确性：hid 每行的值(=原 actor a) 应满足 per_level_index = batch*BATCH + a
# 因为 hid[row] 全 = a，per_level_index[row] 应 = (row 所属 batch)*BATCH + a
ok_map = True
for row in range(hid.shape[0]):
    a_val = int(hid[row, 0])                      # 原 actor 列号 a
    b_val = row // (K * A)                         # 该行所属 batch
    expected_level = b_val * BATCH + a_val
    if per_level_index[row] != expected_level:
        ok_map = False; break
check("per_level_index 正确映射 hidden 行 → env 列", ok_map)
check("per_level_index 覆盖 0..M-1", set(per_level_index.tolist()) == set(range(NB * BATCH)))

# ── dones/advantages (T, M) 重排 ──
def stack_TM(x):  # (NB, T, A) → (T, NB*A)
    return x.transpose(1, 0, 2).reshape(x.shape[1], -1)
dones = np.random.rand(NB, ROLLOUT, A) < 0.2
dones_TM = np.asarray(stack_TM(jnp.asarray(dones)))
check("dones_TM 形状 (T, NB*A)", dones_TM.shape == (ROLLOUT, NB * A))


# ── 真实 config 的 host 内存预算（防 OOM 复发）──
def mb(*dims):
    return np.prod(dims) * 4 / 1e6  # float32 MB
RNB, RROLL, RBATCH, RH, RK = 5, 1000, 1000, 512, 16
hidden_mb = mb(RNB, RK, RBATCH, RH)
dones_mb = mb(RROLL, RNB * RBATCH)
adv_mb = mb(RROLL, RNB * RBATCH)
print(f"\n[真实 config 预算] hidden(subsampled)={hidden_mb:.0f}MB, "
      f"dones={dones_mb:.0f}MB, adv={adv_mb:.0f}MB, 合计≈{hidden_mb+dones_mb+adv_mb:.0f}MB")
# 旧 bug：未抽样 hidden = mb(5,1000,1000,512) ≈ 10240MB（爆 24G）
old_hidden_mb = mb(RNB, RROLL, RBATCH, RH)
print(f"[对照] 未抽样 hidden 会是 {old_hidden_mb:.0f}MB（旧 OOM 元凶）→ 抽样后降 {old_hidden_mb/hidden_mb:.0f}×")
check("hidden subsampled < 500MB（远低于 24G）", hidden_mb < 500)
check("总 host 预算 < 1GB", hidden_mb + dones_mb + adv_mb < 1000)


n_pass = sum(1 for _, ok in results if ok)
print(f"\n{'='*40}\n{n_pass}/{len(results)} 通过")
if n_pass != len(results):
    print("失败项:", [n for n, ok in results if not ok]); raise SystemExit(1)
print("全部通过 ✓")
