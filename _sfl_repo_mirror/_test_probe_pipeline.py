"""探针信号管道 dry-run（CPU）—— 用合成 rollout 统计验证 compute_four_signals + write_reports
全链路：GMM 自拟自评 → 四信号 (4,M) → 相关报告 → 落盘。不需 jaxnav/checkpoint，只验接口与数值健全。

跑法（minimax env 或 sfl env，需 minimax.util.rl.cenie_density + sklearn）：
    python _test_probe_pipeline.py
"""
import os, tempfile
import numpy as np
import jax.numpy as jnp

from probe_orthogonality import compute_four_signals, write_reports

PASS, FAIL = "PASS", "FAIL"
results = []
def check(name, cond):
    results.append((name, bool(cond))); print(f"[{PASS if cond else FAIL}] {name}")


# ── 合成一批 rollout 统计：M 个 level、T 步、d 维 hidden ──
rng = np.random.default_rng(0)
M, T, d = 64, 32, 16

# success_rate 跨 [0,1]；让前 5 个 level incomplete（num_episodes=0）
success_rate = rng.uniform(0, 1, M).astype(np.float32)
num_episodes = rng.integers(1, 6, M).astype(np.int32)
num_episodes[:5] = 0                                    # 前 5 个 incomplete

# dones/advantages: (T, M)，喂 positive_value_loss。dones 为 bool（同官方 traj_batch.done）。
dones = (rng.uniform(0, 1, (T, M)) < 0.1)
dones[-1, :] = True                                    # 保证每 level 至少一个 done（除 incomplete）
dones[:, :5] = False                                   # incomplete level 无 done
dones = dones.astype(bool)
advantages = rng.normal(0, 1, (T, M)).astype(np.float32)

# hidden_feats: (T*M, d)，per_level_index 指明每行属于哪个 level
hidden_feats = rng.normal(0, 1, (T * M, d)).astype(np.float32)
per_level_index = np.tile(np.arange(M), T).astype(np.int64)   # (T*M,)

mat, names = compute_four_signals(
    success_rate=jnp.asarray(success_rate),
    num_episodes=jnp.asarray(num_episodes),
    dones=jnp.asarray(dones),
    advantages=jnp.asarray(advantages),
    hidden_feats=jnp.asarray(hidden_feats),
    per_level_index=jnp.asarray(per_level_index),
    n_levels=M,
)
mat_np = np.asarray(mat)

check("形状: (4, M)", mat.shape == (4, M))
check("names 顺序对", names == ['difficulty', 'learnability', 'PVL', 'CENIE'])
check("incomplete level (前5) 四信号全 -inf",
      np.all(np.isneginf(mat_np[:, :5])))
check("complete level 至少 difficulty/learnability 有限",
      np.all(np.isfinite(mat_np[0, 5:])) and np.all(np.isfinite(mat_np[1, 5:])))
check("CENIE 行 complete 位有限（GMM 自拟自评成功）",
      np.all(np.isfinite(mat_np[3, 5:])))
check("无 NaN（finite 区）", not np.any(np.isnan(mat_np[np.isfinite(mat_np)])))

# difficulty/learnability 数值对（与直接公式比，complete 位）
p = success_rate[5:]
check("difficulty 数值 = -(p-0.5)²", np.allclose(mat_np[0, 5:], -((p - 0.5) ** 2), atol=1e-5))
check("learnability 数值 = p(1-p)", np.allclose(mat_np[1, 5:], p * (1 - p), atol=1e-5))


# ── 报告落盘 ──
out = tempfile.mkdtemp(prefix="probe_test_")
rep = write_reports(mat, names, out, threshold=0.5)
check("verdict 合法", rep["verdict"] in ("ready_pool_sufficient", "must_implement_expensive_signal"))
check("corr_matrix.csv 存在", os.path.exists(os.path.join(out, "corr_matrix.csv")))
check("signals.npz 存在", os.path.exists(os.path.join(out, "signals.npz")))
check("verdict.txt 存在", os.path.exists(os.path.join(out, "verdict.txt")))
# 相关矩阵对角=1、对称
corr = rep["corr"]
check("corr 对角=1", np.allclose(np.diag(corr), 1.0, atol=1e-5))
check("corr 对称", np.allclose(corr, corr.T, atol=1e-5))
check("corr |ρ|≤1", np.all(np.abs(corr) <= 1.0 + 1e-6))

# npz 可复读
loaded = np.load(os.path.join(out, "signals.npz"), allow_pickle=True)
check("npz 复读 mat 形状对", loaded["mat"].shape == (4, M))

print(f"\n[probe out dir] {out}")
n_pass = sum(1 for _, ok in results if ok)
print(f"{'='*40}\n{n_pass}/{len(results)} 通过")
if n_pass != len(results):
    print("失败项:", [n for n, ok in results if not ok]); raise SystemExit(1)
print("全部通过 ✓")
