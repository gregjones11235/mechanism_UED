"""auction_bid.py 单元测试（CPU，验证三 bug 修复 + 数值性质）。

复现 mechanism_UED/方案B_多generator设计.md §6.3 的 numpy 推导，落成断言。
跑法（Oscar，sfl 环境）：  python _test_auction_bid.py
"""
import jax
import jax.numpy as jnp
import numpy as np

from auction_bid import (
    standardize_per_estimator, topk_bids, auction_weights, mix_scores,
)

PASS, FAIL = "PASS", "FAIL"
results = []

def check(name, cond):
    results.append((name, bool(cond)))
    print(f"[{PASS if cond else FAIL}] {name}")


# ---- 场景①：量级悬殊（future_routes §1.1 炸弹）：PVL~1e-2, SFL~0, CENIE~-1e2 ----
np.random.seed(0)
M = 256
pvl = np.random.uniform(0.005, 0.02, M)
sfl = np.zeros(M)
cenie = -np.random.uniform(300, 400, M)
per_est = jnp.asarray(np.stack([pvl, sfl, cenie], axis=0))  # (3, 256)

z = standardize_per_estimator(per_est)
# z 每行 finite 部分均值≈0、方差≈1
zf = np.asarray(z)
check("z-score: 每 estimator finite 均值≈0",
      np.allclose(zf.mean(axis=1), 0, atol=1e-4))
check("z-score: 每 estimator finite 方差≈1 (非全0行)",
      abs(zf[0].std() - 1) < 1e-3 and abs(zf[2].std() - 1) < 1e-3)

bids = topk_bids(z)
bf = np.asarray(bids)
# bug② 修复核心：bid 不再恒 0（旧实现 z 后全体均值≈0）
check("bug②修复: bids 不恒为0 (有区分度)", np.ptp(bf) > 0.1 or np.abs(bf).max() > 0.5)

w = auction_weights(bids, 1.0)
wf = np.asarray(w)
# bug① 修复核心：CENIE（idx=2，原大负数）拿到非平凡权重，不再被 argmax 压死
check("bug①修复: CENIE(大负数) 拿到实质权重 (>0.05)", wf[2] > 0.05)
check("权重和为1", abs(wf.sum() - 1) < 1e-5)


# ---- 场景②：三正交 estimator，验证 λ-frontier 单调 + 端点 ----
np.random.seed(1)
diff = np.random.uniform(0, 0.25, M)
cover = -np.random.uniform(50, 500, M)
trans = np.random.exponential(2.0, M)
pe2 = jnp.asarray(np.stack([diff, cover, trans], axis=0))
z2 = standardize_per_estimator(pe2)
b2 = topk_bids(z2)

def entropy(w):
    w = np.asarray(w)
    return float(-(w * np.log(w + 1e-12)).sum())

# λ=inf → one-hot argmax（熵≈0）
w_inf = auction_weights(b2, float('inf'))
check("λ=inf → one-hot argmax (熵≈0)", entropy(w_inf) < 1e-3)
check("λ=inf → 选中 argmax(bids)", int(np.argmax(np.asarray(w_inf))) == int(np.argmax(np.asarray(b2))))

# λ→0 → 也趋 one-hot（sharp 端）
w_small = auction_weights(b2, 0.001)
check("λ→0 → 趋 one-hot (max>0.99)", np.asarray(w_small).max() > 0.99)

# λ 大 → 趋 uniform
w_large = auction_weights(b2, 1e6)
check("λ→大 → 趋 uniform (max-1/N<0.01)", abs(np.asarray(w_large).max() - 1/3) < 0.01)

# 单调性：熵随 λ 从小到大先升（sharp→uniform）。检查中间 λ 熵 > 两个 sharp 端
w_mid = auction_weights(b2, 3.0)
check("中间 λ=3 熵 > λ→0 端 (扫到混合区)", entropy(w_mid) > entropy(w_small) + 0.1)

# 三个 estimator 在 λ=1 都参与（min 权重不被压死）
w1 = auction_weights(b2, 1.0)
check("λ=1: 三 estimator 都参与 (min>0.05)", np.asarray(w1).min() > 0.05)


# ---- 场景③：NaN 安全（incomplete level 用 -inf 占位）----
pe3 = np.stack([diff, cover, trans], axis=0).copy()
pe3[:, :10] = -np.inf                      # 前 10 个 level incomplete（所有 estimator）
pe3[0, 10:15] = -np.inf                    # 部分 estimator incomplete
pe3 = jnp.asarray(pe3)
mixed, w3, b3 = mix_scores(pe3, 1.0)
mf = np.asarray(mixed)
check("NaN安全: 全incomplete位 = -inf", np.all(np.isneginf(mf[:10])))
check("NaN安全: finite 位无 NaN/inf", np.all(np.isfinite(mf[15:])))
check("NaN安全: bids/w 无 NaN", np.all(np.isfinite(np.asarray(b3))) and np.all(np.isfinite(np.asarray(w3))))


# ---- 场景④：estimator 无关性（N=5 也能跑）----
np.random.seed(2)
pe5 = jnp.asarray(np.random.randn(5, M) * np.array([[1e3], [1e-3], [1], [10], [0.01]]))
mixed5, w5, b5 = mix_scores(pe5, 1.0)
check("estimator无关: N=5 权重和为1", abs(float(np.asarray(w5).sum()) - 1) < 1e-5)
check("estimator无关: N=5 mixed 形状对 (M,)", mixed5.shape == (M,))


# ---- 汇总 ----
n_pass = sum(1 for _, ok in results if ok)
print(f"\n{'='*40}\n{n_pass}/{len(results)} 通过")
if n_pass != len(results):
    print("失败项:", [n for n, ok in results if not ok])
    raise SystemExit(1)
print("全部通过 ✓")
