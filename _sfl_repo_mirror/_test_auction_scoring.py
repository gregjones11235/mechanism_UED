"""auction 海选打分逻辑单测（CPU）—— 验证 get_learnability_set 的 auction top-K 选择。

抽出 jaxnav_sfl.py 里 auction 打分核心（per_est → mix_scores → argsort top-K），用合成
per-env 信号验证：①开关向后兼容 ②混合分 top-K 选择正确 ③auction_lambda 消融端点
（argmax/uniform 选出不同 level 集）④incomplete(-inf)不入选。

跑法（minimax/sfl env）： python _test_auction_scoring.py
"""
import numpy as np
import jax.numpy as jnp

from auction_bid import mix_scores as auction_mix_scores

PASS, FAIL = "PASS", "FAIL"
results = []
def check(name, cond):
    results.append((name, bool(cond))); print(f"[{PASS if cond else FAIL}] {name}")


def select_topk(per_est, auction_lambda, K, fallback_learnability=None, use_auction=True):
    """复刻 jaxnav_sfl.get_learnability_set 的打分+选择逻辑。"""
    if use_auction:
        mixed, w, bids = auction_mix_scores(per_est, float(auction_lambda))
        score = mixed
    else:
        score = fallback_learnability
    top = jnp.argsort(score)[-K:]
    return np.asarray(top), np.asarray(score)


# ── 合成 M 个 level 的 (difficulty, PVL) per-env 信号 ──
rng = np.random.default_rng(0)
M, K = 500, 100
p = rng.uniform(0, 1, M)
difficulty = -((p - 0.5) ** 2)                          # 难度匹配，p=0.5 最大(=0)
pvl = rng.exponential(0.5, M)                           # 模拟 positive_value_loss 量纲
per_est = jnp.stack([jnp.asarray(difficulty), jnp.asarray(pvl)], axis=0)   # (2, M)
learnability = p * (1 - p)


# ① 向后兼容：use_auction=False → 走 learnability top-K（与原逻辑一致）
top_base, score_base = select_topk(per_est, 1.0, K, fallback_learnability=jnp.asarray(learnability), use_auction=False)
ref_top = np.argsort(learnability)[-K:]
check("向后兼容: 关 auction → learnability top-K 与原逻辑一致", set(top_base) == set(ref_top))


# ② auction 混合分 top-K：mixed 是 z-score 后加权和，能选出 level
top_auc, score_auc = select_topk(per_est, 1.0, K, use_auction=True)
check("auction: 选出 K 个 level", len(set(top_auc)) == K)
check("auction: 混合分 top-K 与 learnability top-K 不同(信号不同维)", set(top_auc) != set(ref_top))


# ③ auction_lambda 消融端点：argmax(inf) vs uniform(大λ) 选出不同 level 集
top_argmax, _ = select_topk(per_est, float('inf'), K, use_auction=True)
top_uniform, _ = select_topk(per_est, 1e6, K, use_auction=True)
# argmax 端只认单 estimator 的 winner → 与 uniform(两信号均权)选出的集合应不同
overlap = len(set(top_argmax) & set(top_uniform))
check("消融: argmax(λ=inf) 与 uniform(λ大) 选出不同 level 集", overlap < K)
# argmax 端：mixed 退化为 z-score 后单 winner estimator 的分 → top-K 应= 该 estimator 的 top-K
# (winner = bids 最大者；难度 z 后与 pvl z 后比，难度方差更集中,具体 winner 由数据定，验集合差异即可)
check("消融: argmax 端 overlap 显著 < uniform (端点确实不同)", overlap < K * 0.95)


# ④ incomplete(-inf) 不入选 top-K
per_est_inc = np.stack([difficulty, pvl], axis=0).copy()
per_est_inc[:, :50] = -np.inf                           # 前 50 个 level 两 estimator 都 incomplete
top_inc, score_inc = select_topk(jnp.asarray(per_est_inc), 1.0, K, use_auction=True)
check("incomplete: -inf level 不入 top-K", not any(i < 50 for i in top_inc))
check("incomplete: 混合分 incomplete 位 = -inf", np.all(np.isneginf(score_inc[:50])))


n_pass = sum(1 for _, ok in results if ok)
print(f"\n{'='*40}\n{n_pass}/{len(results)} 通过")
if n_pass != len(results):
    print("失败项:", [n for n, ok in results if not ok]); raise SystemExit(1)
print("全部通过 ✓")
