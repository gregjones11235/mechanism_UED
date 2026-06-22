"""estimators.py 单元测试（CPU，验证信号数学 + 相关矩阵 NaN 安全 + 决策树判定）。

跑法（Oscar，sfl 环境）：  python _test_estimators.py
对照设计：mechanism_UED/方案B_多generator设计.md §5。
"""
import jax.numpy as jnp
import numpy as np

from estimators import (
    difficulty_match, learnability,
    assemble_signal_matrix, pairwise_correlation, orthogonality_report,
)

PASS, FAIL = "PASS", "FAIL"
results = []

def check(name, cond):
    results.append((name, bool(cond)))
    print(f"[{PASS if cond else FAIL}] {name}")


# ---- 信号①：difficulty_match = -(p-0.5)² 的方向/饱和性质 ----
p = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])
dm = np.asarray(difficulty_match(p))
check("difficulty: p=0.5 取最大 (=0)", abs(dm[2] - 0.0) < 1e-6 and dm[2] == dm.max())
check("difficulty: p=0 与 p=1 都 = -0.25", abs(dm[0] + 0.25) < 1e-6 and abs(dm[4] + 0.25) < 1e-6)
# 关键：difficulty 在 p=0 与 p=1 同值（都 -0.25）——它本身不分方向；方向体现在「梯度指向 0.5」
check("difficulty: 关于 p=0.5 对称", abs(dm[1] - dm[3]) < 1e-6)

# ---- 信号②：learnability = p(1-p)，p∈{0,1} 不可分(都0) ----
lb = np.asarray(learnability(p))
check("learnability: p=0/p=1 都 = 0（饱和/过难不可分）", abs(lb[0]) < 1e-6 and abs(lb[4]) < 1e-6)
check("learnability: p=0.5 取最大 (=0.25)", abs(lb[2] - 0.25) < 1e-6 and lb[2] == lb.max())

# ---- incomplete 标记：num_episodes=0 → -inf ----
ne = jnp.array([0, 3, 5, 0, 2])
dm_inc = np.asarray(difficulty_match(p, num_episodes=ne))
check("incomplete: num_episodes=0 位置 = -inf", np.isneginf(dm_inc[0]) and np.isneginf(dm_inc[3]))
check("incomplete: num_episodes>0 位置保留有限值", np.all(np.isfinite(dm_inc[[1, 2, 4]])))


# ---- assemble：堆叠 + 对齐校验 ----
np.random.seed(0)
M = 200
succ = np.random.uniform(0, 1, M)
s_dm = difficulty_match(jnp.asarray(succ))
s_lb = learnability(jnp.asarray(succ))
s_pvl = jnp.asarray(np.random.exponential(0.5, M))      # 模拟 positive_value_loss 真实量纲
s_cenie = jnp.asarray(np.random.uniform(50, 400, M))    # 模拟 cenie -log p 真实量纲
mat = assemble_signal_matrix([s_dm, s_lb, s_pvl, s_cenie])
check("assemble: 形状 (N=4, M)", mat.shape == (4, M))
try:
    assemble_signal_matrix([s_dm, s_lb[:M-1]])           # 长度不齐应报错
    check("assemble: 长度不齐报错", False)
except ValueError:
    check("assemble: 长度不齐报错", True)


# ---- 相关矩阵①：difficulty 与 learnability 在 success_rate 上的真实关系 ----
# 二者都是 success_rate 的函数，相关性高是真的（不是 bug）——这正是探针该如实报告的。
corr = np.asarray(pairwise_correlation(assemble_signal_matrix([s_dm, s_lb])))
check("corr: 对角 = 1", abs(corr[0, 0] - 1) < 1e-6 and abs(corr[1, 1] - 1) < 1e-6)
check("corr: 对称", abs(corr[0, 1] - corr[1, 0]) < 1e-6)
check("corr: |ρ| ≤ 1", np.all(np.abs(corr) <= 1.0 + 1e-6))

# 与 numpy 标准 Pearson 对照（finite 全集）
ref = np.corrcoef(np.asarray(s_dm), np.asarray(s_lb))[0, 1]
check("corr: 与 numpy corrcoef 一致", abs(float(corr[0, 1]) - ref) < 1e-4)


# ---- 相关矩阵②：构造已知正交/共线，验证判定 ----
np.random.seed(1)
a = np.random.randn(M)
b = np.random.randn(M)                                   # 与 a 独立 → ρ≈0
c = a * 3.0 + 0.01 * np.random.randn(M)                 # 与 a 近共线 → ρ≈1
pe = assemble_signal_matrix([jnp.asarray(a), jnp.asarray(b), jnp.asarray(c)])
cm = np.asarray(pairwise_correlation(pe))
check("corr: 独立信号 ρ≈0", abs(cm[0, 1]) < 0.15)
check("corr: 近共线信号 ρ≈1", cm[0, 2] > 0.95)


# ---- 相关矩阵③：NaN 安全（成对排除 incomplete）----
pe_inc = np.stack([a, b, c], axis=0).copy()
pe_inc[0, :20] = -np.inf                                 # est0 前 20 incomplete
pe_inc[1, 10:30] = -np.inf                               # est1 部分 incomplete（与 est0 部分重叠）
cm2 = np.asarray(pairwise_correlation(jnp.asarray(pe_inc)))
check("corr NaN安全: 无 NaN/inf", np.all(np.isfinite(cm2)))
check("corr NaN安全: 对角仍 = 1", np.all(np.abs(np.diag(cm2) - 1) < 1e-6))

# 零方差信号：与任何信号相关记 0（不误判正交也不误判共线）
const = np.full(M, 7.0)
cm3 = np.asarray(pairwise_correlation(assemble_signal_matrix([jnp.asarray(a), jnp.asarray(const)])))
check("corr: 零方差信号相关记 0", abs(cm3[0, 1]) < 1e-6)


# ---- 决策树判定：orthogonality_report ----
# 现成池有 3 个两两正交 → verdict = ready_pool_sufficient
np.random.seed(2)
o1, o2, o3 = np.random.randn(M), np.random.randn(M), np.random.randn(M)
rep_ok = orthogonality_report(
    assemble_signal_matrix([jnp.asarray(o1), jnp.asarray(o2), jnp.asarray(o3)]),
    names=["difficulty", "PVL", "CENIE"], threshold=0.5)
check("决策树: 三正交 → has_orthogonal_triple=True", rep_ok["has_orthogonal_triple"])
check("决策树: 三正交 → verdict=ready_pool_sufficient",
      rep_ok["verdict"] == "ready_pool_sufficient")

# 复现 maze 的 ρ(PVL,CENIE)=0.73 偏相关场景 → 无正交三元组 → 必须实现贵信号
base = np.random.randn(M)
pvl_c = base + 0.3 * np.random.randn(M)                  # 与 base 高相关
cenie_c = base + 0.3 * np.random.randn(M)                # 与 base 高相关 → 三者两两 ρ 高
rep_bad = orthogonality_report(
    assemble_signal_matrix([jnp.asarray(base), jnp.asarray(pvl_c), jnp.asarray(cenie_c)]),
    names=["difficulty", "PVL", "CENIE"], threshold=0.5)
check("决策树: 三者偏相关 → verdict=must_implement_expensive_signal",
      rep_bad["verdict"] == "must_implement_expensive_signal")
check("决策树: max_abs_offdiag 反映偏相关 (>0.5)", rep_bad["max_abs_offdiag"] > 0.5)


# ---- 汇总 ----
n_pass = sum(1 for _, ok in results if ok)
print(f"\n{'='*40}\n{n_pass}/{len(results)} 通过")
if n_pass != len(results):
    print("失败项:", [n for n, ok in results if not ok])
    raise SystemExit(1)
print("全部通过 ✓")
