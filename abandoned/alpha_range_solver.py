# -*- coding: utf-8 -*-
"""
α 可行域求解器（任务2，main.tex §3.6 "Feasible range of α" 的可执行参照）
==========================================================================
⭐ V4 更新（2026-06-17）：原版基于指数复杂度 K∝e^{2B_s/α}（已被推翻——见
   proof_skeleton §7.9：那个指数是记账失误，走对偶 LogSumExp 收紧为多项式）。
   本版改用 V4 的【多项式】复杂度。旧版的「可行域易空 / B_s<0.5 才非空」结论
   【已作废】——那是基于错误指数率的假警报。多项式率下可行域基本总非空。

给定 (ε_bias 偏差容忍, ε 优化精度, K_budget 计算预算, n 单纯形维数, E_clip 截断底板)，
解析算出可行域 [α_min, α_max]：

  α_max = (ε_bias − E_clip(C)) / log n                （偏差 ≤ ε_bias 的最大 α）
  α_min = Θ( (1/(K_budget·ε⁴))^{1/3} )                （预算内可优化的最小 α；闭式）

复杂度（V4 终判，proof_skeleton §7.9）：
  单样本：  K = O( G_s²σ²Δ_Φ / (α³ ε⁵) )
  大 batch=O(ε⁻²)： K = O( 1/(α³ ε⁴) )    ← 用此式定 α_min（与 NCC 同 ε 阶）
  【多项式 in 1/α，无 e^{2B_s/α}，无截断】

两个量严格区分：
  ε_bias = 正则化偏差容忍（Φ_α 偏离 Φ_0，eq:bias）
  ε      = 优化精度（驻点目标，eq:complexity）—— 与 ε_bias 不同
  E_clip = CENIE 截断引入、与 α 无关的偏差底板（PVL/learnability 取 0）

依赖：numpy。运行：python alpha_range_solver.py
"""

import numpy as np


# ----------------------------------------------------------------------
# 核心：多项式复杂度（V4）与各边界
# ----------------------------------------------------------------------
def K_complexity(alpha, eps, big_batch=True, G_s=1.0, sigma=1.0, Delta=1.0):
    """V4 多项式迭代复杂度。
    big_batch=True:  K = 1/(α³ε⁴)         （内层 batch=O(ε⁻²)，与 NCC 同 ε 阶）
    big_batch=False: K = G_s²σ²Δ/(α³ε⁵)   （单样本）
    """
    if big_batch:
        return 1.0 / (alpha**3 * eps**4)
    return G_s**2 * sigma**2 * Delta / (alpha**3 * eps**5)


def alpha_max(eps_bias, n, E_clip=0.0):
    """偏差侧上界：Φ_α-Φ_0 ≤ α·log n (+E_clip) ≤ ε_bias ⇒ α ≤ (ε_bias-E_clip)/log n。
    返回 None 表示截断底板已超偏差容忍（coverage teacher 的 C 设得太激进，不可接纳）。
    """
    numerator = eps_bias - E_clip
    if numerator <= 0:
        return None
    return numerator / np.log(n)


def alpha_min(eps, K_budget, big_batch=True, G_s=1.0, sigma=1.0, Delta=1.0):
    """预算侧下界：解 K(α) = K_budget 的【闭式】（多项式率，无需求根）。
    big_batch:  1/(α³ε⁴)=K_budget       ⇒ α_min = (1/(K_budget·ε⁴))^{1/3}
    single:     G²σ²Δ/(α³ε⁵)=K_budget   ⇒ α_min = (G²σ²Δ/(K_budget·ε⁵))^{1/3}
    """
    if big_batch:
        return (1.0 / (K_budget * eps**4)) ** (1.0/3.0)
    return (G_s**2 * sigma**2 * Delta / (K_budget * eps**5)) ** (1.0/3.0)


def alpha_star(n, K_budget, big_batch=True, G_s=1.0, sigma=1.0, Delta=1.0):
    """T1（α 单峰）解出的【优化层可达精度最优点】α*。
    ⭐ 这是 Stage 3 该取的 α 锚点——【不是】几何中点 √(α_min·α_max)。

    T1 红队终判（gemini scripts/.../T1_redteam_verdict.md，SymPy 核验）：
      可达精度 P(α)=偏差(α log n) + 预算内优化残差 ε_opt(α)，严格凸 ⇒ 单谷 ⇒ 单峰。
      α* = argmin P(α)。但残差对 α 的幂次随【单样本/大batch】不同，故 α* 两版公式不同：

      大batch  K=1/(α³ε⁴)        ⇒ ε_opt ∝ α^{-3/4} ⇒ α* = (3c/(4 log n))^{4/7}, c=K_budget^{-1/4}
      单样本    K=G²σ²Δ/(α³ε⁵)   ⇒ ε_opt ∝ α^{-3/5} ⇒ α* = (3c/(5 log n))^{5/8}, c=(G²σ²Δ/K_budget)^{1/5}

    ⚠ 几何中点 α_geo=√(α_min·α_max) 与 α* 标度律不同（α*∝K_budget^{-1/7} 或 ^{-1/8}），
       无代数必然联系、差距可任意大 ⇒ 取中点无理论依据、极可能错过真峰。务必用本函数。
    ⚠ L2（真实泛化单峰）证不出，仅工作假设 ⇒ 实操仍需在 α* 邻域补测 1 点验证（α 轴 ×3→×2）。
    """
    logn = np.log(n)
    if big_batch:
        c = K_budget ** (-1.0/4.0)                       # ε_opt = c·α^{-3/4}
        return (3.0 * c / (4.0 * logn)) ** (4.0/7.0)
    c = (G_s**2 * sigma**2 * Delta / K_budget) ** (1.0/5.0)  # ε_opt = c·α^{-3/5}
    return (3.0 * c / (5.0 * logn)) ** (5.0/8.0)


def feasible_range(eps_bias, eps, n, K_budget, E_clip=0.0, big_batch=True,
                   G_s=1.0, sigma=1.0, Delta=1.0):
    """返回 (α_min, α_max, status_str)。"""
    a_max = alpha_max(eps_bias, n, E_clip)
    a_min = alpha_min(eps, K_budget, big_batch, G_s, sigma, Delta)
    if a_max is None:
        return (a_min, None,
                "EMPTY: 截断底板 E_clip ≥ ε_bias（coverage teacher 的 C 太激进，须降 C）——"
                "这是 V4 下【唯一】会导致空区间的退化情形")
    if a_min > a_max:
        return (a_min, a_max,
                f"EMPTY: α_min={a_min:.4g} > α_max={a_max:.4g}（预算极小或 ε_bias 极严；"
                f"多项式率下加预算即可打开，因 α_min∝K_budget^(-1/3)）")
    return (a_min, a_max, f"OK: 可行域 [{a_min:.4g}, {a_max:.4g}]，在其内搜 2-3 个 α 点")


# ----------------------------------------------------------------------
# 示例 + 自检
# ----------------------------------------------------------------------
def _report(name, **kw):
    a_min, a_max, status = feasible_range(**kw)
    print(f"\n[{name}]")
    print(f"  输入: " + ", ".join(f"{k}={v}" for k, v in kw.items()))
    print(f"  α_max = {a_max if a_max is None else round(a_max, 6)}  (偏差侧)")
    print(f"  α_min = {round(a_min, 6)}  (预算侧, 多项式率闭式)")
    print(f"  → {status}")
    # T1 锚点：该取的 α* 与（被否决的）几何中点对照
    big = kw.get('big_batch', True)
    a_star = alpha_star(kw['n'], kw['K_budget'], big_batch=big,
                        G_s=kw.get('G_s', 1.0), sigma=kw.get('sigma', 1.0),
                        Delta=kw.get('Delta', 1.0))
    if a_max is not None:
        a_geo = np.sqrt(a_min * a_max)
        clipped = min(max(a_star, a_min), a_max)   # 落区间外则取最近端点（凸性）
        note = "" if a_min <= a_star <= a_max else "  (α* 在区间外 → 取最近端点)"
        print(f"  ★ α*  = {a_star:.6g}  ← T1 该取的锚点（取此点，非中点）{note}")
        print(f"    α_geo= {a_geo:.6g}  (几何中点，已被 T1 否决：与 α* 差 {a_star/a_geo:.2f}×)")
        print(f"    建议: 在 α*={clipped:.4g} 跑 1 点 + 邻域补 1 点验泛化（α 轴 ×3→×2）")


if __name__ == "__main__":
    print("=" * 72)
    print("α 可行域求解器 (V4 多项式率) —— main.tex §3.6 的可执行参照")
    print("=" * 72)

    # 情形1：PVL/learnability，无截断，中等预算 —— V4 下应非空（旧指数版这里曾为空！）
    _report("情形1: PVL/learnability, 大batch, 中预算",
            eps_bias=0.5, eps=0.1, n=100, K_budget=1e9, E_clip=0.0)

    # 情形2：coverage teacher，截断底板 E_clip 适中
    _report("情形2: coverage teacher, E_clip=0.1",
            eps_bias=0.5, eps=0.1, n=100, K_budget=1e9, E_clip=0.1)

    # 情形3：单样本率（ε⁻⁵），预算需更大
    _report("情形3: 单样本率 (ε^-5)",
            eps_bias=0.5, eps=0.1, n=100, K_budget=1e12, E_clip=0.0, big_batch=False)

    # 情形4：截断底板超偏差容忍 —— V4 下唯一的真空区间
    _report("情形4: E_clip ≥ ε_bias (C 太激进)",
            eps_bias=0.2, eps=0.1, n=100, K_budget=1e9, E_clip=0.25)

    # ---- 自检：边界与复杂度自洽 ----
    print("\n" + "=" * 72)
    print("自检")
    print("=" * 72)
    # α_min 处 K(α_min) 应 = K_budget
    am = alpha_min(0.1, 1e9, big_batch=True)
    K_at = K_complexity(am, 0.1, big_batch=True)
    print(f"  α_min={am:.6g}; K(α_min)={K_at:.6g}, K_budget=1e9")
    assert abs(np.log(K_at) - np.log(1e9)) < 1e-9, "α_min 闭式不自洽！"
    print("  ✅ K(α_min)=K_budget（闭式自洽）")
    # α_max 处偏差 = ε_bias
    amax = alpha_max(0.5, 100, 0.0)
    print(f"  α_max={amax:.6g}; α_max·log n={amax*np.log(100):.6g}, ε_bias=0.5")
    assert abs(amax*np.log(100) - 0.5) < 1e-12, "α_max 不自洽！"
    print("  ✅ α_max·log n=ε_bias（自洽）")
    # V4 关键定性：α_min 随预算多项式衰减（不再指数顶高）
    a1, a2 = alpha_min(0.1, 1e6), alpha_min(0.1, 1e12)
    print(f"  预算 1e6→1e12（×10⁶），α_min {a1:.4g}→{a2:.4g}（比值 {a1/a2:.1f}=100，即 K^(1/3)）")
    print("  ✅ α_min∝K_budget^(-1/3)：加预算即降 α_min、打开可行域（V4 多项式特性）")
    # T1：α* 是 P'(α)=0 的解（残差幂次 3/4），验 P'(α*)≈0
    ast = alpha_star(100, 1e9, big_batch=True)
    c_big = 1e9 ** (-1.0/4.0)
    P_prime = np.log(100) - 0.75 * c_big * ast ** (-7.0/4.0)
    print(f"  α*={ast:.6g}; P'(α*)={P_prime:.3e}")
    assert abs(P_prime) < 1e-9, "α* 不是 P'=0 的解！"
    print("  ✅ P'(α*)=0（α* 是可达精度的驻点，T1 单峰最优）")
    # T1 关键反转：α* 与几何中点标度律不同 ⇒ 随预算变化，二者比值漂移（无固定关系）
    amin_a, amax_a = alpha_min(0.1, 1e9), alpha_max(0.5, 100, 0.0)
    geo = np.sqrt(amin_a * amax_a)
    r1 = alpha_star(100, 1e9) / geo
    amin_b = alpha_min(0.1, 1e12); geo_b = np.sqrt(amin_b * amax_a)
    r2 = alpha_star(100, 1e12) / geo_b
    print(f"  α*/α_geo: K=1e9→{r1:.3f}, K=1e12→{r2:.3f}（比值随预算漂移 ⇒ 无固定关系）")
    print("  ✅ α* 与几何中点标度律不同（α*∝K^-1/7, α_geo∝α_min^0.5∝K^-1/6）：取中点无依据")
    print("\n全部自检通过 ✅")
    print("\n注：旧版『可行域易空 / B_s<0.5 才非空』结论已作废（基于错误指数率）。")
    print("    V4 多项式率下，除非截断底板 E_clip≥ε_bias，可行域基本总非空。")
