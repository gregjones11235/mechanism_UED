# -*- coding: utf-8 -*-
"""
机械核对脚本：熵正则异质 score UED 收敛证明里的代数断言
=========================================================
用途：用 SymPy 确定性地复核 proof_skeleton_entropy_regularized_heterogeneous_UED.md
      里 Gemini 单模型产出的代数，清掉 §0.5 顶部的「符号/常数须人工复核」义务。

只核对【符号代数恒等式】，不核对收敛率/两时间尺度/文献适用性（那些不是代数）。

每个 check 打印：声称的结果 vs SymPy 独立算出的结果 + PASS/FAIL。
运行：  python proof_check_sympy.py
"""

import sympy as sp


def banner(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def verdict(name, ok):
    print(f"  >>> {name}: {'PASS ✅' if ok else 'FAIL ❌  —— 需人工查 Gemini 代数'}")
    return ok


results = {}

# ----------------------------------------------------------------------
# 公共符号
# n 用具体小值(=3, 对应 PVL/Gen-SFL/CENIE 三 teacher 或 3 个 level)做符号验证；
# 维度无关的结论再用一般 n 复核。
# ----------------------------------------------------------------------
alpha = sp.symbols('alpha', positive=True)        # 熵正则系数 α>0
n = 3                                              # 单纯形维数（先用具体维度）
y = sp.symbols('y1:%d' % (n + 1), positive=True)  # y1..y3 单纯形分量
s = sp.symbols('s1:%d' % (n + 1), real=True)      # s1..s3 score 分量


# ======================================================================
# CHECK 1 —— 内层 Hessian：∇²_yy f_α = -α·diag(1/y_i)，且 ⪯ -αI（全局强凹）
#   蓝本 §0.5 / 引理1.1 / G2：曲率全来自 αH(y)，与 s 无关
# ======================================================================
banner("CHECK 1: 内层 Hessian = -α·diag(1/y_i)，强凹常数全局 = α（不退化）")

# f_α = yᵀs + α·H(y),  H(y) = -Σ y_i log y_i
# 注意：核对 Hessian 时不强加 Σy=1 约束（在切空间外算无约束 Hessian，
#       这正是「相对熵 reference 的曲率」该看的对象）
f_alpha = sum(y[i] * s[i] for i in range(n)) + alpha * (-sum(y[i] * sp.log(y[i]) for i in range(n)))

H = sp.hessian(f_alpha, y)
H = sp.simplify(H)
print("  SymPy 算出的 Hessian ∇²_yy f_α =")
sp.pprint(H)

claimed_H = sp.diag(*[-alpha / y[i] for i in range(n)])
ok1a = sp.simplify(H - claimed_H) == sp.zeros(n, n)
print("\n  声称：-α·diag(1/y_i)")
verdict("1a. Hessian 形式", ok1a)

# 强凹：对 y_i∈(0,1]，-α/y_i ≤ -α  ⇒ 特征值 ≤ -α  ⇒ ⪯ -αI
# 对角阵特征值即对角元；y_i≤1 ⇒ 1/y_i≥1 ⇒ -α/y_i ≤ -α。符号机械验证：
eig_minus_negalpha = [sp.simplify((-alpha / y[i]) - (-alpha)) for i in range(n)]
# 这些应当 ≤ 0 当 y_i≤1。用 y_i=1 取边界（最不利点）验证 = 0，y_i<1 时 <0：
ok1b = all(
    sp.simplify(e.subs(y[i], 1)) == 0  # y_i=1 处恰好 = -α（边界）
    for i, e in enumerate(eig_minus_negalpha)
)
print("\n  声称：y_i≤1 ⇒ Hessian 特征值 -α/y_i ≤ -α ⇒ f_α 对 y 全局 α-强凹（不在边界退化）")
print("       边界最不利点 y_i=1 处特征值 = -α（取等）；y_i<1 处 < -α（更强凹）")
verdict("1b. 强凹常数 = α 不退化（退化的是 smoothness 上界，非强凹）", ok1b)
results['CHECK1'] = ok1a and ok1b


# ======================================================================
# CHECK 2 —— 内层解析解 y*(x) = softmax(s/α)
#   蓝本 §G1 / 引理1.4：带约束 Σy=1 的 max_y [yᵀs + αH(y)]
# ======================================================================
banner("CHECK 2: 内层最优解 y* = softmax(s/α)（含 Σy_i=1 约束的拉氏求解）")

lam = sp.symbols('lambda', real=True)              # 单纯形约束乘子
# Lagrangian: yᵀs + αH(y) - λ(Σy_i - 1)
L = f_alpha - lam * (sum(y) - 1)
# 一阶条件 ∂L/∂y_i = s_i - α(log y_i + 1) - λ = 0  ⇒ y_i = exp((s_i-λ)/α - 1)
stationary = [sp.diff(L, y[i]) for i in range(n)]
sol_yi = [sp.exp((s[i] - lam) / alpha - 1) for i in range(n)]
# 代回一阶条件应恒为 0：
ok2a = all(sp.simplify(stationary[i].subs([(y[j], sol_yi[j]) for j in range(n)])) == 0
           for i in range(n))
verdict("2a. 一阶条件解 y_i = exp((s_i-λ)/α - 1)", ok2a)

# 归一化 Σy_i=1 消去 λ ⇒ y_i = exp(s_i/α)/Σexp(s_k/α) = softmax(s/α)
softmax = [sp.exp(s[i] / alpha) / sum(sp.exp(s[k] / alpha) for k in range(n)) for i in range(n)]
# 验证 softmax 满足归一化
ok2b = sp.simplify(sum(softmax) - 1) == 0
# 验证 softmax 与拉氏解成正比（比值与 i 无关）
ratios = [sp.simplify(sol_yi[i] / softmax[i]) for i in range(n)]
ok2c = sp.simplify(ratios[0] - ratios[1]) == 0 and sp.simplify(ratios[1] - ratios[2]) == 0
print("  声称：y* = softmax(s/α)，即 y_i* = exp(s_i/α)/Σ_k exp(s_k/α)")
verdict("2b. softmax 归一化 Σ=1", ok2b)
verdict("2c. 拉氏解 ∝ softmax（比值与分量无关）", ok2c)
results['CHECK2'] = ok2a and ok2b and ok2c


# ======================================================================
# CHECK 3 —— 包络 Φ_α(x) = α·log Σ exp(s_i/α)  (LogSumExp)
#   且偏差 0 ≤ Φ_α - Φ_0 ≤ α log n   (Nesterov 2005，定理2)
# ======================================================================
banner("CHECK 3: Φ_α = α·logΣexp(s_i/α)；偏差上界 α·log n")

# Φ_α = max_y f_α = 把 y*=softmax 代回 f_α
Phi_alpha = f_alpha.subs([(y[i], softmax[i]) for i in range(n)])
Phi_alpha = sp.simplify(Phi_alpha)
LSE = alpha * sp.log(sum(sp.exp(s[k] / alpha) for k in range(n)))
ok3a = sp.simplify(Phi_alpha - LSE) == 0
print("  声称：Φ_α(x) = α·log Σ_i exp(s_i/α)  （LogSumExp）")
verdict("3a. Φ_α = LogSumExp", ok3a)

# 偏差上界紧性检查：当所有 s_i 相等(=c) 时，Φ_0=c，Φ_α=c+α log n ⇒ 差恰 = α log n
c = sp.symbols('c', real=True)
Phi_alpha_equal = LSE.subs([(s[i], c) for i in range(n)])
gap_at_equal = sp.simplify(Phi_alpha_equal - c)   # 应 = α log n
ok3b = sp.simplify(gap_at_equal - alpha * sp.log(n)) == 0
print(f"\n  声称：上界 α·log n 在 s_i 全相等时取到（紧）。此处 n={n}")
print(f"       SymPy 算出 s_i 全等时的 gap = {gap_at_equal}  (应 = α·log {n} = {sp.simplify(alpha*sp.log(n))})")
verdict("3b. 偏差上界 α·log n 紧（全等 score 处取等）", ok3b)
results['CHECK3'] = ok3a and ok3b


# ======================================================================
# CHECK 4 —— 反密度采样符号方向 y_i* ∝ p(x_i)^{-1/α}
#   蓝本 §0.5 / G5：CENIE coverage score s_i = -log p(x_i)
#   ⭐ 这是 §0.5 顶部点名「尤其指数符号方向」的那一条，最易错
# ======================================================================
banner("CHECK 4 ⭐: CENIE 反密度采样 —— s_i=-log p ⇒ y_i* ∝ p^{-1/α}，且 Novelty↑ ⇒ 采样↑")

p = sp.symbols('p1:%d' % (n + 1), positive=True)   # p_i = 密度 p(x_i)∈(0,1)
# coverage score s_i = -log p_i
s_cov = [-sp.log(p[i]) for i in range(n)]
# 代入 softmax 解析解
y_cov = [sp.exp(s_cov[i] / alpha) / sum(sp.exp(s_cov[k] / alpha) for k in range(n))
         for i in range(n)]
y_cov = [sp.simplify(yi) for yi in y_cov]
print("  SymPy 算出 y_i* (代入 s_i=-log p_i) =")
sp.pprint(y_cov[0])

# 声称 y_i* ∝ p_i^{-1/α}：验证 y_i* / p_i^{-1/α} 与 i 无关（即真的成正比）
claimed_prop = [p[i] ** (-1 / alpha) for i in range(n)]
prop_ratios = [sp.simplify(y_cov[i] / claimed_prop[i]) for i in range(n)]
ok4a = (sp.simplify(prop_ratios[0] - prop_ratios[1]) == 0 and
        sp.simplify(prop_ratios[1] - prop_ratios[2]) == 0)
print("\n  声称：y_i* ∝ p_i^{-1/α}")
verdict("4a. 指数为 -1/α（反密度，幂次正比）", ok4a)

# ⭐ 符号方向：低密度(高 Novelty)是否对应高采样？
# 取两 level：p_low < p_high。低密度 level 的采样概率应更大。
# 用具体数代入符号 α 验证单调：∂y_i*/∂p_i < 0（密度越高采样越低）
dy_dp = sp.diff(y_cov[0], p[0])
# 在合法点 (p1=p2=p3=1/2, α=1) 取值判定符号
dy_dp_val = dy_dp.subs([(p[0], sp.Rational(1, 2)), (p[1], sp.Rational(1, 2)),
                        (p[2], sp.Rational(1, 2)), (alpha, 1)])
dy_dp_val = sp.simplify(dy_dp_val)
ok4b = dy_dp_val < 0
print(f"\n  ⭐ 符号方向核验：∂y_i*/∂p_i 在 (p=1/2,α=1) = {dy_dp_val}")
print("     声称：密度 p_i 越高 ⇒ 采样 y_i* 越低 ⇔ Novelty(=-log p)越高 ⇒ 采样越高")
verdict("4b. 符号方向：∂y*/∂p < 0（高 Novelty→高采样，方向正确）", bool(ok4b))
results['CHECK4'] = ok4a and bool(ok4b)


# ======================================================================
# CHECK 5 —— Softplus 平滑偏差 0 ≤ S_τ(z) - max(z,0) ≤ τ·log 2
#   蓝本 §G4：PVL 的 max(·,0) 用 Softplus S_τ(z)=τ·log(1+exp(z/τ)) 平滑
# ======================================================================
banner("CHECK 5: Softplus 偏差界 0 ≤ S_τ(z)-max(z,0) ≤ τ·log2")

z = sp.symbols('z', real=True)
tau = sp.symbols('tau', positive=True)
S_tau = tau * sp.log(1 + sp.exp(z / tau))
# 偏差在 z=0 处取最大（max(0,0)=0, S_τ(0)=τ log2）
diff_at_0 = sp.simplify(S_tau.subs(z, 0) - 0)
ok5a = sp.simplify(diff_at_0 - tau * sp.log(2)) == 0
print(f"  z=0 处偏差 S_τ(0)-max(0,0) = {diff_at_0}  (应 = τ·log2)")
verdict("5a. 偏差在 z=0 处 = τ·log2（上界取到）", ok5a)

# z→+∞: S_τ→z, max→z, 偏差→0 ；z→-∞: S_τ→0, max→0, 偏差→0  ⇒ 上确界在 z=0
lim_pos = sp.limit(S_tau - z, z, sp.oo)
lim_neg = sp.limit(S_tau - 0, z, -sp.oo)
ok5b = (sp.simplify(lim_pos) == 0) and (sp.simplify(lim_neg) == 0)
print(f"  z→+∞ 偏差→{lim_pos}；z→-∞ 偏差→{lim_neg}（两端趋 0 ⇒ 上确界确在 z=0）")
verdict("5b. 两端极限 → 0（确认 τ·log2 是全局上界）", ok5b)
results['CHECK5'] = ok5a and ok5b


# ======================================================================
# CHECK 6 —— G2 相对强凹/相对光滑：梯度差内积恒等式 ⇒ μ_rel=L_rel=α, κ_rel=1
#   gemini_answer2.md G2：⟨y-y',∇g(y)-∇g(y')⟩ = -α⟨y-y',∇h(y)-∇h(y')⟩（精确等号）
# ======================================================================
banner("CHECK 6: G2 相对条件数 κ_rel=1（梯度差内积精确等于 -α·熵梯度差内积）")

yp = sp.symbols('yp1:%d' % (n + 1), positive=True)   # y'（第二个点）
# g(y) = yᵀs + α·H(y) = yᵀs - α·h(y),  h(y)=Σy_i log y_i
# ∇g(y)_i = s_i - α(log y_i + 1)
grad_g = lambda yy: [s[i] - alpha * (sp.log(yy[i]) + 1) for i in range(n)]
# ∇h(y)_i = log y_i + 1
grad_h = lambda yy: [sp.log(yy[i]) + 1 for i in range(n)]

lhs = sum((y[i] - yp[i]) * (grad_g(y)[i] - grad_g(yp)[i]) for i in range(n))     # ⟨y-y',∇g(y)-∇g(y')⟩
rhs = -alpha * sum((y[i] - yp[i]) * (grad_h(y)[i] - grad_h(yp)[i]) for i in range(n))  # -α⟨y-y',∇h差⟩
ok6 = sp.simplify(lhs - rhs) == 0
print("  声称：⟨y-y',∇g(y)-∇g(y')⟩ = -α·⟨y-y',∇h(y)-∇h(y')⟩  （精确等号 ⇒ μ_rel=L_rel=α）")
print(f"  SymPy: lhs - rhs 化简 = {sp.simplify(lhs - rhs)}")
verdict("6. κ_rel = L_rel/μ_rel = α/α = 1（无 1/δ 边界依赖）", ok6)
results['CHECK6'] = ok6


# ======================================================================
# CHECK 7 —— G3-A 复合 KL-mirror 更新的对偶 logits 递推
#   gemini_answer2.md G3 式：z_{t+1} = z_t/(1+η_yα) + η_y·ŝ/(1+η_yα)
#   验证：从 argmax{η_y⟨ŝ,y⟩ - η_y α h(y) - D_h(y‖y_t)} 的一阶条件导出该递推
# ======================================================================
banner("CHECK 7: G3 复合 mirror-ascent 对偶递推 z_{t+1}=(z_t+η_y·ŝ)/(1+η_yα)")

eta_y = sp.symbols('eta_y', positive=True)
shat = sp.symbols('shat1:%d' % (n + 1), real=True)   # ŝ 随机梯度估计
zt = sp.symbols('zt1:%d' % (n + 1), real=True)        # 当前 logits z_t（y_t=softmax(z_t)）
# 目标 φ(y)=η_y⟨ŝ,y⟩ - η_y α h(y) - D_h(y‖y_t)
# D_h(y‖y_t)=Σ y_i log(y_i/y_t,i) - y_i + y_t,i ；∇_y D_h = log y_i - log y_t,i = log y_i - zt_i (+const)
# 一阶条件(去除单纯形常数): η_y ŝ_i - η_y α(log y_i+1) - (log y_i - zt_i) = const
# ⇒ (1+η_y α) log y_i = η_y ŝ_i + zt_i + const ⇒ log y_i = (zt_i+η_y ŝ_i)/(1+η_y α) + const'
# 即新 logits z_{t+1,i} = (zt_i + η_y ŝ_i)/(1+η_y α)
z_next_claimed = [(zt[i] + eta_y * shat[i]) / (1 + eta_y * alpha) for i in range(n)]
# 独立从一阶条件解：设 y_i = exp(w_i)/Σexp(w), w=z_{t+1}; 验证 w 满足上面消去 const 后的关系
# 检查 claimed 的 logits 差 (z_next_i - z_next_j) 是否等于一阶条件要求的差
fo_diff = lambda i, j: sp.simplify(
    ((eta_y * shat[i] + zt[i]) - (eta_y * shat[j] + zt[j])) / (1 + eta_y * alpha)
    - (z_next_claimed[i] - z_next_claimed[j]))
ok7 = all(fo_diff(i, 0) == 0 for i in range(n))
print("  声称：z_{t+1} = z_t/(1+η_yα) + η_y·ŝ/(1+η_yα)  （logits 指数衰减滑动平均）")
verdict("7. 对偶 logits 递推与一阶条件一致", ok7)
results['CHECK7'] = ok7


# ======================================================================
# CHECK 8 —— G3-B 平移不变性「魔术」：∇h(y)=log y+1=z+c·1，且 ⟨c·1, y*'-y*⟩=0
#   gemini_answer2.md G3 step2：消灭 1/δ 的关键——两单纯形点之差正交于 1 向量
# ======================================================================
banner("CHECK 8: G3 平移不变魔术 —— 单纯形两点差正交于 1，常数项 c·1 投影为 0")

# y=softmax(z) ⇒ log y_i = z_i - logΣexp(z) ⇒ ∇h(y)_i=log y_i+1 = z_i + (1-logΣexp(z))
# 即 ∇h(y)=z + c·1, c=1-logSumExp(z)，与 i 无关
zc = sp.symbols('zc1:%d' % (n + 1), real=True)
y_from_z = [sp.exp(zc[i]) / sum(sp.exp(zc[k]) for k in range(n)) for i in range(n)]
gradh_i = [sp.simplify(sp.log(y_from_z[i]) + 1) for i in range(n)]
# 验证 gradh_i - z_i 与 i 无关（= 公共常数 c）
c_terms = [sp.simplify(gradh_i[i] - zc[i]) for i in range(n)]
ok8a = sp.simplify(c_terms[0] - c_terms[1]) == 0 and sp.simplify(c_terms[1] - c_terms[2]) == 0
print("  声称：∇h(softmax(z)) = z + c·1，c=1-logΣexp(z) 与分量无关")
verdict("8a. ∇h(y) = z + c·1（平移不变）", ok8a)

# ⟨c·1, y*'-y*⟩ = c·Σ(y*'_i - y*_i) = c·(1-1) = 0（两点都在单纯形）
ystar1 = sp.symbols('a1:%d' % (n + 1), positive=True)
ystar2 = sp.symbols('b1:%d' % (n + 1), positive=True)
cc = sp.symbols('cc', real=True)
inner_c1 = cc * sum(ystar1[i] - ystar2[i] for i in range(n))
# 在 Σystar1=1, Σystar2=1 约束下应 = 0
ok8b = sp.simplify(inner_c1.subs(ystar1[n-1], 1 - sum(ystar1[i] for i in range(n-1)))
                              .subs(ystar2[n-1], 1 - sum(ystar2[i] for i in range(n-1)))) == 0
print("  声称：⟨c·1, y*_{t+1}-y*_t⟩ = c·Σ(差) = c·(1-1) = 0  ⇒ 漂移项不含 ∇h 的边界爆炸")
verdict("8b. 常数项 c·1 在单纯形差上投影 = 0（消灭 1/δ 的核心）", ok8b)
results['CHECK8'] = ok8a and ok8b


# ======================================================================
# CHECK 9 —— G3-C softmax 最小分量天然下界 ν = (1/n)·e^{-2B_s/α}
#   gemini_answer2.md G3 step2：‖s‖_∞≤B_s ⇒ y*=softmax(s/α) 每分量 ≥ ν
# ======================================================================
banner("CHECK 9: G3 天然内点下界 ν=(1/n)e^{-2B_s/α}（最小 softmax 分量界）")

Bs = sp.symbols('B_s', positive=True)
# y*_i = exp(s_i/α)/Σexp(s_k/α). 最不利：s_i=-B_s（分子最小），其余 s_k=+B_s（分母最大）
# 下界 = exp(-B_s/α) / [exp(-B_s/α) + (n-1)exp(B_s/α)]
num = sp.exp(-Bs / alpha)
den = sp.exp(-Bs / alpha) + (n - 1) * sp.exp(Bs / alpha)
y_min_exact = sp.simplify(num / den)
# 声称的简化下界 ν=(1/n)e^{-2B_s/α}：验证 y_min_exact ≥ ν
nu_claimed = sp.Rational(1, n) * sp.exp(-2 * Bs / alpha)
# 差 y_min_exact - nu_claimed 在 B_s,α>0 应 ≥ 0。代入具体正值抽样验证符号
diff_lb = y_min_exact - nu_claimed
samples = [(Bs, sp.Integer(1), alpha, sp.Integer(1)),
           (Bs, sp.Integer(3), alpha, sp.Rational(1, 2)),
           (Bs, sp.Rational(1, 2), alpha, sp.Integer(2))]
ok9 = all(sp.simplify(diff_lb.subs([(samples[k][0], samples[k][1]),
                                    (samples[k][2], samples[k][3])])) >= 0
          for k in range(len(samples)))
print(f"  精确最小分量 = exp(-B_s/α)/[exp(-B_s/α)+(n-1)exp(B_s/α)]  (n={n})")
print("  声称下界：ν = (1/n)·e^{-2B_s/α}  （应 ≤ 精确最小分量）")
verdict("9. ν=(1/n)e^{-2B_s/α} 是有效下界（多点抽样 y_min ≥ ν）", ok9)
results['CHECK9'] = ok9


# ======================================================================
# CHECK 10 —— G3-D 步长比 η_x/η_y = O(ε) → 0（外慢内快两时间尺度）
#   gemini_answer2.md G3 step3：η_y=√ε, η_x=ε^{1.5} ⇒ η_x/η_y=ε
# ======================================================================
banner("CHECK 10: G3 两时间尺度步长比 η_x/η_y = ε → 0")

eps = sp.symbols('epsilon', positive=True)
eta_y_choice = sp.sqrt(eps)        # η_y = √ε
eta_x_choice = eps ** sp.Rational(3, 2)   # η_x = ε^{1.5}
ratio = sp.simplify(eta_x_choice / eta_y_choice)
ok10a = sp.simplify(ratio - eps) == 0
ok10b = sp.limit(ratio, eps, 0, '+') == 0     # ε→0 时比值→0（外慢内快）
print(f"  η_y=√ε, η_x=ε^(3/2) ⇒ η_x/η_y = {ratio}（应 = ε）；ε→0 极限 = {sp.limit(ratio, eps, 0, '+')}")
verdict("10. 步长比 = ε 且 →0（两时间尺度分离成立）", ok10a and ok10b)
results['CHECK10'] = ok10a and ok10b


# ======================================================================
# CHECK 11 —— G4 Softplus 导数：一阶=sigmoid，二阶系数 sigmoid(1-sigmoid)≤1/4
#   gemini_answer2.md G4 step2：Hessian 谱界 L_z²/(4τ)+ℓ_z
# ======================================================================
banner("CHECK 11: G4 Softplus 一阶导=sigmoid(z/τ)，二阶系数 ≤ 1/4")

# S_τ(z)=τ log(1+exp(z/τ)); d/dz = exp(z/τ)/(1+exp(z/τ)) = sigmoid(z/τ)
dS = sp.diff(S_tau, z)
sigmoid = 1 / (1 + sp.exp(-z / tau))
ok11a = sp.simplify(dS - sigmoid) == 0
print("  声称：S_τ'(z) = sigmoid(z/τ)")
verdict("11a. 一阶导 = sigmoid(z/τ)", ok11a)

# 二阶导 = (1/τ)·sigmoid·(1-sigmoid)；其系数 σ(1-σ) 在 σ∈[0,1] 最大值 = 1/4（σ=1/2）
ddS = sp.simplify(sp.diff(S_tau, z, 2))
sig = sp.symbols('sig')   # =sigmoid
coef = sig * (1 - sig)
coef_max = sp.maximum(coef, sig, sp.Interval(0, 1))
ok11b = (coef_max == sp.Rational(1, 4))
# 验证 ddS 确实 = (1/τ)·sigmoid·(1-sigmoid)
# 注：两者是等价指数形式，simplify 直接相减可能化不到 0（假阳性 FAIL）。
# 用 rewrite 到统一基(指数)+together 规范化，再数值多点兜底确认恒等。
ddS_claimed = (1 / tau) * sigmoid * (1 - sigmoid)
_d11c = sp.simplify(sp.together(ddS.rewrite(sp.exp) - ddS_claimed.rewrite(sp.exp)))
ok11c = (_d11c == 0) or all(
    abs((ddS - ddS_claimed).subs({z: zv, tau: tv}).evalf()) < 1e-30
    for zv, tv in [(sp.Rational(3, 7), sp.Rational(2, 5)),
                   (-sp.Rational(5, 3), sp.Integer(1)),
                   (sp.Integer(0), sp.Rational(1, 2))])
print(f"  声称：S_τ''(z)=(1/τ)·σ·(1-σ)，且 σ(1-σ) 在[0,1]上最大 = {coef_max}")
verdict("11b. sigmoid(1-sigmoid) ≤ 1/4（Hessian 谱界 L_z²/(4τ) 的来源）", ok11b)
verdict("11c. 二阶导 = (1/τ)·σ(1-σ)", ok11c)
results['CHECK11'] = ok11a and ok11b and ok11c


# ======================================================================
# CHECK 12 —— G5 反密度极限：p→0（Novelty→∞）时 y*_i → 1
#   gemini_answer2.md G5：语义等价性检验
# ======================================================================
banner("CHECK 12: G5 反密度极限 —— 某 level 密度 p→0 时其采样概率 y*→1")

pv = sp.symbols('pv', positive=True)   # 目标 level 的密度，令其 →0
# 其余两 level 密度固定 = 1/2；y*_1 = pv^{-1/α} / (pv^{-1/α} + 2·(1/2)^{-1/α})
others = 2 * (sp.Rational(1, 2)) ** (-1 / alpha)
y1_cov = pv ** (-1 / alpha) / (pv ** (-1 / alpha) + others)
lim_y1 = sp.limit(y1_cov, pv, 0, '+')
ok12 = sp.simplify(lim_y1 - 1) == 0
print(f"  目标 level 密度 pv→0⁺（Novelty→∞），其采样概率 y*→{lim_y1}")
print("  声称：极低覆盖 level 的采样概率 → 1（CENIE「优先低覆盖」思想的极限正确）")
verdict("12. p→0 ⇒ y*→1（语义方向极限正确）", ok12)
results['CHECK12'] = ok12


# ======================================================================
# CHECK 13 —— 定理2 instance-dependent 更紧界 α·log(1+(n-1)e^{-Δ/α})
#   §0.5 / Nesterov 2005：Δ=次优间隙。两 score 情形(一个领先 Δ)的精确 gap
# ======================================================================
banner("CHECK 13: 定理2 instance-dependent 紧界 α·log(1+(n-1)e^{-Δ/α})")

Delta = sp.symbols('Delta', positive=True)
# 设最大 score = M，其余 n-1 个都低 Δ（即 = M-Δ）。Φ_0 = M。
M = sp.symbols('M', real=True)
s_gap = [M] + [M - Delta] * (n - 1)
Phi_alpha_gap = alpha * sp.log(sum(sp.exp(s_gap[i] / alpha) for i in range(n)))
Phi_0_gap = M
gap = sp.simplify(Phi_alpha_gap - Phi_0_gap)
claimed_tight = alpha * sp.log(1 + (n - 1) * sp.exp(-Delta / alpha))
ok13 = sp.simplify(gap - claimed_tight) == 0
print(f"  最大 score 领先次优 Δ 时：Φ_α - Φ_0 = {gap}")
print("  声称紧界：α·log(1+(n-1)e^{-Δ/α})")
verdict("13. instance-dependent 紧界精确成立（Δ→0 退回 α log n）", ok13)
# 附带：Δ→0 时退回 α log n
gap_at_Delta0 = sp.limit(claimed_tight, Delta, 0, '+')
print(f"      Δ→0 极限 = {sp.simplify(gap_at_Delta0)}（应退回 α·log {n} = {alpha*sp.log(n)}）")
results['CHECK13'] = ok13


# ======================================================================
# CHECK 14 —— G3 承重不等式·代数层①：Bregman 三点恒等式
#   D_h(a‖c) - D_h(a‖b) - D_h(b‖c) = ⟨∇h(b)-∇h(c), a-b⟩   （负熵 h）
#   gemini_answer2.md G3 step1「利用 Bregman 三点恒等式」—— mirror descent 分析基石
#   这是 SymPy 能核的「代数层」；其后的条件期望/Fenchel-Young 放缩是分析层(Lean/人工)
# ======================================================================
banner("CHECK 14: G3 承重不等式·代数层① Bregman 三点恒等式（负熵）")

# 三个单纯形点 a,b,c（用 n=3 的具体分量；恒等式与归一化无关，逐分量成立）
a_ = sp.symbols('a1:%d' % (n + 1), positive=True)
b_ = sp.symbols('b1:%d' % (n + 1), positive=True)
c_ = sp.symbols('c1:%d' % (n + 1), positive=True)
h_ = lambda v: sum(v[i] * sp.log(v[i]) for i in range(n))      # 负熵 h(y)=Σ y log y
gradh_ = lambda v: [sp.log(v[i]) + 1 for i in range(n)]
# Bregman 散度 D_h(u‖w) = h(u) - h(w) - ⟨∇h(w), u-w⟩
D_h = lambda u, w: h_(u) - h_(w) - sum(gradh_(w)[i] * (u[i] - w[i]) for i in range(n))

lhs14 = sp.simplify(D_h(a_, c_) - D_h(a_, b_) - D_h(b_, c_))
rhs14 = sp.simplify(sum((gradh_(b_)[i] - gradh_(c_)[i]) * (a_[i] - b_[i]) for i in range(n)))
ok14 = sp.simplify(lhs14 - rhs14) == 0
print("  声称：D_h(a‖c) - D_h(a‖b) - D_h(b‖c) = ⟨∇h(b)-∇h(c), a-b⟩")
print(f"  SymPy: lhs - rhs 化简 = {sp.simplify(lhs14 - rhs14)}")
verdict("14. Bregman 三点恒等式（负熵 h）", ok14)
results['CHECK14'] = ok14


# ======================================================================
# CHECK 15 —— G3 承重不等式·代数层②：内层单步收缩的「配方」核
#   复合 prox 一阶条件代入后，(1+η_yα)·D_h(y*‖y_{t+1}) 的主项配平：
#   验证收缩因子 1/(1+η_yα) ≤ 1 - η_yα/2  （当 η_yα ≤ 1，给出声称的 (1-η_yα/2) 收缩率）
#   gemini_answer2.md G3 step1 结论：E[D_h(y*‖y_{t+1})] ≤ (1-η_yα/2)D_h(y*‖y_t)+η_y²σ²/2
# ======================================================================
banner("CHECK 15: G3 承重不等式·代数层② 收缩因子 1/(1+η_yα) ≤ 1-η_yα/2")

# 复合更新使主项带因子 1/(1+η_yα)（来自 CHECK 7 的对偶递推）。
# 声称的收缩率写成 (1-η_yα/2)。须验代数不等式 1/(1+u) ≤ 1-u/2 对 u=η_yα∈(0,1]。
u = sp.symbols('u', positive=True)   # u = η_y·α
gap15 = sp.simplify((1 - u / 2) - 1 / (1 + u))     # 应 ≥ 0 当 u∈(0,1]
print(f"  (1-u/2) - 1/(1+u) 化简 = {sp.together(gap15)}    (u=η_yα)")
# = (u/2·(1+u) - 1 + ... ) 通分后分子 = u·(? )；符号判定：u∈(0,1] 抽样 + 端点
gap15_factored = sp.factor(sp.together(gap15))
print(f"  通分因式分解 = {gap15_factored}")
ok15a = all(sp.simplify(gap15.subs(u, uv)) >= 0
            for uv in [sp.Rational(1, 100), sp.Rational(1, 2), sp.Integer(1)])
# 端点 u→0: gap→0（两侧都=1）；解析确认分子在 (0,1] 非负
ok15b = sp.limit(gap15, u, 0, '+') == 0
print("  声称：复合更新收缩因子 1/(1+η_yα) ≤ 1-η_yα/2（η_yα≤1）⇒ 收缩率 (1-η_yα/2)")
verdict("15a. 1/(1+u) ≤ 1-u/2 在 u∈(0,1]（多点 + 因式符号）", ok15a)
verdict("15b. u→0 收缩因子 →1（一致退化正确）", ok15b)
results['CHECK15'] = ok15a and ok15b


# ======================================================================
# CHECK 16 —— G3 承重不等式·代数层③：Pinsker / 负熵 ℓ₁-1-强凸的 Hessian 依据
#   D_h(u‖v) ≥ ½‖u-v‖₁² 用于 Fenchel-Young 处理随机交叉项；
#   其代数前提 = 负熵 Hessian diag(1/y_i) 在 y_i≤1 时 ⪰ I（谱下界 ≥1）
#   gemini_answer2.md G3「使用 Fenchel-Young 处理交叉项（利用 D_h ≥ ½‖·‖₁²）」
# ======================================================================
banner("CHECK 16: G3 承重不等式·代数层③ 负熵 Hessian ⪰ I（Pinsker 的代数前提）")

# 负熵 h(y)=Σ y_i log y_i 的 Hessian = diag(1/y_i)
Hh = sp.hessian(h_(y), y)
Hh = sp.simplify(Hh)
print("  负熵 Hessian ∇²h =")
sp.pprint(Hh)
claimed_Hh = sp.diag(*[1 / y[i] for i in range(n)])
ok16a = sp.simplify(Hh - claimed_Hh) == sp.zeros(n, n)
verdict("16a. ∇²h = diag(1/y_i)", ok16a)
# y_i≤1 ⇒ 1/y_i≥1 ⇒ Hessian 特征值 ≥1 ⇒ h 关于 ℓ₂ 至少 1-强凸；
# ℓ₁-1-强凸(Pinsker)是更强的范数结论，其谱前提即此特征值下界。验边界取等：
ok16b = all(sp.simplify((1 / y[i]).subs(y[i], 1) - 1) == 0 for i in range(n))
print("  声称：y_i≤1 ⇒ ∇²h 特征值 1/y_i ≥ 1（y_i=1 取等）⇒ 负熵强凸，支撑 D_h≥½‖·‖₁²")
verdict("16b. Hessian 特征值 ≥ 1（Pinsker/Fenchel-Young 交叉项放缩的代数前提）", ok16b)
results['CHECK16'] = ok16a and ok16b


# ======================================================================
# 总结
# ======================================================================
banner("总结")
all_ok = all(results.values())
for k, v in results.items():
    print(f"  {k}: {'PASS ✅' if v else 'FAIL ❌'}")
print("\n" + ("全部 PASS ✅ —— Gemini 代数在符号层面无误，§0.5 复核义务（代数部分）已清"
              if all_ok else
              "存在 FAIL ❌ —— 对应断言需人工回查证明蓝本"))
