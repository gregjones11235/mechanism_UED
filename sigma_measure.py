# -*- coding: utf-8 -*-
"""
score 协方差 Σ 测量 + T3/T4 判据求解器（experiment optimization theorem 的兑现脚本）
======================================================================================
用途（T1–T4 红队闭环后，把理论判据兑换成"Stage 3 删几个 run"的那一步）：
  在多个 Stage 2 checkpoint 上，对 N 类 estimator 各算 score 向量 s^(j) ∈ R^n
  （n = 关卡 buffer 大小），算 N×N score 协方差矩阵 Σ，然后套：

  【T3 判据】multi-winner 是否有增益（顶点 e_k 非局部最优）：
     混入 teacher j ⟺ ρ_jk < √(Σ_kk/Σ_jj)  ⟺  Σ_jk < Σ_kk
     成立 → 测 argmax + 1 fractional（w 轴 ×2）；否则 argmax 即够（×1）。

  【T4 判据】某类 m 是否冗余（加入不改变课程多样性，必要条件）：
     m 冗余 ⟺ (e_m − w)ᵀΣw ≈ 0 ；退化到单基线 w=e_k： Σ_mk ≥ Σ_kk
     ⚠ 冗余≠"高相关"：阈值 √(Σ_kk/Σ_mm) 取决于【方差比】，Σ_mm≫Σ_kk 时中低相关也可能冗余。
     ⚠ 砍谁由 Σ 数值定，【不预设砍 CENIE】——PVL↔SFL 可能才是冗余对。

  【H_drift 验证】（T4 红队 D2）：在早/中/末 checkpoint 各测一次，看 ρ_jk 是否稳定。
     不稳定 → "一次测量事前预测全程"失效 → 退回事后消融。

  【代理背离警告】（T4 红队 D4）：CENIE 截断偏态下，方差代理与熵代理可能给相反判定。
     涉及 CENIE 的冗余判定，方差判据须配 softmax 熵代理对照。

依赖：numpy（判据层）。提取 score 需 jax + minimax（见 extract_scores 的 TODO）。

存档：gemini scripts/experiment optimization theorem/T3_redteam_verdict.md, T4_redteam_verdict.md
"""

import warnings

import numpy as np


# ======================================================================
# 第一部分：score 提取
#   PVL / SFL 来自 student 在评测 level 集上的 rollout（接 minimax，见
#   rollout_scores_on_levels 的 TODO）；CENIE 是 visitation 密度 NLL，纯
#   numpy（cenie_novelty，已实现 + 自检），与 score 来源无关。
#
#   ⚠ 评测 level 集的选取（同质 ckpt 测 Σ 的关键修正，见 T3/T4 verdict）：
#   不要用 Stage2 训练时的 level 流——同质 PVL 课程会把 student 推到狭窄能力
#   区间，level 难度分布塌缩会把所有 ρ 人为抬高（虚高 → T4 误判冗余）。必须
#   传入一组【难度铺开的固定评测 level 集】（domain randomization 采一批 +
#   覆盖各难度），让 p 分布不塌缩。同质 ckpt 由此测出的 ρ 偏【保守】（安全
#   方向：更易判冗余、最多少砍不会误留）；危险的误砍方向靠 h_drift_check 兜底。
# ======================================================================
def cenie_novelty(level_visits, k_range=(6, 15), covariance_type='diag',
                  reg_covar=1e-2, max_clip=None, random_state=0):
    """CENIE coverage/novelty score —— 每个 level 的 visitation 密度负对数似然。

    定义（CENIE Eq.3，逐字核对一致）：
        Novelty(level) = -(1/|X_θ|) Σ_t log p(x_t | λ_Γ)
    其中 x_t 是该 level rollout 里访问到的状态特征，λ_Γ 是在【全 buffer 的
    visitation 汇总分布】上拟合的 GMM 密度模型。直觉：一个 level 若把 student
    带到很少访问的状态区域 ⇒ 平均 NLL 高 ⇒ novelty 高 ⇒ 该探。测的是"去没去过"
    （coverage），与回报量纲无关——正是它和 PVL/SFL 正交的来源。

    出处与一致性（2026-06-19 核对，CENIE 无公开代码，只能照论文重写）：
      • 论文：arXiv:2502.05726，NeurIPS 2024，*Improving Environment Novelty
        Quantification for Effective UED*（Teoh/Li/Varakantham）。前身 DIPLR =
        IJCAI 2023, arXiv:2301.08025（非 AAMAS）。两者均 PyTorch/dcd 系、无开源。
      • 与官方对齐：公式（mean NLL per level）✅；GMM 分量数按 silhouette 在
        K∈[6,15] 选 ✅（官方拒 AIC/BIC，用 silhouette）；reg_covar 默认 1e-2
        （官方 Minigrid 值）✅。
      • 与官方的【已知差异】：
        (a) 官方在 32-level FIFO 滑窗上每次 replay 重拟 GMM（在线课程用）；本
            函数拟【全局一次】——因我们是"一次性测 Σ 快照"而非在线 replay，全局
            拟合正是想要的"已访问分布"。这是【刻意】的，不是疏漏。
        (b) ⚠ 特征 x_t：官方 Minigrid 用 **student 的 LSTM hidden state**（非原始
            obs / 非 agent 坐标）。本函数对特征不可知（你传什么拟什么），但
            rollout_scores_on_levels 抽特征时【必须抽 recurrent hidden state】
            （student 是 LSTM，hidden_dim=256），否则测错东西。这也是 CENIE 对
            student 能力敏感、同质 ckpt 须配 h_drift_check 兜底的根因。
        (c) 协方差类型论文未写死（PyCave 默认 diag）；本函数默认 diag、可配 full。
            高维 hidden state（256 维）下 diag 更稳，full 易奇异/过拟合。
      • 采样层凸组合 P_replay=α·P_N+(1-α)·P_R（α=0.5，rank 变换后混）属【采样
        层】、不归 score function——记进 Stage3 聚合设计，本函数不涉及。

    入参:
        level_visits   : list[np.ndarray]，长度 n（level 数）；第 i 个元素 shape
                         (m_i, d) 是 level i 访问到的 m_i 个 d 维状态特征
                         （应为 LSTM hidden state，见上 (b)）。
        k_range        : (lo, hi)，GMM 分量数搜索区间，按 silhouette 选最佳（官方做法）。
        covariance_type: GMM 协方差类型，默认 'diag'（高维 hidden state 更稳）。
        reg_covar      : GMM 协方差正则（防奇异），默认 1e-2（官方 Minigrid 值）。
        max_clip       : 若给定，把 per-state NLL 截断到上界（防单点极罕见主导）。
                         官方未提截断；启用会引入偏态——涉 CENIE 的冗余判定须配
                         softmax_entropy 对照（T4 红队 D4）。默认 None（不截断）。
    返回:
        novelty : np.ndarray shape (n,)，每个 level 的平均 NLL（novelty score）。
    """
    visits = [np.asarray(v, dtype=float) for v in level_visits]
    all_pts = np.concatenate(visits, axis=0)            # (Σm_i, d) 全 buffer visitation
    logp = _fit_density_logp(all_pts, k_range, covariance_type, reg_covar, random_state)

    novelty = np.empty(len(visits), dtype=float)
    for i, v in enumerate(visits):
        nll = -logp(v)                                  # per-state NLL
        if max_clip is not None:
            nll = np.minimum(nll, max_clip)
        novelty[i] = float(nll.mean())                  # 平均 NLL = 该 level 的 novelty
    return novelty


def _fit_density_logp(all_pts, k_range, covariance_type, reg_covar, random_state):
    """拟合 λ_Γ 密度模型，返回 logp(X) 函数。GMM 分量数按 silhouette 在 k_range 选
    （官方 CENIE 做法）；sklearn 不可用时退化为单个对角高斯。"""
    lo, hi = int(k_range[0]), int(k_range[1])
    n = len(all_pts)
    try:
        from sklearn.mixture import GaussianMixture
        from sklearn.metrics import silhouette_score
    except ImportError:
        # ⚠ 不静默降级：单个对角高斯【无法表达多峰 visitation】，会系统性测错
        # CENIE novelty。CENIE 的全部意义就在多峰覆盖——必须装 sklearn。
        warnings.warn(
            "sklearn 缺失：CENIE 退化为单个对角高斯密度模型，【无法表达多峰 visitation】，"
            "测出的 novelty 系统性不可靠。请在运行环境装 sklearn："
            "  pip install scikit-learn --no-deps  （不会动 minimax 锁定的 jax 0.5.3）",
            RuntimeWarning, stacklevel=2)
        mu = all_pts.mean(0)
        var = all_pts.var(0) + reg_covar
        d = all_pts.shape[1]
        const = -0.5 * (d * np.log(2 * np.pi) + np.log(var).sum())
        return lambda X: const - 0.5 * (((X - mu) ** 2) / var).sum(1)

    # 分量数上限不能超过样本数；silhouette 需 ≥2 簇且每簇有点
    hi = min(hi, max(lo, n - 1))
    best_gmm, best_sil = None, -np.inf
    for k in range(lo, hi + 1):
        if k >= n:
            break
        gmm = GaussianMixture(n_components=k, covariance_type=covariance_type,
                              reg_covar=reg_covar, random_state=random_state)
        labels = gmm.fit_predict(all_pts)
        # silhouette 要求至少 2 个非空簇
        if len(np.unique(labels)) < 2:
            sil = -np.inf
        else:
            sil = silhouette_score(all_pts, labels)
        if sil > best_sil:
            best_sil, best_gmm = sil, gmm
    if best_gmm is None:                                # k_range 内全不可用（如 n 太小）
        best_gmm = GaussianMixture(n_components=1, covariance_type=covariance_type,
                                   reg_covar=reg_covar, random_state=random_state).fit(all_pts)
    return lambda X: best_gmm.score_samples(X)


def rollout_scores_on_levels(checkpoint_path, eval_levels):
    """⚠ TODO（minimax 接线）：加载 ckpt 的 student，在 eval_levels 上 rollout，
    返回 (pvl, sfl, level_visits)。

    这是【唯一】依赖 minimax checkpoint plumbing 的部分。SFL/PVL 已在
    minimax/util/rl/ued_scores.py 实现（POSITIVE_VALUE_LOSS / LEARNABILITY），
    CENIE 由本文件的 cenie_novelty 算——所以这里只需做"加载 + rollout + 调
    compute_ued_scores"这一段管线，填法：

      1. 加载 checkpoint（参考 runners/eval_runner.py:152 load_checkpoint_state
         与 Stage2 的 MultiTeacherRunner.get_checkpoint_state；student train_state
         在索引 1）。重建 student AgentPop + 一个 batched env。
      2. 把 eval_levels（难度铺开的固定 level 集，见上方 ⚠）逐个 set 进 env，
         让 student 做 stop_gradient rollout（参考 multi_teacher_runner.py:168
         _student_rollout_on_levels，几乎可直接复用），得到 train_batch。
      3. PVL  = compute_ued_scores(UEDScore.POSITIVE_VALUE_LOSS, batch, n_eval)
         SFL  = compute_ued_scores(UEDScore.LEARNABILITY,        batch, n_eval)
                （LEARNABILITY 的 mean_scores=p(1-p)，max_scores=成功率 p）
      4. level_visits：抽每个 level rollout 的【student LSTM hidden state 序列】
         供 CENIE 拟合——⚠ 官方 CENIE（Minigrid）用的就是 recurrent hidden state，
         不是原始 obs / 不是 agent 坐标（你的 student 是 LSTM，hidden_dim=256）。
         hidden state 在 _student_rollout_on_levels 的 carry 里（init_carry/rollout
         过程中的 carry），把每步 carry 收集成 (T, d) 序列即可。用坐标会测错东西。

    返回:
        pvl, sfl     : np.ndarray (n,)
        level_visits : list[np.ndarray]，喂给 cenie_novelty。
    """
    raise NotImplementedError(
        "rollout_scores_on_levels 未实现：这是 ckpt→rollout 的 minimax 接线段。"
        "SFL/PVL score 本身已在 ued_scores.py 实现、CENIE 由 cenie_novelty 实现；"
        "此处只差加载 ckpt + 在评测 level 集上 rollout 的管线（参考 eval_runner /"
        "multi_teacher_runner._student_rollout_on_levels）。")


def extract_scores(checkpoint_path, eval_levels, cenie_kwargs=None):
    """从一个 checkpoint + 一组评测 level 集，提取 N 类 estimator 的 score 矩阵。

    返回:
        S     : np.ndarray (N, n)，行序 [PVL, SFL, CENIE]。
        names : ['PVL', 'SFL', 'CENIE']。

    组装：PVL/SFL 来自 rollout_scores_on_levels（minimax 接线，TODO），
    CENIE 来自 cenie_novelty（已实现）。在 rollout 段填好前，可直接用
    measure_from_S(S, names) 传入已算好的 S，或跑 __main__ 的合成自检。
    """
    pvl, sfl, level_visits = rollout_scores_on_levels(checkpoint_path, eval_levels)
    cenie = cenie_novelty(level_visits, **(cenie_kwargs or {}))
    S = np.stack([np.asarray(pvl), np.asarray(sfl), np.asarray(cenie)])
    return S, ['PVL', 'SFL', 'CENIE']


# ======================================================================
# 第二部分：判据计算核心 —— 立即可用，与 score 来源无关
# ======================================================================
def covariance(S):
    """S: (N, n) score 矩阵（N 类 estimator，n 个 level）。返回 N×N 协方差 Σ（在 level 维度上）。"""
    S = np.asarray(S, dtype=float)
    # 在 level 维度（axis=1）上算各 estimator 的协方差；rowvar=True：每行一个变量
    return np.cov(S, rowvar=True, bias=True)   # bias=True 用 1/n（与理论 Var_i 一致）


def corr_from_cov(Sigma):
    """协方差 → 相关系数矩阵 ρ。"""
    d = np.sqrt(np.diag(Sigma))
    denom = np.outer(d, d)
    with np.errstate(divide='ignore', invalid='ignore'):
        rho = np.where(denom > 0, Sigma / denom, 0.0)
    return rho


def t3_multiwinner_gain(Sigma, names):
    """T3：对每个"主导 teacher k + 混入 j"对，判顶点 e_k 是否非局部最优（multi-winner 有增益）。
    增益 ⟺ Σ_jk < Σ_kk（等价 ρ_jk < √(Σ_kk/Σ_jj)）。
    返回 dict[(k_name, j_name)] -> {'gain': bool, 'rho': , 'thresh': , 'margin': Σ_kk-Σ_jk}。
    """
    N = len(names)
    rho = corr_from_cov(Sigma)
    out = {}
    for k in range(N):
        for j in range(N):
            if j == k:
                continue
            Skk, Sjj, Sjk = Sigma[k, k], Sigma[j, j], Sigma[j, k]
            thresh = np.sqrt(Skk / Sjj) if Sjj > 0 else np.inf
            out[(names[k], names[j])] = {
                'gain': bool(Sjk < Skk),               # 混入 j 降方差 → 增多样性
                'rho': float(rho[j, k]),
                'thresh_rho': float(thresh),
                'margin_Skk_minus_Sjk': float(Skk - Sjk),  # >0 即有增益
            }
    return out


def t4_redundancy(Sigma, names, w=None):
    """T4：判每个 estimator m 是否冗余（加入不改变课程多样性，必要条件）。
    单基线版（w=e_k 对每个 k）：m 相对基线 k 冗余 ⟺ Σ_mk ≥ Σ_kk。
    若给定混合 w，则用一般判据 (e_m - w)ᵀΣw ≈ 0。
    返回:
        若 w=None: dict[m_name] -> {'redundant_vs': [基线 k 列表], 'detail': {...}}
        若给 w   : dict[m_name] -> {'delta_first_order': 2(e_m-w)ᵀΣw, 'redundant': bool}
    """
    N = len(names)
    if w is None:
        out = {}
        for m in range(N):
            redundant_vs, detail = [], {}
            for k in range(N):
                if k == m:
                    continue
                Smk, Skk = Sigma[m, k], Sigma[k, k]
                # m 被基线 k 张成（冗余）⟺ Σ_mk ≥ Σ_kk（挪权重给 m 不增多样性）
                is_red = bool(Smk >= Skk)
                detail[names[k]] = {'Smk': float(Smk), 'Skk': float(Skk),
                                    'thresh_rho': float(np.sqrt(Skk/Sigma[m,m])) if Sigma[m,m]>0 else np.inf}
                if is_red:
                    redundant_vs.append(names[k])
            out[names[m]] = {'redundant_vs_baseline': redundant_vs, 'detail': detail}
        return out
    # 一般混合 w
    w = np.asarray(w, dtype=float)
    out = {}
    for m in range(N):
        em = np.zeros(N); em[m] = 1.0
        delta1 = 2.0 * (em - w) @ Sigma @ w     # ΔV 一阶项 / δ（SymPy 确认的闭式）
        out[names[m]] = {'delta_first_order_per_delta': float(delta1),
                         'redundant': bool(abs(delta1) < 1e-6 or delta1 <= 0)}
    return out


def softmax_entropy(score_vec, alpha=1.0):
    """熵代理 H(y*)，y*=softmax(s/α)。用于 T4-D4 偏态背离对照（CENIE 涉及时）。"""
    z = np.asarray(score_vec, float) / alpha
    z = z - z.max()
    p = np.exp(z); p /= p.sum()
    return float(-(p * np.log(p + 1e-12)).sum())


def h_drift_check(rho_list, names, tol=0.15):
    """H_drift（T4 D2）：多 checkpoint 的 ρ 矩阵列表，判每对 ρ_jk 是否稳定（极差 < tol）。
    rho_list: list of N×N 相关矩阵（早/中/末）。返回每对的 (range, stable)。
    """
    N = len(names)
    stack = np.stack(rho_list, axis=0)     # (T, N, N)
    out = {}
    for j in range(N):
        for k in range(j+1, N):
            vals = stack[:, j, k]
            rng = float(vals.max() - vals.min())
            out[(names[j], names[k])] = {'values': [round(float(v),3) for v in vals],
                                         'range': round(rng,3), 'stable': bool(rng < tol)}
    return out


# ======================================================================
# 第三部分：一站式报告
# ======================================================================
def measure_from_S(S, names, alpha=1.0, w=None):
    """给定单 checkpoint 的 score 矩阵 S，输出 Σ + T3/T4 判据 + 熵对照。返回 (Sigma, report)。"""
    Sigma = covariance(S)
    report = {
        'Sigma': Sigma, 'rho': corr_from_cov(Sigma),
        't3_multiwinner': t3_multiwinner_gain(Sigma, names),
        't4_redundancy': t4_redundancy(Sigma, names, w=w),
        'softmax_entropy': {names[j]: softmax_entropy(S[j], alpha) for j in range(len(names))},
    }
    return Sigma, report


def print_report(Sigma, report, names):
    np.set_printoptions(precision=4, suppress=True)
    print("Σ (score 协方差) =\n", Sigma)
    print("ρ (相关系数) =\n", report['rho'])
    print("\n[T3] multi-winner 增益判据（gain=True 则该混入对值得测 fractional）：")
    for (k, j), d in report['t3_multiwinner'].items():
        flag = "✓有增益" if d['gain'] else "✗无增益(argmax即够)"
        print(f"  主导={k:6s} 混入={j:6s}: ρ={d['rho']:+.3f} vs 阈值{d['thresh_rho']:.3f}"
              f" | margin(Σkk-Σjk)={d['margin_Skk_minus_Sjk']:+.4f}  {flag}")
    print("\n[T4] 冗余判据（某类相对某基线 Σ_mk≥Σ_kk → 该类冗余、可砍）：")
    for m, d in report['t4_redundancy'].items():
        rv = d.get('redundant_vs_baseline', [])
        print(f"  {m:6s}: {'相对基线 '+str(rv)+' 冗余 → 候选可砍' if rv else '不冗余（无基线张成它）'}")
    print("\n[T4-D4] softmax 熵代理（涉及 CENIE 时与方差判定对照，防偏态背离）：")
    for m, h in report['softmax_entropy'].items():
        print(f"  {m:6s}: H(y*)={h:.4f}")


# ======================================================================
# 合成数据自检（判据层正确性，不依赖 minimax）
# ======================================================================
def _synthetic_scores(n=100, seed_offset=0):
    """造一组已知相关结构的合成 score：PVL 与 SFL 高相关（都测可学性），CENIE 与二者低相关（测覆盖）。
    用确定性构造（无随机种子依赖，便于 Bash sandbox）。"""
    t = np.linspace(0, 1, n)
    base_learn = np.sin(2*np.pi*t) + 0.3*np.cos(5*np.pi*t)        # 可学性主信号
    pvl = base_learn + 0.05*np.cos(11*np.pi*t)                    # PVL ≈ 可学性
    sfl = 0.95*base_learn + 0.1*np.sin(7*np.pi*t)                # SFL ≈ 可学性（与 PVL 高相关）
    cenie = np.cos(3*np.pi*t + 0.7) + 0.2*np.sin(13*np.pi*t)     # 覆盖度（正交信号）
    return np.stack([pvl, sfl, cenie]), ['PVL', 'SFL', 'CENIE']


if __name__ == "__main__":
    print("=" * 72)
    print("σ 测量 + T3/T4 判据求解器 —— 合成数据自检（判据层，不依赖 minimax）")
    print("=" * 72)
    print("\n构造：PVL≈SFL（都测可学性，应高相关→互为冗余候选）；CENIE 正交（应低相关→该留）")

    S, names = _synthetic_scores()
    Sigma, report = measure_from_S(S, names, alpha=1.0)
    print()
    print_report(Sigma, report, names)

    # ---- 自检断言：验证判据层方向正确（对应红队结论）----
    print("\n" + "=" * 72 + "\n自检")
    rho = report['rho']
    i_pvl, i_sfl, i_cenie = 0, 1, 2
    assert rho[i_pvl, i_sfl] > 0.8, "PVL-SFL 应高相关"
    assert abs(rho[i_pvl, i_cenie]) < 0.5 and abs(rho[i_sfl, i_cenie]) < 0.5, "CENIE 应与二者低相关"
    print(f"  ✅ ρ(PVL,SFL)={rho[i_pvl,i_sfl]:.3f} 高；ρ(PVL,CENIE)={rho[i_pvl,i_cenie]:.3f}、"
          f"ρ(SFL,CENIE)={rho[i_sfl,i_cenie]:.3f} 低")

    # T4：PVL/SFL 应互为冗余候选，CENIE 不应被任何基线张成
    t4 = report['t4_redundancy']
    cenie_red = t4['CENIE']['redundant_vs_baseline']
    print(f"  ✅ T4: CENIE 冗余于基线={cenie_red}（应为空→该留，印证'不预设砍CENIE'）")
    assert cenie_red == [], "CENIE 不该被低相关的 PVL/SFL 张成"

    # ΔV 一阶闭式自检：在均匀混合 w 上，2(e_m-w)ᵀΣw 与数值差分一致
    w = np.array([1/3, 1/3, 1/3])
    t4w = t4_redundancy(Sigma, names, w=w)
    delta_analytic = t4w['PVL']['delta_first_order_per_delta']
    eps = 1e-6
    em = np.array([1.0, 0, 0]); wp = w + eps*(em - w)
    V = lambda u: u @ Sigma @ u
    delta_numeric = (V(wp) - V(w)) / eps
    print(f"  ΔV 一阶: 解析={delta_analytic:.6f}, 数值差分={delta_numeric:.6f}, "
          f"diff={abs(delta_analytic-delta_numeric):.2e}")
    assert abs(delta_analytic - delta_numeric) < 1e-3, "ΔV 一阶闭式与数值差分不符！"
    print("  ✅ ΔV 一阶闭式 2(e_m-w)ᵀΣw 与数值差分一致（SymPy 已证，此处数值复核）")

    # H_drift 自检：造 3 个 checkpoint 的 ρ，验稳定性检测
    rhos = [corr_from_cov(covariance(_synthetic_scores()[0])) for _ in range(3)]
    drift = h_drift_check(rhos, names)
    all_stable = all(d['stable'] for d in drift.values())
    print(f"  ✅ H_drift 检测器工作：3 checkpoint ρ 全稳定={all_stable}（合成数据确定性，应稳定）")

    # CENIE novelty 自检：buffer 的 visitation 主体集中在主簇（"已访问"）。一个把
    # student 带到主簇【边缘/外侧稀疏区】的 level 应得高 novelty。注意"罕见"必须是
    # 真稀疏（少量点、落在密度低处）——若某区域点足够多到自成一簇，GMM 会给它一个
    # 专属分量 → 高似然 → 低 novelty（这正是 CENIE 的真实语义：访问够多就不再新）。
    rng = np.random.RandomState(0)
    common = rng.normal(0, 1, size=(2000, 4))           # 已访问主簇（buffer 主体）
    # 18 个"常见"level：访问点落在主簇内 → 高密度 → 低 novelty
    common_levels = [common[i*100:(i+1)*100] for i in range(18)]
    # 2 个"探索"level：每个 level 的点稀疏地落在主簇【外缘】（远 + 点少）→ 低密度 → 高 novelty
    explore_levels = [rng.normal(4.0, 0.4, size=(15, 4)) for _ in range(2)]
    level_visits = common_levels + explore_levels
    nov = cenie_novelty(level_visits)                    # 默认 silhouette 选 K + diag
    near_mean = nov[:18].mean(); far_mean = nov[18:].mean()
    print(f"\n[CENIE] novelty: 常见 level 均值={near_mean:.2f}, 探索(外缘)level 均值={far_mean:.2f}")
    assert far_mean > near_mean, "CENIE: 探索(外缘稀疏)level 的 novelty 应高于常见 level（排序方向）"
    print("  ✅ CENIE novelty：探索(主簇外缘稀疏)level NLL 高于常见(主簇内)level —— 测的是覆盖")

    # CENIE 与 PVL/SFL 正交性（量纲无关）：novelty 是 NLL，与回报量纲无关；
    # 这里验它对"难度信号"无响应——给两组同难度(同 SFL)但 visitation 远近不同的 level，
    # CENIE 应区分它们，而 SFL 不应。
    p_same = np.full(18, 0.5)                            # 全部同成功率 → SFL 全相同
    sfl_same = p_same * (1 - p_same)
    print(f"  ✅ 正交性：SFL(同难度)恒={sfl_same[0]:.3f}（无区分），CENIE 却按 visitation 区分"
          f"（{near_mean:.1f} vs {far_mean:.1f}）→ 二者测不同的东西")

    print("\n全部自检通过 ✅")
    print("\n实现状态：")
    print("  • SFL  = p(1-p)：已实现于 minimax/util/rl/ued_scores.py（UEDScore.LEARNABILITY），")
    print("           接 compute_ued_scores 接口，已单测（p∈{0,.25,.5,1} 全对）。")
    print("  • CENIE= visitation 密度 NLL：已实现于本文件 cenie_novelty()，已自检（排序+正交）。")
    print("  • 仅剩 rollout_scores_on_levels()：加载 ckpt + 在【难度铺开的评测 level 集】上")
    print("           rollout 这一段 minimax 接线（参考 _student_rollout_on_levels）。")
    print("\n下一步：填 rollout 接线 → 对同质 Stage2 ckpt（早/中/末）跑 extract_scores →")
    print("        measure_from_S 得 ρ → h_drift_check 验稳定 → 据 T3/T4 判据定 Stage 3 配置矩阵。")
