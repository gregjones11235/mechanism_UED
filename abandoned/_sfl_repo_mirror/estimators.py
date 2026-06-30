"""异质 estimator 信号层 —— per-level score 计算 + V1 正交性探针工具（载体中立）。

设计依据：mechanism_UED/方案B_多generator设计.md §5（信号去留汇总表 + 数据驱动决策树）。
本模块只做两件事，**不碰 bid/权重分配**（那是 auction_bid.py 的职责，本模块输出喂给它）：

  1. 把 rollout outcomes 折成 N 个**现成信号**的 per-level score，组装成 (N, M) 矩阵；
  2. V1 探针核心：在该矩阵上算 estimator 两两 Pearson 相关——决策树用它裁决
     「现成池能否撑起 N=3 正交、要不要回头实现贵信号（transition-error 等）」(§5.4)。

—— 信号边界（务必看清，避免拿代理当真信号，否则相关矩阵是假的）——
本模块只内置「纯 outcome 即得、零额外网络头」的两个真信号：
  • difficulty_match  s = -(p-0.5)²   （§5.0 主力；p=per-level success_rate，有方向、饱和/过难可分）
  • learnability      s = p(1-p)        （SFL 原信号，退到打分层当 estimator 之一，§4.3）

另两个维度的真信号**不在本模块重写**，直接复用各自的官方实现，本模块只负责组装：
  • PVL（positive_value_loss，不确定性/regret 维度）：复用 sfl.util.jaxued.jaxued_utils.positive_value_loss
      —— SFL repo 单 critic 无 ensemble，故**不用** value_disagreement(values.std(-1))，
         那是跨 critic 群体的分歧，单 critic 下拿不到；PVL 是真实 UED 信号，非代理。
  • CENIE（覆盖/行为新颖维度）：复用官方 minimax cenie_density.cenie_neg_logp(GMM, hidden)
      —— 特征是 **recurrent hidden state**、密度是 **GMM**、score = -log p(h)（反密度）。
         注：SFL repo 的 ScannedRNN 用 **GRU**（carry 是单数组，非 LSTM 的 (c,h) 元组），
         hidden = GRU state (batch, HIDDEN_SIZE)，官方 CENIE(Minigrid) 同样用 recurrent hidden。
         **不用** outcome 空间 k-NN：低维 outcome 信息量远低于 HIDDEN_SIZE 维 hidden，是另一个信号不是退化版。

→ assemble_signal_matrix 接收**各信号已算好的 per-level 向量**（解耦：本模块不依赖
  cenie/PVL 的具体计算上下文，由调用方/探针把它们算好传进来），统一对齐成 (N, M) 喂 bid 层。

incomplete level（该 level 无完成 episode）用 -inf 占位，与 auction_bid.py 契约一致（NaN 安全）。
纯 JAX，可进 jitted 训练循环；相关矩阵工具是诊断用、可在 host 侧 numpy 跑。
"""
import jax
import jax.numpy as jnp


# ======================================================================
# 现成信号（纯 outcome 即得，零网络头）—— 本模块内置的两个真信号
# ======================================================================
def difficulty_match(success_rate, num_episodes=None, incomplete_value=-jnp.inf):
    """难度匹配信号 s = -(p-0.5)²，p = per-level success_rate。(§5.0 主力)

    有方向（p<0.5 太难 / p>0.5 太易，梯度指向 0.5）、饱和(p=1)与过难(p=0)可分（值不同，
    分别 -0.25），比 p(1-p) 多「方向」信息——这是不拿 p(1-p) 训 generator 的核心理由(§4)。
    打分层取最大值在 p=0.5（最该回放的 frontier level）。

    Args:
      success_rate: (M,) per-level 成功率 p ∈ [0,1]。
      num_episodes: (M,) 可选；该 level 完成的 episode 数，0 → 标记 incomplete(-inf)。
      incomplete_value: incomplete level 的占位值（默认 -inf，对齐 bid 契约）。
    Returns:
      s: (M,) per-level difficulty-match score；incomplete level = incomplete_value。
    """
    p = jnp.asarray(success_rate, jnp.float32)
    s = -((p - 0.5) ** 2)
    if num_episodes is not None:
        complete = jnp.asarray(num_episodes) > 0
        s = jnp.where(complete, s, incomplete_value)
    return s


def learnability(success_rate, num_episodes=None, incomplete_value=-jnp.inf):
    """SFL learnability 信号 s = p(1-p)（Bernoulli 方差）。p = per-level success_rate。

    SFL 原信号（Rutherford et al. 2024）。本方案不拿它训 generator（方向盲，§4.1），
    而是**退到打分层当 N 个 estimator 之一**（§4.3）参与 auction/相关性分析。
    p∈{0,1} 都给 0（饱和/过难不可分），p=0.5 最大。

    Args / Returns 同 difficulty_match。
    """
    p = jnp.asarray(success_rate, jnp.float32)
    s = p * (1.0 - p)
    if num_episodes is not None:
        complete = jnp.asarray(num_episodes) > 0
        s = jnp.where(complete, s, incomplete_value)
    return s


# ======================================================================
# 组装 (N, M) 信号矩阵 —— 喂给 auction_bid.mix_scores / 喂给探针算相关
# ======================================================================
def assemble_signal_matrix(per_level_signals):
    """把 N 个**已算好**的 per-level score 向量堆成 (N, M) 矩阵（第 0 维 = estimator）。

    解耦设计：本函数不计算任何信号，只对齐 + 堆叠。各信号由各自官方实现算好后传进来：
      difficulty_match / learnability  —— 本模块上面两个函数；
      PVL                              —— sfl.util.jaxued.jaxued_utils.positive_value_loss(dones, advantages)；
      CENIE                            —— minimax cenie_density.cenie_neg_logp(gmm, hidden)，
                                          再 reshape 成 per-level (M,)。
    这样本模块不依赖单/多 critic、recurrent/feedforward 等载体细节——谁有就传谁。

    Args:
      per_level_signals: 长度 N 的序列，每项 (M,)（同一组 level、同一顺序）。
                         incomplete level 各项内部已用 -inf 占位。
    Returns:
      per_est: (N, M)，可直接喂 auction_bid.standardize_per_estimator / mix_scores。
    """
    arrs = [jnp.asarray(s, jnp.float32).reshape(-1) for s in per_level_signals]
    M = arrs[0].shape[0]
    for i, a in enumerate(arrs):
        if a.shape[0] != M:
            raise ValueError(
                f"signal {i} 长度 {a.shape[0]} != signal 0 长度 {M}；"
                "所有信号必须对齐同一组 level、同一顺序。")
    return jnp.stack(arrs, axis=0)                       # (N, M)


# ======================================================================
# V1 探针核心：estimator 两两相关矩阵（§5.4 决策树的输入）
# ======================================================================
def pairwise_correlation(per_est):
    """N 个 estimator 在共同 finite level 上的 Pearson 相关矩阵。(§5.4 探针核心)

    相关性**低**才证明 N 个 generator 能分化（正交=多样性硬前提，§5.1）。决策树读它：
      • 现成池里有 3 个两两 |ρ| 低且覆盖核心维度 → 主线用、贵信号列 future work；
      • 都偏相关（像 maze 上 ρ(PVL,CENIE)=0.73）→ 现成池撑不起正交 → 必须回头实现贵信号。

    实现要点（和 auction_bid.standardize 的 NaN 安全策略一致）：
      - 每对 estimator 只在**两者都 finite** 的 level 上算相关（incomplete=-inf 被成对排除）；
      - 标准 Pearson：cov(i,j)/(σ_i σ_j)，分母加 eps 防零方差；
      - 全相同/方差为 0 的 estimator 与任何信号相关记为 0（无方向，无从判正交）。

    Args:
      per_est: (N, M)。-inf 占位 incomplete。
    Returns:
      corr: (N, N) 对称相关矩阵，对角 1.0。NaN 安全。
    """
    x = jnp.asarray(per_est, jnp.float32)
    N, M = x.shape
    fin = jnp.isfinite(x)                                 # (N, M)

    def pair_corr(i, j):
        both = fin[i] & fin[j]                            # (M,) 两者都 finite
        n = jnp.maximum(both.sum(), 1)
        xi = jnp.where(both, x[i], 0.0)
        xj = jnp.where(both, x[j], 0.0)
        mi = xi.sum() / n
        mj = xj.sum() / n
        di = jnp.where(both, x[i] - mi, 0.0)
        dj = jnp.where(both, x[j] - mj, 0.0)
        cov = (di * dj).sum() / n
        si = jnp.sqrt((di * di).sum() / n)
        sj = jnp.sqrt((dj * dj).sum() / n)
        denom = si * sj
        # 任一方差为 0（信号无变化）→ 相关无定义，记 0（不能误判为正交也不能误判共线）。
        r = jnp.where(denom > 1e-8, cov / (denom + 1e-12), 0.0)
        # finite level < 2 时相关无意义，记 0。
        r = jnp.where(both.sum() >= 2, r, 0.0)
        return jnp.clip(r, -1.0, 1.0)

    rows = []
    for i in range(N):
        row = []
        for j in range(N):
            row.append(jnp.where(i == j, 1.0, pair_corr(i, j)))
        rows.append(jnp.stack(row))
    return jnp.stack(rows)                                # (N, N)


def orthogonality_report(per_est, names=None, threshold=0.5):
    """诊断摘要（host 侧用）：相关矩阵 + 「是否有 ≥3 个两两 |ρ|<threshold 的正交子集」判定。

    返回 dict 直接喂日志/csv。这是 §5.4 决策树最终读的那个布尔判定的程序化版本：
    有正交子集 → 主线用现成池；没有 → 必须实现贵信号。
    threshold 默认 0.5（|ρ|≥0.5 视为偏相关；maze 的 0.73 远超线）。
    """
    import itertools
    corr = pairwise_correlation(per_est)
    N = corr.shape[0]
    names = list(names) if names is not None else [f"est{i}" for i in range(N)]
    corr_np = jax.device_get(corr)

    # 找是否存在大小≥3 的两两 |ρ|<threshold 子集（generator 至少要 3 个正交才有多样性卖点）。
    orthogonal_triples = []
    for combo in itertools.combinations(range(N), 3):
        ok = all(abs(float(corr_np[a, b])) < threshold
                 for a, b in itertools.combinations(combo, 2))
        if ok:
            orthogonal_triples.append(tuple(names[k] for k in combo))

    max_abs_offdiag = 0.0
    pairs = {}
    for a, b in itertools.combinations(range(N), 2):
        r = float(corr_np[a, b])
        pairs[f"{names[a]}~{names[b]}"] = r
        max_abs_offdiag = max(max_abs_offdiag, abs(r))

    return {
        "names": names,
        "corr": corr_np,
        "pairwise": pairs,
        "max_abs_offdiag": max_abs_offdiag,
        "has_orthogonal_triple": len(orthogonal_triples) > 0,
        "orthogonal_triples": orthogonal_triples,
        "threshold": threshold,
        # 决策树裁决：有正交三元组 → 现成池够，主线用 + 贵信号列 future work；
        #            没有        → 现成池撑不起，必须实现 transition-error 等贵信号（§5.4）。
        "verdict": ("ready_pool_sufficient" if orthogonal_triples
                    else "must_implement_expensive_signal"),
    }
