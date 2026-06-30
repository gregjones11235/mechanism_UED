"""
CENIE coverage/novelty 的密度模型：拟合（numpy，jit 外）/ 求值（jax，jit 内）分离。

为什么分两半（见 mechanism_UED 的设计决策，方案 B）：
  CENIE novelty score 是 s^(CENIE) = -log p(x | λ_Γ)，p 是在 student 访问到的
  hidden state 上拟合的 GMM 密度（λ_Γ = "已访问分布"）。GMM 的 EM 拟合是任意
  numpy 迭代代码，【不能】进 jax.jit 编译的训练步；而给定 GMM 参数后算 -log p(x)
  只是几个高斯 log 密度的 logsumexp，是纯数值运算、【能】进 jit。

  所以：
    • fit_visitation_gmm(...)  —— numpy/sklearn，由 runner 每 K 步在 Python 循环里
       调一次（jit 外），把访问到的 hidden state 拟成一个 diag-GMM，返回纯数组参数。
    • cenie_neg_logp(...)      —— 纯 jax，每步在 jitted run() 里调，给定上面那组参数，
       对当前 batch 的 hidden state 算 s^(CENIE) = -log p(x)。

  这样 GMM 拟合从不进 jit（避开 callback-in-jit 的开销/脆性），求值进 jit（满足
  熵正则收敛框架要的 score 层线性结构 A5）。

设计要点（对齐 mechanism_UED 的理论裁决）：
  • 不截断：s^(CENIE) = -log p(x)，不做 max(-log p, -C)（proof_skeleton §368 净判：
    mirror 内层天然处理无界，不丢高-Novelty 尾部）。
  • diag 协方差：hidden state 维度高（如 256），full 协方差样本不足易奇异（对齐
    官方 CENIE 的 PyCave 默认 diag）。
  • GMM 分量数按 silhouette 在 K∈[6,15] 选（官方 CENIE 做法，拒 AIC/BIC）。
  • 冷启动（还没拟过 GMM）：返回全 0 的 s^(CENIE)（= 均匀，不偏向任何 level），
    用 GMMParams.valid 标志触发。
"""

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from flax import struct
from jax.scipy.special import logsumexp


class GMMParams(struct.PyTreeNode):
    """一个 diag-GMM 的纯数组参数，可作 pytree 随 runner_state 流转、进 jitted run()。

    维度：K = 分量数（固定为 max_components，未用满的分量用 log_weights=-inf 关掉，
    保证形状静态、jit 友好）；d = hidden state 特征维。
    """
    means: jax.Array          # (K, d)
    variances: jax.Array      # (K, d)  对角协方差的方差（已含 reg_covar）
    log_weights: jax.Array    # (K,)    log π_k；未用的分量为 -inf
    valid: jax.Array          # () bool；False=冷启动（还没拟过）→ neg_logp 返回 0


def init_gmm_params(max_components, feat_dim):
    """冷启动占位参数：valid=False，cenie_neg_logp 会因此返回全 0。"""
    return GMMParams(
        means=jnp.zeros((max_components, feat_dim), dtype=jnp.float32),
        variances=jnp.ones((max_components, feat_dim), dtype=jnp.float32),
        log_weights=jnp.full((max_components,), -jnp.inf, dtype=jnp.float32),
        valid=jnp.array(False),
    )


# ======================================================================
# 求值半（纯 jax，进 jitted run()）
# ======================================================================
@jax.jit
def cenie_neg_logp(gmm: GMMParams, x):
    """s^(CENIE) = -log p(x | λ_Γ)，p 为 diag-GMM。x: (..., d)。返回: (...,)。

    log p(x) = logsumexp_k [ log π_k + Σ_d log N(x_d; μ_kd, σ²_kd) ]
    冷启动（gmm.valid=False）返回 0（均匀，不偏向）。
    """
    # 每个分量的对角高斯 log 密度：-0.5[ d·log(2π) + Σ log σ² + Σ (x-μ)²/σ² ]
    # x: (..., d) → 扩成 (..., 1, d) 与分量维 (K, d) 广播
    xe = x[..., None, :]                                   # (..., 1, d)
    mu = gmm.means                                         # (K, d)
    var = gmm.variances                                    # (K, d)
    d = mu.shape[-1]
    log_norm = -0.5 * (
        d * jnp.log(2.0 * jnp.pi)
        + jnp.sum(jnp.log(var), axis=-1)                   # (K,)
        + jnp.sum((xe - mu) ** 2 / var, axis=-1)           # (..., K)
    )                                                      # (..., K)
    log_comp = gmm.log_weights + log_norm                  # (..., K) 广播 (K,)
    logp = logsumexp(log_comp, axis=-1)                    # (...,)
    neg_logp = -logp
    # 冷启动：valid=False 时整体置 0
    return jnp.where(gmm.valid, neg_logp, jnp.zeros_like(neg_logp))


# ======================================================================
# 拟合半（numpy/sklearn，jit 外，runner 每 K 步调一次）
# ======================================================================
def fit_visitation_gmm(visitation_feats, max_components=15, min_components=6,
                       covariance_type='diag', reg_covar=1e-2, random_state=0):
    """在访问到的 hidden state 上拟合 diag-GMM（λ_Γ）。返回 GMMParams（纯数组）。

    入参:
        visitation_feats : np.ndarray (M, d)，M 个访问到的 hidden state 特征
                           （从最近若干步 rollout 的 carry 收集，已拷回 host）。
        max_components   : K 的上界（也是返回 GMMParams 的静态分量数，未用满的用
                           log_weights=-inf 关掉，保证 jit 形状不变）。
        min_components   : silhouette 搜索下界。
        covariance_type  : 'diag'（默认；高维 hidden state 更稳）。
        reg_covar        : 协方差正则（官方 Minigrid 值 1e-2）。
    返回:
        GMMParams，valid=True；分量数 < max_components 时余下分量 log_weights=-inf。

    分量数按 silhouette 在 [min_components, max_components] 选（官方 CENIE 做法）。
    sklearn 缺失或样本太少则退化为单分量对角高斯（仍是合法密度，valid=True）。
    """
    X = np.asarray(visitation_feats, dtype=np.float64)
    if X.ndim != 2:
        X = X.reshape(X.shape[0], -1)
    M, d = X.shape

    means = np.zeros((max_components, d), dtype=np.float32)
    variances = np.ones((max_components, d), dtype=np.float32)
    log_weights = np.full((max_components,), -np.inf, dtype=np.float32)

    def _fill_from_diag(mu, var, w):
        """把拟好的 (k, d) 参数填进静态 (max_components, d) 槽位。"""
        k = mu.shape[0]
        means[:k] = mu.astype(np.float32)
        variances[:k] = np.maximum(var, reg_covar).astype(np.float32)
        log_weights[:k] = np.log(np.maximum(w, 1e-12)).astype(np.float32)

    try:
        from sklearn.mixture import GaussianMixture
        from sklearn.metrics import silhouette_score

        hi = min(max_components, max(min_components, M - 1))
        best_gmm, best_sil = None, -np.inf
        for k in range(min_components, hi + 1):
            if k >= M:
                break
            gmm = GaussianMixture(n_components=k, covariance_type=covariance_type,
                                  reg_covar=reg_covar, random_state=random_state)
            labels = gmm.fit_predict(X)
            sil = silhouette_score(X, labels) if len(np.unique(labels)) >= 2 else -np.inf
            if sil > best_sil:
                best_sil, best_gmm = sil, gmm
        if best_gmm is None:                              # 样本太少，silhouette 无法选
            best_gmm = GaussianMixture(n_components=1, covariance_type=covariance_type,
                                       reg_covar=reg_covar, random_state=random_state).fit(X)
        # 取出 diag 协方差。covariance_type='diag' 时 covariances_ 形状 (k, d)。
        cov = best_gmm.covariances_
        if covariance_type != 'diag':                     # full/tied/spherical → 取对角近似
            if cov.ndim == 3:                             # full: (k,d,d)
                cov = np.stack([np.diag(c) for c in cov])
            elif cov.ndim == 1:                           # spherical: (k,) → 广播到 d
                cov = np.repeat(cov[:, None], d, axis=1)
        _fill_from_diag(best_gmm.means_, cov, best_gmm.weights_)
    except Exception:
        # 退化退路：单个对角高斯（无 sklearn 或拟合失败时）
        mu = X.mean(0, keepdims=True)
        var = X.var(0, keepdims=True) + reg_covar
        _fill_from_diag(mu, var, np.array([1.0]))

    return GMMParams(
        means=jnp.asarray(means),
        variances=jnp.asarray(variances),
        log_weights=jnp.asarray(log_weights),
        valid=jnp.array(True),
    )
