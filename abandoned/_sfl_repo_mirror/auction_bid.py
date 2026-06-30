"""异质 estimator auction —— bid 标准化与权重分配（载体中立、estimator 无关）。

设计与三个 bug 的诊断见 mechanism_UED/方案B_多generator设计.md §6。本模块只做两件事：
  1. per-level z-score 标准化（量级可比，用于混合 mixed = Σ w_j s^(j)）；
  2. bid = z-score 后 **top-k 分位均值**（保留 estimator 间「推崇强度」差异，修 bug②），
     再 w = softmax(bids/λ) 分配权重（λ 是 softmax 温度：λ→0/∞ argmax 端，λ 大 uniform 端）。

为什么 bid ≠ z 后全体均值：z-score 定义上零均值 ⇒ 全体均值≈0 ⇒ softmax 塌成 uniform、
λ 失效（旧实现 bug②）。top-k 取「该 estimator 对它最看重那批 level 的意见强度」，对应
PLR「优先回放高分 level」语义，且对量级悬殊免疫（z-score 已抹平量级）。

estimator 无关：输入 per_est 形如 (N, ...)，对任意 N 个 estimator 成立。incomplete level
用 -inf 占位（NaN 安全）。纯 JAX，可进 jitted 训练循环。
"""
import jax
import jax.numpy as jnp


def standardize_per_estimator(per_est):
    """对每个 estimator 在其 finite level 上做 z-score（减均值除标准差）。

    Args:
      per_est: (N, M) 或 (N, ...) —— N 个 estimator 的 per-level score，第 0 维是 estimator。
               incomplete level 用 -inf（或任意非 finite）占位，会被排除出均值/方差。
    Returns:
      z: 与 per_est 同形，finite 位是标准化值，非 finite 位原样保留（仍是 -inf）。
    """
    N = per_est.shape[0]
    flat = per_est.reshape(N, -1)                       # (N, M)
    fin = jnp.isfinite(flat)                            # (N, M)
    cnt = jnp.maximum(fin.sum(axis=1), 1)              # (N,)
    mean = jnp.where(fin, flat, 0.0).sum(axis=1) / cnt  # (N,)
    var = jnp.where(fin, (flat - mean[:, None]) ** 2, 0.0).sum(axis=1) / cnt
    std = jnp.sqrt(var) + 1e-8
    z = jnp.where(fin, (flat - mean[:, None]) / std[:, None], flat)
    return z.reshape(per_est.shape)


def topk_bids(z, frac=0.125):
    """bid_j = estimator j 的 z-score 后 top-k 分位均值（k = ceil(frac * #finite_j)）。

    修 bug②：不取全体均值（z 后恒 0），取每 estimator z 分数最高前 k 个的均值——量化
    「该 estimator 对最看重那批 level 的意见强度」。incomplete（-inf）自然排在底部不入 top-k。

    Args:
      z: (N, ...) 已 standardize 的 score。
      frac: top 分位比例（默认 12.5%）。10%~20% 给类似 frontier，不敏感（待 jaxnav 确认）。
    Returns:
      bids: (N,) 每个 estimator 的 bid 摘要。
    """
    N = z.shape[0]
    flat = z.reshape(N, -1)                             # (N, M)
    M = flat.shape[1]
    k = max(1, int(M * frac))
    # 降序排序后取前 k 个；-inf 在降序里排末尾，不会污染 top-k（除非 finite 不足 k）。
    sorted_desc = jnp.sort(flat, axis=1)[:, ::-1]       # (N, M) 降序
    topk = sorted_desc[:, :k]                           # (N, k)
    # 只对 finite 求均值（极端情况 finite < k 时排除 -inf）。
    fin = jnp.isfinite(topk)
    cnt = jnp.maximum(fin.sum(axis=1), 1)
    bids = jnp.where(fin, topk, 0.0).sum(axis=1) / cnt  # (N,)
    return bids


def auction_weights(bids, auction_lambda):
    """w = softmax(bids/λ)。λ 是 softmax 温度。

    语义（与 mechanism_UED 文档 §6 对齐，已修正旧文档写反处）：
      λ=inf（特判）→ one-hot argmax（single-winner 端）；
      λ→0         → 也趋 one-hot argmax（sharp 端）；
      λ 大（有限）  → 趋 uniform（fractional/混合端）。
    即 argmax 端 = {λ=inf} ∪ {λ→0}；uniform 端 = {有限大 λ}。

    Args:
      bids: (N,)。
      auction_lambda: float，inf 表示 single-winner。
    Returns:
      w: (N,) ∈ Δ_N。
    """
    bids = jnp.asarray(bids, dtype=jnp.float32)
    N = bids.shape[0]
    is_inf = jnp.isinf(jnp.asarray(auction_lambda, jnp.float32))
    lam = jnp.maximum(jnp.asarray(auction_lambda, jnp.float32), 1e-6)
    soft = jax.nn.softmax(bids / lam)
    onehot = jax.nn.one_hot(jnp.argmax(bids), N, dtype=jnp.float32)
    return jnp.where(is_inf, onehot, soft)


def mix_scores(per_est, auction_lambda, frac=0.125, incomplete_value=-jnp.inf):
    """完整 auction：per_est → 标准化 → bids → w → 混合 per-level score。

    Args:
      per_est: (N, M) N 个 estimator 的 per-level score，-inf 占位 incomplete。
      auction_lambda: float。
      frac: top-k 分位比例。
    Returns:
      mixed: (M,) 混合后 per-level score（incomplete level 仍为 -inf）。
      w: (N,) auction 权重（诊断/日志用）。
      bids: (N,) bid 摘要（诊断/日志用）。
    """
    z = standardize_per_estimator(per_est)             # (N, M)
    bids = topk_bids(z, frac=frac)                     # (N,)
    w = auction_weights(bids, auction_lambda)          # (N,)
    mixed = jnp.tensordot(w, z, axes=(0, 0))           # (M,)
    # incomplete = 所有 estimator 在该 level 都是 -inf → 混合后置 -inf。
    all_incomplete = jnp.all(~jnp.isfinite(z), axis=0)
    mixed = jnp.where(all_incomplete, incomplete_value, mixed)
    return mixed, w, bids
