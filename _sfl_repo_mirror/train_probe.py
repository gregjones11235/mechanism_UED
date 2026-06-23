"""训练时正交性探针（host 侧）—— 边训边验四信号正交性，免饱和 checkpoint 之患。

为什么放训练循环里（而非事后加载单个 ckpt）：
  方案B §5.4 决策树要的是「现成信号池能否撑起 N=3 正交」。事后挑一个 checkpoint 有两病：
    (1) 现有 jaxnav ckpt 全是**末期**（save 覆盖同名 model.safetensors，win rate≈1.0 饱和），
        success_rate 贴 1 → difficulty/learnability 方差塌 → 相关估计不可信；
    (2) 单点快照看不出正交性是否**随训练阶段漂移**（早期 frontier 挣扎 vs 末期饱和可能完全不同）。
  边训边验：每个 eval epoch 就地算一次四信号相关，记录成相关-vs-训练步的轨迹，
  既绕开饱和（早中期 success_rate 散布 0.3~0.8 正是有效区），又看清漂移。

为什么是 host 侧（jit 外）：
  CENIE 的 GMM 拟合是 numpy/sklearn（fit_visitation_gmm），不能进 jitted train_and_eval_step。
  故本模块的入口 log_orthogonality_step() 在 jaxnav_sfl 的 **Python eval for-loop** 里调
  （line 693 那个循环），消费 get_learnability_set 额外吐出的 probe_data（已 device_get 到 host）。

接线（jaxnav_sfl.py 三处最小改动，详见本仓 INTEGRATION.md）：
  1. get_learnability_set 末尾额外返回 probe_data（success_by_env/num_episodes/dones/advantages/
     hidden/per_level_index 的 host 拷贝）——见 PROBE_DATA_CONTRACT。
  2. train_and_eval_step 把 probe_data 透传出来（jit 内只搬运，不算 GMM）。
  3. eval for-loop 里：probe_log = log_orthogonality_step(probe_data, update_count, out_dir)；
     wandb.log(probe_log.wandb_metrics, step=update_count)。
"""
import os
import csv
import numpy as np
import jax

from probe_orthogonality import compute_four_signals
from estimators import orthogonality_report


# get_learnability_set 需额外吐出的 host 侧数据契约（全部 np 数组，已 device_get）：
PROBE_DATA_CONTRACT = """
probe_data = {
    'success_rate':    (M,)        per-level 成功率 p（success_by_env 对 num_agents 维已聚合）,
    'num_episodes':    (M,)        per-level 完成 episode 数（0 → incomplete）,
    'dones':           (T, M)      rollout done（bool，喂 positive_value_loss）,
    'advantages':      (T, M)      GAE advantages（喂 positive_value_loss）,
    'hidden_feats':    (Nh, d)     本批所有时刻 GRU hidden（host 拟 GMM）,
    'per_level_index': (Nh,)       每个 hidden 行属于哪个 level（0..M-1）,
    'n_levels':        int         M,
}
M = NUM_BATCHES * BATCH_SIZE（jaxnav-sfl.yaml: 5 * 1000 = 5000）。
"""


def _saturation_flags(success_rate, num_episodes):
    """报告 success_rate 分布，判定信号是否饱和（饱和则相关不可信）。"""
    p = np.asarray(success_rate)
    ne = np.asarray(num_episodes)
    complete = ne > 0
    pc = p[complete]
    if pc.size == 0:
        return {"frac_complete": 0.0, "frac_saturated_high": 1.0,
                "frac_saturated_low": 1.0, "p_std": 0.0, "saturated": True}
    frac_hi = float((pc > 0.95).mean())
    frac_lo = float((pc < 0.05).mean())
    p_std = float(pc.std())
    # 饱和的本质 = success_rate 几乎没变异 → 相关无法可靠估计。**唯一可靠度量是 p_std 低**。
    # 关键修正（jaxnav 实测教训）：旧判据 (frac_hi+frac_lo>0.8) 会把**双峰分布**（一批易
    #   success≈1 + 一批难 success≈0，p_std≈0.45 很分散）误报为饱和——双峰下相关完全可信
    #   （两极都有、方差大），不该排除。jaxnav 上 p_std 全程稳在 0.45，却被旧判据标掉大半 epoch。
    # 新判据：
    #   ① p_std < 0.05  → 真饱和（success_rate 几乎恒定，相关不可信）；
    #   ② frac_hi > 0.95 → 单极饱和（几乎所有 level 都解开，只剩极少变异，难分辨）。
    #   双峰（frac_hi+frac_lo 高但 p_std 大）**不算饱和**。
    saturated = (p_std < 0.05) or (frac_hi > 0.95)
    return {
        "frac_complete": float(complete.mean()),
        "frac_saturated_high": frac_hi,
        "frac_saturated_low": frac_lo,
        "p_std": p_std,
        "saturated": saturated,
    }


def log_orthogonality_step(probe_data, update_count, out_dir,
                           threshold=0.5, gmm_kwargs=None):
    """单个 eval epoch 的正交性探针：四信号 → 相关 → 落 csv 行 + 返回 wandb 指标。

    Args:
      probe_data: 见 PROBE_DATA_CONTRACT（host 侧 np 数组）。
      update_count: 当前训练步（用于横轴/csv 行）。
      out_dir: 输出目录（追加写 orthogonality_trace.csv）。**None → 只算 wandb 指标、
               不落 trace 文件**（STAGE4 阶段 1 暴露：p_std 等诊断指标曾耦合在 SAVE_PATH/out_dir
               上，未设 SAVE_PATH 的注入档完全拿不到 probe/p_std；解耦后只有"写 trace 文件给
               总裁决 summarize_trace"才需要 out_dir，wandb 曲线指标无条件可得）。
      threshold: |ρ| 偏相关阈值。
    Returns:
      dict: {wandb_metrics: {...}, report: {...}, saturation: {...}}。
    """
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
    gmm_kwargs = gmm_kwargs or {}

    sat = _saturation_flags(probe_data["success_rate"], probe_data["num_episodes"])

    mat, names = compute_four_signals(
        success_rate=probe_data["success_rate"],
        num_episodes=probe_data["num_episodes"],
        dones=probe_data["dones"],
        advantages=probe_data["advantages"],
        hidden_feats=probe_data["hidden_feats"],
        per_level_index=probe_data["per_level_index"],
        n_levels=probe_data["n_levels"],
        **gmm_kwargs,
    )
    rep = orthogonality_report(mat, names=names, threshold=threshold)
    corr = rep["corr"]

    # —— 追加写相关轨迹 csv（每 eval epoch 一行）。out_dir=None 时跳过文件、只保留 wandb 指标 ——
    if out_dir is not None:
        trace_path = os.path.join(out_dir, "orthogonality_trace.csv")
        pair_keys = list(rep["pairwise"].keys())
        row = {
            "update_count": int(update_count),
            "verdict": rep["verdict"],
            "max_abs_offdiag": round(rep["max_abs_offdiag"], 4),
            "has_orthogonal_triple": int(rep["has_orthogonal_triple"]),
            "p_std": round(sat["p_std"], 4),
            "frac_sat_hi": round(sat["frac_saturated_high"], 4),
            "frac_sat_lo": round(sat["frac_saturated_low"], 4),
            "saturated": int(sat["saturated"]),
        }
        for k in pair_keys:
            row[f"rho[{k}]"] = round(rep["pairwise"][k], 4)
        write_header = not os.path.exists(trace_path)
        with open(trace_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader()
            w.writerow(row)

    # —— wandb 指标（前缀 probe/，相关随训练步成曲线）——
    wandb_metrics = {
        "probe/max_abs_offdiag": rep["max_abs_offdiag"],
        "probe/has_orthogonal_triple": int(rep["has_orthogonal_triple"]),
        "probe/p_std": sat["p_std"],
        "probe/frac_saturated_high": sat["frac_saturated_high"],
        "probe/saturated": int(sat["saturated"]),
    }
    for k in pair_keys:
        wandb_metrics[f"probe/rho/{k}"] = rep["pairwise"][k]

    # 饱和警告（相关不可信时 stdout 提示，避免误读末期 ckpt 的退化相关）。
    if sat["saturated"]:
        print(f"[probe][step {int(update_count)}] ⚠ 信号饱和 "
              f"(p_std={sat['p_std']:.3f}, hi={sat['frac_saturated_high']:.2f})"
              f" → 本步相关不可信，参考早中期 epoch 的 verdict。")
    else:
        print(f"[probe][step {int(update_count)}] verdict={rep['verdict']} "
              f"max|ρ|={rep['max_abs_offdiag']:.2f} "
              f"pairs={ {k: round(v,2) for k,v in rep['pairwise'].items()} }")

    return {"wandb_metrics": wandb_metrics, "report": rep, "saturation": sat}


def summarize_trace(out_dir):
    """训练结束后读 orthogonality_trace.csv，给出**非饱和 epoch** 上的总裁决。

    决策树的最终输入：只看信号有效（未饱和）的那些 epoch 的多数 verdict——
    避开早期没学会 + 末期饱和两端，取中段稳定区的结论。
    """
    trace_path = os.path.join(out_dir, "orthogonality_trace.csv")
    if not os.path.exists(trace_path):
        return {"error": "no trace csv"}
    rows = list(csv.DictReader(open(trace_path, encoding="utf-8")))
    valid = [r for r in rows if r.get("saturated") == "0"]
    if not valid:
        return {"verdict": "inconclusive_all_saturated",
                "note": "所有 epoch 信号都饱和——success_rate 没散开，换更难 test set 或看早期 epoch。",
                "n_epochs": len(rows), "n_valid": 0}
    n_ready = sum(1 for r in valid if r["verdict"] == "ready_pool_sufficient")
    n_must = len(valid) - n_ready
    # 取多数；非饱和 epoch 上多数 ready → 现成池够。
    verdict = "ready_pool_sufficient" if n_ready >= n_must else "must_implement_expensive_signal"
    # 各 pair 在非饱和 epoch 的平均相关（看哪两个信号稳定共线）。
    pair_cols = [c for c in valid[0].keys() if c.startswith("rho[")]
    avg_rho = {c: float(np.mean([float(r[c]) for r in valid])) for c in pair_cols}
    return {
        "verdict": verdict,
        "n_epochs": len(rows),
        "n_valid_nonsaturated": len(valid),
        "n_ready": n_ready,
        "n_must_implement": n_must,
        "avg_rho_nonsaturated": avg_rho,
        "note": "verdict 基于非饱和 epoch 多数；avg_rho 看哪对信号稳定共线（高→该维度冗余）。",
    }
