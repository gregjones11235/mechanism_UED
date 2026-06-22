"""V1 正交性探针 —— 加载 SFL checkpoint，在 jaxnav 上跑一批 rollout，输出四个**真信号**的
per-level 矩阵 + 两两相关，裁决「现成池能否撑起 N=3 正交」(方案B §5.4 数据驱动决策树)。

—— 这是 get_learnability_set 的离线只读变体 ——
不动训练主循环：复用官方 env / ActorCriticRNN(GRU) / cenie_density，只加载一个已训 checkpoint
跑一次评估 rollout，把四信号算出来比相关。**探针不训练、不写回 buffer**。

四个真信号（无任何代理，见 estimators.py 信号边界注释）：
  1. difficulty-match  s=-(p-0.5)²        ← estimators.difficulty_match(success_rate)
  2. learnability      s=p(1-p)            ← estimators.learnability(success_rate)
  3. PVL               positive_value_loss ← jaxued.jaxued_utils.positive_value_loss(dones, advantages)
  4. CENIE             -log p(hidden)      ← 官方 GMM：本批 hidden 自拟(fit_visitation_gmm)自评(cenie_neg_logp)

CENIE 两段（"同一批 rollout 内自拟自评"）：
  收本批所有 GRU hidden (M_steps, d) → host 侧 fit_visitation_gmm 拟一次 GMM(λ_Γ) → 对同批
  per-level 平均 hidden 求 -log p。GMM 拟合是 numpy/sklearn（jit 外），求值是纯 jax。

═══════════════ 在 Oscar 上怎么跑（本机 WSL 无 sfl env / jaxnav，跑不了）═══════════════
  把本文件 + estimators.py + auction_bid.py 放进 SFL repo（与 jaxnav_sfl.py 同级或 PYTHONPATH 可达），
  在 sfl 环境跑：
      python probe_orthogonality.py \
          --checkpoint checkpoints/multi_robot_ued/<run-name>/model.safetensors \
          --config <训练时的 config yaml 或同 jaxnav_sfl.py 默认> \
          --num-batches 4 --out probe_out/
  产物：probe_out/corr_matrix.csv（4x4 相关）、probe_out/signals.npz（per-level 矩阵）、
        probe_out/corr_heatmap.png、probe_out/verdict.txt（ready_pool_sufficient / must_implement_expensive_signal）。
════════════════════════════════════════════════════════════════════════════════════════
"""
import argparse
import os
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

# —— 信号层（本仓三件套）——
from estimators import (
    difficulty_match, learnability,
    assemble_signal_matrix, orthogonality_report,
)
# —— PVL 真信号：复用官方 jaxued 实现，不重写 ——
#    sfl repo 内 jaxued 在 sfl.util.jaxued 下（jaxnav_plr.py 同款 import 路径）。
from sfl.util.jaxued.jaxued_utils import positive_value_loss
# —— CENIE 真信号：复用官方 minimax GMM 密度（拟合/求值分离）——
#    cenie_density.py 自包含（仅 numpy/jax/flax），已复制到 sfl/train/ 当本地模块，
#    避免在 sfl env 装完整 minimax 包（依赖冲突）。源： minimax/util/rl/cenie_density.py。
from cenie_density import fit_visitation_gmm, cenie_neg_logp


# ──────────────────────────────────────────────────────────────────────
# 信号计算：从一批 rollout 的 per-level outcomes + hidden 折出 (4, M) 矩阵
# ──────────────────────────────────────────────────────────────────────
def compute_four_signals(success_rate, num_episodes, dones, advantages,
                         hidden_feats, per_level_index, n_levels,
                         gmm_max_components=15, gmm_min_components=6,
                         gmm_reg_covar=1e-2):
    """把一批 rollout 统计折成四信号 per-level 矩阵 (4, M)。

    Args:
      success_rate : (M,) per-level 成功率 p。
      num_episodes : (M,) per-level 完成 episode 数（0 → incomplete）。
      dones        : (T, M) rollout done 序列（喂 positive_value_loss）。
      advantages   : (T, M) GAE advantages（喂 positive_value_loss）。
      hidden_feats : (T*M_actors, d) 本批所有时刻的 GRU hidden（host 侧拟 GMM 用）。
      per_level_index: (T*M_actors,) 每个 hidden 行属于哪个 level（0..M-1），把 -log p 还原 per-level。
      n_levels     : M。
    Returns:
      mat   : (4, M) [difficulty, learnability, PVL, CENIE] per-level score，-inf 占位 incomplete。
      names : ['difficulty', 'learnability', 'PVL', 'CENIE']。
    """
    p = jnp.asarray(success_rate, jnp.float32)
    ne = jnp.asarray(num_episodes)

    s_diff = difficulty_match(p, num_episodes=ne)               # (M,)
    s_lear = learnability(p, num_episodes=ne)                   # (M,)
    # PVL：官方信号，per-level（incomplete 自带 -inf）。
    s_pvl = positive_value_loss(jnp.asarray(dones), jnp.asarray(advantages))   # (M,)

    # CENIE：本批 hidden 自拟自评（host 侧 numpy/sklearn 拟，纯 jax 求值）。
    feats_host = np.asarray(jax.device_get(hidden_feats))       # (Nh, d)
    gmm = fit_visitation_gmm(feats_host, max_components=gmm_max_components,
                             min_components=gmm_min_components, reg_covar=gmm_reg_covar)
    nlp = cenie_neg_logp(gmm, jnp.asarray(feats_host))          # (Nh,) -log p，反密度
    # 按 per_level_index 把 -log p 聚到 per-level（mean NLL，官方 _score_one_estimator 同口径）。
    pli = np.asarray(jax.device_get(per_level_index)).astype(np.int64)
    nlp_host = np.asarray(jax.device_get(nlp))
    sum_by_level = np.zeros(n_levels, dtype=np.float64)
    cnt_by_level = np.zeros(n_levels, dtype=np.float64)
    np.add.at(sum_by_level, pli, nlp_host)
    np.add.at(cnt_by_level, pli, 1.0)
    s_cenie = np.where(cnt_by_level > 0, sum_by_level / np.maximum(cnt_by_level, 1), -np.inf)
    # incomplete（无完成 episode）的 level 也标 -inf，与其它三信号对齐。
    ne_host = np.asarray(jax.device_get(ne))
    s_cenie = np.where(ne_host > 0, s_cenie, -np.inf)
    s_cenie = jnp.asarray(s_cenie, jnp.float32)

    mat = assemble_signal_matrix([s_diff, s_lear, s_pvl, s_cenie])   # (4, M)
    return mat, ['difficulty', 'learnability', 'PVL', 'CENIE']


# ──────────────────────────────────────────────────────────────────────
# rollout：复用 get_learnability_set 的评估循环，额外 emit hidden + advantages
# ──────────────────────────────────────────────────────────────────────
def collect_probe_batch(rng, network, network_params, env, config, t_config):
    """跑一批评估 rollout，返回算四信号所需的 per-level 统计 + 全时刻 hidden。

    结构刻意对齐 jaxnav_sfl.get_learnability_set 的 _batch_step（同一套 reset/scan/
    _calc_outcomes_by_agent），差别仅在：① _env_step 把 hidden emit 到 transition；
    ② 额外算 GAE advantages 喂 PVL；③ 返回 hidden 给 host 拟 GMM。

    返回 dict：success_rate / num_episodes / dones / advantages / hidden_feats /
              per_level_index / n_levels / env_instances。

    NOTE: 本函数体依赖 SFL repo 内的 batchify/unbatchify/Transition/ScannedRNN/
          _calculate_gae 等符号——在 Oscar 上 import jaxnav_sfl 的同名工具即可。这里写成
          需在 repo 内补一行 `from jaxnav_sfl import batchify, unbatchify, Transition, ScannedRNN`
          （或直接把本函数贴进 jaxnav_sfl.py）。占位实现见 build_rollout_in_repo()。
    """
    raise NotImplementedError(
        "collect_probe_batch 需在 SFL repo 内绑定 jaxnav_sfl 的 env/network 工具——"
        "见 build_rollout_in_repo()，把它接到 get_learnability_set 的 rollout 上。")


def build_rollout_in_repo():
    """把探针 rollout 接进 jaxnav_sfl.py 的最小补丁说明（贴进 repo 时照此改 get_learnability_set）。

    在 jaxnav_sfl.get_learnability_set 的 _env_step 里，Transition 已带 value/done/info；
    只需三处加料即可复用它当探针 rollout，无需重写：

    (1) emit hidden —— _env_step 里 hstate 已在 carry，把 `hstate` 也塞进 transition：
          transition = Transition(..., carry=hstate)   # ScannedRNN GRU state (BATCH_ACTORS, HIDDEN_SIZE)
        （Transition 加一个 carry 字段；非 recurrent 时置 None。）

    (2) 算 advantages 喂 PVL —— scan 完 traj_batch 后，用 repo 既有的 _calculate_gae(traj_batch, last_val)
        （主循环 line 417 已有）算 advantages: (ROLLOUT_STEPS, BATCH_ACTORS)。

    (3) per-level 还原 —— success_by_env 已是 (num_agents, BATCH_SIZE)；hidden 是
        (ROLLOUT_STEPS, BATCH_ACTORS, d)，per_level_index = 每个 (t, actor) 展平后对应的
        env 列号（actor = agent*BATCH_SIZE + env），reshape 即得。

    然后调 compute_four_signals(...) 出 (4, M) 矩阵。把它和 env_instances 一起返回，
    探针主函数再做相关 + 报告。这样训练主循环的 p(1-p) 计分一行不动（向后兼容）。
    """
    raise NotImplementedError("说明性函数，不执行。照 docstring 三步改 get_learnability_set。")


# ──────────────────────────────────────────────────────────────────────
# 报告：相关矩阵 → csv / npz / png / verdict
# ──────────────────────────────────────────────────────────────────────
def write_reports(mat, names, out_dir, threshold=0.5):
    """mat (4,M) → 相关矩阵 + 决策树裁决，落 csv/npz/png/verdict.txt。"""
    os.makedirs(out_dir, exist_ok=True)
    rep = orthogonality_report(mat, names=names, threshold=threshold)
    corr = rep["corr"]                                          # (4,4) numpy

    # corr_matrix.csv
    with open(os.path.join(out_dir, "corr_matrix.csv"), "w", encoding="utf-8") as f:
        f.write("," + ",".join(names) + "\n")
        for i, n in enumerate(names):
            f.write(n + "," + ",".join(f"{corr[i, j]:.4f}" for j in range(len(names))) + "\n")

    # signals.npz（per-level 原始矩阵，便于复跑/换 threshold）
    np.savez(os.path.join(out_dir, "signals.npz"),
             mat=np.asarray(jax.device_get(mat)), names=np.array(names))

    # verdict.txt
    with open(os.path.join(out_dir, "verdict.txt"), "w", encoding="utf-8") as f:
        f.write(f"verdict: {rep['verdict']}\n")
        f.write(f"has_orthogonal_triple: {rep['has_orthogonal_triple']}\n")
        f.write(f"orthogonal_triples: {rep['orthogonal_triples']}\n")
        f.write(f"max_abs_offdiag: {rep['max_abs_offdiag']:.4f}\n")
        f.write(f"threshold: {rep['threshold']}\n")
        f.write("pairwise:\n")
        for k, v in rep["pairwise"].items():
            f.write(f"  {k}: {v:+.4f}\n")
        # 决策树解读（§5.4）：
        if rep["verdict"] == "ready_pool_sufficient":
            f.write("\n→ 现成池有 ≥3 个两两正交信号，主线用现成池；transition-error/co-learnability\n"
                    "  列 future work（论文诚实写两维度留作扩展）。\n")
        else:
            f.write("\n→ 现成池撑不起 N=3 正交（像 maze ρ(PVL,CENIE)=0.73）。按 §5.4：贵信号\n"
                    "  (transition-error 等) 必须实现，不能因实现贵而绕过——正交是 generator\n"
                    "  分化的硬前提，撑不起=mode collapse=多样性卖点落空。\n")

    # corr_heatmap.png
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4.5, 4))
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                        color="white" if abs(corr[i, j]) > 0.5 else "black", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(f"estimator corr  |  {rep['verdict']}")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "corr_heatmap.png"), dpi=130)
        plt.close(fig)
    except Exception as e:
        print(f"[warn] heatmap 跳过（matplotlib 不可用）：{e}")

    return rep


def main():
    ap = argparse.ArgumentParser(description="V1 estimator 正交性探针（SFL/jaxnav）")
    ap.add_argument("--checkpoint", required=True, help="model.safetensors 路径")
    ap.add_argument("--config", default=None, help="训练 config yaml；省略则用 jaxnav_sfl 默认")
    ap.add_argument("--num-batches", type=int, default=4, help="评估 batch 数（越多越稳）")
    ap.add_argument("--threshold", type=float, default=0.5, help="|ρ| 偏相关阈值")
    ap.add_argument("--out", default="probe_out", help="输出目录")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # ―― 以下 4 步需在 SFL repo 内绑定官方组件（本机无 sfl env / jaxnav 跑不了，故只在 Oscar 执行）――
    # 1) load_params + 重建 ActorCriticRNN(GRU) 与训练同 config；
    # 2) 建 jaxnav env（同训练 env_name）；
    # 3) collect_probe_batch ×num_batches（按 build_rollout_in_repo 三步改好的 rollout）；
    # 4) compute_four_signals → 拼 (4, M_total) → write_reports。
    #
    # 留作 repo 内补全（占位，避免在无依赖环境误跑）：
    raise SystemExit(
        "probe_orthogonality.main: 请在 SFL repo (Oscar, sfl env) 内运行，并按 "
        "build_rollout_in_repo() 把 collect_probe_batch 接到 jaxnav_sfl 的 env/network 上。\n"
        "信号计算 compute_four_signals / 报告 write_reports 已可直接复用，无需改。")


if __name__ == "__main__":
    main()
