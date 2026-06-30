"""
读 measure_sigma_on_ckpt --all_archives 跑出的 <prefix>_<step>.npy（每个 step 一个 S 矩阵），
沿训练时间画 ρ 漂移 + 各 step 的方差，并跑 h_drift_check 判稳定。

核心诊断点：SFL 那一行的方差是否在中段 > 0（末期已知塌成 0 → ρ=0/0 未定义）。
若中段方差 > 0，则"用中段 ckpt 测 Σ 救 SFL"这条先验成立。

用法：
  python run_h_drift_on_archives.py sigma_S/meplr-arch-s0
  （会自动 glob sigma_S/meplr-arch-s0_*.npy + 读 sigma_S/meplr-arch-s0.names.json）
"""
import sys
import re
import glob
import json
import os
import numpy as np

import sigma_measure as sm


def load_steps(prefix):
    names_path = prefix + '.names.json'
    with open(names_path) as f:
        names = json.load(f)
    items = []
    for p in glob.glob(prefix + '_*.npy'):
        m = re.search(r'_(\d+)\.npy$', os.path.basename(p))
        if m:
            items.append((int(m.group(1)), p))
    items.sort(key=lambda t: t[0])
    return names, items


def main():
    prefix = sys.argv[1] if len(sys.argv) > 1 else 'sigma_S/meplr-arch-s0'
    names, items = load_steps(prefix)
    assert items, f'没找到 {prefix}_*.npy'
    N = len(names)
    print(f'prefix={prefix}  estimators={names}  共 {len(items)} 个 step\n')

    # 表头
    var_cols = '  '.join(f'var[{n}]'.rjust(11) for n in names)
    print(f"{'step':>7} | {var_cols} | {'ρ(PVL,SFL)':>11} {'ρ(PVL,CEN)':>11} {'ρ(SFL,CEN)':>11}")
    print('-' * 100)

    rho_list = []
    steps = []
    for step, p in items:
        S = np.load(p)  # (N, n)
        Sigma = sm.covariance(S)
        rho = sm.corr_from_cov(Sigma)
        variances = np.diag(Sigma)
        rho_list.append(rho)
        steps.append(step)

        # 假设 names 顺序 = [PVL, SFL, CENIE]，但用 index 通用
        def rp(a, b):
            return f'{rho[a, b]:+.3f}'
        idx = {n: i for i, n in enumerate(names)}
        # 三对（若名字不全则跳过对应列）
        pairs = []
        for (a, b) in [(0, 1), (0, 2), (1, 2)]:
            pairs.append(f'{rho[a, b]:+.3f}'.rjust(11))
        var_str = '  '.join(f'{v:11.5f}' for v in variances)
        print(f'{step:>7} | {var_str} | ' + ' '.join(pairs))

    # SFL 方差专项（核心诊断）
    sfl_i = names.index('SFL') if 'SFL' in names else 1
    print('\n=== 核心诊断：SFL 方差随训练 ===')
    for step, p in items:
        S = np.load(p)
        v = float(np.var(S[sfl_i]))
        flag = '塌缩(ρ未定义)' if v < 1e-12 else 'OK 有方差'
        print(f'  step={step:>6}: var(SFL)={v:.3e}  {flag}')

    # h_drift_check
    print('\n=== h_drift_check（ρ 跨 step 稳定性，tol=0.15）===')
    drift = sm.h_drift_check(rho_list, names, tol=0.15)
    for pair, d in drift.items():
        flag = '稳定' if d['stable'] else '★漂移'
        print(f"  {pair[0]:6s}-{pair[1]:6s}: range={d['range']:.3f}  {flag}  values={d['values']}")


if __name__ == '__main__':
    main()
