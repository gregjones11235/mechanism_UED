### Q1：`1/ν = n·e^{2B_s/α}` 是 Bregman 几何的内禀，还是欧氏式放缩的残留？

**(a) 判断与分析**：**是欧氏/二次方式放缩的残留，这是一次非常糟糕的记账松弛。**
在原本的推导中，使用了类似 $D_h(p \| q) \le \frac{1}{2\nu} \|p - q\|_1^2$（其中 $\nu = \min q_i$）的放缩。这本质上是 Pinsker 不等式的**反向**：强凸性给出了下界（$D_h \ge \frac{1}{2}\| \cdot \|_1^2$），但若要用 $\ell_1$ 范数给 Bregman 散度找**上界**，就必须依赖熵函数在单纯形上的局部光滑性。由于单纯形边界上熵梯度的奇异性，Hessian $\nabla^2 h(y) = \text{diag}(1/y)$ 的最大特征值爆炸到 $1/\nu$。
因此，把内层的漂移先退化到 $\ell_1$ 欧氏几何，再强行转回 Bregman 几何，是产生指数项 `e^{2B_s/α}` 的唯一原因。这完全丢掉了 KL 几何“自动适应单纯形边界”的核心优势。

**(b) 正确的度量**：跨步漂移的正确度量应直接利用共轭对偶和原映射 $s \mapsto y^* = \text{softmax}(s/\alpha)$ 的 KL 稳定性。**这个稳定性常数中绝对不含 $1/\nu$**（见下方 Q2 推导）。

---

### Q2：重推漂移项、给出不含 `e^{·/α}` 的多项式率

我们需要在完全不经过 $\ell_1$ 范数和 $\nu$ 的前提下，直接 Bound $D_{KL}(y^*(x_{t+1}) \| y^*(x_t))$。

**推导如下（完全基于 Bregman 散度的对偶性质，无需引入未经证明的新引理）：**
1. **共轭对偶关系**：对于单纯形上的负熵 $h(y) = \sum y_i \log y_i$，其凸共轭（Convex Conjugate）是 Log-Sum-Exp 函数 $Z(u) = \log \sum \exp(u_i)$。
2. **对偶散度恒等式**：根据 Bregman 散度的标准对偶性质，原空间的散度等于对偶空间中梯度的散度（反向）：
   $$D_h(y_1 \| y_2) = D_Z(\nabla h(y_2) \| \nabla h(y_1))$$
3. **代入最优点**：已知内层精确最优解 $y^*(x) = \text{softmax}(s(x)/\alpha)$。注意到 $\text{softmax}(u) = \nabla Z(u)$，且 $u = \nabla h(y)$。因此对于 $y^*(x)$，其在对偶空间的预映射（Pre-image）直接就是 $u(x) = s(x)/\alpha$。
   $$D_{KL}(y^*(x_{t+1}) \| y^*(x_t)) = D_Z \left( \frac{s(x_t)}{\alpha} \Big\| \frac{s(x_{t+1})}{\alpha} \right)$$
4. **利用对偶函数的全局光滑性**：计算 $Z(u)$ 的 Hessian：
   $$\nabla^2 Z(u) = \text{diag}(\text{softmax}(u)) - \text{softmax}(u)\text{softmax}(u)^\top$$
   对于任意向量 $v$，二次型 $v^\top \nabla^2 Z(u) v$ 表示向量 $v$ 在概率分布 $\text{softmax}(u)$ 下的方差。方差严格小于等于二阶矩 $\sum p_i v_i^2 \le \sum v_i^2 = \|v\|_2^2$。这意味着 $\nabla^2 Z(u) \preceq I$。
   因此，**$Z(u)$ 在整个 $\mathbb{R}^n$ 上关于 $\ell_2$ 范数是全局 1-光滑的**（无需截断，也不依赖边界与概率最小值）。
5. **完成多项式 Bound**：基于 $Z$ 的 1-光滑性，有 $D_Z(u_2 \| u_1) \le \frac{1}{2}\|u_2 - u_1\|_2^2$。代入即可得：
   $$D_{KL}(y^*(x_{t+1}) \| y^*(x_t)) \le \frac{1}{2} \left\| \frac{s(x_t)}{\alpha} - \frac{s(x_{t+1})}{\alpha} \right\|_2^2 = \frac{1}{2\alpha^2} \|s(x_{t+1}) - s(x_t)\|_2^2$$
   再结合前提 A3（$s(x)$ 是 $L_s$-Lipschitz 的），最终得到：
   $$D_{KL}(y^*(x_{t+1}) \| y^*(x_t)) \le \frac{L_s^2}{2\alpha^2} \|x_{t+1} - x_t\|_2^2$$

**外层期望漂移放缩**：
已知外层是 SGD 更新 $x_{t+1} = x_t - \eta_x \hat{\nabla}_x f_\alpha$，在给定 $\mathcal{F}_t$ 下：
$$ \mathbb{E}[D_{KL}(y^*_{t+1} \| y^*_t) | \mathcal{F}_t] \le \frac{L_s^2}{2\alpha^2} \mathbb{E}[\|x_{t+1} - x_t\|_2^2 | \mathcal{F}_t] = O\left( \frac{\eta_x^2 L_s^2 G_\Phi^2}{\alpha^2} \right) $$

**新复杂度 $K$**：
漂移常数从先前的 $\frac{L_s^2}{\nu \alpha^2} \propto \frac{e^{2B_s/\alpha}}{\alpha^2}$ 直接被收紧到了 $\frac{L_s^2}{\alpha^2}$。消除了所有指数项 $1/\nu$ 的依赖后，原本包含 $n \cdot e^{2B_s/\alpha} / (\alpha^3 \epsilon^3)$ 的项将替换为 $1 / (\alpha^3 \epsilon^3)$ （具体次幂取决于两时间尺度步长设定的耦合关系，但绝对是 $1/\alpha$ 的多项式）。
新的总复杂度为：**$K = O\left( \text{poly}(1/\alpha) \cdot \epsilon^{-3} \right)$**。
这与 NCC 论文的 $O(1/\alpha^2)$ 在关于 $\alpha$ 的多项式阶次上是同级别的，彻底拔除了指数爆炸。

---

### Q3：三选一并论证

我明确选择 **(A) 可收紧**。

**论证**：
1. **理论上的内禀性**：指数项 `e^{2B_s/α}` 根本不是该非凸-强凹目标函数的内禀难度，更不是 Mirror Ascent / Bregman 几何带来的缺陷。相反，它是之前证明过程中试图在局部将 Bregman 散度强制上界化为 $\ell_1$ 欧氏距离的“记账失误”。
2. **Mirror 路线的优越性**：如 Q2 推导所示，当我们严格保持在 Bregman 几何下操作（通过对偶 Log-Sum-Exp 的全局光滑性直接 bound KL 漂移）时，多项式率顺理成章地得出。
3. **对比 NCC 截断路线**：本工作（KL-mirror 无截断路线）不仅能像 NCC 一样拿到关于 $1/\alpha$ 的多项式率，甚至在内层几何上更加自洽和优雅——它避免了欧氏投影 $P_{\Delta_\delta}$ 所需的硬截断排序操作，直接通过 Softmax 完美贴合单纯形。这保留了原有设计的所有 Novelty，且无需退回到截断策略。

---

### 结尾总判

`e^{2B_s/α}` 是 **(A) 可收紧** 的，它纯粹是上一轮试图用欧氏度量界定 Bregman 散度漂移时的记账松弛，重置到正确的共轭对偶几何后可直接得到与截断法同阶的多项式率。