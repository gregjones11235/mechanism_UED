### 0. 先证伪结论（F1–F4 裁决）

在着手证明前，对可分性可能塌方的路径进行排查裁决：

- **F1（$\alpha_{\min}$ 经 $O(\cdot)$ 常数强依赖 $\lambda$）**：**不塌（但需降级/重定义）**。由于 $\bar{s} = \sum w_j(\lambda)s^{(j)}$ 是凸组合，只要各个独立 teacher 的 Lipschitz 常数 $G_s^{(j)}$、方差 $\sigma_j^2$ 和分数极值 $\max s^{(j)}$ 均有限且存在 $\lambda$-无关的最坏情况上界（如利用 **Z3** 次可加性），$O(\cdot)$ 前因子就不会发散。但由于它们确实显式依赖 $\lambda$，$\alpha_{\min}$ 在严格的代数解析意义下**必然残留对 $\lambda$ 的弱依赖**，除非我们用最坏常数将其重定义。
- **F2（$E_{\text{clip}}$ 不是弱耦合）**：**条件不塌**。$E_{\text{clip}}$ 是截断带来的偏差底板。根据截断积分的莱布尼茨法则（详见后文新引理），$|\partial E_{\text{clip}}/\partial C|$ 被重尾分布的尾部概率有界控制（$\le 1$）。只要拍卖机制赋予的 $\lambda \mapsto C(\lambda)$ 映射是非奇异的（即 $|\partial C/\partial \lambda| < \infty$），弱耦合成立。若机制存在使 $\partial C/\partial \lambda \to \infty$ 的突变，此档塌方。
- **F3（$\lambda$ 的作用不止于 $w(\lambda)$）**：**不塌（基于前提）**。任务书 §0 已明确将本任务的 $\lambda$ 作用完全封装在凸组合权重 $w(\lambda)$ 中。但必须指出：若实际机制允许 $\lambda$ 改变候选集大小 $n$（例如自适应过滤 level），则 $\alpha_{\max}$ 分母中的 $\log n$ 会变为 $\log n(\lambda)$，可分性将彻底崩塌。
- **F4（独立搜索：空区间塌方）**：**塌方风险存在**。如果强行采用 **Z3** 给出 $\lambda$-无关的最坏情况常数去定义 $\tilde{\alpha}_{\min}$，当各 teacher 间方差或 Lipschitz 常数差异巨大时，最坏常数会极大推高下界，可能导致 $\tilde{\alpha}_{\min} > \alpha_{\max}$。此时虽然 $\alpha$ 的可行区间严格不依赖 $\lambda$，但**该区间变成了空集**，可分性在优化实操层面失去意义。

**结论**：T2 框架整体不塌，但 T2-A 无法在"保留 instance-dependent 常数"的前提下做到严格可分，必须承认主阶可分或退化到最坏常数。以下进行正式证明。

---

### 1. T2-A 证明（无 coverage：PVL + SFL learnability）

**目标**：证 $\alpha$ 容许区间 $[ \alpha_{\min}, \alpha_{\max} ]$ 的端点与 $\lambda$ 严格可分（或阐明其退化）。
**设定**：$E_{\text{clip}} \equiv 0$（**B4**）。

**推导 1：核查 $\alpha_{\max}$ 对 $\lambda$ 的依赖**
根据 **B3**，$\alpha_{\max}(\lambda) = \epsilon_{\text{bias}} / \log n$。
我们需要核查 $\log n$ 这个常数是否由于 $\bar{s}(x; \lambda)$ 潜入了对 $\lambda$ 的依赖。
根据 **B2**（Nesterov 2005 偏差界），偏差包络界 $\Phi_\alpha(x;\lambda) - \Phi_0(x;\lambda) \le \alpha \log \sum_{i=1}^n \exp(0) = \alpha \log n$。
这里的上限 $\log n$ 纯粹由 $n$ 维单纯形 $\Delta_n$ 上香农熵 $H(y)$ 的最大值（在均匀分布处取到）决定，完全不依赖于具体的分数向量 $\bar{s}$，自然也与 $\lambda$ 无关。
因此，$\partial \alpha_{\max}/\partial \lambda = 0$ 成立，**$\alpha_{\max}$ 严格不依赖 $\lambda$**。

**推导 2：核查 $\alpha_{\min}$ 对 $\lambda$ 的依赖**
根据 **B3**，$\alpha_{\min} = (1 / (K_{\text{budget}} \cdot \epsilon^4))^{1/3}$。然而，其推导母式是 $K = O(G_s^2 \sigma^2 \Delta_\Phi / (\alpha^3 \epsilon^5))$。
根据禁糊清单 (a) 与 (b)，我们必须拆解隐藏常数 $G_s(\lambda), \sigma^2(\lambda), \Delta_\Phi(\lambda)$：
- $G_s(\lambda)$：$\bar{s}$ 的 Lipschitz 常数。由 **Z3**，凸组合的 Lipschitz 常数满足 $G_s(\lambda) \le \sum w_j(\lambda) G_s^{(j)} \le \max_j G_s^{(j)}$。
- $\sigma^2(\lambda)$：分数梯度的方差。随机梯度方差受凸组合权重控制，存在全局上界 $\sigma^2(\lambda) \le \max_j \sigma_j^2$。
- $\Delta_\Phi(\lambda)$：初始次优度 $\Phi_\alpha(x_0) - \Phi_\alpha(x^*)$。由于 teacher 的得分幅值必定有界（设 $\sup_{x, j} |s^{(j)}| \le B_s$），$\Delta_\Phi(\lambda) \le 2B_s + \alpha \log n$。

这些常数**实质上（显式地）随着 $\lambda$ 的权重 $w(\lambda)$ 移动而变化**。因此：
- 若保持 instance-dependent 使得 $\alpha_{\min}$ 最紧，则 $\alpha_{\min}$ **经 $O$ 常数弱依赖 $\lambda$**。
- 若要实现绝对的 $\lambda$-解耦，必须用 $\lambda$-无关的最坏情况常数 $C_{\text{worst}} = (\max G_s^{(j)})^2 (\max \sigma_j^2) (2B_s + \alpha \log n)$ 重新定义 $\tilde{\alpha}_{\min} = (C_{\text{worst}} / (K_{\text{budget}} \epsilon^5))^{1/3}$。这样定义的 $\tilde{\alpha}_{\min}$ 满足 $\partial \tilde{\alpha}_{\min}/\partial \lambda = 0$。

**结论 3：区间可分性论定**
在不放缩到最坏情况常数时，区间在 **"主阶意义下"**（指不含 $O$ 内常数显式表达式部分）与 $\lambda$ 解耦。在使用 $\lambda$-无关最坏常数重定义后，端点严格不移动，但需承担区间缩短（甚至如 F4 所述变为空集）的风险。

---

### 2. T2-B 证明（含 coverage / CENIE）

**目标**：证明弱耦合有界，并给出余量吸收条件。
**设定**：$E_{\text{clip}}(C(\lambda)) > 0$（**B4**）。

**推导 1：显式偏导与 $B_{\text{couple}}$ 上界**
由链式法则计算漂移率：
$$ \frac{\partial \alpha_{\max}}{\partial \lambda} = -\frac{1}{\log n} \cdot \frac{\partial E_{\text{clip}}}{\partial C} \cdot \frac{\partial C}{\partial \lambda} $$
> **⚠ 此处需新引理，我不确定它成立：截断底板对阈值的偏导数有界引理。**
> *引理内容*：设分数为 $\max(-\log p_i, -C)$（假定对应下截断，等价于原分数下界被限制）。偏差底板 $E_{\text{clip}}(C) = \mathbb{E}_{y^*}[s - s_{\text{clip}}]$。若 $s$ 服从某种分布，$E_{\text{clip}}$ 本质上是积分 $\int_{-\infty}^{-C} \mathbb{P}(s < t) dt$ 或类似的尾部截断。由莱布尼茨积分法则，$\partial E_{\text{clip}}/\partial C = \pm \mathbb{P}(s \text{ outside } C)$。无论具体形式如何，因它是概率测度下的差值，必然有：
> $$ \left| \frac{\partial E_{\text{clip}}}{\partial C} \right| \le 1 $$
> 成立条件：底层 teacher 给出分数的概率密度或质量在截断点处是可测且有界的。

基于此引理，并假设拍卖机制调整 $C$ 是 $L_C$-Lipschitz 的（即 $|\partial C/\partial \lambda| \le L_C$），我们得到显式上界：
$$ \left| \frac{\partial \alpha_{\max}}{\partial \lambda} \right| \le \frac{1}{\log n} \cdot 1 \cdot L_C := B_{\text{couple}} $$

**推导 2：$\alpha$ 余量吸收的充分条件**
在网格搜索中，若 $\lambda$ 在某个局部调参区间内扫荡，其最大移动范围为 $\Delta \lambda$。由于 $\alpha_{\min}$ 处理同 T2-A（最坏常数替代法），只有 $\alpha_{\max}$ 随 $\lambda$ 漂移。
最大漂移量为 $\Delta \alpha_{\max} = B_{\text{couple}} \cdot \Delta \lambda$。
**不需要坐标式复核的充分条件是**：在确定 $\alpha$ 时，预留充足的安全余量。即，选择一个保守的 $\alpha_{\text{target}}$，使得：
$$ \alpha_{\text{target}} \le \alpha_{\max}(\lambda_0) - B_{\text{couple}} \cdot \Delta \lambda $$
只要满足此不等式，即使 $\lambda$ 在 $\Delta \lambda$ 范围内任意变动导致 $\alpha_{\max}$ 发生下降缩紧，$\alpha_{\text{target}}$ 依然会被稳稳包裹在合法的 $[\alpha_{\min}, \alpha_{\max}(\lambda)]$ 内部。

---

### 3. 最终判定输出

`T2 是否成立（A 档：仅主阶；B 档：成立）；若证伪，命门在 F4；T2-B 弱耦合界 B_couple=L_C / log n；坐标式复核可否省（A 档：可省 / B 档：预留余量后可省）`。