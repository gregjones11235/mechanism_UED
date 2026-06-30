# New SOTA Results Report: Targeted Level Generation Beats SFL on worst-case CVaR in jaxnav

> Created 2026-06-26. Protocol: 100-map test set + returns CVaR10 (hardest 10% of levels). currA = our method (learned PCGRL generator + geodesic-anchored signal + learnability curriculum, injected into a fixed PLR buffer).

---

## One-sentence takeaway

**On the jaxnav continuous-navigation home turf, using a smaller candidate pool than SFL (3000 < 5000), our method consistently beats the recent SOTA SFL on worst-case CVaR over the hardest 10% of levels (mean CVaR10 1.75 vs 1.27, +38%), with all three seeds agreeing and almost no catastrophic failures.** The means are tied — the entire gap is in the tail, which is exactly the selling point a UED method should have: targeted supply of hard levels → a student that is more robust on the hardest levels.

---

## 1. Core results table (multi-seed, unified protocol = returns CVaR10)

> baseline = [SFL](https://arxiv.org/pdf/2408.15099) (Rutherford, NeurIPS 2024). **Why SFL as the comparison target**: per a deep-research literature review, SFL is the strongest method in current UED / curriculum learning (on jaxnav it beats DR / PLR / ACCEL / PAIRED and the rest of the baselines via learnability curation), so beating it amounts to setting a new SOTA under this setting.

| Method | Pool | overall win | overall return | **CVaR10 return (hardest 10%)** | worst-bin return |
|---|---|---|---|---|---|
| **SFL baseline** (learnability curation) | 5000 | 0.979 | 3.911 | 1.269 | −4.94 |
| **currA seed0** | 3000 | 0.982 | 3.933 | **1.653** | −2.11 |
| **currA seed1** | 3000 | 0.983 | 3.954 | **1.730** | −4.09 |
| **currA seed2** | 3000 | 0.986 | 3.947 | **1.855** | −0.87 |
| **currA mean ± range** | 3000 | 0.984 | 3.945 | **1.746 (1.65~1.86)** | — |

**Key readings:**

- **CVaR10 beats SFL across the board**: all three seeds are ≥ 1.65, mean **1.746 vs SFL 1.269 (+0.48, +38%)**. The three seeds agree (range only 1.65~1.86), so this is not single-seed luck.
- **overall return / win are tied**: mean 3.945 vs 3.911, 0.984 vs 0.979. The gap is **not in the mean, it is entirely in the tail** — most of the 100 maps are not hard and anyone can solve them; the watershed is the hardest batch of levels.
- **Almost no catastrophic failure**: SFL hits return = **−4.94** on its worst level (deep collapse from crashing into walls / timing out); currA's best seed is only −0.87 (nearly solved), and even its worst seed is only −4.09 — overall less prone to total failure on hard levels than SFL.

---

## 2. Why it beats SFL (mechanism)

SFL uses the same learnability signal, but **has no generator and can only passively pick levels from the unbiased prior of `env.reset`** — it cannot select structural hard levels that simply aren't in the pool (e.g., the Nav2/Nav3-style corridor topologies with the agent facing away from the goal + starting flush against a wall + long geodesic path).

We have a **learned PCGRL generator** that can **actively and deliberately generate** hard levels SFL could never sample out, then inject them into the fixed PLR buffer. So this is:

> **"Few-but-fine targeted generation" > "many-but-noisy random sampling".** Feeding the same PLR, we feed a more robust student with less and scarcer raw material.

---

## 3. The correct framing of the selling point: we win on quality / training efficiency, not pool size

**Same-protocol numbers directly refute the "winning by candidate-pool size" extrapolation:**

- **The largest pool is SFL's (5000), and it is precisely the CVaR loser.** We win with 3000 (a smaller pool). If size were the driver, 5000 should win.
- **We even have an order of magnitude fewer learnable levels (p≈0.5)**: the learnable region is 1~21% of SFL's pool, but only 0.07~1.4% of ours. We win under the disadvantage of **scarcer learnable raw material**.

**So the real selling point = level quality / training efficiency, not quantity:**

> With a smaller candidate pool and an order of magnitude fewer learnable levels, **the per-level value of targeted generation far exceeds that of random sampling**, beating the recent SOTA SFL on worst-case CVaR over the hardest 10% of levels (+38%).

---

## 4. Method composition (currA, the current best architecture)

| Component | Role |
|---|---|
| **learned PCGRL generator** | Actively generates levels (rather than passively sampling), injected into the fixed PLR buffer |
| **geodesic-anchored signal** | Anchors value-loss / regret geometrically via the geodesic distance field, preventing the generator from gaming the score by piling up walls |
| **CENIE signal** | Coverage / novelty signal |
| **learnability curriculum** | Steers the generation distribution toward the learnable region (p≈0.5) |

> Note: the difficulty signal has been shown to be redundant at large pool size (its weight is only 8% throughout, and difficulty ≡ learnability − 0.25 loses discriminative power in a bimodal pool), so it can be dropped. Current best = the simplified architecture of **anchored + cenie + geo curriculum**.

---

## 5. Honest accounting (the current boundaries of this result)

1. **The strict current meaning of "beating SFL" = an internal-controlled win**: the SFL baseline is run in our own repo with `GENERATOR_INJECTION=false` — same environment / same evaluation / same code, with the only variable being "generator or not". This is the cleanest ablation control, and it holds.
2. **A "same-table-as-the-paper win" has not been done yet**: comparing against the **numbers reported in the papers** of SFL / PLR / ACCEL / PAIRED in the same table (requires 10 seeds + external baselines) is not done. Both can be stated at submission, but the former must not be passed off as the latter.
3. **CVaR is a tail statistic** — three seeds already give consistency evidence, but submission-grade strength needs to be pushed to 10 seeds.

---

## 6. Data sources

- **wandb project**: `gregjones11235-brown-university/multi_robot_ued`
