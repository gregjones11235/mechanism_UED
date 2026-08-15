# Talk Script (EN) — Skill-Preflight UED, season final
**2026-08-13 · Mason** | Supersedes `演讲稿_final_talk_20260807.md` in full — see §0 for what changed and why.
Organised as **content blocks with timings**, not fixed slide numbers, so it maps onto whatever deck we land on. Core = **9:00**; with the two optional blocks = **12:00**.

---

## §0 What changed since the 8/07 script (read this first)

| 8/07 script said | Status now | Why |
|---|---|---|
| "The stack adds +0.90 over vanilla" | **Superseded** | That gap bundles a supply change. Clean ladder: base2e9 43.12 → (parents 2→15, **+1.90**) → base15 45.02 → (**whole stack, −0.29**) → longStack 44.73 |
| "The true graph was a lock, not a ladder" | **Superseded** | gateOff kept the lock (151 scheduler cycles, **zero** targeting of the three mobs) yet recovered the floor-2 cluster in full ⇒ the lock is real but **not the binding constraint** |
| "Machinery load-bearing, knowledge not" | **Retired** (already, 8/07 night) | Never revive in any form |
| "Repair arm fired, lands this week" | **Never happened** | The arms actually fired on 8/08 were gateOff and repairF; landed 8/10 |
| Q&A: "attribution awaits the repair arm" | **Resolved** | Four single-variable arms have since adjudicated it |

**Everything below is built on the current verdicts.** Numbers are last-10 offline official unless marked.

---

## Block 1 — The question (0:45)

> This season asked one question: **what does an LLM-written curriculum actually buy at full budget?** Not "does it help" — it does — but which part of it is doing the work.
>
> Our setup: DiCode's pipeline, student, environment and budget all unchanged; the teacher swapped to a **local 14B** on a single card. Everything I report is 2×10⁹ student steps, evaluated offline on a fixed set of 1024 held-out worlds, last ten checkpoints.

---

## Block 2 — The instrument, before any result (1:15)

> Before results, the ruler — every verdict here rests on it.
>
> The noise floors are **measured, not assumed**: seed band 1.13; last-10 estimator 0.68; within-run checkpoint jitter 0.17 to 0.80 depending on the arm. Crucially the band is **not constant** — it peaks around 7 inside the take-off window and falls below 1 afterwards, so every effect is judged against the band **at its own position**.
>
> This discipline was bought with damage. **In-training evaluation misled us four times this season**, once structurally — an evaluation leak whose books reconciled to the decimal. So: offline multi-checkpoint only, criteria frozen before the data, nulls verified for dose, code parsed with an AST rather than regexes.
>
> One more, added last week: **per-skill claims now carry per-skill error bars.** Aggregate return is a low-resolution instrument here; the per-skill table is the one to read, and it needs its own bars.

---

## Block 3 — The five-arm picture and the accident (1:45)

> Five arms, one protocol. Vanilla 14B, no stack: **43.12**. With our full stack, three seeds: **44.01 ± 0.79** — parity with their 30B point, 44.38 ± 0.66, across model families; parity, not "beats".
>
> Then the control arm that broke the season. We shuffled 90% of the edges of the hand-authored prerequisite graph, expecting the mechanisms to fail. It scored **45.66 — the highest arm of the season.**
>
> A control arm outscoring the treatment is either a mistake or a message. The rest of this talk is what it turned out to be.

---

## Block 4 — Taking the accident apart: four single-variable arms (2:30) **[core]**

> We answered it with four arms, each changing exactly one thing, every criterion frozen before the data.
>
> **First: what did the shuffle actually break?** Two things read that graph — the scheduler and a validation gate. We built `gateOff`: true graph, gate switched off entirely. Eight absolute thresholds, all frozen in advance, **all passed**. The conditional hit rate on the floor-2 combat cluster moved from 0.44 / 0.03 / 0.09 to **0.79 / 0.24 / 0.72** — the shuffled arm's own regime.
>
> **And here is the sharp part.** In `gateOff` the scheduler was still locked out of those skills — across 151 cycles it targeted them **zero times** — and the cluster recovered anyway. So the lock we had diagnosed a week earlier was real, but it was **not** the binding constraint.
>
> **The binding constraint was a single clause in the gate.** A rule intended to stop levels from handing the agent unearned prerequisites was treating a *starting floor* as such a hand-out. Its LLM repair then rewrote **24 of 27** trained combat levels from floor 2 down to the surface — descriptions untouched, code changed. The curriculum was starving the exact cluster it was built to teach.
>
> **Second: is it repairable?** `surgical` removes that one clause and leaves every other repair path intact — 76 gate cycles, every one carrying the removal marker. All four thresholds passed, conditional hits 0.80 and 0.60. **The mechanism chain closes end to end.**

---

## Block 5 — What the shuffle really was (1:15) **[core]**

> One number explains the accident. We measured **coupling** — how often a newly generated level actually matches the target the scheduler asked for. In the true-graph stack: **60%**. In the shuffled arm: **6.3%**.
>
> Shuffling did not steer the curriculum somewhere better. It **removed 90% of the steering**, and the generator fell back on its own prior.
>
> The line I would like remembered: **shufGraph's score was set by an emergence engine that had lost its steering wheel.** That is also why it is not a recommendation.

---

## Block 6 — The season's verdict (1:30) **[core]**

> If steering was doing so little, the obvious question is what the whole stack was worth. The last arm answers it. `base15` is the **vanilla pipeline** with exactly one change — a generation-supply parameter the upstream default sets to 2, raised to 15. **No scheduler, no preflight, no gate.**
>
> It landed on top of the shuffled arm: **45.83 against 45.85** on the last-three window. Its floor-2 cluster is the strongest of the season.
>
> So the ladder, finally clean:
>
> **43.12 → (one supply parameter, +1.90) → 45.02 → (our entire mechanism stack, −0.29) → 44.73.**
>
> **One untuned parameter in the baseline is worth more than the whole curriculum machinery.** For calibration: on their own teacher ladder, going from 30B to 80B is worth +1.50.

---

## Block 7 — What survives, and it is not nothing (1:00) **[core]**

> Two things survive that arithmetic.
>
> **First, a priced bill of mechanism costs** — each with intervention-level evidence: the floor clause starves the floor-2 cluster; the scheduler cultivates a 75%-single-crop diet; and what we call the **loop tax** — the gate's remaining repairs strip subsidies that are not violations, task-level success drops to zero, and because retention scores levels by `sr·(1−sr)`, **a level the agent cannot touch is scored identically to one it has mastered, and is evicted the same way.** That blind spot belongs to any system that retains levels by a learnability criterion — it is not specific to our environment.
>
> **Second, one measured positive.** The scheduler does buy the top of the iron chain — armour +10.0, four-point-four sigma. In a season whose aggregate net is zero, that is the one component-level gain that clears its own error bar.

---

## Block 8 — Honest closing (1:00)

> What we can defend: across twelve pre-registered interventions — teacher-side machinery, student-side rewards, supply — **not one moved the 2×10⁹ return outside the seed band**, with a single exception, and that exception is a supply parameter, not a mechanism.
>
> What we cannot: we never varied the teacher. Every arm this season ran the same 14B, so "the ceiling is set by model capability" is a hypothesis with **zero** evidence in our data — and three measurements pointing the other way, including the fact that the depth wall stands for a 235B teacher too.
>
> Certification is in flight: three seeds of the recommended configuration land tomorrow.
>
> And the sentence I would leave you with is about method rather than results: **this season's conclusion was overturned by its own data four times, and each overturn is on the record with a timestamp.** That is what the frozen-criteria discipline was for.

---

## Optional blocks (add if the slot is 12 minutes)

### Block 4b — The trade nobody avoided (1:00)
> Three arms that unlocked the deep cluster by three completely different routes paid **the same** surface-skill cost: −54, −69, −55 points. When three different mechanisms produce one bill, the bill is not theirs — it comes from the replay pool budget. Breadth for depth is intrinsic to the architecture, not to any of our choices.

### Block 6b — Where the ceiling actually binds (2:00)
> Our own configuration audit found the pool saturated for **85% of every run** — capacity 100, roughly 750 generated tasks competing for those slots across a season, 650 evicted. Retention is a learnability tournament with a fixed number of seats. That is where the next experiments go, and the dose design is pre-registered.

---

## Timing table

| Block | Budget | Cumulative |
|---|---|---|
| 1 Question | 0:45 | 0:45 |
| 2 Instrument | 1:15 | 2:00 |
| 3 Five arms + accident | 1:45 | 3:45 |
| 4 Four arms | 2:30 | 6:15 |
| 5 Coupling | 1:15 | 7:30 |
| 6 Verdict | 1:30 | 9:00 |
| 7 What survives | 1:00 | 10:00 |
| 8 Closing | 1:00 | **11:00** |
| *(4b + 6b optional)* | *+3:00* | *14:00* |

**Never cut**: Blocks 4, 5, 6, 7 — the four pillars of the current story.
**Cut first**: Block 2's second half (keep "in-training evaluation misled us four times"), then Block 3's arm-by-arm numbers (keep 43.12 / 44.01 / 45.66).

---

## Q&A — updated red lines and standard answers

**Red lines (never say):**
1. "Shuffling improves the curriculum" — single arm, single permutation.
2. "Machinery is load-bearing, knowledge is not" — retired 8/07, never revive.
3. "The true graph is a lock" **as the explanation** — the lock is real but not binding; the floor clause is.
4. "The ceiling is set by the 14B" — the teacher axis was never varied.
5. Any per-skill claim without its error bar.

**Standard answers:**

- **"Isn't 45.66 just a lucky permutation?"** → We answered it mechanistically rather than by re-running it: coupling fell from 60% to 6.3%, and the shuffled scheduler emitted only 12 unique targets in 151 cycles. It is not a better curriculum; it is a curriculum that stopped steering.
- **"Does this refute DiCode?"** → No. Their numbers stand. What we identify is a **confound in the teacher-scaling interpretation**: the ladder was measured under a configuration tuned for the largest teacher. And note our results are among the strongest defences of LLM-written curricula that exist — 45.02 from a 14B against 34.6 for the best non-LLM lineage arm at the same budget. The credit moves from "bigger teacher" and "more machinery" to "an LLM writing code-level tasks at all, plus supply configuration."
- **"Your recommended config has n=1."** → Correct, and stated on every slide it appears. Three seeds land tomorrow; the band will be reported whatever it says.
- **"Why is the stack net negative if the mechanism arms all passed?"** → Because passing a mechanism threshold and moving aggregate return are different claims. The per-skill separations are 6 to 22 sigma; the aggregate difference is inside the seed band. That gap is itself our result about return being a low-resolution instrument.
- **"What would change your mind?"** → A teacher ladder on the same pipeline. We have never run one, and it is the most valuable untested axis we have.

---

## Numbers card (for the podium)

| | last-10 | last-3 |
|---|---|---|
| base2e9 (vanilla, parents=2) | 43.12 | 43.35 |
| **base15** (vanilla, parents=15) | **45.02** | **45.83** |
| longStack s1 / stack n=3 | 44.73 / **44.01 ± 0.79** | 45.13 |
| shufGraph | 45.66 | 45.85 |
| gateOff | 45.06 | 45.33 |
| repairF | 44.37 | 45.09 |
| surgical | 44.86 | 46.47 |

Anchors (their SE, 8 seeds): 30B **44.38 ± 0.66** · 80B **45.88 ± 0.40** · 235B **48.55 ± 0.49** · Goal-Only 42.82 ± 0.39.
Conditional hits (mob ÷ mines): longStack .44/.03/.09 · gateOff .79/.24/.72 · surgical .80/.34/.60 · base15 .82/.54/.79 · shuf .79/.58/.79.
Season cost: ≈955 GPU·h measured/estimated, 63.4 h per full arm, teacher token cost $0 (local).
