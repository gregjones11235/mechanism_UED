# Focused Milestone-Chain Curriculum for Deep Craftax Achievements

*A design pitch. The goal is to break through the tier-3/4 achievement ceiling that DiCode itself never broke — by teaching an automated curriculum teacher to behave less like a scheduler and more like a player who gets stuck on a hard boss, focuses, and grinds through it.*

---

## TL;DR

- **Problem.** DiCode's LLM curriculum teacher (and our previous iterations of it) all plateau at ~10% success on tier-3/4 Craftax achievements (gnome warrior, diamond sword). The paper treated 0%→10% as success and stopped. We want to break that ceiling.
- **Root cause.** DiCode compresses "already-mastered" prerequisites into the level's initial state so the agent only trains the *final hop*. This is correct for short chains but **cuts long chains**: the intermediate steps of a deep achievement are things the student *also* doesn't know, so once compressed away, the held-out agent has no way to fill the gap → it never transfers.
- **Core idea (H1).** Instead of spreading effort thinly across all 23 tier-3 goals, the teacher **focuses on one hard combat goal at a time, preserves its full un-mastered prerequisite chain, breaks through it, and distills a transferable "playing style"** — positioning, gear-up timing, kiting, escaping. That style then accelerates the next combat goal. Gear-crafting goals are deterministic *enablers* within a combat goal's chain, not independent targets.
- **How the teacher stays honest.** We do **not** feed it a curriculum answer key. It reasons out the attack chain from (a) the game's crafting/world mechanics — which the baseline teacher already sees — and (b) the student's own behavior. It decides *what's a hard wall* and *what order to attack* by "feeling stuck" from success rates, exactly as a player would.

---

## 1. Why the baseline cuts long chains (the thing we're fixing)

DiCode splits every level's achievements into two sets: **Relevant** (what the agent must actively achieve here — the level's win condition) and **Completed** (prerequisites baked into the initial inventory, so the agent doesn't redo them). "Compressing a long chain" = stuffing the prerequisites into the starting state and marking them Completed, leaving only the new dependency to learn.

This is a *reasonable* design for three reasons: it fixes credit assignment (long chains have sparse rewards, RL can't walk them from scratch); it enforces one-main-change-per-level (no difficulty spikes, levels survive the learnability filter); and it keeps average success rate near the ZPD sweet spot of ~0.5.

It works beautifully for **short chains**: iron armour, reaching floor 2. The compressed-away prerequisites are things a competent held-out agent genuinely already has (tier-1/2 skills are 90%+), so it fills them in itself and the skill transfers.

It **fails for long chains**. For a deep combat goal, the intermediate steps — surviving a dark floor, reaching floor 2, gearing up mid-run — are things the student *doesn't* reliably know either. Compress them away, and at held-out evaluation the agent is dropped in front of the boss with no way to have gotten there. It can't fill the gap. Nothing transfers. This is exactly why every arm plateaus at the same ~10% the DiCode paper reported. **Breaking a ceiling DiCode never broke is the novelty and the comparison value of this work.**

## 2. Core hypothesis H1: combat breakthroughs distill a transferable style

- **Playing style** = the set of *dynamic* strategies a student picks up while cracking a hard combat goal: positioning, when to gear up, pulling one enemy at a time, when to run, lighting dark floors, routing resources, combining deep navigation with resource management.
- **Only combat goals produce style.** Beating a tough elite mob requires genuine dynamic decision-making. Gear-crafting goals (diamond sword/armour) are deterministic recipes — they are the *key that makes a fight attemptable* (an enabler), indispensable but style-less.
- **H1:** breaking through one combat goal — *including its complete prerequisite chain, in one focused push* — distills a style core that transfers to the next combat goal and accelerates it. Spreading effort across all tier-3 goals (our previous flaw) leaves every style half-formed, so everything stalls at ~10%.
- **Testable prediction:** combat goals attacked *later* in the sequence should see their held-out success rate climb *faster*. That acceleration is the direct evidence for style transfer, and it is the central claim of this design.

## 3. The attack unit and its ordering

- **Attack one combat goal at a time.** Gear goals are folded into whichever combat goal's chain they serve, not attacked independently. The combat/gear distinction is a benign category label the game already provides (COMBAT vs CRAFTING), not a curriculum prior.
- **The teacher decides what's a "hard wall" by feeling stuck — no tier table.** A hard wall = a combat achievement whose success rate is flat-near-zero for a long time *while its mechanistic prerequisites are already in place* — i.e. the student is stuck *on this*, not merely *hasn't gotten here yet*. Critically, the teacher never consults an achievement→tier mapping; it infers depth itself from success rates plus game mechanics, the same signals the baseline teacher already uses.
- **Never attack easy ground.** Achievements the student is already saturated on (collect_wood at 90%+) are off-limits as attack targets. Attacking only ever aims at the deep walls where all arms plateau.
- **Early-stage safeguard — two kinds of "low success rate".** Early on, the student is weak overall, so *easy* achievements also sit near zero. Judging "stuck" purely from "low and not rising" would mistakenly attack trivial early goals. So we distinguish:
  - **(A) A genuine hard wall** — the student has clearly passed the early stage (many achievements at mid-to-high success), this goal's prerequisites are in place, yet it stays flat-near-zero → attack it.
  - **(B) Early immaturity** — the student is broadly weak, and this low goal is simply *not its turn yet* → wait, follow the normal curriculum.
  The teacher is told this litmus test explicitly, and a maturity gate refuses to open any focus until the student is broadly competent — so even a misjudgment can't drag focus onto an easy level.
- **Attacking is a dependency tree, not a linear queue.** Within the deep achievements there is a spatially interwoven dependency DAG (see §5), where gear tasks routinely sit downstream of combat tasks. The teacher picks a combat wall, backtracks along the mechanistic/spatial dependencies, and attacks the *deepest point whose prerequisites are all already consolidated*.
- **Adaptive backtracking (the unfolding rule).** At each link on the way down, look at the student's held-out success rate:
  - **A mastered link** → don't attack it; it can legitimately be compressed into Completed (compressing something genuinely mastered doesn't hurt transfer).
  - **A flat-near-zero link** → unfold it into a sub-goal that *stays in the level and is trained on* (compressing it would recreate the chain-cutting flaw).
  This prunes mastered links automatically (no infinite backtracking) while guaranteeing un-mastered links are always trained. Depth here, too, is inferred from success rates + mechanics, never a tier table.

## 4. The intervention point: a stricter Completed/Relevant admission rule

We do **not** overturn the scaffold. DiCode's core is that the teacher decides which achievements go into Completed; the previous flaw was that it also treated *intermediate chain links* as mastered and compressed them. Our single change:

- A prerequisite may be compressed into **Completed only if the student's held-out success rate is genuinely high**.
- Intermediate chain steps the student **hasn't reliably mastered** (surviving dark floors, reaching floor 2, …) are **forced to stay in Relevant** — actually trained on in the level — and forbidden from being compressed away.

Everything else — the Completed/Relevant split, the initial-state tool, the compilation pipeline, learnability-based elimination — is unchanged. This keeps the comparison fair (same scaffolding framework), and simply replaces the teacher's gut call about "mastered" with a judgment grounded in genuine held-out mastery.

## 5. Knowledge-leakage boundary (what the teacher may and may not see)

The teacher must reason out the chain, not look it up. The boundary:

| Form | Example | Allowed? |
|---|---|---|
| (a) Full tech-tree / curriculum-chain answer key | "the optimal path to gnome_warrior = […]" | **No** — heavy leakage, the baseline has no such thing |
| (b) Crafting / world-mechanics dependency graph | "diamond sword needs a diamond; diamonds are on floor 2/7" | **Yes** — the baseline teacher already receives the full game rules |
| (c) Student behavioral evidence | "what else the student achieved in the same episode it beat a deep goal" | **Yes** — purely empirical, from the student's own trajectories |

(b) is not new leakage: the baseline teacher's prompt already injects the complete game rules (constants, mobs, mechanics, world-gen). We only *structure* that scattered text into a graph for easier reasoning — no new information. The novelty lives entirely in **the teacher inferring the chain and its order itself**, from (b) + (c), rather than being handed (a).

**The real Craftax structure makes "gear is downstream of combat" a systemic pattern**, which is why the teacher has to reason rather than pattern-match a recipe table:

- 9 vertical floors: Overworld → Dungeon → **Gnomish Mines (pitch dark; gnome warriors/archers; diamonds)** → Sewers/Vaults (enchantment tables) → Troll/Fire/Ice → Boss.
- **The diamond chain is interwoven:** crafting diamond gear needs a diamond; diamonds are effectively unobtainable in the Overworld (0.1%), so in practice you mine them on the *dark, gnome-infested floor 2*. So *crafting diamond gear (tier-3) is downstream of surviving gnome combat (tier-3)*.
- **The enchantment chain is interwoven:** enchanting needs tables on floors 3/4, which you can only reach by passing through floor 2.
- **Dark floors impose hidden thresholds:** floors 2/5/7/8 are pitch black → an *implicit* dependency on tier-2 torches that appears in *no crafting recipe*. Only world structure (b) + student behavior (c) can surface a "you need light here" dependency — direct evidence that the teacher must reason, not look up.

## 6. The mechanism modules

Four modules turn the above into a system. The first is the memory the whole thing hangs on; the next three are the evidence and consolidation around it.

### 6.1 A persistent battle notebook (the core data structure)

The attack dependency tree is **not** re-derived every session — it's a persistent "battle notebook" the teacher maintains and updates across sessions, like the notes a player builds up while stuck on a boss. It must persist because: behavioral evidence (§6.2) is sparse and needs to accumulate; a single breakthrough spans many sessions, so the teacher must remember *what it's currently attacking and which links are done*; and H1's "style" inherently depends on memory that grows over time.

The notebook holds:

| Block | Content | Update rule |
|---|---|---|
| **Current focus** | Which combat wall is being attacked | Changes only on breakthrough or abandonment → focus can't drift |
| **Prerequisite tree** | The backtracked chain + each link's mastery status | Re-marked each session from fresh success rates + co-occurrence |
| **Style note** | Free-text know-how for cracking this wall (see §6.3) | Refined every session; the reusable payload |
| **Verified-chain library** | Chains already broken through + their evidence + style | Grows on each breakthrough; the foundation for tier-4 |
| **Protected set** | Skills subject to rehearsal (§6.4) | Added on each verified breakthrough |

**How the human-intuition / hard-constraint split works.** Code owns the hard constraints and the skeleton: the notebook's persistence, the scope constraint (focus can't be an easy/saturated level), the per-link mastered/not-mastered marking (from a success-rate threshold), and the minimum condition to switch focus (only after K sessions of no progress). These can't be overridden. The LLM owns the judgment: read the previous notebook page + this session's new evidence, infer/update the prerequisite chain, decide whether to switch focus and what to attack next, judge at the semantic level whether a link is really a prerequisite. The LLM *updates* the notebook under a fixed schema rather than rewriting it from scratch — so we get the player's growing-intuition feel *and* a backstop against the well-known problem of LLM agent memory drifting into self-contradiction over a long run.

### 6.2 Student co-occurrence evidence

For each deep achievement the student *does* occasionally beat, we record — across sessions — **which other achievements it reached in the same successful episode**. This tells the teacher which skills empirically travel together with a win, so the inferred prerequisite chain comes from *real trajectories* rather than imagination. It's layered on top of (b) mechanics as an enhancement: it only kicks in once a deep goal is beaten often enough to be trustworthy (a relative success-rate floor guards against single-sample noise), and otherwise falls back cleanly to mechanics-only reasoning.

### 6.3 Behavior fingerprint — grounding the style in what the student actually did

Co-occurrence answers *which* achievements accompany a win. It cannot answer **how the student behaved to get the win** — the positioning, action mix, and pacing that actually worked. That "how" is precisely what the **style note** (§6.1) is meant to carry, and it's the physical embodiment of H1's transferable style. Without a source of truth for it, the teacher would just *invent* a plausible-sounding tactic from game lore.

The **behavior fingerprint** closes that gap cheaply. For each deep achievement, over the winning held-out episodes, we accumulate the per-episode **action counts** and **episode lengths**. Queried, it yields a compact summary: *"when the student beats this wall, a typical winning episode runs ~84 steps, is 60% movement, and heavily uses PLACE_STONE and DO with little combat — i.e. a mining/craft route, not a fight."* The teacher folds this into the style note as **grounded evidence** instead of a guess. It's not full trajectories (positioning is only partially captured by an action histogram) — that's the accepted cost of the cheap route — but it still tells the teacher the real motion/craft/combat mix. Like co-occurrence, it accumulates across sessions (single-session fingerprints are noisy), and it's suppressed below a success-rate floor, falling back to mechanics-based know-how.

**Why this matters for H1:** the style note is the *one field that carries style forward* to a later, similar wall. The skill/link/success-rate fields are just a ledger. If the note is empty or fabricated, the "transferable style" H1 depends on has no real substance. The behavior fingerprint is what makes it real.

### 6.4 Consolidation — preventing forgetting so style accumulates

Once a combat goal is broken through and its chain verified multiple times, it's consolidated to prevent catastrophic forgetting (a root cause of the previous iteration's collapse). Two-layer division of labor:

- **Student side — rehearsal (the safety net).** A broken-through chain's skills go into the protected set. When — and *only* when — forgetting is detected (a protected skill's success rate drops after peaking), extra review levels teaching those skills are **appended** to the training batch. Rehearsal is pure addition: the attack batch is never shrunk, so attack firepower stays full and review is a top-up, capped so it can't run away. If nothing is being forgotten, rehearsal does nothing and all firepower stays on the current wall.
- **Teacher side — the verified-chain library (agent memory).** A mastered, consolidated chain can be *legitimately* reused as a Completed prerequisite when building tier-4 levels (it's genuinely mastered and under active rehearsal), and it informs the ordering of the next attack.
- **The split:** the library knows *what to consolidate and what to attack next*; rehearsal makes sure the student's weights actually *remember* it.

*(Out of scope: giving the RL student its own persistent external memory. That would touch the "student stays a fixed RL network" foundation and is a separate research problem.)*

## 7. Ownership and scope

This whole package — attack tree, combat/gear classification, chain inference, verified-chain library, consolidation scheduling — is a **pure additive upgrade to the existing curriculum teacher (the "modeler")**, not a new agent. If its responsibilities eventually outgrow one prompt, or diagnosis and planning start to interfere, we split it into "state diagnosis" + "attack planning". Upgrade first; split later only if needed.

We validate in phases: (1) the stricter Completed admission rule + a prompt demanding the complete chain — enough to test H1 quickly; (2) layer on the behavioral evidence once it's dense enough; (3) if promising but insufficient, progressive multi-level chains that leave one more intermediate link per level (an extension of DiCode's existing level-evolution lineage, not a new structure).

## 8. Success criteria

1. **Does tier-3 held-out per-skill success rate break the ~10% ceiling** — the wall shared by all current arms and by the DiCode paper itself?
2. **Do combat goals attacked later in the sequence climb faster** — the direct signature of style transfer, and the core evidence for H1?
