# Skill-Preflight UED — repo guide

Branch: `skill-preflight-ued_Mason` on `gregjones11235/mechanism_UED`
Maintainer: Mason · last updated 2026-08-08

Everything this branch adds is **flag-gated and off by default**, so the vanilla DiCode
path is unchanged. If you pass none of the `+skill_preflight.*` / `+training.sil_*`
flags, you get stock behaviour.

**What's in here**

| Component | Where | Purpose |
|---|---|---|
| SkillGraph prereq scheduler | `src/dicode/skill_preflight/skill_scheduler.py` | picks which achievements the teacher targets each session |
| Preflight task filter | `src/dicode/skill_preflight/` | rejects generated tasks that don't compile / don't reach the target |
| Scaffold gate | same | blocks two-channel teaching (task text leaking the solution) |
| Death forensics ("necropsy") | `src/dicode/necropsy.py` | per-death diagnostics attached to the official eval |
| SIL collector + BC | `experiments/analysis/sil_collect.py`, `src/dicode/sil_bc.py` | capture successful segments, clone them back into the policy |
| Verdict tools | `experiments/analysis/necro_verdict.py`, `iron_triage.py` | read a corpse and print the verdict tables |

---

## 1. Quickstart

### 1.1 Pod

A100 SXM (80 GB). On CUDA-13 pods you must point the loader at the bundled CUDA-12 libs
or JAX won't see the GPU:

```bash
export LD_LIBRARY_PATH=$(python -c "import site,glob,os;print(':'.join(glob.glob(os.path.join(site.getsitepackages()[0],'nvidia','*','lib'))))"):$LD_LIBRARY_PATH
```

### 1.2 Python environment

```bash
python -m venv /workspace/venv && source /workspace/venv/bin/activate
pip install -e /workspace/mechanism_UED/dicode_src

# pyproject.toml carries NO version constraints, so a fresh venv resolves to whatever
# is newest that day, and plain `jax`/`jaxlib` are CPU-only wheels. Pin by hand:
pip install "jax[cuda12]==0.6.2" jaxlib==0.6.2 orbax-checkpoint==0.11.18 \
            flax==0.10.7 optax==0.2.5 chex==0.1.89 craftax==1.4.5

python -c "import jax; assert jax.default_backend()=='gpu'; print(jax.devices())"
```

> `uv.lock` agrees on these versions but contains **no CUDA plugin** — `uv sync` alone
> still yields CPU-only JAX.

Working set on the maintainer's pod (regenerate with
`pip freeze | grep -Ei "^(jax|orbax|flax|optax|chex|craftax|numpy|ml_dtypes)"`):

```
python 3.12.3
jax / jaxlib / jax-cuda12-plugin / jax-cuda12-pjrt  0.6.2
orbax-checkpoint 0.11.18   flax 0.10.7   optax 0.2.5   chex 0.1.89
craftax 1.4.5   numpy 2.5.1   ml_dtypes 0.5.4
```

> **`craftax==1.4.5` is the pin that matters for the science.** The orbax/jax pins decide
> whether you can *read* someone's checkpoint; a different craftax is a different
> environment, which makes scores **incomparable**, not merely unreadable.

> **Checkpoint compatibility.** `sharding passed to deserialization should be
> specified... Got None` when restoring someone else's checkpoint is an orbax version
> difference **or a device/topology mismatch** (forcing `JAX_PLATFORMS=cpu` on a
> GPU-written checkpoint reproduces it exactly — always restore on a GPU), not a corrupt
> file. The two pins that matter for reading each other's
> checkpoints are `orbax-checkpoint==0.11.18` and `jax==0.6.2`; numpy / ml_dtypes patch
> versions can differ without trouble.

Housekeeping: delete `WANDB_API_KEY=unused` from `.env` if present. If the Craftax
texture cache is corrupt, delete `texture_cache.pbz2` and let it rebuild.

### 1.3 Ollama (teacher LLM + embeddings)

`OLLAMA_KEEP_ALIVE`, `OLLAMA_CONTEXT_LENGTH` and `OLLAMA_MODELS` are **server-side**
variables — setting them in the training env does nothing. Start one server per card:

```bash
pkill ollama; sleep 2; pgrep -af ollama          # MUST be empty — see the port trap below

nohup env OLLAMA_HOST=127.0.0.1:11434 CUDA_VISIBLE_DEVICES=0 \
     OLLAMA_KEEP_ALIVE=-1 OLLAMA_CONTEXT_LENGTH=32768 \
     OLLAMA_MODELS=/root/ollama_models ollama serve > /root/ollama_0.log 2>&1 &
sleep 6 && grep -E "OLLAMA_KEEP_ALIVE|OLLAMA_CONTEXT_LENGTH|OLLAMA_MODELS" /root/ollama_0.log

OLLAMA_HOST=127.0.0.1:11434 ollama pull qwen2.5-coder:14b   # hydra name: local_qwen14b
OLLAMA_HOST=127.0.0.1:11434 ollama pull nomic-embed-text
grep "offloaded 49/49" /root/ollama_0.log        # all layers on GPU; if not, don't launch
```

For a second card repeat with `OLLAMA_HOST=127.0.0.1:11435 CUDA_VISIBLE_DEVICES=1` and
point that arm's `GENERATION_SERVER_URL` / `EMBEDDING_SERVER_URL` at `:11435`.

> **Port trap (cost us a 9 GB download to the wrong disk, with no error).** If a server
> already holds `:11434`, a second `serve` exits with `bind: address already in use`, and
> a later `pull` **silently attaches to the old server** and inherits *its*
> `OLLAMA_MODELS`. Always: pkill → confirm `pgrep` empty → start → grep the three
> variables out of the new log → only then pull.
>
> `ollama pull` downloads **via the server**, so restarting `serve` aborts an in-flight
> pull. Retries logged against `/embeddings` and `/chat` during training are benign.
>
> VRAM budget per card: trainer ~61.3 GB + ollama ~15.9 GB ≈ 77 GB of 80 GB. A larger
> teacher does not fit locally alongside the trainer.

### 1.4 Smoke test before you burn a night

```bash
cd /workspace/mechanism_UED/dicode_src && source /workspace/venv/bin/activate
python -c "import jax; print(jax.devices())"                 # should list the GPU
python -c "import craftax; print(craftax.__version__)"       # must be 1.4.5
curl -s http://localhost:11434/v1/models | head -c 200        # ollama up
python -c "from dicode.skill_preflight.prereq_graph import DIRECT_PREREQS as G; import hashlib,json; \
print(hashlib.md5(json.dumps({k:sorted(v) for k,v in sorted(G.items())}).encode()).hexdigest()[:12])"
# expect cfbb1c9a4558 (unshuffled graph) — see 6b
```

---

## 2. Launching a run

```bash
FORK=/workspace/mechanism_UED/dicode_src/outputs/<run_name>
mkdir -p $FORK/rl_checkpoints

tmux new-session -d -s <run_name> "cd /workspace/mechanism_UED/dicode_src && \
source /workspace/venv/bin/activate && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75 \
  CUDA_VISIBLE_DEVICES=0 \
  GENERATION_SERVER_URL=http://localhost:11434/v1 \
  EMBEDDING_SERVER_URL=http://localhost:11434/v1 && \
timeout 27000 python experiments/training/run_dicode.py \
  hydra.run.dir=$FORK seed=1 \
  use_wandb=true wandb_entity=mechanism_UED wandb_project=Skill_Preflight_UED \
  training.total_timesteps=2000000000 \
  gen_manager/llm@gen_manager.task_generator=local_qwen14b \
  gen_manager/llm@gen_manager.env_generator=local_qwen14b \
  gen_manager.embedding_model.model=nomic-embed-text \
  gen_manager.task_generator.max_tokens=8192 \
  gen_manager.env_generator.max_tokens=8192 \
  gen_manager.task_generator.base_url=http://localhost:11434/v1 \
  gen_manager.env_generator.base_url=http://localhost:11434/v1 \
  gen_manager.embedding_model.base_url=http://localhost:11434/v1 \
  dicode_manager.additional_num_parents=15 \
  +skill_preflight.use_scheduler=true \
  +skill_preflight.use_preflight=true \
  +skill_preflight.mastery_threshold=0.6 \
  +skill_preflight.frontier_mode=prereq \
  +skill_preflight.prereq_threshold=0.3 \
  +skill_preflight.use_scaffold_gate=true \
  +skill_preflight.r3_mastered_exemption=true \
  > /root/run_<run_name>.log 2>&1"
```

> **Two cards.** `CUDA_VISIBLE_DEVICES` is per-arm and the old template omitted it; on a
> two-card pod, set `0` / `1` and point each arm at its own ollama port.
>
> **base_url as a hydra override too.** Environment variables never appear in
> `.hydra/overrides.yaml`, so pass the three `base_url`s through hydra as well — that is
> what makes an arm's teacher endpoint reconstructable months later.
>
> **Thresholds.** The block above is the 2026-summer base stack: `mastery_threshold=0.6`
> plus `r3_mastered_exemption`. Early-season arms used `0.2`; the threshold line is
> closed — 0.6 vs 0.2 from scratch landed inside the noise band, the only real difference
> being that 0.6 roughly doubles mid-tier crafting volatility. Say which you used.
>
> **Upstream-vanilla baseline** = drop every `+skill_preflight.*` flag and set
> `dicode_manager.additional_num_parents=2` (the module default). Measured wall clock:
> full stack 60.3–66.0 h per 2e9 arm; vanilla ~52 h.
>
> **Checkpoints belong on the container disk.** `/workspace` is MooseFS: buffered writes
> are fast but `fsync` is ~1000× slower than local disk. Keep the venv and repo on
> `/workspace`, put `hydra.run.dir` and `OLLAMA_MODELS` on `/root`, and rsync back every
> 30 min (the container disk is wiped by every pod restart). Hang probe:
> `timeout 10 ls /workspace >/dev/null 2>&1; echo $?` (124 = hung).

Two fingerprints to check ~15 min in, before you go to sleep:

```bash
grep -a "Restored Optimizer Step Count" /root/run_<run_name>.log   # lineage
grep -a "SIL-BC. phase start"          /root/run_<run_name>.log   # SIL arms only
grep -a "offloaded 49/49"              /root/ollama_0.log         # teacher fully on GPU
grep -a "\[Preflight\] ERROR (kept all, gate inactive!)" /root/run_<run_name>.log   # MUST be empty
```

That last one is the expensive silence: if `validation` is missing from the config the
preflight gate raises, the exception is swallowed into that single line, **every**
generated task is admitted, and the arm is scientifically void while looking healthy.

### Flags that matter

| Flag | Default | Effect |
|---|---|---|
| `+skill_preflight.use_scheduler` | off | turn the SkillGraph scheduler on |
| `+skill_preflight.use_preflight` | off | filter generated tasks before training on them |
| `+skill_preflight.frontier_mode` | `prereq` | `prereq` = target a skill only when its direct prerequisites are above `prereq_threshold` |
| `+skill_preflight.mastery_threshold` | **0.6** | a skill stops being a target once its success rate passes this |
| `+skill_preflight.prereq_threshold` | 0.3 | bar each direct prerequisite must clear |
| `+skill_preflight.use_scaffold_gate` | off | reject tasks whose text leaks the solution |
| `+skill_preflight.r3_mastered_exemption` | off | let rule 3 grant an already-mastered direct prerequisite |
| `dicode_manager.additional_num_parents` | **2** | extra archive parents drawn per cycle for mutation; base stack uses **15** (→ ~25 parents/cycle, measured 1899–1900 tasks per 2e9 arm) |
| `+training.sil_coef` | 0 (no-op) | weight of the self-imitation BC loss |
| `+training.sil_buffer` | — | path to a golden-segment buffer |
| `+training.sil_burn` | 48 | burn-in steps before the BC loss applies (0 = ablation) |

Note `max_target_achievements = 6` in `skill_scheduler.py`: eligible skills are sorted
hardest-first and truncated to six. **The slot cap, not the threshold, usually decides
who actually gets taught** — check the frontier lists in the log before concluding that a
threshold change did nothing:

```bash
grep -a "SkillGraph" /workspace/run_<run_name>.log | tail -5
```

---

## 3. Evaluation

The official protocol is 1024 frozen held-out Craftax worlds, seed 0.

```bash
python experiments/training/eval_checkpoints.py \
  hydra.run.dir=/tmp/<tag> use_wandb=false seed=0 \
  gen_manager/llm@gen_manager.task_generator=local_qwen14b \
  gen_manager/llm@gen_manager.env_generator=local_qwen14b \
  gen_manager.embedding_model.model=nomic-embed-text \
  +eval.ckpt_root=<run_dir>/rl_checkpoints \
  "+eval.steps=[1700]" +eval.tag=<TAG> +eval.details=true \
  2>&1 | grep -aE "RESULT|EVAL_DONE"

python experiments/analysis/necro_verdict.py /tmp/<tag>/eval_<TAG>_seed0_details.json
python experiments/analysis/iron_triage.py   /tmp/<tag>/eval_<TAG>_seed0_details.json
```

Gotchas: eval params need a leading `+`; you must repeat the same `gen_manager`
overrides as training; `use_wandb=false` is fine here (it is **not** fine for training).

`necro_verdict` prints death-floor distribution, per-floor step/damage/exchange ledger,
death context (melee/ranged proximity, food, drink), and the kill matrix.
`iron_triage` prints per-achievement rates and the return premium of achievers vs
non-achievers. Premiums are **correlational upper bounds** — achievers are stronger
episodes for many reasons — so don't read them as "teaching this skill buys N points".

---

## 4. SIL pipeline (optional)

Collect golden segments from a donor checkpoint, then clone them back:

```bash
python experiments/analysis/sil_collect.py \
  hydra.run.dir=/tmp/collect use_wandb=false seed=0 \
  +sil.ckpt_root=<donor_run>/rl_checkpoints '+sil.step=400' \
  +sil.mode=descend \
  +sil.tag=TAG +sil.out=/workspace/golden_buffer +sil.rollouts=8
```

Four trigger modes:

| mode | fires when | teaches |
|---|---|---|
| `descend` | agent crosses into floor 2 | going down |
| `stay` | 64 consecutive steps on floor 2 | staying alive down there |
| `resource` | drinking while thirsty | resource management |
| `skill` | target achievement's first flip (`+sil.skill=MAKE_IRON_ARMOUR`) | the prep → craft sequence |

Then train with `+training.sil_coef=1.0 +training.sil_buffer=/workspace/golden_buffer`.
The buffer is capped at 512 segments, admitted by episode return.

**Conditioning matters.** The collector must feed the same multi-hot achievement vector
the official eval uses; an all-zero vector is out-of-distribution and silently suppresses
the behaviour you're trying to capture. This produced one wrong verdict before it was
caught — if collection rates look impossible, check the conditioning first.

---

## 5. Operating rules (learned the hard way)

**Resuming**
- Never resume in place. Always: new `hydra.run.dir`, copy the latest checkpoint to
  `rl_checkpoints/0`, and copy `task_graph.graphml` + `runtime_analysis/` from the run
  you're continuing. In-place resume silently loses checkpoints — the segment-local
  counter races orbax's `latest_step` guard and there is no error.
- Keep `training.total_timesteps=2000000000` on every fork and resume, **and set the
  learning rate the position calls for** (or restore the anneal counter). Optimizer reset
  zeroes the anneal counter, so a fork trains at the full initial LR where the position
  wants ~1/10 of it; the arm then dies at a *fixed total-step position* with a fixed
  signature: `value_loss → 1.5e10`, `grad_norm → 2.5e10`, `entropy → 0`,
  `return ≡ −0.90`.
  <br>**Correction to earlier guidance.** This was previously written up as an unclamped
  anneal driving the LR through zero into gradient *ascent*. A dedicated crash test bench
  falsified that: three candidates — value-loss normalisation, a critic/​trunk gradient
  firewall, and the anneal clamp itself — all still died at the same position, while an
  arm at 1/10 LR traversed it healthy. The clamp remains as a harmless guardrail; the
  root-cause wording in commit `ff6b956` needs an erratum.

**Checkpoints and shutdown**
- Orbax writes asynchronously and lags by roughly 20 minutes. At a cutoff, wait for the
  final `Checkpointing` directory to appear on disk before killing anything.
- `tmux kill-session` does not reliably kill the trainer — it leaves orphans still
  holding the GPU (this cost two hours of a card before anyone noticed). Kill by PID:
  `kill $(pgrep -f "outputs/<arm_name>")`, then verify `pgrep` is empty and the card's
  memory has dropped before launching anything else.

**Budgets**
- About 25 minutes per session. A 17-session arm wants `timeout 27000`; SIL arms are
  slower, use 27000–30000.

**Logging and analysis**
- `global_env_steps` resets on resume — plot against `session`, not steps.
- Training runs need `use_wandb=true`; the `false` path is broken for training.
- `wandb` `scan_history(keys=[...])` hits a broken API path. Scan everything and filter
  client-side.
- Don't pass `+validation=default`.
- **In-loop `evaluation/*` used to read the training env's slot**, so any arm wrapped in a
  reward modifier reported inflated return with clean achievement bits (measured +3.4,
  matching `2.0 × (0.86 + 0.73)` to the decimal). Fixed: every session now runs a real
  held-out evaluation for `evaluation/*`, and training-side metrics were renamed
  `evaluation_shaped/*`, with a structural test pinning it. If you ever see the two used
  interchangeably, treat it as a bug.
- **`ep_len` alone is not evidence.** Two same-config arms differed by 55–82 % in episode
  length while their returns differed by 0.30–1.13.

**Editing**
- Python files in the repo are CRLF. Scripted edits must use `open(..., newline='')` or
  you'll rewrite every line ending.

---

## 6. How we judge results

This is the part most worth adopting, because it changed several of our own conclusions.

**Measured noise floor (this system, official eval):**

| Source | Magnitude |
|---|---|
| Checkpoint jitter (last 10 sessions of an arm) | SD 0.37 – 0.54 |
| Seed difference, same arm, position-matched | 1.13 pts (0.68 after averaging) |
| Same-config re-run, single early offline checkpoint | ±2.31 |
| **Band vs training stage** | **~7 inside the skill-takeoff window; 0.3 – 1.1 after ~s15** |
| Achievement rate — armour | 1.7 pp |
| Achievement rate — pickaxe / sword | 5 – 6 pp (±9 pp at takeoff) |
| Floor-2 residence steps | effect ~17 vs noise ~4 |

The band is **not a constant**: measured at six budgets it is a peak, near zero before
takeoff, ~7 while one arm's iron chain has ignited and the other's hasn't, then settling
to ~1. Read every difference against the band for *its* stage — most early-window gaps
are unadjudicable no matter how large they look. Cross-check: three seeds of the same arm
differ pairwise by 1.05 on average, and the comparison paper's SE × √n lands at 1.1–1.4.

Consequences:

1. **A single corpse is an observation, never a verdict.** Report score as a mean over
   the last N sessions, or over three adjacent checkpoints. This is free and it halves
   the seeds needed to resolve a 1-point difference (8 → 4).
2. **Prefer behaviour needles.** Floor-2 residence has ~4x signal-to-noise at a single
   seed; score has ~0.9 and cannot resolve anything we've built.
3. **Pre-register the criterion before ignition** — the metric, the threshold, and what
   the reverse result would mean. We've had two attractive findings die this way, both
   correctly.
4. **A null only counts if the intervention reached the tested path.** Before concluding
   "X doesn't work", verify X actually changed what it was supposed to change (e.g. that
   a curriculum flag actually altered the frontier lists).
5. **Only offline, multi-checkpoint evaluation adjudicates.** In-training curves misled us
   four times in one season — three by noise (wrong direction or wrong magnitude versus
   the offline re-measurement) and once structurally (the leakage above). Training curves
   are for spotting that something is alive, not for deciding what is true.

---

## 6b. Prerequisite graph and the shuffle ablation

`DIRECT_PREREQS` in `src/dicode/skill_preflight/prereq_graph.py` is the one-hop
prerequisite graph over the 67 Craftax achievements (94 edges). It feeds **both** the
scheduler (`frontier_mode=prereq`) and the scaffold gate, so a change here moves two
mechanisms at once.

The file carries an **environment-gated shuffle patch**: with `SP_SHUFFLE_PREREQ` unset it
is a byte-for-byte no-op; set it (e.g. `SP_SHUFFLE_PREREQ=20260804`) and the graph is
randomly rewired by double-edge swaps that preserve the node set, the edge count, every
node's in- and out-degree, and acyclicity — only *which* achievement gates *which* is
scrambled. It prints a fingerprint line on import:

```
[PREREQ-SHUFFLE] seed=20260804 nodes=67 edges=94 swaps=1880/2566 changed=79/94 degseq=OK
```

Graph fingerprints: original `cfbb1c9a4558`, seed-20260804 shuffle `1354f4e59b14`. Self-check:

```bash
python -c "from dicode.skill_preflight.prereq_graph import DIRECT_PREREQS as G; import hashlib,json; \
print(len(G), sum(len(v) for v in G.values()), \
hashlib.md5(json.dumps({k:sorted(v) for k,v in sorted(G.items())}).encode()).hexdigest()[:12])"
# unset  -> 67 94 cfbb1c9a4558
```

**Invariant: the edge list must be `sorted()`.** The first shuffled arm iterated
`frozenset`s, so `PYTHONHASHSEED` gave **every process its own graph**; the fingerprint
line caught it and the arm was voided and re-run. Keep the sort, and require that any
shuffled arm's log contains the fingerprint line — no line, no arm.

One structural finding worth knowing before you touch thresholds: under the true graph
with `prereq_threshold=0.3`, **27 achievements are permanently ineligible as targets**,
because a gate's best-ever mastery across every arm we ran stays below the bar (e.g. the
whole floor-2 combat cluster sits behind `enter_gnomish_mines`, which peaks at 0.12).
Correct, conservative knowledge combined with reachability-blind gating behaves as a
**lock** on exactly the skills that most need curriculum. Treat "the graph is true" and
"the graph is usable by the scheduler" as separate properties.

---

## 7. Where things live

- Run tracking: wandb project `mechanism_UED/Skill_Preflight_UED`
- Training outputs: `dicode_src/outputs/<run_name>/rl_checkpoints/`
- Golden buffers: `/workspace/golden_*`
- Logs: `/workspace/run_<run_name>.log`
- Preflight cost work (baselines, criteria, traps): [PREFLIGHT_COST.md](PREFLIGHT_COST.md)
- Design notes and weekly reports live alongside this file; the SIL design card and the
  handoff notes have the full derivations.
- **Number disputes are settled by the data compendium** (数据总账, 2026-08-07 final
  edition, held by Mason). Where this guide and the compendium disagree, the compendium
  wins.

### Reference runs

| arm | wandb id | last session |
|---|---|---|
| fork source / resumeAB | `2oyy46uv` | 154 |
| from-scratch 2e9, threshold 0.2 | `hdodsb5l` | 66 |
| 14B-only baseline (3e8, no skill_preflight) | `mc75k0nx` | 23 |
| threshold 0.4 (3e8) | `506cdgz3` | 37 |
| threshold 0.6 (late fork) | `w8mlwsi8` | 172 |
| placebo seed-1 | `5qnqbjjb` | 186 |
| placebo seed-2 | `r369pox1` | 174 |
| phi tail | `gejgawhc` | 175 |
| SIL, dual-donor buffer | `2awp5kbt` | 170 |
| SIL, baseline-only buffer | `vnpcrp0y` | 172 |
| LOCK seed-1 | `5dfg8rr1` | 174 |
| SKILL (armour library) | `pgh95yfl` | 167 |

> Every one of these shows `state: crashed` in wandb. That is **normal** — it is how a
> `timeout`-terminated trainer exits, and the checkpoints are intact. The one genuine
> failure is `5qnqbjjb`, which ran past the anneal horizon: its return collapses from
> 42.19 at session 172 to −0.90 at session 173. That is the learning-rate-crossing-zero
> failure described in section 5, and it is what the clamp fix on this branch prevents.

Questions → Mason. If something in section 5 bites you anyway, tell me and I'll add it.
