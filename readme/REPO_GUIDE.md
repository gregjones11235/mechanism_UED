# Skill-Preflight UED — repo guide

Branch: `skill-preflight-ued_Mason` on `gregjones11235/mechanism_UED`
Maintainer: Mason · last updated 2026-07-25

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
```

Working set on the maintainer's pod (regenerate with
`pip freeze | grep -Ei "^(jax|orbax|flax|optax|chex|craftax|numpy|ml_dtypes)"`):

```
python 3.11.15
jax / jaxlib / jax-cuda12-plugin / jax-cuda12-pjrt  0.6.2
orbax-checkpoint 0.11.18   flax 0.10.7   optax 0.2.5   chex 0.1.89
craftax 1.4.5   numpy 2.5.1   ml_dtypes 0.5.4
```

> **Checkpoint compatibility.** `sharding passed to deserialization should be
> specified... Got None` when restoring someone else's checkpoint is an orbax version
> difference, not a corrupt file. The two pins that matter for reading each other's
> checkpoints are `orbax-checkpoint==0.11.18` and `jax==0.6.2`; numpy / ml_dtypes patch
> versions can differ without trouble.

Housekeeping: delete `WANDB_API_KEY=unused` from `.env` if present. If the Craftax
texture cache is corrupt, delete `texture_cache.pbz2` and let it rebuild.

### 1.3 Ollama (teacher LLM + embeddings)

```bash
export OLLAMA_MODELS=/workspace/ollama_models
ollama serve > /workspace/ollama_server.log 2>&1 &
ollama pull qwen2.5-coder:14b     # served to hydra as local_qwen14b
ollama pull nomic-embed-text
```

Set `OLLAMA_KEEP_ALIVE=-1` in the training env or the model unloads between sessions and
every session pays the reload.

### 1.4 Smoke test before you burn a night

```bash
cd /workspace/mechanism_UED/dicode_src && source /workspace/venv/bin/activate
python -c "import jax; print(jax.devices())"                 # should list the GPU
curl -s http://localhost:11434/v1/models | head -c 200        # ollama up
```

---

## 2. Launching a run

```bash
FORK=/workspace/mechanism_UED/dicode_src/outputs/<run_name>
mkdir -p $FORK/rl_checkpoints

tmux new-session -d -s <run_name> "cd /workspace/mechanism_UED/dicode_src && \
source /workspace/venv/bin/activate && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75 OLLAMA_KEEP_ALIVE=-1 \
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
  +skill_preflight.use_scheduler=true \
  +skill_preflight.use_preflight=true \
  +skill_preflight.mastery_threshold=0.2 \
  +skill_preflight.frontier_mode=prereq \
  +skill_preflight.prereq_threshold=0.3 \
  +skill_preflight.use_scaffold_gate=true \
  > /workspace/run_<run_name>.log 2>&1"
```

> The template above passes `mastery_threshold=0.2`, which is the value our arms have
> used historically — **not** the module default (`0.6`). Keep 0.2 only if you are
> reproducing an existing arm; for a new experiment decide deliberately and say which
> you used.

Two fingerprints to check ~15 min in, before you go to sleep:

```bash
grep -a "Restored Optimizer Step Count" /workspace/run_<run_name>.log   # lineage
grep -a "SIL-BC. phase start"          /workspace/run_<run_name>.log   # SIL arms only
```

### Flags that matter

| Flag | Default | Effect |
|---|---|---|
| `+skill_preflight.use_scheduler` | off | turn the SkillGraph scheduler on |
| `+skill_preflight.use_preflight` | off | filter generated tasks before training on them |
| `+skill_preflight.frontier_mode` | `prereq` | `prereq` = target a skill only when its direct prerequisites are above `prereq_threshold` |
| `+skill_preflight.mastery_threshold` | **0.6** | a skill stops being a target once its success rate passes this |
| `+skill_preflight.prereq_threshold` | 0.3 | bar each direct prerequisite must clear |
| `+skill_preflight.use_scaffold_gate` | off | reject tasks whose text leaks the solution |
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
- Keep `training.total_timesteps=2000000000` on every fork and resume. It defines the LR
  anneal horizon; changing it changes the schedule, and past the horizon an unclamped
  anneal drives the learning rate through zero and turns Adam into gradient *ascent*.
  (This caused seven crashes before it was found; the clamp fix is on this branch.)

**Checkpoints and shutdown**
- Orbax writes asynchronously and lags by roughly 20 minutes. At a cutoff, wait for the
  final `Checkpointing` directory to appear on disk before killing anything.
- `tmux kill-session` does not reliably kill the trainer. Kill the PIDs from
  `pgrep -af run_dicode` and verify with `pgrep` afterwards.

**Budgets**
- About 25 minutes per session. A 17-session arm wants `timeout 27000`; SIL arms are
  slower, use 27000–30000.

**Logging and analysis**
- `global_env_steps` resets on resume — plot against `session`, not steps.
- Training runs need `use_wandb=true`; the `false` path is broken for training.
- `wandb` `scan_history(keys=[...])` hits a broken API path. Scan everything and filter
  client-side.
- Don't pass `+validation=default`.

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
| Achievement rate — armour | 1.7 pp |
| Achievement rate — pickaxe / sword | 5 – 6 pp |
| Floor-2 residence steps | effect ~17 vs noise ~4 |

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

---

## 7. Where things live

- Run tracking: wandb project `mechanism_UED/Skill_Preflight_UED`
- Training outputs: `dicode_src/outputs/<run_name>/rl_checkpoints/`
- Golden buffers: `/workspace/golden_*`
- Logs: `/workspace/run_<run_name>.log`
- Design notes and weekly reports live alongside this file; the SIL design card and the
  handoff notes have the full derivations.

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
