#!/usr/bin/env python3
"""
FIX for the ★B preflight hook (B-1) in run_dicode.py.

WHY the old B-1 was broken (observed in run_AB.log):
    [Preflight] skipped (kept all): [Errno 2] No such file or directory: '.../task_5'
  - new_task_ids contains task IDs, not file paths;
  - generated tasks are NOT .py files at all — their code strings live in the
    archive graph nodes and are exec'd via load_tasks_from_env_codes;
  - the policy expects task-conditioned observations (obs + 67-dim task vector),
    so a bare MiniCraftaxTrain env would be wrong even if it loaded.

WHAT the new B-1 does (all production code paths, nothing hand-built):
  1. load_tasks_from_env_codes(archive, new_task_ids)  -> which ids actually load
  2. evaluate_new_tasks(config, rng, rl_train_state, ok_ids, archive, embedding_model)
       -> batched frozen-policy rollouts on the new tasks (env-from-code,
          embeddings, masks all handled by the existing machinery)
  3. calculate_scores_from_snapshot(...)               -> per-task SR
  4. route(sr, any_partial_progress)                   -> accept / reject per Mason's policy
  5. rejected: status 'preflight_<reason>' + set_task_active_status(id, False)
     (deactivation matters: PLR sampling uses is_active, so without it the gate
      would only delay a bad task by one session, not remove it)

REQUIREMENT: config must contain `validation.rollout_updates`
  (group file exists: conf/validation/default.yaml, rollout_updates: 40).
  If the experiment defaults don't include it, add `+validation=default` to the
  run command (drop the flag if Hydra says it's already in defaults).
  The hook guards this and prints an actionable error instead of silently passing.

Usage:
    python fix_preflight_hook.py [path/to/dicode_src]
Idempotent; validates with ast.parse; asserts every referenced symbol exists.
"""
import ast
import os
import sys


def find_root():
    if len(sys.argv) > 1:
        return sys.argv[1].rstrip("/")
    for cand in ("dicode_src", "./dicode_src", "/workspace/mechanism_UED/dicode_src"):
        if os.path.isdir(cand):
            return cand.rstrip("/")
    sys.exit("Could not locate dicode_src. Pass its path as an argument.")


ROOT = find_root()
RD = os.path.join(ROOT, "experiments/training/run_dicode.py")
GM = os.path.join(ROOT, "src/dicode/dreaming/gen_manager.py")
for p in (RD, GM):
    if not os.path.isfile(p):
        sys.exit(f"Missing file: {p}")


def read_lf(path):
    text = open(path, "rb").read().decode("utf-8")
    return text.replace("\r\n", "\n"), ("\r\n" in text)


def write_from_lf(path, text_lf, crlf):
    open(path, "wb").write((text_lf.replace("\n", "\r\n") if crlf else text_lf).encode("utf-8"))


# --- sanity: every symbol the new hook relies on must exist in the repo -----
def assert_in(path, needle, what):
    if needle not in open(path, encoding="utf-8", errors="replace").read():
        sys.exit(f"SANITY FAIL: {what} not found in {path}")

assert_in(os.path.join(ROOT, "src/dicode/evaluation/online_evaluation.py"),
          "def evaluate_new_tasks", "evaluate_new_tasks")
assert_in(os.path.join(ROOT, "src/dicode/scoring.py"),
          "def calculate_scores_from_snapshot", "calculate_scores_from_snapshot")
assert_in(os.path.join(ROOT, "src/dicode/task_utils.py"),
          "def load_tasks_from_env_codes", "load_tasks_from_env_codes")
assert_in(os.path.join(ROOT, "src/dicode/skill_preflight/preflight.py"),
          "def route", "route()")
assert_in(GM, "def set_task_active_status", "TaskArchive.set_task_active_status")
assert_in(os.path.join(ROOT, "conf/validation/default.yaml"),
          "rollout_updates", "conf/validation/default.yaml rollout_updates")
print("sanity: all referenced symbols exist ✓")

# --- the OLD (broken) B-1 block, verbatim ------------------------------------
OLD = (
    "        # [B] Preflight Gate: cold-rollout new tasks with the CURRENT policy; keep only\n"
    "        #     learnable ones. Single-threaded here, policy in hand (no worker-thread races).\n"
    "        #     Flag-gated (default off) -> baseline behaviour unchanged.\n"
    "        if config.get(\"skill_preflight\", {}).get(\"use_preflight\", False) and new_task_ids:\n"
    "            try:\n"
    "                from dicode.dreaming.gen_manager import Task\n"
    "                from dicode.dreaming.utils import smart_absolute_path\n"
    "                from dicode.wrappers import BatchEnvWrapper\n"
    "                from dicode.skill_preflight.preflight import cold_preflight\n"
    "                _target_ach = _sched.target_achievements if _sched is not None else []\n"
    "                _kept = []\n"
    "                for _tid in new_task_ids:\n"
    "                    rng, _pf_rng = jax.random.split(rng)\n"
    "                    _raw = Task(smart_absolute_path(_tid)).env\n"
    "                    _eparams = _raw.default_params.replace(\n"
    "                        max_timesteps=config.evaluation.get(\"max_timesteps\", 8192))\n"
    "                    _env = BatchEnvWrapper(_raw, num_envs=config.evaluation.num_envs)\n"
    "                    _res = cold_preflight(_env, _eparams, rl_train_state, _pf_rng, config, _target_ach)\n"
    "                    if _res.action == \"accept\":\n"
    "                        _kept.append(_tid)\n"
    "                    else:\n"
    "                        gen_manager.archive.update_node_status(_tid, f\"preflight_{_res.reason}\")\n"
    "                        print(f\"  [Preflight] reject {_tid}: {_res.reason} (sr={_res.sr:.2f})\")\n"
    "                print(f\"  [Preflight] kept {len(_kept)}/{len(new_task_ids)} new tasks\")\n"
    "                new_task_ids = _kept\n"
    "            except Exception as e:\n"
    "                print(f\"  [Preflight] skipped (kept all): {e}\")\n"
)

# --- the NEW B-1 block --------------------------------------------------------
NEW = (
    "        # [B] Preflight Gate (v2): score new tasks with the CURRENT policy and keep\n"
    "        #     only learnable ones. Reuses the codebase's own machinery end-to-end:\n"
    "        #     load_tasks_from_env_codes (env from archived code) -> evaluate_new_tasks\n"
    "        #     (batched frozen-policy rollouts, embeddings, masks) ->\n"
    "        #     calculate_scores_from_snapshot (per-task SR) -> route() (accept/reject).\n"
    "        #     Flag-gated (default off) -> baseline behaviour unchanged.\n"
    "        if config.get(\"skill_preflight\", {}).get(\"use_preflight\", False) and new_task_ids:\n"
    "            try:\n"
    "                from dicode.evaluation import evaluate_new_tasks\n"
    "                from dicode.scoring import calculate_scores_from_snapshot\n"
    "                from dicode.task_utils import load_tasks_from_env_codes\n"
    "                from dicode.skill_preflight.preflight import route\n"
    "\n"
    "                if \"validation\" not in config:\n"
    "                    raise RuntimeError(\n"
    "                        \"config.validation missing - add `+validation=default` to the \"\n"
    "                        \"run command (preflight needs validation.rollout_updates)\")\n"
    "\n"
    "                _pf_t0 = time.time()\n"
    "                # Resolve which ids actually load, in order (index-aligns with scores)\n"
    "                _pf_classes, _pf_ok_ids = load_tasks_from_env_codes(\n"
    "                    gen_manager.archive, new_task_ids)\n"
    "                # Ids whose code failed to load: keep (same as baseline; they will be\n"
    "                # skipped again by the training loader anyway)\n"
    "                _kept = [t for t in new_task_ids if t not in _pf_ok_ids]\n"
    "\n"
    "                if _pf_ok_ids:\n"
    "                    rng, _pf_rng = jax.random.split(rng)\n"
    "                    _pf_raw = evaluate_new_tasks(\n"
    "                        config, _pf_rng, rl_train_state, _pf_ok_ids,\n"
    "                        gen_manager.archive, gen_manager.selector.embedding_model,\n"
    "                    )\n"
    "                    _pf_swd = _pf_raw.get(\"scoring_window_data\")\n"
    "                    if _pf_swd is None:\n"
    "                        print(\"  [Preflight] WARNING: rollouts returned no scoring data; \"\n"
    "                              \"keeping all new tasks\")\n"
    "                        _kept = list(new_task_ids)\n"
    "                    else:\n"
    "                        _pf_scores = calculate_scores_from_snapshot(\n"
    "                            _pf_swd, len(_pf_ok_ids),\n"
    "                            _pf_raw[\"task_achievement_mask\"],\n"
    "                            _pf_raw[\"task_completed_mask\"],\n"
    "                            config,\n"
    "                        )\n"
    "                        for _pf_i, _tid in enumerate(_pf_ok_ids):\n"
    "                            _sr = float(_pf_scores.get(str(_pf_i), {}).get(\"sr\", -1.0))\n"
    "                            # sr < 0 => no episode finished => no partial progress\n"
    "                            _d = route(max(_sr, 0.0), any_partial_progress=(_sr >= 0.0))\n"
    "                            if _d.action == \"accept\":\n"
    "                                _kept.append(_tid)\n"
    "                                _clip = min(max(_sr, 0.0), 1.0)\n"
    "                                gen_manager.archive.update_node_learnability(\n"
    "                                    _tid, _clip * (1.0 - _clip))\n"
    "                            else:\n"
    "                                gen_manager.archive.update_node_status(\n"
    "                                    _tid, f\"preflight_{_d.reason}\")\n"
    "                                gen_manager.archive.set_task_active_status(_tid, False)\n"
    "                                print(f\"  [Preflight] reject {_tid}: {_d.reason} \"\n"
    "                                      f\"(sr={_sr:.2f})\")\n"
    "                print(f\"  [Preflight] kept {len(_kept)}/{len(new_task_ids)} new tasks \"\n"
    "                      f\"({time.time() - _pf_t0:.1f}s)\")\n"
    "                new_task_ids = _kept\n"
    "            except Exception as e:\n"
    "                print(f\"  [Preflight] ERROR (kept all, gate inactive!): {e}\")\n"
)

text, crlf = read_lf(RD)
if "Preflight Gate (v2)" in text:
    print("fix: already applied, skipping.")
elif text.count(OLD) == 1:
    write_from_lf(RD, text.replace(OLD, NEW), crlf)
    print(f"fix: applied to run_dicode.py (CRLF={crlf}).")
else:
    sys.exit(f"fix: old B-1 block matched {text.count(OLD)}x (need 1). "
             "File diverged; apply manually.")

ast.parse(open(RD, encoding="utf-8").read())
print("ast.parse OK: run_dicode.py")
print("\nDone. Notes:")
print(" - add `+validation=default` to the +A+B command (drop it if Hydra says duplicate)")
print(" - watch Session 3 for `[Preflight] kept X/Y new tasks (Ns)`")
print(" - `[Preflight] ERROR (kept all...)` means the gate is OFF - stop and debug")
