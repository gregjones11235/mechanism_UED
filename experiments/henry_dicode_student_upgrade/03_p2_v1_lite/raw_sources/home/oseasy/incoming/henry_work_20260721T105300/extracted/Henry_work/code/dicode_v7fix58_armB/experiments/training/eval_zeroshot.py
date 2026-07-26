"""Zero-shot arbitration eval (2026-07-14).

Question: is ckpt-8100's TRUE zero-shot SR on the kobold R0 level ~15% (the
fix4.2/fix4.5 first readings) or ~0.5% (the fix4.6 roll-1/roll-2 first readings)?

Method: restore ckpt 8100 (golden copy), run the EXACT production rollout stack
(run_training_session) with lr=0 (adam step is a no-op -> pure rollout
collection), on [the exact task_800 kobold R0 level from roll-1's graphml,
OriginalTask as harness sanity]. Metrics via the production scoring path, so the
reported SR has byte-identical semantics to the runs' performance_history 'sr'.
"""
import glob
import json
import types
import xml.etree.ElementTree as ET

import hydra
import jax
import jax.numpy as jnp

GRAPHML_GLOB = "/oscar/scratch/jzhu223/dicode_outputs/v7fix4_s0/backup_fix46roll1_*/task_graph.graphml"
CKPT_ROOT = "/oscar/scratch/jzhu223/dicode_outputs/eval_zeroshot_ckpt"  # staged by wrapper
NUM_UPDATES = 30  # 30 x 128 envs x 1024 steps, 80% on kobold -> ~3000 episodes


def _extract_kobold_code() -> str:
    ns = "{http://graphml.graphdrawing.org/xmlns}"
    path = sorted(glob.glob(GRAPHML_GLOB))[-1]
    keymap, code = {}, None
    for _ev, el in ET.iterparse(path, events=("end",)):
        if el.tag == ns + "key":
            keymap[el.get("id")] = el.get("attr.name")
        elif el.tag == ns + "node":
            attrs = {keymap.get(d.get("key"), d.get("key")): (d.text or "")
                     for d in el.findall(ns + "data")}
            if str(attrs.get("system_built", "")).lower() == "true":
                for v in attrs.values():
                    if "DEFEAT_KOBOLD" in v and "set_starting_floor(3)" in v:
                        code = v
                        break
            el.clear()
            if code:
                break
    assert code, f"no kobold R0 system level found in {path}"
    print(f"level source: {path}")
    return code


@hydra.main(version_base="1.2", config_path="../../conf", config_name="config")
def main(config):
    import wandb
    wandb.init(mode="disabled")  # ppo's _log_callback calls wandb.log unconditionally

    from dicode.ppo_tr import run_training_session
    from dicode.setup import _load_agent_state
    from dicode.task_utils import get_achievement_multi_hot
    from dicode.training import (
        _create_achievement_masks,
        extract_and_format_original_metrics,
        process_training_metrics,
    )
    from minicraftax.tasks.seed_tasks.original import Env as OriginalTask

    assert float(config.training.lr) == 0.0 and float(config.training.min_lr) == 0.0, \
        "pass training.lr=0 training.min_lr=0 — this eval must not learn"

    code = _extract_kobold_code()
    print("=== LEVEL CODE (verbatim from roll-1 graphml) ===")
    print(code)
    mod = types.ModuleType("kobold_r0_eval")
    exec(code, mod.__dict__)

    task_classes = [mod.Env, OriginalTask]
    task_ids = ["kobold_r0_zero_shot", "original_craftax"]

    ach_mask, comp_mask = _create_achievement_masks(task_classes)

    cot = config.training.condition_on_task
    if cot and cot != "embedding":
        ach_lists = [cls(None, None).relevant_achievements for cls in task_classes]
        emb = jnp.array([get_achievement_multi_hot(a) for a in ach_lists])
    elif not cot:
        emb = None
    else:
        raise RuntimeError("embedding conditioning needs the LLM — unexpected here")
    print("conditioning:", cot, "| emb:", None if emb is None else emb.shape)

    train_state = _load_agent_state(config, CKPT_ROOT)
    print("weights restored from", CKPT_ROOT)

    dist = jnp.array([0.8, 0.2])  # 80% of envs on the kobold level
    rng = jax.random.PRNGKey(int(config.seed))
    results = run_training_session(
        config,
        rng,
        task_classes,
        num_training_updates=NUM_UPDATES,
        train_state=train_state,
        task_embeddings=emb,
        task_distribution_proportions=dist,
        global_update_step=0,
    )

    # force_include [0, 1]: the kobold task ALSO reports its full achievement breakdown —
    # the death-cause data (did episodes place torches? fight anything? starve? time out?).
    tm = process_training_metrics(
        task_ids, results["metrics"], len(task_classes), ach_mask, comp_mask,
        config, force_include_achievements_indices=[0, 1],
    )

    def _clean(d):
        # NO truncation (run 3966171's str(v)[:200] guillotined achievement_srs — the whole
        # point of this rerun). Dicts pass through whole; json default=str catches the rest.
        return {k: (float(v) if hasattr(v, "item") else v) for k, v in (d or {}).items()}

    print("=== KOBOLD R0 ZERO-SHOT (ckpt 8100, lr=0) — full breakdown incl achievements ===")
    print("NOTE: collect_*/make_torch/make_*_pickaxe/enter_* at ~100% can be kit/floor "
          "pre-credits (calculate_inventory_achievements runs at RESET), not behaviour. "
          "Behaviour reads: place_torch/place_table/place_stone/wake_up/eat_*/defeat_*/"
          "make_wood_sword/fire_bow.")
    print(json.dumps(_clean(tm.get("kobold_r0_zero_shot")), indent=1, default=str, sort_keys=True))
    print("=== ORIGINAL TASK (harness sanity, expect mean_return ~40) ===")
    print(json.dumps(_clean(extract_and_format_original_metrics(tm, "original_craftax")),
                     indent=1, default=str))
    print("=== VERDICT KEY: kobold success/sr ~15% -> training-window sensitivity; "
          "~0.5-2% -> takeoff-dice (15% was within-session bootstrap) ===")


if __name__ == "__main__":
    main()
