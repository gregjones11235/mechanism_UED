"""Standalone official DiCode evaluation over multiple checkpoint steps.

Reuses the exact official eval path (online_evaluation.run_session_evaluation ->
craftax_evaluation.main): full Craftax, 1024 procedurally-generated held-out
worlds, reports mean_return (paper metric, DiCode=48.33).

Usage (Hydra, same conf as run_dicode):
  python experiments/training/eval_checkpoints.py \
    hydra.run.dir=<tmp> \
    <gen_manager llm overrides same as training> \
    use_wandb=false \
    eval.ckpt_root=/oscar/scratch/.../rl_checkpoints \
    'eval.steps=[9900,10100,10200,10300,10400]' \
    eval.tag=CARM seed=0
"""
import os
import json

import hydra
import jax
import jax.numpy as jnp
from omegaconf import DictConfig, OmegaConf

from dicode.evaluation.online_evaluation import run_session_evaluation
from dicode.dreaming.gen_manager import GenManager
from dicode.utils.general.train_state_utils import load_weights_only
from dicode.task_utils import EMBEDDING_SIZE
from minicraftax.envs.craftax import CraftaxAugObsTrain


@hydra.main(version_base="1.2", config_path="../../conf/", config_name="config")
def main(config: DictConfig) -> None:
    ckpt_root = config.eval.ckpt_root
    steps = list(config.eval.steps)
    tag = config.eval.get("tag", "run")
    seed = int(config.seed)

    print(f"=== [eval_checkpoints] tag={tag} seed={seed} ===")
    print(f"ckpt_root={ckpt_root}")
    print(f"steps={steps}")
    print(f"condition_on_task={config.training.condition_on_task} "
          f"conditioning_type={config.training.conditioning_type}")

    # GenManager needed only so run_session_evaluation can access the embedding
    # model when conditioning_type == 'embedding'. For one_hot it is untouched,
    # but we build it the same way training does to guarantee identical env dims.
    gen_manager = GenManager(config)

    # dummy env matching TRAINING obs dims (drives network init in load_weights_only).
    # CRITICAL: training used conditioning_type=one_hot with a 67-dim multi-hot
    # (task_embeddings.shape[1]==67, verified from training logs "Using embedding size: 67"),
    # NOT gen_manager.embedding_model.embedding_size (=1024). Using 1024 here would
    # build a wrong-sized obs and fail checkpoint restore.
    if config.training.conditioning_type == "embedding":
        emb_size = config.gen_manager.embedding_model.embedding_size
    else:  # one_hot
        emb_size = EMBEDDING_SIZE  # 67
    print(f"emb_size used for dummy env = {emb_size}")
    dummy_env = CraftaxAugObsTrain(
        condition_on_task=config.training.condition_on_task,
        conditioning_type=config.training.conditioning_type,
        embedding_size=emb_size,
        task_embeddings=jnp.zeros((1, emb_size)),
    )

    results = {}
    raw_details = {}
    for step in steps:
        ckpt_path = os.path.join(ckpt_root, str(step))
        if not os.path.isdir(ckpt_path):
            print(f"[skip] step {step}: not found at {ckpt_path}")
            continue
        print(f"\n===== EVAL step {step} =====")
        rl_train_state = load_weights_only(
            checkpoint_path=ckpt_path,
            env=dummy_env,
            env_params=dummy_env.default_params,
            config=config.training,
        )
        # deterministic per-step seed so both runs see the SAME held-out worlds
        rng = jax.random.PRNGKey(seed)
        _, metrics = run_session_evaluation(
            config, rng, rl_train_state, gen_manager, step, 0,
            detail=bool(config.eval.get("details", False)),
        )
        mr = float(metrics.get("mean_return", float("nan")))
        mp = float(metrics.get("mean_performance", float("nan")))
        el = float(metrics.get("average_episode_length", float("nan")))
        print(f"[RESULT] tag={tag} step={step} mean_return={mr:.4f} "
              f"mean_performance={mp:.4f} avg_ep_len={el:.1f}")
        _det = metrics.pop("_details", None)
        if _det is not None:
            raw_details[step] = _det
        results[step] = {"mean_return": mr, "mean_performance": mp,
                         "average_episode_length": el,
                         "skills": {k: float(v) for k, v in metrics.items()
                                    if k.startswith("skill_")}}

    print("\n===== SUMMARY (tag=%s) =====" % tag)
    for step in sorted(results):
        print(f"  step {step}: mean_return={results[step]['mean_return']:.4f} "
              f"mean_performance={results[step]['mean_performance']:.4f}")
    out_json = os.path.join(os.getcwd(), f"eval_{tag}_seed{seed}.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[saved] {out_json}")
    if raw_details:
        import numpy as _np
        det_all = {str(k): {kk: _np.asarray(vv).tolist() for kk, vv in v.items()}
                   for k, v in raw_details.items()}
        det_json = out_json.replace(".json", "_details.json")
        with open(det_json, "w") as f:
            json.dump(det_all, f)
        print(f"[saved] {det_json}")
    print("EVAL_DONE")


if __name__ == "__main__":
    main()
