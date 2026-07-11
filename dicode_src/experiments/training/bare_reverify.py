"""C-2 dynamic-layer prototype: bare re-verification of scaffolded tasks.

For each selected task from a task_graph.graphml:
  1. keep the ORIGINAL code;
  2. build a BARE variant (AST-guided strip):
       S1: remove builder.set_player_inventory(...) lines
       S2: replace `self.completed_achievements = [...]` with `= []`
       S4 (--only with +reverify.strip_floor=true): set_starting_floor(n) -> (0)
     (S3 mob spawns are KEPT: removing them could make combat tasks vacuously
      impossible; we test whether the policy solves the task without handed
      tools/premarks, not without the task's subject matter.)
  3. inject both variants into the GenManager archive as temp nodes;
  4. evaluate BOTH with the SAME frozen checkpoint + SAME rng via the
     production `evaluate_new_tasks` pipeline; report paired SR.

Interpretation:
  orig SR high & bare SR ~0  -> scaffolds are load-bearing (causal demo)
  bare SR ~= orig SR         -> tasks don't depend on scaffolds; plateau story
                                needs revisiting BEFORE Friday.

Usage (Hydra, same conf as run_dicode; venv on pod):
  python experiments/training/bare_reverify.py \
    hydra.run.dir=/workspace/eval_out/reverify \
    gen_manager/llm@gen_manager.task_generator=local_qwen14b \
    gen_manager/llm@gen_manager.env_generator=local_qwen14b \
    gen_manager.embedding_model.model=nomic-embed-text \
    use_wandb=false \
    +reverify.graphml=/abs/path/task_graph.graphml \
    +reverify.ckpt=/abs/path/rl_checkpoints/2400 \
    '+reverify.tasks=[task_17,task_19,task_20]' \
    +reverify.strip_floor=false
"""
import ast
import json
import os

import hydra
import jax
import jax.numpy as jnp
import networkx as nx
from omegaconf import DictConfig

from dicode.evaluation.online_evaluation import evaluate_new_tasks
from dicode.scoring import calculate_scores_from_snapshot
from dicode.dreaming.gen_manager import GenManager
from dicode.utils.general.train_state_utils import load_weights_only
from dicode.task_utils import EMBEDDING_SIZE
from minicraftax.envs.craftax import CraftaxAugObsTrain


# ---------------------------------------------------------------- strip
def strip_scaffolds(code: str, strip_floor: bool) -> str:
    """AST-guided source edit: blank S1 lines, empty S2 list, optionally S4->0."""
    tree = ast.parse(code)
    lines = code.split("\n")
    kill_spans = []      # (lineno, end_lineno) 1-based inclusive -> blank
    premark_spans = []   # replace with `= []`
    floor_spans = []     # replace arg with 0

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "set_player_inventory":
                kill_spans.append((node.lineno, node.end_lineno))
            elif strip_floor and node.func.attr == "set_starting_floor":
                floor_spans.append((node.lineno, node.end_lineno))
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            t = node.targets[0]
            if (isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                    and t.value.id == "self" and t.attr == "completed_achievements"):
                premark_spans.append((node.lineno, node.end_lineno))

    for lo, hi in kill_spans:
        for i in range(lo - 1, hi):
            lines[i] = ""
    for lo, hi in premark_spans:
        indent = lines[lo - 1][: len(lines[lo - 1]) - len(lines[lo - 1].lstrip())]
        lines[lo - 1] = f"{indent}self.completed_achievements = []"
        for i in range(lo, hi):
            lines[i] = ""
    for lo, hi in floor_spans:
        for i in range(lo - 1, hi):
            lines[i] = lines[i].replace("set_starting_floor(", "set_starting_floor(0) #(", 1) \
                if "set_starting_floor(" in lines[i] else lines[i]

    out = "\n".join(lines)
    ast.parse(out)  # must stay valid
    return out


@hydra.main(version_base="1.2", config_path="../../conf/", config_name="config")
def main(config: DictConfig) -> None:
    graphml = config.reverify.graphml
    ckpt = config.reverify.ckpt
    task_ids = list(config.reverify.tasks)
    strip_floor = bool(config.reverify.get("strip_floor", False))
    seed = int(config.seed)

    print(f"=== [bare_reverify] tasks={task_ids} strip_floor={strip_floor} seed={seed} ===")
    print(f"graphml={graphml}\nckpt={ckpt}")

    # --- load codes + build bare variants -----------------------------------
    g = nx.read_graphml(graphml)
    pairs = {}  # tid -> (orig_code, bare_code)
    for tid in task_ids:
        if tid not in g.nodes or not g.nodes[tid].get("code"):
            print(f"[skip] {tid}: not in graph / no code"); continue
        orig = g.nodes[tid]["code"]
        bare = strip_scaffolds(orig, strip_floor)
        changed = sum(a != b for a, b in zip(orig.split("\n"), bare.split("\n")))
        print(f"  {tid}: bare variant built ({changed} lines changed)")
        pairs[tid] = (orig, bare)
    if not pairs:
        raise SystemExit("no valid tasks")

    # --- GenManager + inject both variants into its archive -----------------
    gen_manager = GenManager(config)
    orig_ids, bare_ids = [], []
    for tid, (orig, bare) in pairs.items():
        oid, bid = f"rv_orig__{tid}", f"rv_bare__{tid}"
        gen_manager.archive.graph.add_node(oid, code=orig)
        gen_manager.archive.graph.add_node(bid, code=bare)
        orig_ids.append(oid); bare_ids.append(bid)

    # --- frozen policy (exact eval_checkpoints restore path) -----------------
    emb_size = (config.gen_manager.embedding_model.embedding_size
                if config.training.conditioning_type == "embedding" else EMBEDDING_SIZE)
    dummy_env = CraftaxAugObsTrain(
        condition_on_task=config.training.condition_on_task,
        conditioning_type=config.training.conditioning_type,
        embedding_size=emb_size,
        task_embeddings=jnp.zeros((1, emb_size)),
    )
    rl_train_state = load_weights_only(
        checkpoint_path=ckpt, env=dummy_env,
        env_params=dummy_env.default_params, config=config.training,
    )

    # --- paired evaluation (same rng => paired conditions) -------------------
    def eval_ids(ids, label):
        rng = jax.random.PRNGKey(seed)
        raw = evaluate_new_tasks(config, rng, rl_train_state, ids,
                                 gen_manager.archive,
                                 gen_manager.selector.embedding_model)
        swd = raw.get("scoring_window_data")
        if swd is None:
            print(f"[{label}] no scoring data"); return {}
        scores = calculate_scores_from_snapshot(
            swd, len(ids), raw["task_achievement_mask"],
            raw["task_completed_mask"], config)
        return {ids[i]: float(scores.get(str(i), {}).get("sr", -1.0))
                for i in range(len(ids))}

    sr_orig = eval_ids(orig_ids, "orig")
    sr_bare = eval_ids(bare_ids, "bare")

    # --- report ---------------------------------------------------------------
    print("\n===== BARE RE-VERIFICATION (frozen ckpt %s) =====" % os.path.basename(ckpt))
    print(f"{'task':<12} {'SR(orig)':>9} {'SR(bare)':>9} {'collapse':>9}")
    rows = {}
    for tid in pairs:
        o = sr_orig.get(f"rv_orig__{tid}", float("nan"))
        b = sr_bare.get(f"rv_bare__{tid}", float("nan"))
        print(f"{tid:<12} {o:>9.3f} {b:>9.3f} {o - b:>9.3f}")
        rows[tid] = {"sr_orig": o, "sr_bare": b}
    out = os.path.join(os.getcwd(), "bare_reverify.json")
    json.dump(rows, open(out, "w"), indent=2)
    print(f"[saved] {out}\nREVERIFY_DONE")


if __name__ == "__main__":
    main()
