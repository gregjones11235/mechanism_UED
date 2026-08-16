p = "/workspace/mechanism_UED/dicode_src/experiments/analysis/sil_collect.py"
src = open(p, newline='').read()
def sub1(old, new):
    global src
    assert src.count(old) == 1, f"anchor x{src.count(old)}: {old[:60]!r}"
    src = src.replace(old, new)
sub1("from dicode.task_utils import EMBEDDING_SIZE\n",
     "from dicode.task_utils import EMBEDDING_SIZE, get_achievement_multi_hot\n")
sub1("""    env = CraftaxAugObsTrain(
        condition_on_task=config.training.condition_on_task,
        conditioning_type=config.training.conditioning_type,
        embedding_size=emb,
        task_embeddings=jnp.zeros((1, emb)),
    )
""",
"""    # [v1.2] Condition EXACTLY like the official bare-chain eval (one_hot ->
    # multi-hot of the original task's relevant achievements). v1 fed zeros:
    # OOD conditioning that suppressed descent for BOTH donors (training-seed
    # reached2 2.6-4.4% vs official touched2 15-16%).
    if config.training.conditioning_type == "one_hot":
        emb_vec = jnp.asarray(get_achievement_multi_hot(
            CraftaxAugObsTrain().relevant_achievements))
    else:
        emb_vec = jnp.zeros((emb,))
    print(f"[SIL-COLLECT] conditioning={config.training.conditioning_type} "
          f"vec_sum={float(emb_vec.sum()):.0f}")
    env = CraftaxAugObsTrain(
        condition_on_task=config.training.condition_on_task,
        conditioning_type=config.training.conditioning_type,
        embedding_size=emb,
        task_embeddings=jnp.tile(emb_vec[None, :], (num_envs, 1)),
    )
""")
open(p, "w", newline='').write(src)
print("conditioning patch OK (2 anchors)")
