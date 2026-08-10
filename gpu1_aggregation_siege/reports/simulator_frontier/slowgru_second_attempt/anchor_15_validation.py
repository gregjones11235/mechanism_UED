import hashlib
import importlib.util
import json
import pathlib
import sys
import time

import jax
import jax.numpy as jnp
from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
from minicraftax.envs.multitask import MultiTaskMiniCraftaxEnv
from minicraftax.tasks.seed_tasks import collecting, combat, crafting


root = pathlib.Path(sys.argv[1]).resolve()
dynamic_paths = sorted((root / "generated").glob("slot_*.py"))
if len(dynamic_paths) != 12:
    raise RuntimeError(f"expected 12 dynamic modules, found {len(dynamic_paths)}")
dynamic_modules = []
for index, path in enumerate(dynamic_paths):
    spec = importlib.util.spec_from_file_location(f"e3_dynamic_{index}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dynamic_modules.append(module)

static_params = StaticEnvParams()
base_params = EnvParams()
classes = [collecting.Env, combat.Env, crafting.Env]
classes.extend(module.Env for module in dynamic_modules)
for name, cls in zip(("collecting", "combat", "crafting"), classes[:3]):
    task_params = cls(static_params, base_params).get_task_params()
    print("ANCHOR", name, task_params, flush=True)
for index, cls in enumerate(classes[3:], 3):
    print("DYNAMIC", index, cls(static_params, base_params).get_task_params(), flush=True)

env_params = EnvParams(max_timesteps=8)
env = MultiTaskMiniCraftaxEnv(
    task_classes=classes,
    static_env_params=static_params,
    params=env_params,
    condition_on_task=True,
    conditioning_type="embedding",
    embedding_size=67,
)
stacked_lengths = [
    int(getattr(env.stacked_task_params, field).shape[0])
    for field in env.stacked_task_params.__dataclass_fields__
]
if len(classes) != 15 or any(length != 15 for length in stacked_lengths):
    raise RuntimeError(f"stacked TaskParams mismatch: {stacked_lengths}")
print("STACKED", len(classes), stacked_lengths, flush=True)
for task_id in (0, 1, 2, 3):
    _, state = env.reset_env(
        jax.random.PRNGKey(100 + task_id),
        env_params,
        task_id,
        jnp.zeros((1, 67)),
    )
    if int(state.task_id) != task_id:
        raise RuntimeError(f"reset task id mismatch: {task_id} vs {state.task_id}")
    print("RESET", task_id, state.task_params, flush=True)

result = {
    "schema": "e3_gpu1_anchor_15_validation/v1",
    "gpu_uuid": "GPU-3c7a2864-755b-7045-b293-6f80e748283f",
    "module_count": len(classes),
    "resets": [0, 1, 2, 3],
    "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
raw = json.dumps(result, sort_keys=True, separators=(",", ":")).encode("utf-8")
(root / "evidence" / "anchor_15_result.json").write_bytes(raw)
print("RESULT_SHA256", hashlib.sha256(raw).hexdigest(), flush=True)
