"""Dependency-light production per-slot Craftax task materializer.

The materializer deliberately keeps controller supplied anchor bindings
separate from dynamic planner output.  It never invents anchor science and it
never falls back to the test runtime.
"""
from __future__ import annotations
import hashlib, json, math
from collections.abc import Mapping

TASKPARAM_FIELDS = ("passive_spawn_multiplier","melee_spawn_multiplier","ranged_spawn_multiplier","mob_health_multiplier","mob_damage_multiplier","melee_trigger_distance","monsters_killed_to_clear_level","needs_depletion_multiplier","health_recover_multiplier","health_loss_multiplier","mana_recover_multiplier","growing_plants_age")
INT_FIELDS = {"melee_trigger_distance","monsters_killed_to_clear_level","growing_plants_age"}
LOWER_BOUNDS = {"passive_spawn_multiplier":0,"melee_spawn_multiplier":0,"ranged_spawn_multiplier":0,"mob_health_multiplier":.01,"mob_damage_multiplier":0,"melee_trigger_distance":1,"monsters_killed_to_clear_level":0,"needs_depletion_multiplier":0,"health_recover_multiplier":.01,"health_loss_multiplier":0,"mana_recover_multiplier":.01,"growing_plants_age":2}
ANCHOR_REQUIRED_FIELDS = (
    "anchor_id", "base_env_entrypoint", "base_env_hash", "taskparams",
    "world_set_ref", "seed_policy_ref", "reset_protocol",
)


def canonical_sha256(value) -> str:
    blob = json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()

def resolve_taskparams(taskparam_ranges, *, distribution_id: str, plan_hash: str):
    if not isinstance(taskparam_ranges, dict) or set(taskparam_ranges) != set(TASKPARAM_FIELDS):
        raise ValueError("TaskParams allowlist mismatch")
    out = {}
    for name in TASKPARAM_FIELDS:
        value = taskparam_ranges[name]
        if isinstance(value, bool) or isinstance(value, (int,float)):
            lo = hi = value
        elif isinstance(value, (list,tuple)) and len(value)==2:
            lo,hi = value
        else: raise ValueError(f"invalid TaskParams range: {name}")
        if isinstance(lo,bool) or isinstance(hi,bool) or not all(isinstance(x,(int,float)) and math.isfinite(float(x)) for x in (lo,hi)) or lo>hi:
            raise ValueError(f"invalid TaskParams value: {name}")
        if name in INT_FIELDS and any(not isinstance(x, int) for x in (lo, hi)):
            raise ValueError(f"integer TaskParams range required: {name}")
        mid=(float(lo)+float(hi))/2
        out[name]=(int(math.floor(mid + .5)) if name in INT_FIELDS else float(mid))
        if out[name] < LOWER_BOUNDS[name]: raise ValueError(f"TaskParams below server clamp bound: {name}")
    return out

def render_slot_env_module(taskparams: dict, *, distribution_id: str, plan_hash: str) -> tuple[str,str]:
    if not distribution_id or not plan_hash: raise ValueError("slot identity required")
    if set(taskparams) != set(TASKPARAM_FIELDS): raise ValueError("TaskParams fields mismatch")
    checked = resolve_taskparams({k: taskparams[k] for k in TASKPARAM_FIELDS}, distribution_id=distribution_id, plan_hash=plan_hash)
    literal=", ".join(f"{k}={taskparams[k]!r}" for k in TASKPARAM_FIELDS)
    code=("# schema=production_task_materializer/v1\n"
          f"DISTRIBUTION_ID={distribution_id!r}\nPLAN_HASH={plan_hash!r}\n"
          "from minicraftax.tasks.seed_tasks.collecting import Env as BaseEnv\n"
          "from minicraftax.craftax_state import TaskParams\n\n"
          "class Env(BaseEnv):\n    def get_task_params(self):\n"
          f"        return TaskParams({', '.join(f'{k}={checked[k]!r}' for k in TASKPARAM_FIELDS)})\n")
    return code, hashlib.sha256(code.encode()).hexdigest()


def require_anchor_bindings(anchor_manifest):
    if not isinstance(anchor_manifest, (list, tuple)) or len(anchor_manifest) != 3:
        raise RuntimeError("BLOCKED_ANCHOR_EXECUTABLE_BINDING_MISSING")
    seen = set()
    validated = []
    for anchor in anchor_manifest:
        if not isinstance(anchor, Mapping) or any(k not in anchor for k in ANCHOR_REQUIRED_FIELDS):
            raise RuntimeError("BLOCKED_ANCHOR_EXECUTABLE_BINDING_MISSING")
        anchor_id = str(anchor["anchor_id"])
        if not anchor_id or anchor_id in seen:
            raise RuntimeError("BLOCKED_ANCHOR_EXECUTABLE_BINDING_MISSING")
        if not str(anchor["base_env_entrypoint"]).count(":") == 1:
            raise RuntimeError("BLOCKED_ANCHOR_EXECUTABLE_BINDING_MISSING")
        if not isinstance(anchor["base_env_hash"], str) or len(anchor["base_env_hash"]) != 64:
            raise RuntimeError("BLOCKED_ANCHOR_EXECUTABLE_BINDING_MISSING")
        if anchor["reset_protocol"] != "STANDARD_RESET":
            raise RuntimeError("BLOCKED_ANCHOR_EXECUTABLE_BINDING_MISSING")
        params = anchor["taskparams"]
        if not isinstance(params, Mapping):
            raise RuntimeError("BLOCKED_ANCHOR_EXECUTABLE_BINDING_MISSING")
        # Reuse the exact production validator for all 12 fields.  Scalars are
        # accepted here because the controller binds observed anchor values.
        resolved = resolve_taskparams(dict(params), distribution_id=anchor_id,
                                      plan_hash=str(anchor.get("manifest_hash", anchor_id)))
        record = dict(anchor)
        record["taskparams"] = resolved
        seen.add(anchor_id)
        validated.append(record)
    return tuple(validated)


def load_executable_anchor_manifest(path: str):
    """Load a controller-issued executable anchor manifest from JSON.

    The formal runner supplies this path only after signature verification;
    absence or a legacy schema fails closed and never invents anchors.
    """
    if not path:
        raise RuntimeError("BLOCKED_SHARED_ANCHOR_MANIFEST")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("BLOCKED_ANCHOR_EXECUTABLE_BINDING_MISSING") from exc
    if not isinstance(payload, Mapping) or payload.get("schema") != "e3_executable_anchor_manifest/v1":
        raise RuntimeError("BLOCKED_ANCHOR_EXECUTABLE_BINDING_MISSING")
    manifest_hash = payload.get("manifest_hash")
    signature_ref = payload.get("controller_signature_ref")
    if not isinstance(manifest_hash, str) or len(manifest_hash) != 64 or not signature_ref:
        raise RuntimeError("BLOCKED_ANCHOR_EXECUTABLE_BINDING_MISSING")
    if str(signature_ref).startswith("TEST_ONLY") or str(signature_ref) in {"e3-smoke", "0" * 64}:
        raise RuntimeError("BLOCKED_SHARED_ANCHOR_MANIFEST")
    anchors = require_anchor_bindings(payload.get("anchors"))
    canonical_payload = {"schema": payload["schema"],
                         "controller_signature_ref": str(signature_ref),
                         "anchors": [dict(a) for a in anchors]}
    if canonical_sha256(canonical_payload) != manifest_hash:
        raise RuntimeError("BLOCKED_ANCHOR_EXECUTABLE_BINDING_MISSING")
    return {"schema": payload["schema"], "manifest_hash": manifest_hash,
            "controller_signature_ref": str(signature_ref),
            "anchors": anchors}
