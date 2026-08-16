#!/usr/bin/env python3
"""D3Q CPU-JAX candidate validation (isolated subprocess).

Runs ONLY inside a child process spawned by ``d3q_slot_runner`` with
``CUDA_VISIBLE_DEVICES=''``, ``JAX_PLATFORMS=cpu`` and
``PYTHONPATH=<mason worktree>/dicode_src/src`` so the main runner never imports
jax and the Mason-baseline ``dicode.dreaming.gen_manager.Task`` is used.

Mirrors the frozen ``llm_replay_harness.cpu_jax_validation`` semantics:

* ``Task(code_file)`` loads the candidate as a MiniCraftax task;
* a ``jax.jit(..., backend="cpu")`` function resets the env, samples one
  action, steps once and verifies every ``state.inventory`` field is int32;
* timings are split into load / compile / execute phases.

Output: a JSON result file with ``valid``, ``message`` (sanitized),
``load_s``, ``compile_s``, ``execute_s`` and ``total_s``.  The process exits 0
only when the result file was written.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-file", required=True)
    parser.add_argument("--result-file", required=True)
    args = parser.parse_args(argv)

    code_path = Path(args.code_file)
    result_path = Path(args.result_file)
    started = time.monotonic()

    try:
        import jax
        import jax.numpy as jnp

        from dicode.dreaming.gen_manager import Task

        load_start = time.monotonic()
        temp_file = None
        module_name = None
        load_start = time.monotonic()
        compile_start = None
        compile_end = None
        execute_end = None
        temp_file = None
        module_name = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as handle:
                handle.write(code_path.read_text(encoding="utf-8"))
                temp_file = handle.name
            try:
                cpu_device = jax.devices("cpu")[0]
            except IndexError:
                cpu_device = jax.local_devices(backend="cpu")[0]
            with jax.default_device(cpu_device):
                temp_task = Task(temp_file)
                env = temp_task.env
                module_name = temp_task.task.__module__
                params = env.default_params
                key = jax.random.PRNGKey(0)

                def _validate_on_cpu_impl(rng):
                    rng, reset_key = jax.random.split(rng)
                    obs, state = env.reset(reset_key, params)
                    action = env.action_space(params).sample(rng)
                    obs, state, reward, done, info = env.step(rng, state, action, params)
                    for field_name, value in state.inventory.__dict__.items():
                        if hasattr(value, "dtype") and value.dtype != jnp.int32:
                            raise ValueError(
                                f"Inventory field '{field_name}' has type "
                                f"{value.dtype}, expected int32."
                            )
                    return reward

                _validate_on_cpu = jax.jit(_validate_on_cpu_impl, backend="cpu")
                compile_start = time.monotonic()
                lowered = _validate_on_cpu.lower(key)
                compiled = lowered.compile()
                compile_end = time.monotonic()
                output = compiled(key)
                output.block_until_ready()
                execute_end = time.monotonic()
        finally:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except OSError:
                    pass
            if module_name:
                sys.modules.pop(module_name, None)

        result = {
            "valid": True,
            "message": "",
            "error_class": None,
            "load_s": round(compile_start - load_start, 6) if compile_start else None,
            "compile_s": round(compile_end - compile_start, 6) if compile_start and compile_end else None,
            "execute_s": round(execute_end - compile_end, 6) if compile_end and execute_end else None,
            "total_s": round(time.monotonic() - started, 6),
        }
    except Exception as exc:
        result = {
            "valid": False,
            "message": f"Compilation error: {exc}",
            "error_class": "cpu_jax_error",
            "load_s": None,
            "compile_s": None,
            "execute_s": None,
            "total_s": round(time.monotonic() - started, 6),
        }

    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0 if result["valid"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
