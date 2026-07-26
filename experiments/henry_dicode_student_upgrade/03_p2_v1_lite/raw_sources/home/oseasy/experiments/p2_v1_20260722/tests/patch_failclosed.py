#!/usr/bin/env python3
"""Make stage4_continue_launcher.main() FAIL CLOSED on a mid-session crash.

Bug (diagnosed from Stage A): when run_session catches an exception it still
returns with ``global_step`` advanced PAST the last successfully-applied update
(e.g. global_step=4096 while only 1 update was applied).  main() then
UNCONDITIONALLY saved that inconsistent state as ``checkpoints/4096`` and the
process exited 0 — masking the crash and writing a corrupt checkpoint that can
never be used as a resume point.

Fix (surgical; the successful path stays bit-identical):
  1. Guard the per-session checkpoint save: only save when crash_info is None;
     otherwise set ckpt_path=None, mark session_failed, and print a loud
     FAIL-CLOSED banner (NO corrupt checkpoint written).
  2. Initialize session_failed=False before the session loop.
  3. At the end of main(), sys.exit(1) if any session crashed, else fall through
     (exit 0) exactly as before.

The manifest / training_log / source_hashes / crash json are still written on a
crash (with checkpoint_path=null and crash_info populated) so the evidence is
preserved; only the corrupt checkpoint write is suppressed and the exit code is
made non-zero so the orchestrator hard gates catch it.

Each old-string must match EXACTLY ONCE.  Fails loudly otherwise.
"""
import io

PATH = "/home/oseasy/experiments/p2_v1_20260722/src/stage4_continue_launcher.py"

with io.open(PATH, encoding="utf-8") as f:
    src = f.read()


def rep(old, new, label):
    n = src.count(old)
    assert n == 1, f"[{label}] anchor matched {n} times (need exactly 1)"
    return src.replace(old, new, 1)


# ── Edit 1: guard the per-session checkpoint save (fail closed) ────────
OLD_SAVE = (
    "            # Save full checkpoint after every session\n"
    "            ckpt_path = save_stage4_checkpoint(\n"
    "                ts, replay, rng, global_step, cfg,\n"
    "                session_index, session_dir, P2_V1_GLOBAL_STEP_START,\n"
    "                action_rng_state=action_rng_state(action_rng),\n"
    "                update_count=update_count,\n"
    "                source_checkpoint=source_checkpoint,\n"
    "                source_checkpoint_sha256=source_checkpoint_sha256,\n"
    "                pending_state=pending.state_dict(),\n"
    "                collector_state={\n"
    "                    \"env_state\": c_env_state,\n"
    "                    \"obsv\": c_obsv,\n"
    "                    \"memories\": c_memories,\n"
    "                    \"mem_mask\": c_mem_mask,\n"
    "                    \"mem_idx\": c_mem_idx,\n"
    "                })\n"
)

NEW_SAVE = (
    "            # Save full checkpoint after every SUCCESSFUL session.  A session\n"
    "            # that crashed mid-update has an INCONSISTENT state (global_step\n"
    "            # advanced past the last applied update), so we FAIL CLOSED: do\n"
    "            # NOT write a corrupt checkpoint, and propagate a non-zero exit so\n"
    "            # the orchestrator's hard gates catch it.  The manifest/log/crash\n"
    "            # json are still written below (checkpoint_path=null) as evidence.\n"
    "            if crash_info is None:\n"
    "                ckpt_path = save_stage4_checkpoint(\n"
    "                    ts, replay, rng, global_step, cfg,\n"
    "                    session_index, session_dir, P2_V1_GLOBAL_STEP_START,\n"
    "                    action_rng_state=action_rng_state(action_rng),\n"
    "                    update_count=update_count,\n"
    "                    source_checkpoint=source_checkpoint,\n"
    "                    source_checkpoint_sha256=source_checkpoint_sha256,\n"
    "                    pending_state=pending.state_dict(),\n"
    "                    collector_state={\n"
    "                        \"env_state\": c_env_state,\n"
    "                        \"obsv\": c_obsv,\n"
    "                        \"memories\": c_memories,\n"
    "                        \"mem_mask\": c_mem_mask,\n"
    "                        \"mem_idx\": c_mem_idx,\n"
    "                    })\n"
    "            else:\n"
    "                ckpt_path = None\n"
    "                session_failed = True\n"
    "                print(f\"\\n!! SESSION {session_index} CRASHED — \"\n"
    "                      f\"{crash_info['error']}\")\n"
    "                print(\"!! FAIL CLOSED: NOT saving checkpoint (state \"\n"
    "                      \"inconsistent: global_step advanced past last applied \"\n"
    "                      \"update). Will exit non-zero.\")\n"
)

src = rep(OLD_SAVE, NEW_SAVE, "1-guard-save")

# ── Edit 2: initialize session_failed before the session loop ──────────
OLD_LOOP = (
    "        while max_sessions is None or session_index < max_sessions:\n"
    "            session_index += 1\n"
)
NEW_LOOP = (
    "        session_failed = False\n"
    "        while max_sessions is None or session_index < max_sessions:\n"
    "            session_index += 1\n"
)
src = rep(OLD_LOOP, NEW_LOOP, "2-init-session-failed")

# ── Edit 3: non-zero exit at the end of main() if any session crashed ──
OLD_END = (
    "    print(f\"  Output:            {session_dir}\")\n"
    "    print(f\"{'='*60}\")\n"
)
NEW_END = (
    "    print(f\"  Output:            {session_dir}\")\n"
    "    print(f\"{'='*60}\")\n"
    "\n"
    "    if session_failed:\n"
    "        print(\"\\nEXIT 1: at least one session crashed — NO valid checkpoint \"\n"
    "              \"was saved for it. See crash_session_*.json and the \"\n"
    "              \"orchestrator log.\")\n"
    "        sys.exit(1)\n"
)
src = rep(OLD_END, NEW_END, "3-exit-nonzero")

with io.open(PATH, "w", encoding="utf-8") as f:
    f.write(src)

print("PATCH OK: launcher fail-closed (guard save + session_failed + exit 1)")
