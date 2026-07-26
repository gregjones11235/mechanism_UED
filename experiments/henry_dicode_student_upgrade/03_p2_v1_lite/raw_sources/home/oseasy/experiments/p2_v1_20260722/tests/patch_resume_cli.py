#!/usr/bin/env python3
"""Wire production --resume-from (方案B exact resume) into stage4_continue_launcher.main().

Surgical, verifiable edits only.  Each old-string must match EXACTLY ONCE.
The fresh-start path (no --resume-from) is left bit-identical: the [4b] else
branch reproduces the previous `pending=None; c_*=None` defaults, and the loop
is unchanged.  Fails loudly if any anchor is missing or non-unique.
"""
import io
import sys

PATH = "/home/oseasy/experiments/p2_v1_20260722/src/stage4_continue_launcher.py"

with io.open(PATH, encoding="utf-8") as f:
    src = f.read()


def rep(old, new, label):
    n = src.count(old)
    assert n == 1, f"[{label}] anchor matched {n} times (need exactly 1)"
    return src.replace(old, new, 1)


# ── Edit A: main() signature ─────────────────────────────────────────
src = rep(
    "def main(max_sessions: Optional[int] = None,\n"
    "         num_updates: int = NUM_UPDATES_PER_SESSION):\n",
    "def main(max_sessions: Optional[int] = None,\n"
    "         num_updates: int = NUM_UPDATES_PER_SESSION,\n"
    "         resume_from: Optional[int] = None):\n",
    "A-signature")

# ── Edit C: remove pre-loop pending/collector init block ─────────────
# (moved into the [4b] branch below; anchor includes the unique comment)
src = rep(
    "    # 方案B: persistent cross-rollout/cross-session collector state.  Starts None\n"
    "    # (true fresh start -> first session resets the env); carried across sessions\n"
    "    # and checkpointed so a resumed run continues bit-exactly.\n"
    "    pending = None\n"
    "    c_env_state = None\n"
    "    c_obsv = None\n"
    "    c_memories = None\n"
    "    c_mem_mask = None\n"
    "    c_mem_idx = None\n"
    "\n"
    "    try:\n",
    "    # 方案B: persistent cross-rollout/cross-session collector state.  Initialized\n"
    "    # in the [4b/6] block above: None for a true fresh start (first session resets\n"
    "    # the env), or restored from the P2-v1 checkpoint when --resume-from is given.\n"
    "    # Carried across sessions and checkpointed so a resumed run continues bit-exactly.\n"
    "\n"
    "    try:\n",
    "C-preloop")

# ── Edit B: insert [4b/6] resume-override branch after [4/6] ─────────
src = rep(
    "    source_checkpoint_sha256 = fresh[\"source_checkpoint_sha256\"]\n",
    "    source_checkpoint_sha256 = fresh[\"source_checkpoint_sha256\"]\n"
    "\n"
    "    # ── [4b/6] 方案B exact-resume override (optional --resume-from) ──\n"
    "    # When resume_from is given, replace the fresh-start state with the state\n"
    "    # restored from P2-v1's OWN checkpoint (CKPT_ROOT/<resume_from>) so this run\n"
    "    # CONTINUES bit-exactly instead of restarting.  The fresh init in [4/6] is\n"
    "    # still run to recover the session175 source provenance for the manifest.\n"
    "    resuming = resume_from is not None\n"
    "    if resuming:\n"
    "        print(f\"\\n[4b/6] 方案B exact-resume from P2-v1 checkpoint \"\n"
    "              f\"step={resume_from} ...\")\n"
    "        rr = restore_p2_v1_checkpoint(\n"
    "            CKPT_ROOT, resume_from, network, cfg, obs_dim)\n"
    "        ts = rr[\"train_state\"]\n"
    "        replay = rr[\"replay_buffer\"]\n"
    "        rng = rr[\"rng\"]\n"
    "        action_rng = restore_action_rng(\n"
    "            rr[\"action_rng_state\"], seed=P2_V1_MASTER_SEED)\n"
    "        global_step = int(rr[\"global_step\"])\n"
    "        update_count = int(rr[\"update_count\"])\n"
    "        pending = PendingEpisodeBuffers.from_state_dict(rr[\"pending_state\"])\n"
    "        _cs = rr[\"collector_state\"]\n"
    "        c_env_state = _cs[\"env_state\"]\n"
    "        c_obsv = _cs[\"obsv\"]\n"
    "        c_memories = _cs[\"memories\"]\n"
    "        c_mem_mask = _cs[\"mem_mask\"]\n"
    "        c_mem_idx = _cs[\"mem_idx\"]\n"
    "        print(f\"  resumed: global_step={global_step} \"\n"
    "              f\"update_count={update_count} replay={len(replay)} \"\n"
    "              f\"pending={pending.total_pending_transitions()} \"\n"
    "              f\"(continues WITHOUT env.reset)\")\n"
    "    else:\n"
    "        # True fresh start: first session resets the env (collector state None).\n"
    "        pending = None\n"
    "        c_env_state = None\n"
    "        c_obsv = None\n"
    "        c_memories = None\n"
    "        c_mem_mask = None\n"
    "        c_mem_idx = None\n",
    "B-resume-branch")

# ── Edit D: manifest start / resume_from_step ────────────────────────
src = rep(
    "                \"start\": \"weights_only_from_session175\",\n"
    "                \"source_checkpoint\": source_checkpoint,\n"
    "                \"source_checkpoint_sha256\": source_checkpoint_sha256,\n"
    "                \"p2_v1_start_step\": P2_V1_GLOBAL_STEP_START,\n"
    "                \"resume_from_step\": P2_V1_GLOBAL_STEP_START,\n",
    "                \"start\": ((\"resume_from_p2v1_step_%d\" % resume_from)\n"
    "                          if resuming else \"weights_only_from_session175\"),\n"
    "                \"source_checkpoint\": source_checkpoint,\n"
    "                \"source_checkpoint_sha256\": source_checkpoint_sha256,\n"
    "                \"p2_v1_start_step\": P2_V1_GLOBAL_STEP_START,\n"
    "                \"resume_from_step\": (resume_from if resuming\n"
    "                                     else P2_V1_GLOBAL_STEP_START),\n",
    "D-manifest-start")

# ── Edit D2: manifest optimizer_fresh_at_start ───────────────────────
src = rep(
    "                \"optimizer_fresh_at_start\": True,\n",
    "                \"optimizer_fresh_at_start\": (not resuming),\n",
    "D2-manifest-optfresh")

# ── Edit E: argparse --resume-from + main() call ─────────────────────
src = rep(
    "    parser.add_argument(\n"
    "        \"--smoke-test\", action=\"store_true\",\n"
    "        help=\"Run weights-only fresh-start verification only (no training)\")\n"
    "    args = parser.parse_args()\n"
    "\n"
    "    if args.smoke_test:\n"
    "        success = smoke_test()\n"
    "        sys.exit(0 if success else 1)\n"
    "    else:\n"
    "        main(max_sessions=args.max_sessions, num_updates=args.num_updates)\n",
    "    parser.add_argument(\n"
    "        \"--resume-from\", type=int, default=None,\n"
    "        help=\"P2-v1 self-checkpoint step to resume from (方案B exact resume; \"\n"
    "             \"continues bit-exactly WITHOUT env.reset). Default None = fresh \"\n"
    "             \"weights-only start from session175.\")\n"
    "    parser.add_argument(\n"
    "        \"--smoke-test\", action=\"store_true\",\n"
    "        help=\"Run weights-only fresh-start verification only (no training)\")\n"
    "    args = parser.parse_args()\n"
    "\n"
    "    if args.smoke_test:\n"
    "        success = smoke_test()\n"
    "        sys.exit(0 if success else 1)\n"
    "    else:\n"
    "        main(max_sessions=args.max_sessions, num_updates=args.num_updates,\n"
    "             resume_from=args.resume_from)\n",
    "E-argparse")

with io.open(PATH, "w", encoding="utf-8") as f:
    f.write(src)

print("PATCH OK: all 6 edits applied (A, C, B, D, D2, E)")
