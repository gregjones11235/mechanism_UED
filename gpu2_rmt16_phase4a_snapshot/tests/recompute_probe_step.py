#!/usr/bin/env python3
"""Phase4A-v2 §二 / GATE 1 — OFFLINE recompute of the L512 probe first_ge512 PRECISE resolved step.

Re-derives each probe episode's authoritative `completion_resolved_env_step` from the EXISTING
episode records (written by the frozen 16384-step probe), WITHOUT rerunning the probe:

    resolved = update_index * num_envs * rollout_steps
             + rollout_step * num_envs
             + env_id
             + 1

Constants for the frozen probe: num_envs=16, rollout_steps=128. Each record carries
(update_index, rollout_step, env_id, length, episode_id). The reachability conclusion is
re-checked (any length>=512 => reachable) and is expected to remain BOTH=PASS; this script
only CORRECTS the step provenance, it does NOT change the verdict.

Usage:
    python recompute_probe_step.py <episodes.jsonl> [label]
    python recompute_probe_step.py --selftest          # synthetic method self-test (no data needed)
Prints a JSON report; exit 0 on success.
"""
import json
import sys

NUM_ENVS = 16
ROLLOUT_STEPS = 128
GE_LEN = 512


def completion_resolved_env_step(update_index, num_envs, rollout_steps, rollout_step, env_id):
    """Authoritative precise resolved env step (1-indexed). See phase4a_v2_counters."""
    return (int(update_index) * int(num_envs) * int(rollout_steps)
            + int(rollout_step) * int(num_envs) + int(env_id) + 1)


def completion_global_step_deprecated(update_index, num_envs, rollout_steps, rollout_step):
    """Old, non-precise formula (kept only to show the before/after delta)."""
    return int(update_index) * (int(num_envs) * int(rollout_steps)) + int(rollout_step)


def recompute_from_records(records, num_envs=NUM_ENVS, rollout_steps=ROLLOUT_STEPS, ge_len=GE_LEN):
    """records: chronological list of episode dicts (file order == collection order)."""
    total = len(records)
    ge = [r for r in records if int(r["length"]) >= ge_len]
    first = None
    for r in ge:  # first in chronological (file) order
        ui = int(r["update_index"]); rs = int(r["rollout_step"]); eid = int(r["env_id"])
        resolved = completion_resolved_env_step(ui, num_envs, rollout_steps, rs, eid)
        deprecated = completion_global_step_deprecated(ui, num_envs, rollout_steps, rs)
        first = dict(
            episode_id=int(r["episode_id"]), length=int(r["length"]),
            update_index=ui, rollout_step=rs, env_id=eid,
            first_ge512_resolved_env_step=resolved,
            first_ge512_global_step_deprecated=deprecated,
            step_correction_delta=resolved - deprecated,
        )
        break
    return dict(
        num_envs=num_envs, rollout_steps=rollout_steps, ge_len=ge_len,
        total_completed_episodes=total,
        count_ge512=len(ge),
        reachable=bool(len(ge) > 0),
        first_ge512=first,
    )


def _selftest():
    # Synthetic records: lengths [100, 600, 200]; the 600-step ep is first >=512.
    recs = [
        dict(episode_id=0, length=100, update_index=0, rollout_step=10, env_id=2),
        dict(episode_id=1, length=600, update_index=2, rollout_step=5, env_id=3),
        dict(episode_id=2, length=200, update_index=3, rollout_step=1, env_id=0),
    ]
    rep = recompute_from_records(recs)
    # resolved = 2*16*128 + 5*16 + 3 + 1 = 4096 + 80 + 4 = 4180
    expected = 2 * 16 * 128 + 5 * 16 + 3 + 1
    assert rep["count_ge512"] == 1, rep
    assert rep["reachable"] is True, rep
    assert rep["first_ge512"]["first_ge512_resolved_env_step"] == expected == 4180, rep
    # deprecated = 2*2048 + 5 = 4101; delta = 79
    assert rep["first_ge512"]["first_ge512_global_step_deprecated"] == 4101, rep
    assert rep["first_ge512"]["step_correction_delta"] == 79, rep
    # empty / not-reachable case
    rep2 = recompute_from_records([dict(episode_id=0, length=100, update_index=0,
                                        rollout_step=0, env_id=0)])
    assert rep2["reachable"] is False and rep2["first_ge512"] is None, rep2
    return rep


def main(argv):
    if len(argv) > 1 and argv[1] == "--selftest":
        rep = _selftest()
        print("RECOMPUTE_SELFTEST=PASS")
        print(json.dumps(rep, indent=2))
        return 0
    if len(argv) < 2:
        print("usage: recompute_probe_step.py <episodes.jsonl> [label] | --selftest", file=sys.stderr)
        return 2
    path = argv[1]
    label = argv[2] if len(argv) > 2 else path
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    rep = recompute_from_records(records)
    rep["source"] = label
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
