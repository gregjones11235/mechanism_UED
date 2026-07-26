"""D052 Phase 2.5 REAL migration-bundle reconciliation (canonical_v2 side).

Read-only tooling that binds the REAL Phase 2.5 Canonical Migration Bundle
(Modeler CC, server oseasy@172.25.14.221, SHA256-verified 2026-07-26) to the
canonical_v2 schemas in this worktree:

  * real_bundle            -- bundle location, integrity verification, judgment-hash
                              tamper-evidence formula
  * judgment_adapter       -- read-only bundle-judgment -> canonical RoleJudgment
                              adapter (+ audit envelope + glm role-normalization log)
  * prompt_profile_contract-- offline verification of the real prompt registry and
                              frozen StudentProfile contract
  * replay                 -- deterministic offline B/C historical replay (no LLM)

OFFLINE. NO LLM. NO TRAINING. Bundle originals are never modified.
"""
