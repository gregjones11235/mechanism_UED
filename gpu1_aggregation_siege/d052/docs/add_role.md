# How to add a role

canonical_v2 has four roles. Three are SCORING roles (produce a per-candidate
headline score that selectors consume); the Modeler is an ADVISORY role (runs once
per session over the student state, never per-candidate scoring).

| Role | Kind | Headline score key | Pinned provider/model |
|---|---|---|---|
| Tutor | scoring | `progression_score` | dashscope / qwen-turbo |
| Critic | scoring | `critic_penalty` (+ `critic_reject`) | deepseek / deepseek-chat |
| Explorer | scoring | `novelty_score` | zhipu / glm-4.5-air |
| Modeler | advisory | — (siege foci / recommendation) | zhipu / glm-4.5 |

## Steps to add (or re-pin) a role

1. **Enum** — add the role to `RoleName` (all roles) and, if it is a scoring role,
   to `ScoringRole` in `d052/schemas/roles.py`. The Modeler must NOT be a
   `ScoringRole` (it never enters per-candidate selection scoring).
2. **Headline key** — add the role's headline score key to `HEADLINE_SCORE_KEY` in
   `d052/roles/protocol.py`. Every judgment for the role MUST carry this key in
   `scores`; `RoleJudgment` validation enforces it.
3. **Registry pin** — add a `RoleDefinition` to `ROLE_REGISTRY` with
   `provider`, `exact_model_id`, `prompt_version` (`canonical_v2.roles.v1`),
   `output_schema` (`role_judgment_v2`), and `is_scoring_role`. These pins must be
   reconciled with the legacy `model_manifest.py` / `llm_roles.py` (the registry
   already reconciles the two; keep them consistent). `assert_registry_consistency()`
   runs at import and fails on duplicate/missing roles.
4. **Critic special-case** — only the Critic carries `critic_reject` (required for
   critic, forbidden for others). Selectors consume it via the critic policy.
5. **Normalization** — scoring roles are normalized per-role independently by
   `d052/normalization/rank_percentile_v1`. Add the role's column to the matrix;
   no cross-role scale sharing.
6. **Selectors** — a scoring role enters S1/S2 via `SelectorConfig.roles` and the
   composite mean; Copeland/Auction aggregate all `role_scores`. No selector change
   is needed just to add a scoring role (it flows through `role_scores`).
7. **Tests** — extend GATE 6 (`tests/test_roles_and_profiling.py`): registry
   membership, headline key presence, batch validation, and (for the Modeler) the
   machine-facts/judgment firewall (no tier labels handed to the LLM).

## Constraints

- No new heavy dependency. Judgments are validated by Pydantic (`extra=forbid`).
- Every judgment records `provider` / `exact_model_id` / `prompt_version` for audit.
- The Modeler firewall is structural: `MachineFacts` carries facts only (held-out
  SR series, forgetting prefilter); mastery tiers are a deterministic downstream
  derivation and are NEVER passed to the LLM/selector.
