"""CC4 Tier3 scaffolded evaluation environment (V1).

Owner: CC4 (TIER3_EVALUATION_OWNER). This package is the SINGLE authorized
implementation of the Tier3 decomposed evaluation machinery: corridor
predicates, scaffold builder, front/back evaluator, boss-area judgement,
state bank, metric schema and failure taxonomy. CC2 only trains Students /
produces checkpoints; CC3 only consumes the StudentProfile CC4 emits. No
second implementation of these components is permitted (see reports).

Status discipline (round V1): every component is labelled honestly as one of
IMPLEMENTED_STATIC / TESTED_SYNTHETIC / TESTED_REAL_ENV / MATERIALIZED /
EVALUATED / SCIENTIFICALLY_VALIDATED. This round reaches at most
IMPLEMENTED_STATIC + TESTED_SYNTHETIC (+ possible TESTED_REAL_ENV_RESET on a
JAX/craftax host). No scaffold result may replace the full DEFEAT_KOBOLD task
(SCAFFOLDED_RESULTS_CAN_REPLACE_FULL_TASK=false) and no Tier3 breakthrough /
SOTA / SOTA-comparison claim is authorized this round.

The package reuses the CC4 V3 canonical-world machinery (runtime source
identity, EnvState serializer, field manifest, two-process orchestration,
SHA256SUMS, fail-closed). It extends V3 ONLY additively; it never rewrites the
frozen 54 remediation files or the original SHA256SUMS, and
GLOBAL_WORLD_SET_HASH stays BLOCKED_SOURCE_UNVERIFIED (owned solely by the
seed42 canonical world materializer).
"""

SCHEMA_PREFIX = "mechanism_UED"

# Scenario / identity-class constants (frozen V1).
SCENARIO_FULL = "full"
SCENARIO_FRONT_L2 = "front_l2"
SCENARIO_BACK_L2 = "back_l2"

IDENTITY_FULL = "CANONICAL_S4_EVALUATION"
IDENTITY_FRONT = "TIER3_FRONT_DIAGNOSTIC_SCAFFOLD"
IDENTITY_BACK = "TIER3_BACK_DIAGNOSTIC_SCAFFOLD"

# The one class admissible as exact evidence (owned by the V3 seed42
# canonical world materializer). Scaffold state banks must NEVER claim it.
IDENTITY_CANONICAL_EXACT = "CANONICAL_EVALUATOR_EXACT_WORLD_SET"

ACTION_MODE_FROZEN = "greedy_argmax"

__all__ = [
    "SCHEMA_PREFIX",
    "SCENARIO_FULL",
    "SCENARIO_FRONT_L2",
    "SCENARIO_BACK_L2",
    "IDENTITY_FULL",
    "IDENTITY_FRONT",
    "IDENTITY_BACK",
    "IDENTITY_CANONICAL_EXACT",
    "ACTION_MODE_FROZEN",
]
