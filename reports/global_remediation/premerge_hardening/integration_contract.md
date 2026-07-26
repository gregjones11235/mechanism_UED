# Integration & scientific-claims contract (round-6 revision -- eleven)

- UTC: `2026-07-26T15:10:12Z`

## Round-6 new/revised files
- `tools/global_evaluation/world_key_manifest_prototype.py (git mv from materialize_world_set_twice.py; DEPRECATED; fails closed)`
- `tools/global_evaluation/materialize_craftax_world_set_twice.py (actual serializer + materializer; static)`
- `tools/global_evaluation/world_materializer_negative_tests.py (10 negative tests)`
- `reports/global_remediation/premerge_hardening/world_generation_path_audit.{md,json}`
- `reports/global_remediation/premerge_hardening/host_run_boundary.{md,json}`
- `reports/global_remediation/premerge_hardening/{world_manifest_evidence_tier,integration_contract,pure_logic_gate_report}.{md,json} (revised)`
- `reports/global_remediation/world_set_materialization_runbook.{md,json} (revised)`

## World-set status
- OLD_WORLD_KEY_PROTOTYPE = **DEPRECATED_INVALID_FOR_WORLD_SET_HASH**
- WORLD_GENERATION_SOURCE_PATH = **FOUND**
- CRAFTAX_WORLD_MATERIALIZER_CODE = **IMPLEMENTED**
- CRAFTAX_WORLD_MATERIALIZER_REAL_RUN = **NOT_RUN**
- GLOBAL_WORLD_RECIPE = **PASS**
- GLOBAL_WORLD_SET_HASH = **BLOCKED_SOURCE_UNVERIFIED**
- statement = **actual materializer code IMPLEMENTED + statically tested, but NO real world materialized on this host; world_set_hash NOT upgraded.**

## ALLOWED claims (round 6)
- the canonical world-generation path is LOCATED and line/SHA-anchored (FOUND)
- an actual stable serializer + actual-reset materializer is IMPLEMENTED and statically tested (self-test/anchor/negative FAIL=0)
- the materialization gate REJECTS key-only output (GATE17 PASS)
- the deprecated key prototype is sealed (fails closed) with history preserved

## FORBIDDEN claims (round 6)
- the world set has been really materialized (NOT_RUN)
- a world_set_hash exists / is verified (BLOCKED_SOURCE_UNVERIFIED)
- evaluation_seed binding to real worlds is fully verified on hardware (PARTIAL_ENVIRONMENT_BLOCKED)
- materializer and evaluator share a literal builder function (strict BLOCKED)
- checkpoints re-evaluated / training reproduced / Exact Resume bit-exact / matched Replay run (all still NOT_RUN)

- CC2_FILES_TOUCHED=False ; CC3_FILES_TOUCHED=False ; PUSH_PERFORMED=False
