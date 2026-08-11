"""Static LLM-Authored UED teacher (V1).

Research line: WITHOUT simulator probe feedback, WITHOUT environment-state
save/restore, WITHOUT a world model, an LLM designs the curriculum using ONLY
the Student's normal training performance, behavior summaries, capability
profiles, and static Craftax rule knowledge.

Hard invariants enforced by this package (see reports/static_llm_ued/
design_contract.md):

* I1 the LLM only analyzes the Student and authors environment designs/code;
* I2 no per-candidate multi-episode simulator probe (legality is
  ``EnvGenerator.check_compilation`` only);
* I3 no candidate success-rate / regret / partial-progress feedback to the LLM;
* I4 no Frontier Archive; I5 no EnvState save/restore;
* I6 no multi-branch simulator search; I7 no world model / imagined rollout.

The package is OFF by default; it activates only for ``teacher=static_llm``.
Pure-logic modules (schemas, guards, plan_cache, invocation_gate, llm_client,
roles) import ONLY the standard library so they are testable offline; the
controller module is the single seam that touches the GenManager/jax world.
"""

STATIC_LLM_UED_VERSION = "dicode.teachers.static_llm_ued.v1"

__all__ = ["STATIC_LLM_UED_VERSION"]
