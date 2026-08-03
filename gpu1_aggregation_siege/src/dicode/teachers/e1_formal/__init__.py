"""E1 formal teacher: Behavior-Aware Regret-Guided LLM-UED (formal Direction One).

Demotion note (E1-S): the former Static-LLM-UED direction is preserved
UNMODIFIED in ``dicode.teachers.static_llm`` and kept only as the E1-S
weak ablation. E1 REUSES it via imports (guards, schemas, contract
consumer) and never modifies or deletes it.

Layering rule: every pure-logic module in this package is standard
library only (no jax / craftax / pydantic / networkx). The jax edge is
confined to ``gen_manager.py`` / ``archive_view.py`` and the evaluation
seam in ``dicode.evaluation.candidate_evaluation``.
"""

E1_FORMAL_VERSION = "e1_formal.v1"
