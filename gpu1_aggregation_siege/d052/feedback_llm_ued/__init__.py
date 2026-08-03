"""SIMULATOR-GROUNDED FEEDBACK-ADAPTIVE LLM-UED (D052).

A curriculum-design closed loop in which an LLM (curriculum designer) is the
decision maker and a Craftax-style simulator is the VERIFIER and FEEDBACK
SOURCE (not the primary teacher):

    plan_k
      -> candidate environment generation
      -> simulator probe (Student + Reference, staged fast->full)
      -> expected-vs-observed comparison
      -> LLM feedback diagnosis
      -> RETAIN / MUTATE / RETIRE / REQUEST_CONTROL
      -> plan_{k+1}

This is deliberately DIFFERENT from the two sibling lines:

  * vs CC2 static LLM-UED — here the candidate probe RESULTS are fed back to
    the LLM and the LLM revises the next plan from them (not generate-then-
    accept/reject);
  * vs CC1 Simulator-Centric Frontier-UED — no EnvState restore, no Frontier
    State Archive, no multi-branch search from one mid-state, no PPO-from-
    deep-state. Candidates are standard-reset environment-level TaskParams;
    the LLM stays the curriculum leader and the simulator only validates and
    reports.

Authorization state THIS ROUND: TRAINING / FORMAL_EVALUATION / REAL_LLM /
REAL_SIMULATOR probes are all NOT authorized — the loop runs against a
deterministic symbolic probe runner and a deterministic mock LLM backend, and
the controller re-asserts every flag at startup (fail-closed). Real Craftax /
real LLM seams exist behind explicit adapters but stay BLOCKED.
"""

FEEDBACK_LLM_UED_VERSION = "d052.feedback_llm_ued.v1"
