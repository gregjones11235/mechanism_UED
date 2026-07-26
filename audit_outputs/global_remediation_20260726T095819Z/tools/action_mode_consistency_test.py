#!/usr/bin/env python
"""GATE1 support: verify a policy's ACTUAL sampling behavior matches its DECLARED action_mode.

Pure-logic + numpy (runs without JAX). A JAX policy can be wrapped into a `sampler(k)->int array` callable.

Rule (CANONICAL_EVALUATOR_V1 A3):
  - declared 'argmax'     => K draws at fixed state must yield EXACTLY 1 unique action vector.
  - declared 'stochastic' => K draws (distinct rng keys) must yield >1 unique action vector with high prob
                             (a truly stochastic categorical with >=2 non-degenerate logits will vary).
A mismatch => action_mode INCONSISTENT (hard-fail). Never infer mode from a dead branch/default.
"""
import argparse, numpy as np

def check_argmax(sampler, state, K=8):
    draws = [tuple(np.asarray(sampler(state, i)).ravel().tolist()) for i in range(K)]
    uniq = set(draws)
    return {"declared": "argmax", "unique_actions_over_K": len(uniq),
            "consistent": len(uniq) == 1,
            "detail": "argmax must be deterministic: exactly 1 unique action over K draws"}

def check_stochastic(sampler, state, K=32):
    draws = [tuple(np.asarray(sampler(state, i)).ravel().tolist()) for i in range(K)]
    uniq = set(draws)
    return {"declared": "stochastic", "unique_actions_over_K": len(uniq),
            "consistent": len(uniq) > 1,
            "detail": "stochastic must vary: >1 unique action over K distinct-key draws "
                      "(if ==1, logits may be degenerate OR mode is secretly argmax)"}

def numpy_categorical_sampler(logits):
    """Builds a sampler(state, key)->action using numpy softmax sampling (stochastic) for demonstration."""
    def sampler(state, key):
        rng = np.random.default_rng(key)
        p = np.exp(logits - logits.max(-1, keepdims=True)); p = p / p.sum(-1, keepdims=True)
        return np.array([rng.choice(len(row), p=row) for row in p])
    return sampler

def numpy_argmax_sampler(logits):
    def sampler(state, key):
        return np.argmax(logits, -1)
    return sampler

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true", help="run built-in numpy demo checks")
    a = ap.parse_args()
    if a.self_test:
        logits = np.array([[1.0, 2.0, 1.5], [0.1, 0.2, 5.0], [2.0, 2.0, 2.0]])  # non-degenerate
        r1 = check_stochastic(numpy_categorical_sampler(logits), state=None, K=32)
        r2 = check_argmax(numpy_argmax_sampler(logits), state=None, K=8)
        # a degenerate stochastic that always picks same => would FAIL stochastic check (correct behavior)
        deg = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
        r3 = check_stochastic(numpy_categorical_sampler(deg), state=None, K=32)
        import json
        print(json.dumps({"stochastic_nondeg": r1, "argmax": r2, "stochastic_degenerate(expect inconsistent)": r3}, indent=2))
        ok = r1["consistent"] and r2["consistent"] and (not r3["consistent"])
        print("SELF_TEST_PASS" if ok else "SELF_TEST_FAIL")
        raise SystemExit(0 if ok else 1)
    print("usage: --self-test  (wrap a JAX policy into sampler(state,key)->actions to test a real policy)")

if __name__ == "__main__":
    main()
