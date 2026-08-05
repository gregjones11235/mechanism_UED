"""P0-15 (§19 seam coverage): Arm-C selection-boundary tie classification.

READ-ONLY investigation — the frozen evidence (judgments, ranking_C.json,
expected_behavior.json, selector_config.json, the ORIGINAL selector
source) is opened read-only and never modified. The classification is a
pure function of the frozen artifacts:

* the C arm's final Soft-Copeland scores for d052_r3_0007 and
  d052_r3_0028 are EXACTLY equal (0.81423508564678215) and straddle the
  k=8 selection boundary;
* ranking_C.json assigns rank 8 to 0007 and rank 9 to 0028 (stable index
  order), while expected_behavior.json C_selected8 contains 0028 and NOT
  0007 — the rank column and the selection contradict each other;
* numpy's default quicksort is NOT a deterministic tie-break: it picks
  0028 here (reproducing C_selection_hash=868a57268d66b90b), while
  stable/mergesort/heapsort all pick 0007 (matching the rank column);
* classification: FROZEN_EVIDENCE_INTERNALLY_INCONSISTENT — no single
  deterministic tie-break reproduces both frozen artifacts, so the legacy
  replay contract is not fixed and stays
  LEGACY_REPLAY_BLOCKED_NON_PRODUCTION.

All fixtures are REAL but read-only (the frozen bundle is opened for
reading only); NO real LLM call, NO simulator episode, and NO passing
test flips a REAL_* flag.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from d052.reconciliation import arm_c_tie_resolution as T
from d052.reconciliation.real_bundle import (
    BUNDLE_REL,
    REPO_ROOT,
    load_judgments,
)
from d052.reconciliation.replay import (
    WEIGHTS,
    _build_signals,
    _load_aggregation,
)

BUNDLE = REPO_ROOT / BUNDLE_REL
AGG_PATH = (REPO_ROOT / "orchestration/experiments/"
            "d052_modeler_shadow_v1/replay_inputs/aggregation_original.py")


def _scores():
    agg = _load_aggregation(AGG_PATH)
    records = load_judgments("C", BUNDLE)
    ids, sig = _build_signals(records)
    return ids, agg._aggregate_soft_copeland(sig, WEIGHTS, 1.0)


def _frozen_bytes(name):
    return (BUNDLE / name).read_bytes()


class TestTieFacts:
    def test_exact_tie_at_the_boundary(self):
        ids, scores = _scores()
        i7, i28 = ids.index(T.TIED_A), ids.index(T.TIED_B)
        assert float(scores[i7]) == float(scores[i28])
        assert float(scores[i7]) == 0.81423508564678215

    def test_quicksort_selects_0028_and_reproduces_frozen_hash(self):
        ids, scores = _scores()
        sel = [ids[i] for i in np.argsort(-scores)[:8]]
        assert T.TIED_A not in sel
        assert T.TIED_B in sel
        sel_hash = T._sha16(json.dumps(sorted(sel)))
        assert sel_hash == T.EXPECTED_C_SELECTION_HASH == "868a57268d66b90b"

    def test_stable_sorts_select_the_other_candidate(self):
        ids, scores = _scores()
        for kind in ("stable", "mergesort", "heapsort"):
            sel = [ids[i] for i in np.argsort(-scores, kind=kind)[:8]]
            assert T.TIED_A in sel, kind
            assert T.TIED_B not in sel, kind

    def test_rank_column_contradicts_frozen_selection(self):
        ranking = json.loads((BUNDLE / "ranking_C.json")
                             .read_text(encoding="utf-8"))
        rank_by_id = {row["task_id"]: row["rank"] for row in ranking}
        eb = json.loads((BUNDLE / "expected_behavior.json")
                        .read_text(encoding="utf-8"))
        c_selected8 = eb["C_selected8"]
        #: rank 8 = 0007 (higher) but the selection holds 0028 — the two
        #: frozen artifacts disagree about the k=8 boundary candidate
        assert rank_by_id[T.TIED_A] < rank_by_id[T.TIED_B]
        assert T.TIED_B in c_selected8
        assert T.TIED_A not in c_selected8


class TestClassification:
    def test_classification_is_internally_inconsistent(self):
        result = T.classify_arm_c_tie()
        assert result["classification"] \
            == T.FROZEN_EVIDENCE_INTERNALLY_INCONSISTENT
        assert result["exact_tie"] is True
        assert result["boundary_flip"] is True
        assert result["stable_disagrees_with_quicksort"] is True
        assert result["quicksort_selection_hash"] \
            == T.EXPECTED_C_SELECTION_HASH
        assert result["frozen_evidence_untouched"] is True

    def test_frozen_evidence_is_not_modified(self):
        #: the classification is a pure read — byte content unchanged
        before = {
            name: _frozen_bytes(name) for name in (
                "ranking_C.json", "expected_behavior.json",
                "selector_config.json")}
        agg_before = AGG_PATH.read_bytes()
        T.classify_arm_c_tie()
        for name, content in before.items():
            assert (BUNDLE / name).read_bytes() == content, name
        assert AGG_PATH.read_bytes() == agg_before

    def test_legacy_replay_anchors_still_reproduce(self):
        #: the read-only classification must not disturb the historical
        #: replay contract (anchors + frozen hashes unchanged)
        from d052.reconciliation.replay import run_replay
        r = run_replay()
        assert r["ALL_ANCHORS_PASS"] is True
        rec = r["recomputed"]
        assert rec["C_selection_hash"] == T.EXPECTED_C_SELECTION_HASH
        assert rec["B_selection_hash"] == "82571538e5299ea9"
        assert r["checks"]["C_determinism_bitidentical"] is True


class TestPosture:
    def test_no_real_capability_flags(self):
        from d052.feedback_llm_ued import constants as C
        for name in C.NEVER_TRUE_REAL_CAPABILITY_FLAGS:
            assert getattr(C, name) is False, name

    def test_production_path_unaffected(self):
        #: the legacy Arm-C tie stays out of the production path
        from d052.feedback_llm_ued import constants as C
        assert C.LEGACY_REPLAY_BLOCKED_NON_PRODUCTION == \
            "LEGACY_REPLAY_BLOCKED_NON_PRODUCTION"
