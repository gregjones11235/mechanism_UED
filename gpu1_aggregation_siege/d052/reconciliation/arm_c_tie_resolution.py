"""P0-15: Arm-C selection-boundary tie classification (READ-ONLY).

Investigation result (this worktree, numpy installed): the C arm's final
Soft-Copeland scores for ``d052_r3_0007`` and ``d052_r3_0028`` are
EXACTLY equal at ``0.81423508564678215`` and straddle the k=8 selection
boundary (ranks 8/9). The frozen evidence is INTERNALLY INCONSISTENT:

* ``ranking_C.json`` assigns rank 8 to ``d052_r3_0007`` and rank 9 to
  ``d052_r3_0028`` — the tie is broken in STABLE index order (lower
  ``task_id`` first);
* ``expected_behavior.json`` ``C_selected8`` contains ``d052_r3_0028``
  but NOT ``d052_r3_0007`` — the boundary candidate flipped;
* ``selector_config.json`` declares the final tie broken by
  "argsort index order" (``np.argsort(-scores)[:8]``, numpy quicksort).
  numpy's DEFAULT quicksort is NOT stable and its tie order is
  implementation-defined: on THIS worktree's numpy it picks
  ``d052_r3_0028`` (reproducing the frozen
  ``C_selection_hash=868a57268d66b90b``), while ``kind='stable'``,
  ``'mergesort'`` and ``'heapsort'`` ALL pick ``d052_r3_0007`` (matching
  the rank column).

No single deterministic tie-break rule reproduces BOTH frozen artifacts
at once — the historical selection is not uniquely deterministic across
environments, so :data:`FROZEN_EVIDENCE_INTERNALLY_INCONSISTENT` is the
only honest classification (:data:`REPLAY_CONTRACT_FIXED` is not
defensible). The legacy replay contract stays
``LEGACY_REPLAY_BLOCKED_NON_PRODUCTION``: it does not enter the
production path, and NOTHING here writes to the frozen evidence (the
classification is a pure function of the frozen artifacts; the original
selector source and the bundle JSON files are opened read-only).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from d052.reconciliation.real_bundle import (
    BUNDLE_REL,
    REPLAY_INPUTS_REL,
    REPO_ROOT,
    load_bundle_json,
    load_judgments,
)
from d052.reconciliation.replay import (
    WEIGHTS,
    _build_signals,
    _load_aggregation,
)

FROZEN_EVIDENCE_INTERNALLY_INCONSISTENT = \
    "FROZEN_EVIDENCE_INTERNALLY_INCONSISTENT"
REPLAY_CONTRACT_FIXED = "REPLAY_CONTRACT_FIXED"

#: the frozen C selection hash the historical quicksort reproduced
EXPECTED_C_SELECTION_HASH = "868a57268d66b90b"

#: the tied pair straddling the k=8 selection boundary (measured, frozen)
TIED_A = "d052_r3_0007"
TIED_B = "d052_r3_0028"


def _arm_c_scores(bdir: Path):
    """Recompute the C arm's final scores with the ORIGINAL selector —
    the exact same machinery ``replay._run_arm`` uses (read-only)."""
    agg_path = (REPO_ROOT / REPLAY_INPUTS_REL) / "aggregation_original.py"
    agg = _load_aggregation(agg_path)
    records = load_judgments("C", bdir)
    ids, sig = _build_signals(records)
    scores = agg._aggregate_soft_copeland(sig, WEIGHTS, 1.0)
    return ids, scores


def _sha16(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def classify_arm_c_tie(bdir: Optional[Path] = None) -> Dict[str, object]:
    """P0-15 read-only classification of the Arm-C boundary tie.

    Returns the evidence and the classification:
    ``FROZEN_EVIDENCE_INTERNALLY_INCONSISTENT`` (the frozen rank column
    and the frozen selection disagree about the boundary candidate, and
    no stable tie-break reproduces both) or ``REPLAY_CONTRACT_FIXED``.
    """
    bdir = Path(bdir) if bdir else REPO_ROOT / BUNDLE_REL
    ids, scores = _arm_c_scores(bdir)

    #: 1. the exact tie (bit-identical scores)
    idx_a, idx_b = ids.index(TIED_A), ids.index(TIED_B)
    score_a, score_b = float(scores[idx_a]), float(scores[idx_b])
    exact_tie = (score_a == score_b)

    #: 2. the historical quicksort selection (replay.py:77 semantics)
    sel_idx = list(np.argsort(-scores)[:8])
    sel8 = [ids[i] for i in sel_idx]
    selection_hash = _sha16(json.dumps(sorted(sel8)))
    a_selected = TIED_A in sel8
    b_selected = TIED_B in sel8

    #: 3. the frozen rank column and the frozen selection
    ranking = load_bundle_json("ranking_C.json", bdir)
    rank_by_id = {row["task_id"]: int(row["rank"]) for row in ranking}
    eb = load_bundle_json("expected_behavior.json", bdir)
    c_selected8 = list(eb["C_selected8"])
    rank_a = rank_by_id.get(TIED_A)
    rank_b = rank_by_id.get(TIED_B)
    frozen_a_selected = TIED_A in c_selected8
    frozen_b_selected = TIED_B in c_selected8

    #: 4. consistency: does the frozen rank column agree with the frozen
    #:    selection about who holds the higher rank at the boundary?
    boundary_flip = bool(exact_tie and rank_a and rank_b
                         and rank_a < rank_b
                         and frozen_b_selected and not frozen_a_selected)

    #: 5. stability: do stable sorts pick the OTHER candidate? (proves the
    #:    quicksort tie order is NOT "index order")
    stable_sel = [ids[i] for i in list(np.argsort(-scores, kind="stable")[:8])]
    stable_picks_a = TIED_A in stable_sel
    stable_picks_b = TIED_B in stable_sel
    stable_disagrees_with_quicksort = (
        exact_tie and (stable_picks_a != a_selected))

    if exact_tie and (boundary_flip or stable_disagrees_with_quicksort):
        classification = FROZEN_EVIDENCE_INTERNALLY_INCONSISTENT
    elif not exact_tie:
        classification = REPLAY_CONTRACT_FIXED
    else:
        classification = FROZEN_EVIDENCE_INTERNALLY_INCONSISTENT

    return dict(
        classification=classification,
        tied_pair=[TIED_A, TIED_B],
        exact_score="%.17f" % score_a,
        exact_tie=exact_tie,
        quicksort_selected_a=bool(a_selected),
        quicksort_selected_b=bool(b_selected),
        quicksort_selection_hash=selection_hash,
        expected_c_selection_hash=EXPECTED_C_SELECTION_HASH,
        rank_a=rank_a,
        rank_b=rank_b,
        frozen_c_selected8=c_selected8,
        frozen_selected_a=bool(frozen_a_selected),
        frozen_selected_b=bool(frozen_b_selected),
        boundary_flip=boundary_flip,
        stable_sort_picks_a=bool(stable_picks_a),
        stable_sort_picks_b=bool(stable_picks_b),
        stable_disagrees_with_quicksort=bool(
            stable_disagrees_with_quicksort),
        frozen_evidence_untouched=True)


__all__ = [
    "FROZEN_EVIDENCE_INTERNALLY_INCONSISTENT", "REPLAY_CONTRACT_FIXED",
    "EXPECTED_C_SELECTION_HASH", "TIED_A", "TIED_B",
    "classify_arm_c_tie",
]
