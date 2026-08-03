"""C9 tests: G4 cross-implementation Soft Copeland parity gate.

THIS TEST HAS NO SKIP PATH. It enforces, against the canonical d052
implementation on THIS branch (read-only import on the test side only
— the E1 runtime never imports d052 selectors):

1. the pinned source SHA256s equal the branch's canonical bytes AND
   the supervisor-frozen report ``reports/d052_canonical_artifacts_
   SHA256SUMS`` (mismatch => COPELAND_SOURCE_SHA_MISMATCH hard stop;
   the pin values are never self-updated to bypass);
2. per-candidate Copeland score vectors are EXACTLY equal;
3. the full pairwise outcome matrices are EXACTLY equal;
4. the final rankings are EXACTLY equal under every critic policy;
5. the canonical result hashes are EXACTLY equal (including the
   shortfall path);
6. both implementations are input-order independent.

Line-ending note (audited): the frozen SHAs are over the committed
LF blob bytes. A Windows checkout materializes them with CRLF, so the
hash here is computed over ``data.replace(b"\\r\\n", b"\\n")`` — the
canonical blob bytes — never over the raw working-copy bytes.

All fixture scores are binary rationals so strength sums are exact
and tie relations are immune to float summation order. Every fixture
candidate is labeled FIXTURE — no real evaluation data exists this
round.
"""
import hashlib
import os

import pytest

from dicode.teachers.e1_formal import selector as S

# read-only d052 imports (TEST SIDE ONLY; canonical implementation)
from d052.legacy import canonical_constants
from d052.schemas.selector import CriticPolicy, SelectorConfig, SelectorType
from d052.selectors.base import CandidateSignals, SelectorSignals
from d052.selectors.base import rank_by as d052_rank_by
from d052.selectors.copeland import _strength as d052_strength
from d052.selectors.copeland import copeland_scores as d052_copeland_scores
from d052.selectors.copeland import (
    select_soft_copeland as d052_select_soft_copeland,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SHA_REPORT = os.path.join(REPO_ROOT, "reports", "d052_canonical_artifacts_SHA256SUMS")

SEED = 20260803
POOL_HASH = "ab" * 32

#: FIXTURE candidates: 7 (>= 6), with exact ties (alpha/beta; epsilon/
#: eta), one hard veto, one soft penalty, one empty role_scores. All
#: scores are binary rationals -> exact float arithmetic.
FIXTURE_CANDIDATES = [
    {
        "candidate_id": "c_alpha",
        "role_scores": {"tutor": 0.75, "critic": 0.5, "explorer": 0.625},
    },
    {
        "candidate_id": "c_beta",
        "role_scores": {"tutor": 0.5, "critic": 0.75, "explorer": 0.625},
    },
    {
        "candidate_id": "c_gamma",
        "role_scores": {"tutor": 0.25, "explorer": 0.375},
    },
    {"candidate_id": "c_delta", "role_scores": {}},
    {
        "candidate_id": "c_epsilon",
        "role_scores": {"tutor": 0.5},
        "critic_reject": True,
    },
    {
        "candidate_id": "c_zeta",
        "role_scores": {"tutor": 0.875, "explorer": 0.25},
        "critic_penalty": 0.25,
    },
    {
        "candidate_id": "c_eta",
        "role_scores": {"tutor": 0.5, "critic": 0.5},
    },
]

POLICIES = (
    (S.CRITIC_HARD_VETO, CriticPolicy.HARD_VETO),
    (S.CRITIC_SOFT_PENALTY, CriticPolicy.SOFT_PENALTY),
    (S.CRITIC_SCORE_ONLY, CriticPolicy.SCORE_ONLY),
)


def _canonical_blob_bytes(relpath):
    path = os.path.join(REPO_ROOT, relpath)
    with open(path, "rb") as handle:
        data = handle.read()
    # normalize CRLF working-copy bytes to the committed LF blob bytes
    return data.replace(b"\r\n", b"\n")


def _d052_signals(candidates):
    return SelectorSignals(
        pool_hash=POOL_HASH,
        candidates=[CandidateSignals(**cand) for cand in candidates],
    )


def _e1_signals(candidates):
    enriched = [
        dict(
            cand,
            provenance="CANDIDATE_EVALUATION",
            has_real_probe=True,
        )
        for cand in candidates
    ]
    return S.consume_candidate_signals(enriched, "parity")


def _d052_config(policy, k):
    return SelectorConfig(
        selector=SelectorType.SOFT_COPELAND,
        critic_policy=policy,
        k=k,
        seed=SEED,
    )


def _permuted(candidates, mode):
    if mode == "reverse":
        return list(reversed(candidates))
    if mode == "rotate":
        return candidates[3:] + candidates[:3]
    return list(candidates)


class TestSourcePinGate:
    """Pin SHA vs branch canonical bytes vs frozen supervisor report."""

    PINS = (
        (S.COPELAND_SOURCE_PATH, S.COPELAND_SOURCE_SHA256),
        (S.COPELAND_CONSTANTS_PATH, S.COPELAND_CONSTANTS_SHA256),
        (S.COPELAND_BASE_PATH, S.COPELAND_BASE_SHA256),
    )

    def test_pinned_shas_match_branch_canonical_sources(self):
        for relpath, pinned in self.PINS:
            digest = hashlib.sha256(_canonical_blob_bytes(relpath)).hexdigest()
            assert digest == pinned, (
                "COPELAND_SOURCE_SHA_MISMATCH: pinned SHA for "
                f"{relpath} does not match the branch canonical bytes; "
                "hard stop — the pin must NOT be self-updated to bypass "
                "(report to supervisor)"
            )

    def test_pins_match_supervisor_frozen_report(self):
        report = {}
        with open(SHA_REPORT, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                sha, path = line.split(None, 1)
                report[path] = sha
        for relpath, pinned in self.PINS:
            key = f"gpu1_aggregation_siege/{relpath}"
            assert key in report, f"missing frozen report entry {key}"
            assert report[key] == pinned, (
                "COPELAND_SOURCE_SHA_MISMATCH: selector pin diverges "
                f"from the frozen report for {relpath}"
            )

    def test_protocol_version_pin_matches_canonical_constants(self):
        assert (
            canonical_constants.CANONICAL_PROTOCOL_VERSION
            == S.COPELAND_PROTOCOL_VERSION
            == "canonical_v2"
        )


class TestParityGate:
    """Four-way exact equality + order independence (NO skip)."""

    def test_per_candidate_score_vectors_exactly_equal(self):
        d052 = d052_copeland_scores(_d052_signals(FIXTURE_CANDIDATES))
        e1 = S.copeland_scores(_e1_signals(FIXTURE_CANDIDATES))
        assert e1 == d052
        # ties are real: alpha and beta share one strength class
        assert d052["c_alpha"] == d052["c_beta"]
        assert d052["c_epsilon"] == d052["c_eta"]

    def test_full_pairwise_matrices_exactly_equal(self):
        signals = _d052_signals(FIXTURE_CANDIDATES)
        by = signals.by_id()
        ids = sorted(by)
        expected = {}
        for i, a in enumerate(ids):
            sa = d052_strength(by[a])
            for b in ids[i + 1:]:
                sb = d052_strength(by[b])
                expected[(a, b)] = a if sa > sb else (b if sa < sb else "tie")
        got = S.pairwise_matrix(_e1_signals(FIXTURE_CANDIDATES))
        assert got == expected
        assert got[("c_alpha", "c_beta")] == "tie"
        assert len(expected) == len(ids) * (len(ids) - 1) // 2

    def test_final_order_exactly_equal_under_every_policy(self):
        d052_signals = _d052_signals(FIXTURE_CANDIDATES)
        base_scores = d052_copeland_scores(d052_signals)
        e1_signals = _e1_signals(FIXTURE_CANDIDATES)
        for e1_policy, d052_policy in POLICIES:
            d052_ranked, d052_rejected = d052_rank_by(
                d052_signals,
                _d052_config(d052_policy, len(FIXTURE_CANDIDATES)),
                lambda sig: base_scores[sig.candidate_id],
            )
            e1_ranked, e1_rejected = S.rank_candidates(e1_signals, e1_policy)
            assert e1_ranked == [
                (cid, composite) for cid, composite in d052_ranked
            ], f"ranking diverged under {e1_policy}"
            assert e1_rejected == d052_rejected, e1_policy

    def test_canonical_result_hashes_exactly_equal(self):
        d052_signals = _d052_signals(FIXTURE_CANDIDATES)
        e1_signals = _e1_signals(FIXTURE_CANDIDATES)
        for e1_policy, d052_policy in POLICIES:
            for k in (1, 3, 5, 7, 9):  # 9 > pool -> shortfall path
                d052_result = d052_select_soft_copeland(
                    _d052_config(d052_policy, k), d052_signals
                )
                e1_result = S.select_soft_copeland(
                    e1_signals, k=k, seed=SEED, critic_policy=e1_policy
                )
                assert e1_result.selection_hash == d052_result.selection_hash, (
                    f"COPELAND_PARITY_MISMATCH: hash diverged "
                    f"(policy={e1_policy}, k={k})"
                )
                assert list(e1_result.selected_ids) == d052_result.selected_ids
                assert e1_result.status == d052_result.selection_status.value
                assert e1_result.eligible_count == d052_result.eligible_count
                assert (
                    e1_result.rejected_by_critic
                    == tuple(d052_result.rejected_by_critic)
                )
                assert e1_result.candidate_count_in == len(FIXTURE_CANDIDATES)

    @pytest.mark.parametrize("mode", ["reverse", "rotate"])
    def test_both_implementations_are_order_independent(self, mode):
        permuted = _permuted(FIXTURE_CANDIDATES, mode)
        for e1_policy, d052_policy in POLICIES:
            base_result = d052_select_soft_copeland(
                _d052_config(d052_policy, 5),
                _d052_signals(FIXTURE_CANDIDATES),
            )
            d052_permuted = d052_select_soft_copeland(
                _d052_config(d052_policy, 5), _d052_signals(permuted)
            )
            e1_permuted = S.select_soft_copeland(
                _e1_signals(permuted), k=5, seed=SEED, critic_policy=e1_policy
            )
            assert d052_permuted.selected_ids == base_result.selected_ids
            assert d052_permuted.selection_hash == base_result.selection_hash
            assert list(e1_permuted.selected_ids) == base_result.selected_ids
            assert e1_permuted.selection_hash == base_result.selection_hash

    def test_e1_promotion_gate_requires_real_probes(self):
        # parity replica core is pure; the E1 gate layers real-evidence
        # enforcement on top (blocked path feeds anchors + REUSE batch)
        signals = S.consume_candidate_signals(
            [
                {
                    "candidate_id": "c_alpha",
                    "role_scores": {"tutor": 0.75},
                    "provenance": "CANDIDATE_EVALUATION",
                    "has_real_probe": True,
                },
                {
                    "candidate_id": "c_beta",
                    "role_scores": {"tutor": 0.5},
                    "provenance": "CANDIDATE_EVALUATION",
                    "has_real_probe": True,
                },
            ],
            "parity",
        )
        outcome = S.select_dynamic_batch(signals, k=2, seed=SEED)
        assert outcome.status == S.STATUS_OK
        no_probe = S.consume_candidate_signals(
            [
                {
                    "candidate_id": "c_alpha",
                    "role_scores": {"tutor": 0.75},
                    "provenance": "CANDIDATE_EVALUATION",
                    "has_real_probe": False,
                }
            ],
            "parity",
        )
        with pytest.raises(S.SelectorError) as excinfo:
            S.select_dynamic_batch(no_probe, k=1, seed=SEED)
        assert excinfo.value.code == "SELECTION_BLOCKED_NO_REAL_EVIDENCE"


def _third_implementations():
    """CC3 extension slot (plan G4).

    If the supervisor provides the CC3 ``bagr_ued`` soft_copeland
    source (or its SHA), register a callable here with the same
    (signals, k, seed, policy) -> (scores, selected, hash) surface and
    the parity loop below folds it into the SAME fixture. Empty this
    round: CC3 source is not present in this worktree.
    """
    return ()


class TestCC3ExtensionSlot:
    def test_slot_exists_and_is_empty_this_round(self):
        assert _third_implementations() == ()

    def test_parity_loop_mechanism(self):
        # same fixture, N implementations: with zero registered third
        # implementations the loop body must not run (vacuous truth is
        # asserted explicitly so a future registration is exercised).
        ran = 0
        for impl in _third_implementations():  # pragma: no branch
            ran += 1
            assert callable(impl)
        assert ran == 0
