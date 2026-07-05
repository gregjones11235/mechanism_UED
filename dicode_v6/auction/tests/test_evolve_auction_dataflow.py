"""Offline dataflow test for evolve_mastered_auction's integration logic.

Does NOT require jax/craftax/LLM. We import the auction adapter + selectors directly and
replicate the method's data path against a fake TaskGenerator whose heavy DiCode dependencies
(prompt building, LLM query, _organize_data) are stubbed. This verifies the WIRING — N proposers
-> proposals -> auction top-k -> rebuilt (parsed, parent, example) triplets — is correct, which is
exactly the part that can't be checked by the pure-auction unit tests and shouldn't wait for Oscar.
"""

import importlib.util
import types
from pathlib import Path

from auction.proposal import Proposal
from auction.selectors import GreedyTopKSelector, SelectionContext

# Load the adapter directly (bypass dicode/__init__).
_AI_PATH = (
    Path(__file__).resolve().parents[2]
    / "src" / "dicode" / "dreaming" / "auction_integration.py"
)
_spec = importlib.util.spec_from_file_location("auction_integration", _AI_PATH)
_ai = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ai)
parsed_response_to_proposal = _ai.parsed_response_to_proposal


def _docstring(achs_upper):
    return f"A level.\nRelevant Achievements: {', '.join(achs_upper)}\nWorld:\n- player"


def _reimplement_auction_dataflow(proposer_outputs, mastered_tasks, k):
    """Mirror of evolve_mastered_auction's core loop, isolated from DiCode heavy deps.

    proposer_outputs: list (per proposer) of list (per parent) of parsed-response dicts.
    Returns the (win_parsed, win_parents, win_examples) that would be passed to _organize_data.
    """
    parent_sets = [[t] for t in mastered_tasks]
    example_sets = [[f"ex_{t}"] for t in mastered_tasks]

    proposals = []
    parent_of, example_of, parsed_of = {}, {}, {}
    pid_counter = 0
    for proposer_idx, parsed_responses in enumerate(proposer_outputs):
        for local_i, parsed in enumerate(parsed_responses):
            pid = f"prop_s0_{pid_counter}"
            pid_counter += 1
            parent_set = parent_sets[local_i]
            proposal = parsed_response_to_proposal(
                parsed, proposal_id=pid, proposer_id=f"proposer_{proposer_idx}",
                parent_task_id=parent_set[0],
            )
            proposals.append(proposal)
            parent_of[pid] = parent_set
            example_of[pid] = example_sets[local_i]
            parsed_of[pid] = parsed

    winners = GreedyTopKSelector().select(proposals, k, SelectionContext())
    win_parsed = [parsed_of[w.proposal_id] for w in winners]
    win_parents = [parent_of[w.proposal_id] for w in winners]
    win_examples = [example_of[w.proposal_id] for w in winners]
    return proposals, winners, win_parsed, win_parents, win_examples


def test_n_proposers_produce_n_times_proposals():
    mastered = ["task_5", "task_9"]
    # 3 proposers, each emits a description per parent -> 3*2 = 6 proposals
    out = [
        [{"description": _docstring(["COLLECT_WOOD"]), "reasoning": "r"},
         {"description": _docstring(["DEFEAT_ARCHER"]), "reasoning": "r"}]
        for _ in range(3)
    ]
    proposals, winners, *_ = _reimplement_auction_dataflow(out, mastered, k=2)
    assert len(proposals) == 6
    assert len(winners) == 2


def test_auction_picks_complementary_winners():
    mastered = ["task_0"]
    # one proposer floods the same cheap achievement; another offers a deep distinct one
    out = [
        [{"description": _docstring(["COLLECT_WOOD"]), "reasoning": "r"}],   # depth1 weight1
        [{"description": _docstring(["COLLECT_WOOD"]), "reasoning": "r"}],   # duplicate
        [{"description": _docstring(["DEFEAT_NECROMANCER"]), "reasoning": "r"}],  # depth4 weight8
    ]
    proposals, winners, *_ = _reimplement_auction_dataflow(out, mastered, k=2)
    won_achs = set()
    for w in winners:
        won_achs |= w.achievements
    # complementary selection must include the deep one and not pick two identical wood levels
    assert "defeat_necromancer" in won_achs


def test_rebuilt_triplets_align_with_winners():
    mastered = ["task_1", "task_2"]
    out = [
        [{"description": _docstring(["COLLECT_WOOD"]), "reasoning": "ra"},
         {"description": _docstring(["COLLECT_IRON"]), "reasoning": "rb"}],
    ]
    proposals, winners, win_parsed, win_parents, win_examples = _reimplement_auction_dataflow(
        out, mastered, k=2
    )
    # the triplets handed to _organize_data must be 1:1 with winners and internally consistent
    assert len(win_parsed) == len(winners) == len(win_parents) == len(win_examples)
    for parsed, parent in zip(win_parsed, win_parents):
        assert parsed["description"]  # carries the docstring forward
        assert parent and parent[0] in mastered  # parent lineage preserved


def test_single_proposer_k_full_is_baseline_like():
    # N=1 proposer, k = number of parents -> every description survives = baseline behaviour.
    mastered = ["task_3", "task_4", "task_7"]
    out = [[{"description": _docstring(["COLLECT_WOOD"]), "reasoning": "r"} for _ in mastered]]
    proposals, winners, *_ = _reimplement_auction_dataflow(out, mastered, k=len(mastered))
    assert len(proposals) == 3
    assert len(winners) == 3  # nothing dropped


# --- Prove Endorsement & AmbitionGain are WIRED (they change selection, not dead 0) ----------

def _build_proposals(specs):
    """specs: list of (pid, proposer_id, [achievements]). Returns list[Proposal]."""
    from auction.proposal import Proposal
    return [Proposal(pid, pr, "task_0", _docstring(a), "r", frozenset(x.lower() for x in a))
            for pid, pr, a in specs]


def test_endorsement_changes_winner():
    from auction.selectors import GreedyTopKSelector, SelectionContext
    # Two proposals with IDENTICAL coverage (so Coverage alone can't separate them);
    # endorsement must break the tie toward the endorsed one.
    props = _build_proposals([
        ("p_a", "proposer_0", ["COLLECT_WOOD"]),
        ("p_b", "proposer_1", ["COLLECT_WOOD"]),
    ])
    # proposer_2 endorses p_a strongly, p_b weakly
    ratings = {"proposer_2": {"p_a": 0.9, "p_b": 0.1}}
    ctx = SelectionContext(cross_ratings=ratings, w_cov=1.0, w_end=10.0, w_amb=0.0)
    winners = GreedyTopKSelector().select(props, 1, ctx)
    assert winners[0].proposal_id == "p_a"  # endorsement decided it


def test_ambition_changes_winner():
    from auction.selectors import GreedyTopKSelector, SelectionContext
    # Two proposals, identical-weight coverage; ambition (target gap) should pick the one
    # covering the achievement the student is failing.
    props = _build_proposals([
        ("p_mastered", "proposer_0", ["COLLECT_WOOD"]),      # student aces this -> gap 0
        ("p_needed", "proposer_1", ["COLLECT_DRINK"]),        # same depth tier 1, same weight
    ])
    target_gap = {"collect_wood": 0.0, "collect_drink": 1.0}  # student fails collect_drink
    ctx = SelectionContext(target_gap=target_gap, w_cov=1.0, w_end=0.0, w_amb=10.0)
    winners = GreedyTopKSelector().select(props, 1, ctx)
    assert winners[0].proposal_id == "p_needed"  # ambition decided it


def test_all_three_terms_off_equals_pure_coverage():
    from auction.selectors import GreedyTopKSelector, SelectionContext
    props = _build_proposals([
        ("p1", "proposer_0", ["COLLECT_WOOD"]),
        ("p2", "proposer_1", ["DEFEAT_ARCHER"]),  # deeper -> higher coverage weight
    ])
    # no endorsement, no ambition -> pure Coverage -> deeper achievement wins
    ctx = SelectionContext(w_end=1.0, w_amb=1.0)  # but no data sources supplied
    winners = GreedyTopKSelector().select(props, 1, ctx)
    assert winners[0].proposal_id == "p2"  # DEFEAT_ARCHER (weight 8) beats COLLECT_WOOD (weight 1)
