#!/usr/bin/env python3
"""Standalone unit tests for the aggregation selector module.

Tests all aggregation strategies, robust normalization, budget caps,
retention trigger, entropy computation, and edge cases.

Usage:
    cd /root/experiments/dreaming-in-code-coop
    PYTHONPATH=src:$PYTHONPATH python scripts/test_aggregation_selector.py
"""

import sys
import os
import json
import tempfile

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from dicode.mechanisms.aggregation import (
    apply_budget_caps,
    compute_curriculum_entropy,
    compute_forgetting_stats,
    compute_signal_scores,
    robust_normalize,
    sample_curriculum,
    select_tasks_with_aggregation,
)


# ==============================================================================
# Fake / mock helpers
# ==============================================================================


class FakeConfig:
    """Minimal fake OmegaConf-like object for testing."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def get(self, key, default=None):
        return self.__dict__.get(key, default)


class FakeNodeView:
    """Mimics networkx NodeView: callable with (data=bool) and subscriptable."""

    def __init__(self, nodes_data):
        self._data = nodes_data

    def __call__(self, data=False):
        if data:
            return self._data.items()
        return self._data.keys()

    def __getitem__(self, node_id):
        return self._data[node_id]

    def __contains__(self, node_id):
        return node_id in self._data


class FakeGraph:
    """Fake networkx DiGraph for testing signal computation."""

    def __init__(self):
        self.nodes_data = {}
        self.nodes = FakeNodeView(self.nodes_data)

    def add_node(self, node_id, **attrs):
        self.nodes_data[node_id] = attrs

    def has_node(self, node_id):
        return node_id in self.nodes_data


class FakeArchive:
    """Fake archive with graph and lock for testing."""

    class FakeLock:
        def __enter__(self, *args, **kwargs):
            return self

        def __exit__(self, *args, **kwargs):
            pass

    def __init__(self):
        self.graph = FakeGraph()
        self._lock = self.FakeLock()


class FakeGenManager:
    """Fake GenManager for testing select_tasks_with_aggregation."""

    def __init__(self):
        self.archive = FakeArchive()
        self.session_idx = 5


def _make_fake_active_pool(gen_manager, n_tasks=20):
    """Populate the fake archive with n active tasks."""
    for i in range(n_tasks):
        tid = f"task_{i + 1}"
        source_type = "learnable" if i < n_tasks // 2 else "mastered"
        gen_manager.archive.graph.add_node(
            tid,
            is_active=True,
            priority_score=0.1 + 0.04 * i,  # 0.1 to 0.86
            session_last_trained=max(0, i - 5),
            status=source_type if source_type == "mastered" else "B",
            type=source_type,
            performance_history=[{"sr": 0.3 + 0.03 * i}] * (i % 4 + 1),
            description=f"Task {i + 1}",
        )


# ==============================================================================
# Tests
# ==============================================================================


def test_robust_normalize_basic():
    """robust_normalize should center and scale with median/IQR."""
    values = np.array([1.0, 2.0, 3.0, 4.0, 100.0])  # 100 is outlier
    result = robust_normalize(values)
    assert len(result) == 5, f"Expected 5 values, got {len(result)}"
    # Outlier should be clipped
    assert result[-1] <= 3.0, f"Outlier should be clipped to 3.0, got {result[-1]}"
    print("  PASS: test_robust_normalize_basic")


def test_robust_normalize_no_variance():
    """robust_normalize should handle zero variance."""
    values = np.array([5.0, 5.0, 5.0])
    result = robust_normalize(values)
    assert np.allclose(result, 0.0), f"Expected zeros, got {result}"
    print("  PASS: test_robust_normalize_no_variance")


def test_robust_normalize_nan_handling():
    """robust_normalize should handle NaN and Inf."""
    values = np.array([1.0, np.nan, 3.0, np.inf, -np.inf])
    result = robust_normalize(values)
    assert not np.any(np.isnan(result)), f"Got NaN in result: {result}"
    assert not np.any(np.isinf(result)), f"Got Inf in result: {result}"
    print("  PASS: test_robust_normalize_nan_handling")


def test_compute_signal_scores():
    """compute_signal_scores should return all expected keys."""
    gen_manager = FakeGenManager()
    _make_fake_active_pool(gen_manager, n_tasks=10)
    task_ids = list(gen_manager.archive.graph.nodes_data.keys())

    signals = compute_signal_scores(task_ids, gen_manager.archive.graph)

    for key in ["progression", "retention", "novelty", "critic_penalty",
                "monopoly_penalty", "source_ids", "skill_counts"]:
        assert key in signals, f"Missing key: {key}"
        assert len(signals[key]) == 10, f"{key} length should be 10, got {len(signals[key])}"

    # Progression should match priority scores
    for i, tid in enumerate(task_ids):
        expected = gen_manager.archive.graph.nodes_data[tid]["priority_score"]
        assert abs(signals["progression"][i] - expected) < 1e-6, \
            f"Progression mismatch at {i}: {signals['progression'][i]} vs {expected}"

    print("  PASS: test_compute_signal_scores")


def test_compute_forgetting_stats_no_forgetting():
    """No forgetting when current >= best."""
    stats = compute_forgetting_stats(
        current_old_success=0.8,
        best_old_success_so_far=0.75,
    )
    assert stats["forgetting_index"] == 0.0, f"Expected 0.0, got {stats['forgetting_index']}"
    assert not stats["anti_forgetting_mode"], "Anti-forgetting should be False"
    print("  PASS: test_compute_forgetting_stats_no_forgetting")


def test_compute_forgetting_stats_with_forgetting():
    """Forgetting detected when current < best."""
    stats = compute_forgetting_stats(
        current_old_success=0.5,
        best_old_success_so_far=0.8,
    )
    assert abs(stats["forgetting_index"] - 0.3) < 1e-9, f"Expected 0.3, got {stats['forgetting_index']}"
    assert stats["anti_forgetting_mode"], "Anti-forgetting should be True"
    print("  PASS: test_compute_forgetting_stats_with_forgetting")


def test_aggregate_scores_rank_reasonable_candidates():
    """Soft Copeland should rank higher-scored tasks higher."""
    # Create tasks where task_0 is clearly best
    task_ids = ["task_1", "task_2", "task_3"]
    # Simulate the aggregation flow: create signals directly
    progression = np.array([0.9, 0.5, 0.1])
    retention = np.array([0.8, 0.5, 0.2])
    novelty = np.array([0.7, 0.5, 0.3])

    signals = {
        "progression": progression,
        "retention": retention,
        "novelty": novelty,
        "critic_penalty": np.zeros(3),
        "monopoly_penalty": np.zeros(3),
        "source_ids": np.array(["seed", "seed", "seed"]),
        "skill_counts": np.ones(3),
    }

    weights = {
        "w_progression": 0.34,
        "w_retention": 0.33,
        "w_novelty": 0.33,
        "w_critic": 0.01,
        "w_monopoly": 0.01,
    }

    from dicode.mechanisms.aggregation import _aggregate_soft_copeland
    scores = _aggregate_soft_copeland(signals, weights, temperature=1.0)

    # Best task should have highest score
    assert scores[0] > scores[1], f"task_1 should beat task_2: {scores}"
    assert scores[1] > scores[2], f"task_2 should beat task_3: {scores}"
    print("  PASS: test_aggregate_scores_rank_reasonable_candidates")


def test_retention_trigger_activates():
    """Retention weight should increase when forgetting_index > trigger."""
    stats = compute_forgetting_stats(0.4, 0.8)  # forgetting_index=0.4

    # Simulate trigger logic
    retention_trigger = 0.15
    weights = {
        "w_progression": 0.34,
        "w_retention": 0.33,
        "w_novelty": 0.33,
    }

    anti_forgetting = False
    if stats["forgetting_index"] > retention_trigger:
        anti_forgetting = True
        boost_factor = 1.0 + min(2.0, stats["forgetting_index"] / retention_trigger)
        weights["w_retention"] *= boost_factor
        weights["w_progression"] *= 0.7
        weights["w_novelty"] *= 0.7

    assert anti_forgetting, "Anti-forgetting should be triggered"
    assert weights["w_retention"] > 0.33, f"Retention weight should increase: {weights['w_retention']}"
    assert weights["w_progression"] < 0.34, f"Progression weight should decrease: {weights['w_progression']}"
    print("  PASS: test_retention_trigger_activates")


def test_budget_caps_reduce_monopoly():
    """Budget caps should penalize tasks from over-represented sources."""
    n = 10
    task_ids = [f"task_{i}" for i in range(n)]
    scores = np.ones(n, dtype=np.float64)
    # 8 tasks from same source, 2 from different sources
    source_ids = np.array(["dominant"] * 8 + ["minor_1"] + ["minor_2"])

    result, budget_info = apply_budget_caps(
        scores, task_ids, source_ids, max_source_share=0.5
    )

    # Only up to ceil(10 * 0.5) = 5 tasks from "dominant" should keep full score
    # The remaining 3 dominant tasks should be penalized (score halved)
    dominant_indices = np.where(source_ids == "dominant")[0]
    penalized = sum(1 for i in dominant_indices if result[i] < 1.0)

    assert penalized >= 2, f"Expected at least 2 penalized dominant tasks, got {penalized}"
    assert budget_info["source_caps_applied"] > 0, "Source caps should have been applied"
    print("  PASS: test_budget_caps_reduce_monopoly")


def test_sample_curriculum_returns_correct_k():
    """sample_curriculum should return exactly k tasks."""
    n = 20
    task_ids = [f"task_{i}" for i in range(n)]
    scores = np.random.rand(n).astype(np.float64)

    for k in [0, 1, 5, 10, 20]:
        selected, probs = sample_curriculum(scores, task_ids, k, temperature=1.0)
        expected = min(k, n)
        assert len(selected) == expected, f"k={k}: expected {expected}, got {len(selected)}"
        if k > 0:
            assert len(probs) == n, f"probs should have {n} elements, got {len(probs)}"
            assert abs(probs.sum() - 1.0) < 1e-6, f"probs should sum to 1.0, got {probs.sum()}"

    print("  PASS: test_sample_curriculum_returns_correct_k")


def test_empty_candidate_pool():
    """Empty candidate pool should not crash."""
    gen_manager = FakeGenManager()

    config = FakeConfig(
        aggregation=FakeConfig(
            enabled=True,
            mode="robust_weighted",
            w_progression=0.34,
            w_retention=0.33,
            w_novelty=0.33,
            w_critic=0.01,
            w_monopoly=0.01,
            retention_trigger=0.15,
            temperature=1.0,
            entropy_regularization=0.0,
            max_source_share=0.5,
            max_signal_share=0.5,
        ),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        result = select_tasks_with_aggregation(
            gen_manager=gen_manager,
            config=config,
            n=10,
            log_dir=tmpdir,
        )
        assert result == [], f"Expected empty list, got {result}"

    print("  PASS: test_empty_candidate_pool")


def test_missing_metrics_no_crash():
    """Missing metrics in node data should not crash."""
    gen_manager = FakeGenManager()
    # Add a task with no priority_score
    gen_manager.archive.graph.add_node(
        "bare_task",
        is_active=True,
        # No priority_score, no session_last_trained
    )

    signals = compute_signal_scores(["bare_task"], gen_manager.archive.graph)
    assert len(signals["progression"]) == 1
    assert signals["progression"][0] == 0.0, "Missing priority_score should default to 0"
    print("  PASS: test_missing_metrics_no_crash")


def test_entropy_computation():
    """Entropy should be max for uniform, min for peaked distribution."""
    # Uniform scores -> high entropy
    uniform = np.ones(10, dtype=np.float64)
    entropy_uniform = compute_curriculum_entropy(uniform)
    assert entropy_uniform > 0, f"Uniform entropy should be > 0, got {entropy_uniform}"

    # One dominant score -> low entropy
    peaked = np.array([100.0] + [0.0] * 9, dtype=np.float64)
    entropy_peaked = compute_curriculum_entropy(peaked, temperature=1.0)
    assert entropy_peaked < entropy_uniform, \
        f"Peaked entropy ({entropy_peaked}) should be < uniform ({entropy_uniform})"

    print("  PASS: test_entropy_computation")


def test_all_aggregation_modes_no_crash():
    """All six aggregation modes should run without crashing."""
    gen_manager = FakeGenManager()
    _make_fake_active_pool(gen_manager, n_tasks=15)

    modes = [
        "raw_weighted",
        "robust_weighted",
        "soft_copeland",
        "budgeted_soft_copeland",
        "budgeted_retention_trigger",
        "entropy_regularized",
    ]

    for mode in modes:
        config = FakeConfig(
            aggregation=FakeConfig(
                enabled=True,
                mode=mode,
                w_progression=0.34,
                w_retention=0.33,
                w_novelty=0.33,
                w_critic=0.01,
                w_monopoly=0.01,
                retention_trigger=0.15,
                temperature=1.0,
                entropy_regularization=0.1,
                max_source_share=0.5,
                max_signal_share=0.5,
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result = select_tasks_with_aggregation(
                gen_manager=gen_manager,
                config=config,
                n=5,
                log_dir=tmpdir,
            )
            # With 15 active tasks, we should get at least 1 result
            assert len(result) <= 5, f"Mode '{mode}': got {len(result)} tasks, expected <= 5"
            assert len(result) > 0, f"Mode '{mode}': got no tasks"

    print("  PASS: test_all_aggregation_modes_no_crash")


def test_diagnostics_jsonl_written():
    """Diagnostics JSONL should be written when log_dir is provided."""
    gen_manager = FakeGenManager()
    _make_fake_active_pool(gen_manager, n_tasks=10)

    config = FakeConfig(
        aggregation=FakeConfig(
            enabled=True,
            mode="robust_weighted",
            w_progression=0.34,
            w_retention=0.33,
            w_novelty=0.33,
            w_critic=0.01,
            w_monopoly=0.01,
            retention_trigger=0.15,
            temperature=1.0,
            entropy_regularization=0.0,
            max_source_share=0.5,
            max_signal_share=0.5,
        ),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        result = select_tasks_with_aggregation(
            gen_manager=gen_manager,
            config=config,
            n=5,
            log_dir=tmpdir,
        )

        jsonl_path = os.path.join(tmpdir, "aggregation_selector.jsonl")
        assert os.path.exists(jsonl_path), f"JSONL not found at {jsonl_path}"

        with open(jsonl_path) as f:
            lines = f.readlines()
            assert len(lines) >= 1, "JSONL should have at least 1 line"

            row = json.loads(lines[0])
            assert "timestamp" in row
            assert "aggregation_mode" in row
            assert row["aggregation_mode"] == "robust_weighted"
            assert "selected_task_ids" in row
            assert len(row["selected_task_ids"]) <= 5
            assert "curriculum_entropy" in row
            assert "forgetting_index" in row

    print("  PASS: test_diagnostics_jsonl_written")


def test_entropy_regularized_sampling():
    """Entropy-regularized sampling should produce more uniform distribution."""
    # Create skewed scores
    n = 50
    task_ids = [f"task_{i}" for i in range(n)]
    scores = np.array([10.0] + [0.0] * (n - 1), dtype=np.float64)

    # Without regularization
    _, probs_no_reg = sample_curriculum(
        scores, task_ids, k=10, temperature=1.0, entropy_regularization=0.0
    )

    # With regularization
    _, probs_with_reg = sample_curriculum(
        scores, task_ids, k=10, temperature=1.0, entropy_regularization=0.3
    )

    # Regularized probs should be less peaked (lower max probability)
    assert probs_with_reg.max() < probs_no_reg.max(), \
        f"Regularized max prob ({probs_with_reg.max()}) should be < no-reg ({probs_no_reg.max()})"

    print("  PASS: test_entropy_regularized_sampling")


# ==============================================================================
# Main
# ==============================================================================

def main():
    """Run all tests."""
    print("=" * 70)
    print("AGGREGATION SELECTOR UNIT TESTS")
    print("=" * 70)

    tests = [
        test_robust_normalize_basic,
        test_robust_normalize_no_variance,
        test_robust_normalize_nan_handling,
        test_compute_signal_scores,
        test_compute_forgetting_stats_no_forgetting,
        test_compute_forgetting_stats_with_forgetting,
        test_aggregate_scores_rank_reasonable_candidates,
        test_retention_trigger_activates,
        test_budget_caps_reduce_monopoly,
        test_sample_curriculum_returns_correct_k,
        test_empty_candidate_pool,
        test_missing_metrics_no_crash,
        test_entropy_computation,
        test_all_aggregation_modes_no_crash,
        test_diagnostics_jsonl_written,
        test_entropy_regularized_sampling,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  FAIL: {test_fn.__name__}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 70}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print(f"{'=' * 70}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
