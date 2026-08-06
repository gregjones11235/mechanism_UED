# -*- coding: utf-8 -*-
"""TEST_ONLY / SYNTHETIC fixtures.  NOT_REAL_EXECUTION.

BUG-E3-10 regression: ``materialize_and_register`` must register the 15
curriculum tasks with a LOADABLE env code (the FULL module source), so DiCode's
``load_tasks_from_env_codes`` can exec it and define the ``Env`` class.
BUG-E3-05: the registration API is ``TaskArchive.record_new_task`` +
node.code (never a nonexistent ``register_task``).
"""

import types

import pytest

from dicode.simulator_frontier.canonical_dicode_runtime import (
    _env_module_source,
    callable_source_sha256,
    compile_canonical_15_plus_1,
    materialize_and_register,
    mint_frontier_distribution_environment_adapter,
)
from dicode.simulator_frontier.errors import InvalidEvidenceError
from dicode.simulator_frontier.frontier_distributions import (
    FrontierDistribution,
)

SYNTHETIC_FIXTURE_NOT_SCIENTIFIC_CONTENT = True


def _distributions():
    dists = []
    for slot in (f"D{i:02d}" for i in range(12)):
        dists.append(FrontierDistribution(
            distribution_id=f"plan-001::{slot}",
            bucket=("bucket",),
            eligible_states=("state-1",),
            start_state_weights={"state-1": 1.0},
            taskparam_ranges={"p": [0.0, 1.0]},
            seed_distribution={"s": [0, 1]},
            stochasticity_range={"e": [0.0, 0.1]},
            memory_mode="SAVED_POLICY_MEMORY",
            goal_family="FRONTIER:LEARNABLE_FRONTIER",
            evidence_hash="e" * 64,
            retention_constraint="anchor_ratio>=0.20",
        ))
    return dists


def _plan():
    return compile_canonical_15_plus_1(
        plan_id="plan-001",
        distributions=_distributions(),
        non_target_anchor_ids=("anchor_a", "anchor_b", "anchor_c"),
        original_task_anchor_id="ORIGINAL_TASK_ANCHOR",
        original_task_id="original_craftax",
        env_adapter_id="adapter-001",
        memory_bindings={
            slot: {"memory_mode": "SAVED_POLICY_MEMORY"}
            for slot in [d.distribution_id for d in _distributions()]
            + ["anchor_a", "anchor_b", "anchor_c"]
        },
        anchor_memory_binding={"memory_mode": "SAVED_POLICY_MEMORY"},
    )


def _adapter():
    from minicraftax.tasks.seed_tasks.collecting import Env
    return mint_frontier_distribution_environment_adapter(
        adapter_id="adapter-001",
        env_entrypoint="minicraftax.tasks.seed_tasks.collecting:Env",
        env_implementation_hash=callable_source_sha256("env", Env),
        taskparam_apply_entrypoint=(
            "dicode.simulator_frontier._dicode_test_runtime:"
            "synthetic_taskparam_apply"),
        taskparam_implementation_hash=callable_source_sha256(
            "taskparam",
            __import__(
                "dicode.simulator_frontier._dicode_test_runtime",
                fromlist=["synthetic_taskparam_apply"])
            .synthetic_taskparam_apply),
    )


class TestEnvModuleSource:
    def test_module_source_is_loadable(self):
        code = _env_module_source("minicraftax.tasks.seed_tasks.collecting:Env")
        module = types.ModuleType("collecting_test")
        exec(code, module.__dict__)
        assert hasattr(module, "Env")
        assert "class Env" in code


class _FakeArchive:
    def __init__(self):
        import networkx as nx
        self.graph = nx.DiGraph()

    def record_new_task(self, child_task, parent_tasks, description,
                        session_id):
        if not self.graph.has_node(child_task):
            self.graph.add_node(child_task, code="", is_active=False)

    def get_task_codes(self, task_paths):
        return {
            task: str(self.graph.nodes[task].get("code", ""))
            for task in task_paths if self.graph.has_node(task)
        }


class TestMaterializeAndRegister:
    def test_registers_15_loadable_tasks(self):
        archive = _FakeArchive()
        plan = _plan()
        registered = materialize_and_register(
            _adapter(), plan, archive, session_idx=0)
        assert len(registered) == 15
        assert set(registered) == set(plan.curriculum_slots)
        from dicode.task_utils import load_tasks_from_env_codes
        classes, ids = load_tasks_from_env_codes(archive, registered)
        assert len(ids) == 15
        assert len(classes) == 15
        assert all(isinstance(c, type) and c.__name__ == "Env" for c in classes)

    def test_registration_api_is_record_new_task(self):
        archive = _FakeArchive()
        assert not hasattr(archive, "register_task")
        plan = _plan()
        materialize_and_register(_adapter(), plan, archive, session_idx=0)
        for slot in plan.curriculum_slots:
            assert archive.graph.has_node(slot)
            assert "code" in archive.graph.nodes[slot]
            assert archive.graph.nodes[slot]["code"]

    def test_rejects_non_plan(self):
        archive = _FakeArchive()
        with pytest.raises(InvalidEvidenceError):
            materialize_and_register(_adapter(), {"not": "plan"}, archive)
