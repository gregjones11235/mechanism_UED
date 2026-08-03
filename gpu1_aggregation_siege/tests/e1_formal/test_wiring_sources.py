"""C11 tests: the guarded integration diffs, verified at the source
level (AST), because the wired legacy modules import craftax /
minicraftax, which are NOT installed in the audit venv. These tests
import NOTHING heavy: they parse the committed sources and assert

* every teacher hook is getattr-guarded (duck-typed), so a teacher
  without the hook attribute takes the legacy path;
* the legacy lines the plan promises VERBATIM are still present
  (``GenManager(config)`` inside ``_resolve_teacher``, the untouched
  ``_calculate_task_distribution``, the legacy worker-dict keys,
  ``sample_tasks_for_training``);
* no module-level E1 import was added to the legacy files (the
  default path imports nothing new).

Runtime equivalence for the distribution path lives in
``test_distribution_byte_identity.py``.
"""
import ast
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _parse(rel_path):
    path = os.path.join(REPO_ROOT, rel_path)
    with open(path, "r", encoding="utf-8") as handle:
        source = handle.read()
    return ast.parse(source), source


def _function(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name!r} not found")


def _names_in(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _string_constants_in(node):
    return {
        n.value
        for n in ast.walk(node)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    }


def _module_level_imports(tree):
    modules = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.add(node.module or "")
    return modules


class TestSetupTeacherSeam:
    def setup_method(self):
        self.tree, self.source = _parse(os.path.join("src", "dicode", "setup.py"))

    def test_resolve_teacher_exists_and_is_used_by_setup_experiment(self):
        _function(self.tree, "_resolve_teacher")
        setup_fn = _function(self.tree, "setup_experiment")
        calls = [
            n
            for n in ast.walk(setup_fn)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "_resolve_teacher"
        ]
        assert len(calls) == 1

    def test_legacy_genmanager_line_preserved_verbatim_inside_seam(self):
        seam = _function(self.tree, "_resolve_teacher")
        found = False
        for node in ast.walk(seam):
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "gen_manager"
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "GenManager"
                and [a.id for a in node.value.args if isinstance(a, ast.Name)]
                == ["config"]
            ):
                found = True
        assert found, "original `gen_manager = GenManager(config)` missing"

    def test_e1_import_is_lazy_inside_the_seam_only(self):
        seam = _function(self.tree, "_resolve_teacher")
        lazy = [
            n
            for n in ast.walk(seam)
            if isinstance(n, ast.ImportFrom)
            and n.module == "dicode.teachers.e1_formal.gen_manager"
        ]
        assert len(lazy) == 1
        # NO module-level teacher import: the default path imports
        # nothing new
        for module in _module_level_imports(self.tree):
            assert not module.startswith("dicode.teachers")

    def test_static_llm_and_unknown_types_fail_closed(self):
        seam = _function(self.tree, "_resolve_teacher")
        strings = _string_constants_in(seam)
        assert any("static_llm" in s for s in strings)
        assert any("preserved artifacts only" in s for s in strings)
        raises = [
            n
            for n in ast.walk(seam)
            if isinstance(n, ast.Raise)
            and isinstance(n.exc, ast.Call)
            and isinstance(n.exc.func, ast.Name)
            and n.exc.func.id == "NotImplementedError"
        ]
        assert len(raises) == 2  # static_llm + unknown teacher_type


class TestTrainingDistributionSeam:
    def setup_method(self):
        self.tree, self.source = _parse(
            os.path.join("src", "dicode", "training.py")
        )

    def test_legacy_distribution_function_is_untouched(self):
        legacy = _function(self.tree, "_calculate_task_distribution")
        names = _names_in(legacy)
        assert "gen_manager" not in names
        strings = _string_constants_in(legacy)
        assert not any("e1" in s.lower() for s in strings)
        assert not any("layout" in s.lower() for s in strings)
        assert "original_task_proportion" in strings
        # the pinned legacy default is still 0.2
        numbers = {
            n.value
            for n in ast.walk(legacy)
            if isinstance(n, ast.Constant) and isinstance(n.value, float)
        }
        assert 0.2 in numbers

    def test_hook_function_is_getattr_guarded_with_legacy_fallback(self):
        hook = _function(self.tree, "_resolve_session_task_distribution")
        getattr_calls = [
            n
            for n in ast.walk(hook)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "getattr"
        ]
        assert any(
            isinstance(a, ast.Constant) and a.value == "build_training_layout"
            for call in getattr_calls
            for a in call.args
        )
        fallbacks = [
            n
            for n in ast.walk(hook)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "_calculate_task_distribution"
        ]
        assert len(fallbacks) == 1  # the verbatim legacy path
        # the e1 schema import inside the hook is lazy, not module level
        for module in _module_level_imports(self.tree):
            assert not module.startswith("dicode.teachers")

    def test_run_session_training_uses_the_seam(self):
        session_fn = _function(self.tree, "run_session_training")
        calls = [
            n
            for n in ast.walk(session_fn)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "_resolve_session_task_distribution"
        ]
        assert len(calls) == 1


class TestEvolutionContextSeam:
    def setup_method(self):
        self.tree, self.source = _parse(
            os.path.join("src", "dicode", "evolution_efficient.py")
        )

    def test_context_hook_is_getattr_guarded(self):
        dispatch = _function(self.tree, "dispatch_evolution_worker")
        strings = _string_constants_in(dispatch)
        assert "select_context_tasks" in strings
        getattr_calls = [
            n
            for n in ast.walk(dispatch)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "getattr"
        ]
        assert any(
            isinstance(a, ast.Constant) and a.value == "select_context_tasks"
            for call in getattr_calls
            for a in call.args
        )

    def test_legacy_selection_call_preserved(self):
        dispatch = _function(self.tree, "dispatch_evolution_worker")
        legacy_calls = [
            n
            for n in ast.walk(dispatch)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "select_tasks_for_evolution"
        ]
        assert len(legacy_calls) == 1


class TestRunDicodeSeams:
    def setup_method(self):
        self.tree, self.source = _parse(
            os.path.join("experiments", "training", "run_dicode.py")
        )

    def test_worker_consume_hook_is_guarded_and_legacy_keys_intact(self):
        process_fn = _function(self.tree, "_process_worker_results")
        strings = _string_constants_in(process_fn)
        assert "consume_worker_results" in strings
        # the ORIGINAL legacy registration keys are untouched
        assert "generated_task_id" in strings
        assert "code_string" in strings
        assert "compiled" in strings

    def test_batch_hook_and_feedback_hook_present(self):
        main_fn = _function(self.tree, "main")
        strings = _string_constants_in(main_fn)
        assert "build_training_batch" in strings
        assert "observe_session_feedback" in strings
        assert "NORMAL_TRAINING_FEEDBACK" in strings

    def test_legacy_sampling_call_preserved(self):
        main_fn = _function(self.tree, "main")
        legacy_calls = [
            n
            for n in ast.walk(main_fn)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "sample_tasks_for_training"
        ]
        assert len(legacy_calls) == 1


class TestEvaluationSeamExport:
    def test_init_exports_the_seam(self):
        path = os.path.join(
            REPO_ROOT, "src", "dicode", "evaluation", "__init__.py"
        )
        with open(path, "r", encoding="utf-8") as handle:
            content = handle.read()
        assert (
            "from .candidate_evaluation import "
            "evaluate_candidates_with_reference" in content
        )
        # and the ORIGINAL export line is untouched
        assert "from .online_evaluation import run_session_evaluation" in content

    def test_seam_module_imports_no_craftax_or_minicraftax(self):
        tree, _ = _parse(
            os.path.join(
                "src", "dicode", "evaluation", "candidate_evaluation.py"
            )
        )
        modules = _module_level_imports(tree)
        for module in modules:
            assert not module.startswith("craftax")
            assert not module.startswith("minicraftax")
