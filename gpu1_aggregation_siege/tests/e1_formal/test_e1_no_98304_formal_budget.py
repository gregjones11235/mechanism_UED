"""CC2-Director tests: 98304 is NOT a formal budget.

The frozen DiCode resolved config (conf/training/default.yaml) defines
the ONLY formal timeline (total_timesteps = 2_005_401_600). 98304 may
appear ONLY in checkpoint paths / checkpoint steps / Student candidate
identity (the pinned PERSISTENT_RMT16_ORIGINAL_VTRACE_98304 id). The
98304-as-budget literals must be gone from the production sources.
"""
import io
import os
import sys
import tokenize

from dicode.teachers.e1_formal import budget_semantics as BS

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)


def _executable_source(path):
    """The module source with ALL docstrings (STRING tokens) and
    comments stripped — docstrings/comments may legitimately document
    the 98304 removal as history; executable code must not."""
    with open(path, "r", encoding="utf-8") as handle:
        source = handle.read()
    kept = []
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type in (tokenize.STRING, tokenize.COMMENT):
            continue
        kept.append(tok.string)
    return "".join(kept)


def _source_files():
    return [
        os.path.join(
            REPO_ROOT, "scripts", "e1_production_runtime.py"
        ),
        os.path.join(
            REPO_ROOT, "scripts", "run_e1_longrun.py"
        ),
        os.path.join(
            REPO_ROOT,
            "src",
            "dicode",
            "teachers",
            "e1_formal",
            "budget_semantics.py",
        ),
    ]


class TestNo98304FormalBudget:
    def test_dicode_timeline_is_the_only_formal_horizon(self):
        # conf/training/default.yaml is the frozen resolved config
        import e1_production_runtime as RT

        total = RT.resolve_dicode_total_timesteps(RT.SIEGE_ROOT)
        assert total == 2_005_401_600

    def test_no_longrun_total_env_steps_constant(self):
        source = open(
            os.path.join(
                REPO_ROOT, "scripts", "e1_production_runtime.py"
            ),
            "r",
            encoding="utf-8",
        ).read()
        assert "LONGRUN_TOTAL_ENV_STEPS" not in source

    def test_no_98304_as_a_formal_budget_literal(self):
        # no 98304-as-budget literal in EXECUTABLE code (docstrings may
        # document the removal as history); the pinned Student identity
        # lives in the config, not these files
        for path in _source_files():
            code = _executable_source(path)
            assert "98304" not in code, (
                f"{path} executable code still carries a 98304 literal"
            )

    def test_budget_semantics_built_on_timesteps_not_98304(self):
        # the budget fields are timesteps on the DiCode timeline
        assert "total_timesteps" in BS.BUDGET_FIELD_NAMES
        assert "final_total_timesteps" in BS.BUDGET_FIELD_NAMES
        code = _executable_source(
            os.path.join(
                REPO_ROOT,
                "src",
                "dicode",
                "teachers",
                "e1_formal",
                "budget_semantics.py",
            )
        )
        assert "98304" not in code
