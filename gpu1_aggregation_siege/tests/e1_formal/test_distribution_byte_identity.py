"""C11 byte-identity proof for the legacy task-distribution path.

Three layers of evidence:

1. ``layout.legacy_distribution_mirror`` is bit-identical to the
   legacy formula computed with the SAME python-float expression
   structure (n = 0..32, several original proportions);
2. the mirror matches the jnp expression structure of the untouched
   ``training._calculate_task_distribution`` within float32 rounding
   (jax runs CPU-only here; craftax is absent, so the legacy module
   itself cannot be imported in the audit venv);
3. guarded by ``pytest.importorskip`` (plan: craftax-dependent tests
   skip when craftax is absent), the REAL legacy functions are
   exercised in a full environment:
   ``_calculate_task_distribution`` == mirror, and
   ``_resolve_session_task_distribution`` with a hookless teacher ==
   ``_calculate_task_distribution`` for every n (fake-GenManager
   no-hook equivalence), plus the E1 override semantics (pinned
   weights applied exactly when they cover the session and sum to 1,
   legacy distribution otherwise — never renormalized).
"""
import numpy as np
import pytest

jax_numpy = pytest.importorskip("jax.numpy")

from dicode.teachers.e1_formal import layout as L  # noqa: E402


def _legacy_formula_python(num_curriculum_tasks, original_proportion):
    """The legacy expression structure in pure python (float64)."""
    if num_curriculum_tasks > 0:
        other = (1.0 - original_proportion) / num_curriculum_tasks
        proportions = [other] * num_curriculum_tasks + [original_proportion]
    else:
        proportions = [1.0]
    total = sum(proportions)
    return [p / total for p in proportions]


def _legacy_formula_jnp(num_curriculum_tasks, original_proportion):
    """Mirror of the UNTOUCHED jnp expression in training.py."""
    if num_curriculum_tasks > 0:
        other = (1.0 - original_proportion) / num_curriculum_tasks
        proportions = jax_numpy.concatenate([
            jax_numpy.full(num_curriculum_tasks, other),
            jax_numpy.array([original_proportion]),
        ])
    else:
        proportions = jax_numpy.array([1.0])
    return proportions / jax_numpy.sum(proportions)


class TestMirrorEquivalence:
    @pytest.mark.parametrize("n", list(range(0, 33)))
    @pytest.mark.parametrize("original_proportion", [0.2, 0.0, 0.35, 0.5])
    def test_mirror_bit_identical_to_python_formula(self, n, original_proportion):
        mirror = L.legacy_distribution_mirror(n, original_proportion)
        replica = _legacy_formula_python(n, original_proportion)
        assert mirror == replica  # exact float64 bit equality

    @pytest.mark.parametrize("n", list(range(0, 33)))
    def test_mirror_matches_jnp_formula_within_float32_rounding(self, n):
        mirror = L.legacy_distribution_mirror(n, 0.2)
        legacy = np.asarray(_legacy_formula_jnp(n, 0.2), dtype=np.float64)
        assert legacy.shape == (len(mirror),)
        assert np.allclose(legacy, np.array(mirror), rtol=0.0, atol=1e-6)
        # every entry strictly positive and normalized in both
        assert np.all(legacy > 0.0)
        assert abs(float(legacy.sum()) - 1.0) <= 1e-6


class _HooklessTeacher:
    """Fake legacy-style GenManager: NO build_training_layout."""


class _ConfigStub:
    def __init__(self, original_proportion=0.2):
        self.dicode_manager = {
            "original_task_proportion": original_proportion
        }


class TestRealLegacyFunctionsFullEnvironment:
    """Runtime equivalence against the REAL training.py functions.

    Skipped in the audit venv (craftax/minicraftax absent => the
    legacy module cannot import). In the full environment these prove
    the default path is numerically unchanged.
    """

    def _training(self):
        return pytest.importorskip("dicode.training")

    def test_real_calculate_matches_mirror_n0_32(self):
        training = self._training()
        config = _ConfigStub()
        for n in range(0, 33):
            real = np.asarray(
                training._calculate_task_distribution(config, n),
                dtype=np.float64,
            )
            mirror = np.array(L.legacy_distribution_mirror(n, 0.2))
            assert np.allclose(real, mirror, rtol=0.0, atol=1e-6), n

    def test_no_hook_teacher_is_byte_identical_to_legacy(self):
        training = self._training()
        config = _ConfigStub()
        teacher = _HooklessTeacher()
        for n in range(0, 33):
            legacy = np.asarray(
                training._calculate_task_distribution(config, n),
                dtype=np.float64,
            )
            via_seam = np.asarray(
                training._resolve_session_task_distribution(
                    config,
                    teacher,
                    n,
                    [f"task_{i}" for i in range(n)] + ["original_craftax"],
                ),
                dtype=np.float64,
            )
            assert np.array_equal(legacy, via_seam), n

    def test_e1_override_applies_pinned_weights_exactly(self):
        training = self._training()
        from test_gen_manager_duck import _manager  # real E1 teacher

        teacher = _manager()
        config = _ConfigStub()
        dynamic = [f"dyn_{i:02d}" for i in range(12)]
        session_ids = dynamic + list(L.ANCHOR_TASK_IDS)
        dist = np.asarray(
            training._resolve_session_task_distribution(
                config, teacher, len(session_ids) - 1, session_ids
            ),
            dtype=np.float64,
        )
        assert dist.shape == (16,)
        # the seam returns jnp arrays (weak float32 like the legacy
        # path); compare in identical float32 rounding space. The
        # exact-rational pin proof lives in test_layout_distribution.
        expected = np.array(
            [float(1 / 16)] * 12 + [float(1 / 20)] * 3 + [float(1 / 10)],
            dtype=np.float32,
        ).astype(np.float64)
        assert np.array_equal(dist, expected)

    def test_duplicate_original_never_dilutes_pinned_weights(self):
        training = self._training()
        from test_gen_manager_duck import _manager

        teacher = _manager()
        config = _ConfigStub()
        dynamic = [f"dyn_{i:02d}" for i in range(12)]
        # run_session_training appends original_craftax unconditionally;
        # an E1 batch already carries it, so the session has it twice.
        # The pinned layout must NOT be renormalized over the duplicate
        # — the legacy distribution applies unchanged instead.
        session_ids = dynamic + list(L.ANCHOR_TASK_IDS) + ["original_craftax"]
        dist = np.asarray(
            training._resolve_session_task_distribution(
                config, teacher, len(session_ids) - 1, session_ids
            ),
            dtype=np.float64,
        )
        legacy = np.asarray(
            training._calculate_task_distribution(
                config, len(session_ids) - 1
            ),
            dtype=np.float64,
        )
        assert np.array_equal(dist, legacy)

    def test_fail_closed_layout_refusal_falls_back_to_legacy(self):
        training = self._training()
        config = _ConfigStub()

        class _BrokenLayoutTeacher:
            anchor_task_ids = L.ANCHOR_TASK_IDS

            def build_training_layout(self, ids):
                from dicode.teachers.e1_formal.layout import (
                    LayoutError,
                    LAYOUT_BAD_DYNAMIC_SET,
                )

                raise LayoutError(LAYOUT_BAD_DYNAMIC_SET, "refused")

        dist = np.asarray(
            training._resolve_session_task_distribution(
                config,
                _BrokenLayoutTeacher(),
                4,
                ["a", "b", "c", "d", "original_craftax"],
            ),
            dtype=np.float64,
        )
        legacy = np.asarray(
            training._calculate_task_distribution(config, 4), dtype=np.float64
        )
        assert np.array_equal(dist, legacy)
